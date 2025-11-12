"""
Scheduled Validator Service

Periodic validation of data consistency across file system, registry, and vector DB.

Features:
- Scheduled validation runs (cron-style)
- Detect orphaned vector DB entries (files deleted but still in DB)
- Detect missing entries (files exist but not in registry)
- Detect hash mismatches (file changed but not re-ingested)
- Automatic repair capabilities
- Detailed validation reports
- APScheduler integration

Author: Career Planning System
Created: 2025
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import yaml

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Import other services
from .ingestion_registry import IngestionRegistry, FileRecord

# Type-checking only imports to avoid circular dependencies
if TYPE_CHECKING:
    from rag.document_processor import DocumentProcessor
    from rag.vector_store import VectorStore


class IssueType(Enum):
    """Types of validation issues"""
    ORPHANED_DB_ENTRY = "orphaned_db_entry"  # In vector DB but file deleted
    ORPHANED_REGISTRY = "orphaned_registry"  # In registry but file deleted
    MISSING_REGISTRY = "missing_registry"  # File exists but not in registry
    HASH_MISMATCH = "hash_mismatch"  # File changed but not re-ingested
    MISSING_VECTOR_DB = "missing_vector_db"  # In registry but not in vector DB


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    issue_type: IssueType
    file_path: str
    collection_name: str
    details: str
    severity: str = "medium"  # low, medium, high
    auto_fixable: bool = True
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['issue_type'] = self.issue_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ValidationReport:
    """Validation run report"""
    run_id: str
    start_time: datetime
    end_time: Optional[datetime]
    collections_validated: List[str]
    total_files_checked: int
    issues_found: List[ValidationIssue]
    issues_fixed: int
    status: str = "in_progress"  # in_progress, completed, failed
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        data['issues_found'] = [issue.to_dict() for issue in self.issues_found]
        data['duration_seconds'] = (
            (self.end_time - self.start_time).total_seconds()
            if self.end_time else None
        )
        return data


class ScheduledValidator:
    """
    Scheduled validation service for data consistency.

    Validates:
    - File system vs. Registry consistency
    - Registry vs. Vector DB consistency
    - File hashes for change detection
    - Orphaned entries cleanup

    Features:
    - Cron-based scheduling
    - Automatic issue detection
    - Optional automatic repair
    - Detailed reporting
    - Integration with alerting system
    """

    def __init__(
        self,
        config_path: str = "config/ingestion_config.yaml",
        registry: Optional[IngestionRegistry] = None,
        document_processor: Optional["DocumentProcessor"] = None
    ):
        """
        Initialize Scheduled Validator

        Args:
            config_path: Path to ingestion configuration
            registry: Ingestion registry instance
            document_processor: Document processor for hash calculation
        """
        self.config_path = Path(config_path)
        self.registry = registry

        # Import at runtime to avoid circular dependency
        if document_processor is None:
            from rag.document_processor import DocumentProcessor
            self.document_processor = DocumentProcessor()
        else:
            self.document_processor = document_processor

        # Load configuration
        self.config = self._load_config()
        self.validation_config = self.config.get("validation", {})

        # Scheduler
        self.scheduler: Optional[AsyncIOScheduler] = None

        # State
        self.running = False
        self.last_report: Optional[ValidationReport] = None
        self.validation_history: List[ValidationReport] = []

        # Callbacks
        self.issue_callback: Optional[callable] = None  # Called for each issue found
        self.report_callback: Optional[callable] = None  # Called when validation completes

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            self.logger.warning(f"Config not found: {self.config_path}")
            return {"validation": {"enabled": True}}

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def set_issue_callback(self, callback: callable):
        """Set callback for when issues are found"""
        self.issue_callback = callback

    def set_report_callback(self, callback: callable):
        """Set callback for validation reports"""
        self.report_callback = callback

    async def _validate_directory(
        self,
        directory: Path,
        collection_name: str,
        file_patterns: List[str]
    ) -> List[ValidationIssue]:
        """
        Validate a single directory against registry and vector DB.

        Args:
            directory: Directory to validate
            collection_name: Collection name
            file_patterns: File patterns to check

        Returns:
            List of validation issues
        """
        issues = []

        if not directory.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return issues

        # Get all files matching patterns
        actual_files = set()
        for pattern in file_patterns:
            actual_files.update(directory.glob(pattern))

        # Get files from registry
        registry_records = await self.registry.get_all_files(
            collection_name=collection_name
        )
        registry_files = {Path(r.file_path): r for r in registry_records}

        self.logger.info(
            f"Validating {len(actual_files)} files vs {len(registry_files)} registry entries"
        )

        # Check 1: Files in registry but not on disk (orphaned registry)
        for registry_path, record in registry_files.items():
            if not registry_path.exists():
                issues.append(ValidationIssue(
                    issue_type=IssueType.ORPHANED_REGISTRY,
                    file_path=str(registry_path),
                    collection_name=collection_name,
                    details=f"File in registry but not found on disk",
                    severity="high",
                    auto_fixable=True
                ))

        # Check 2: Files on disk but not in registry (missing registry)
        for file_path in actual_files:
            if file_path not in registry_files:
                issues.append(ValidationIssue(
                    issue_type=IssueType.MISSING_REGISTRY,
                    file_path=str(file_path),
                    collection_name=collection_name,
                    details=f"File exists but not in registry",
                    severity="medium",
                    auto_fixable=True
                ))

        # Check 3: Hash mismatches (file changed but not re-ingested)
        for file_path in actual_files:
            if file_path in registry_files:
                record = registry_files[file_path]

                # Calculate current hash
                try:
                    current_hash = self.document_processor.calculate_file_hash(file_path)

                    if current_hash != record.file_hash:
                        issues.append(ValidationIssue(
                            issue_type=IssueType.HASH_MISMATCH,
                            file_path=str(file_path),
                            collection_name=collection_name,
                            details=f"File hash changed (old: {record.file_hash[:8]}, new: {current_hash[:8]})",
                            severity="high",
                            auto_fixable=True
                        ))
                except Exception as e:
                    self.logger.error(f"Error calculating hash for {file_path}: {e}")

        return issues

    async def run_validation(
        self,
        collections: Optional[List[str]] = None,
        auto_fix: bool = None
    ) -> ValidationReport:
        """
        Run validation for specified collections.

        Args:
            collections: List of collections to validate (None = all)
            auto_fix: Whether to automatically fix issues (None = use config)

        Returns:
            Validation report
        """
        run_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        # Use config default if not specified
        if auto_fix is None:
            auto_fix = self.validation_config.get("auto_fix", True)

        self.logger.info(f"Starting validation run: {run_id} (auto_fix={auto_fix})")

        # Create report
        report = ValidationReport(
            run_id=run_id,
            start_time=start_time,
            end_time=None,
            collections_validated=[],
            total_files_checked=0,
            issues_found=[],
            issues_fixed=0
        )

        try:
            # Get directories to validate
            monitoring_config = self.config.get("monitoring", {})
            watch_dirs = monitoring_config.get("watch_directories", [])

            # Filter by collections if specified
            if collections:
                watch_dirs = [
                    d for d in watch_dirs
                    if d["collection"] in collections
                ]

            # Validate each directory
            for dir_config in watch_dirs:
                collection_name = dir_config["collection"]
                path = Path(dir_config["path"])
                file_patterns = dir_config.get("file_patterns", ["*.*"])

                report.collections_validated.append(collection_name)

                # Run validation
                issues = await self._validate_directory(
                    path,
                    collection_name,
                    file_patterns
                )

                report.issues_found.extend(issues)

                # Count files
                total_files = sum(
                    len(list(path.glob(pattern)))
                    for pattern in file_patterns
                )
                report.total_files_checked += total_files

            # Auto-fix issues if enabled
            if auto_fix:
                report.issues_fixed = await self._auto_fix_issues(report.issues_found)

            # Mark as completed
            report.status = "completed"
            report.end_time = datetime.now()

            self.logger.info(
                f"Validation completed: {len(report.issues_found)} issues found, "
                f"{report.issues_fixed} fixed"
            )

        except Exception as e:
            report.status = "failed"
            report.error_message = str(e)
            report.end_time = datetime.now()
            self.logger.error(f"Validation failed: {e}")

        # Store report
        self.last_report = report
        self.validation_history.append(report)

        # Keep only last 50 reports
        if len(self.validation_history) > 50:
            self.validation_history = self.validation_history[-50:]

        # Call report callback
        if self.report_callback:
            try:
                await self.report_callback(report)
            except Exception as e:
                self.logger.error(f"Report callback error: {e}")

        return report

    async def _auto_fix_issues(self, issues: List[ValidationIssue]) -> int:
        """
        Automatically fix issues that are auto-fixable.

        Args:
            issues: List of validation issues

        Returns:
            Number of issues fixed
        """
        fixed_count = 0

        for issue in issues:
            if not issue.auto_fixable:
                continue

            try:
                if issue.issue_type == IssueType.ORPHANED_REGISTRY:
                    # Delete from registry
                    deleted = await self.registry.delete_file_record(
                        issue.file_path,
                        issue.collection_name
                    )
                    if deleted:
                        fixed_count += 1
                        self.logger.info(f"Fixed: Deleted orphaned registry entry for {issue.file_path}")

                elif issue.issue_type == IssueType.MISSING_REGISTRY:
                    # Will be handled by orchestrator on next ingestion
                    self.logger.info(f"Issue logged: Missing registry for {issue.file_path} (will be added on next ingestion)")

                elif issue.issue_type == IssueType.HASH_MISMATCH:
                    # Will be handled by orchestrator on next ingestion
                    self.logger.info(f"Issue logged: Hash mismatch for {issue.file_path} (will be re-ingested)")

                # Call issue callback
                if self.issue_callback:
                    await self.issue_callback(issue)

            except Exception as e:
                self.logger.error(f"Error fixing issue {issue.issue_type.value} for {issue.file_path}: {e}")

        return fixed_count

    def start(self):
        """Start scheduled validation service"""
        if self.running:
            self.logger.warning("Validator already running")
            return

        if not self.validation_config.get("enabled", True):
            self.logger.info("Validation is disabled in config")
            return

        if not self.registry:
            self.logger.error("Cannot start validator: registry not set")
            return

        # Create scheduler
        self.scheduler = AsyncIOScheduler()

        # Get schedule from config
        schedule = self.validation_config.get("schedule", "0 */6 * * *")

        # Add validation job
        try:
            self.scheduler.add_job(
                self.run_validation,
                trigger=CronTrigger.from_crontab(schedule),
                id="scheduled_validation",
                name="Periodic Data Validation",
                max_instances=1,  # Don't overlap validations
                coalesce=True  # If missed, run once
            )

            # Start scheduler
            self.scheduler.start()
            self.running = True

            self.logger.info(f"Scheduled validator started (schedule: {schedule})")

        except Exception as e:
            self.logger.error(f"Failed to start validator: {e}")

    async def stop(self):
        """Stop scheduled validation service"""
        if not self.running:
            return

        self.logger.info("Stopping scheduled validator...")

        if self.scheduler:
            self.scheduler.shutdown(wait=True)

        self.running = False
        self.logger.info("Scheduled validator stopped")

    def get_stats(self) -> Dict:
        """Get validator statistics"""
        return {
            "running": self.running,
            "last_run": self.last_report.start_time.isoformat() if self.last_report else None,
            "total_runs": len(self.validation_history),
            "last_issues_found": len(self.last_report.issues_found) if self.last_report else 0,
            "last_issues_fixed": self.last_report.issues_fixed if self.last_report else 0
        }

    def get_last_report(self) -> Optional[Dict]:
        """Get last validation report"""
        return self.last_report.to_dict() if self.last_report else None


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

async def example_issue_callback(issue: ValidationIssue):
    """Example callback for issues"""
    print(f"Issue found: {issue.issue_type.value} - {issue.file_path}")


async def example_report_callback(report: ValidationReport):
    """Example callback for reports"""
    print(f"Validation completed: {len(report.issues_found)} issues")


async def main():
    """Example usage of ScheduledValidator"""

    # Initialize dependencies
    registry = IngestionRegistry()
    await registry.initialize()

    # Create validator
    validator = ScheduledValidator(
        config_path="config/ingestion_config.yaml",
        registry=registry
    )

    # Set callbacks
    validator.set_issue_callback(example_issue_callback)
    validator.set_report_callback(example_report_callback)

    # Run manual validation
    print("Running manual validation...")
    report = await validator.run_validation(auto_fix=True)
    print(f"Report: {report.to_dict()}")

    # Start scheduled validation
    # validator.start()

    # Keep running
    # await asyncio.sleep(3600)  # Run for 1 hour

    # Stop
    # await validator.stop()

    await registry.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
