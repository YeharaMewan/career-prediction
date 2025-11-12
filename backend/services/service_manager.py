"""
Service Manager

Central orchestration point for the automatic data ingestion system.
Starts, stops, and manages all services in the correct order.

Services managed:
1. Ingestion Registry (database)
2. Metrics Collector (statistics)
3. File Monitor (watch directories)
4. Scheduled Validator (periodic checks)
5. Ingestion Orchestrator (process files)

Features:
- Ordered startup/shutdown
- Service health monitoring
- Graceful error handling
- Connection of components
- Status reporting

Author: Career Planning System
Created: 2025
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from enum import Enum
import signal
import yaml

from .ingestion_registry import IngestionRegistry
from .metrics_collector import MetricsCollector
from .file_monitor import FileMonitor, FileChangeEvent
from .scheduled_validator import ScheduledValidator
from .ingestion_orchestrator import IngestionOrchestrator


class ServiceStatus(Enum):
    """Status of a service"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class ServiceManager:
    """
    Manages the lifecycle of all ingestion services.

    Responsibilities:
    - Initialize all components in correct order
    - Connect services together (e.g., file monitor → orchestrator)
    - Start services with proper dependencies
    - Monitor service health
    - Graceful shutdown
    - Error recovery
    """

    def __init__(self, config_path: str = "config/ingestion_config.yaml"):
        """
        Initialize Service Manager

        Args:
            config_path: Path to ingestion configuration
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Service instances
        self.registry: Optional[IngestionRegistry] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.file_monitor: Optional[FileMonitor] = None
        self.scheduled_validator: Optional[ScheduledValidator] = None
        self.orchestrator: Optional[IngestionOrchestrator] = None

        # Service status
        self.status: Dict[str, ServiceStatus] = {
            "registry": ServiceStatus.STOPPED,
            "metrics_collector": ServiceStatus.STOPPED,
            "file_monitor": ServiceStatus.STOPPED,
            "scheduled_validator": ServiceStatus.STOPPED,
            "orchestrator": ServiceStatus.STOPPED
        }

        # Overall system status
        self.running = False
        self.start_time: Optional[datetime] = None

        # Shutdown event
        self.shutdown_event = asyncio.Event()

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            self.logger.warning(f"Config not found: {self.config_path}")
            return {}

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    async def _initialize_services(self):
        """Initialize all service instances"""
        self.logger.info("Initializing services...")

        try:
            # 1. Initialize Registry (database)
            self.status["registry"] = ServiceStatus.STARTING
            self.logger.info("Initializing ingestion registry...")

            registry_config = self.config.get("registry", {})
            self.registry = IngestionRegistry(
                db_path=registry_config.get("database_path", "knowledge_base/ingestion_registry.db"),
                track_history=registry_config.get("track_history", True),
                max_history_per_file=registry_config.get("max_history_per_file", 20)
            )
            await self.registry.initialize()
            self.status["registry"] = ServiceStatus.RUNNING
            self.logger.info("✓ Registry initialized")

            # 2. Initialize Metrics Collector
            self.status["metrics_collector"] = ServiceStatus.STARTING
            self.logger.info("Initializing metrics collector...")

            metrics_config = self.config.get("metrics", {})
            self.metrics_collector = MetricsCollector(
                metrics_file=metrics_config.get("metrics_file", "logs/metrics.json"),
                persist_interval=metrics_config.get("persist_interval", 300)
            )
            self.metrics_collector.start()
            self.status["metrics_collector"] = ServiceStatus.RUNNING
            self.logger.info("✓ Metrics collector initialized")

            # 3. Initialize Ingestion Orchestrator
            self.status["orchestrator"] = ServiceStatus.STARTING
            self.logger.info("Initializing ingestion orchestrator...")

            processing_config = self.config.get("processing", {})
            self.orchestrator = IngestionOrchestrator(
                config_path=str(self.config_path),
                registry=self.registry,
                metrics_collector=self.metrics_collector,
                max_workers=processing_config.get("max_concurrent_workers", 3),
                batch_size=processing_config.get("embedding_batch_size", 50)
            )
            self.orchestrator.start()
            self.status["orchestrator"] = ServiceStatus.RUNNING
            self.logger.info("✓ Orchestrator initialized")

            # 4. Initialize File Monitor
            self.status["file_monitor"] = ServiceStatus.STARTING
            self.logger.info("Initializing file monitor...")

            monitoring_config = self.config.get("monitoring", {})
            self.file_monitor = FileMonitor(
                config_path=str(self.config_path),
                debounce_seconds=monitoring_config.get("debounce_seconds", 5)
            )

            # Connect file monitor to orchestrator
            self.file_monitor.set_event_callback(self.orchestrator.handle_file_event)

            self.file_monitor.start()
            self.status["file_monitor"] = ServiceStatus.RUNNING
            self.logger.info("✓ File monitor initialized")

            # 5. Initialize Scheduled Validator
            self.status["scheduled_validator"] = ServiceStatus.STARTING
            self.logger.info("Initializing scheduled validator...")

            self.scheduled_validator = ScheduledValidator(
                config_path=str(self.config_path),
                registry=self.registry
            )
            self.scheduled_validator.start()
            self.status["scheduled_validator"] = ServiceStatus.RUNNING
            self.logger.info("✓ Scheduled validator initialized")

            self.logger.info("All services initialized successfully!")

        except Exception as e:
            self.logger.error(f"Service initialization failed: {e}")
            raise

    async def start(self):
        """Start all services"""
        if self.running:
            self.logger.warning("Services already running")
            return

        self.logger.info("=" * 60)
        self.logger.info("Starting Career Planning Ingestion System")
        self.logger.info("=" * 60)

        try:
            await self._initialize_services()

            self.running = True
            self.start_time = datetime.now()

            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("✓ All services are running!")
            self.logger.info("=" * 60)
            self.logger.info("")
            self.logger.info("Automatic data ingestion is now active.")
            self.logger.info("Monitoring directories for file changes...")
            self.logger.info("")

            # Print configuration summary
            self._print_configuration_summary()

        except Exception as e:
            self.logger.error(f"Failed to start services: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop all services gracefully"""
        if not self.running:
            return

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Shutting down Career Planning Ingestion System")
        self.logger.info("=" * 60)

        self.running = False

        # Stop services in reverse order
        shutdown_timeout = self.config.get("performance", {}).get("shutdown_timeout", 60)

        # 1. Stop File Monitor (stop receiving new events)
        if self.file_monitor:
            self.logger.info("Stopping file monitor...")
            self.status["file_monitor"] = ServiceStatus.STOPPING
            try:
                await self.file_monitor.stop(timeout=10)
                self.status["file_monitor"] = ServiceStatus.STOPPED
                self.logger.info("✓ File monitor stopped")
            except Exception as e:
                self.logger.error(f"Error stopping file monitor: {e}")
                self.status["file_monitor"] = ServiceStatus.ERROR

        # 2. Stop Scheduled Validator
        if self.scheduled_validator:
            self.logger.info("Stopping scheduled validator...")
            self.status["scheduled_validator"] = ServiceStatus.STOPPING
            try:
                await self.scheduled_validator.stop()
                self.status["scheduled_validator"] = ServiceStatus.STOPPED
                self.logger.info("✓ Scheduled validator stopped")
            except Exception as e:
                self.logger.error(f"Error stopping validator: {e}")
                self.status["scheduled_validator"] = ServiceStatus.ERROR

        # 3. Stop Orchestrator (finish pending tasks)
        if self.orchestrator:
            self.logger.info("Stopping ingestion orchestrator (waiting for pending tasks)...")
            self.status["orchestrator"] = ServiceStatus.STOPPING
            try:
                await self.orchestrator.stop(timeout=shutdown_timeout)
                self.status["orchestrator"] = ServiceStatus.STOPPED
                self.logger.info("✓ Orchestrator stopped")
            except Exception as e:
                self.logger.error(f"Error stopping orchestrator: {e}")
                self.status["orchestrator"] = ServiceStatus.ERROR

        # 4. Stop Metrics Collector
        if self.metrics_collector:
            self.logger.info("Stopping metrics collector...")
            self.status["metrics_collector"] = ServiceStatus.STOPPING
            try:
                await self.metrics_collector.stop()
                self.status["metrics_collector"] = ServiceStatus.STOPPED
                self.logger.info("✓ Metrics collector stopped")
            except Exception as e:
                self.logger.error(f"Error stopping metrics collector: {e}")
                self.status["metrics_collector"] = ServiceStatus.ERROR

        # 5. Close Registry
        if self.registry:
            self.logger.info("Closing ingestion registry...")
            self.status["registry"] = ServiceStatus.STOPPING
            try:
                await self.registry.close()
                self.status["registry"] = ServiceStatus.STOPPED
                self.logger.info("✓ Registry closed")
            except Exception as e:
                self.logger.error(f"Error closing registry: {e}")
                self.status["registry"] = ServiceStatus.ERROR

        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"✓ All services stopped (uptime: {uptime:.0f}s)")
        self.logger.info("=" * 60)

        # Print final statistics
        self._print_final_statistics()

    def _print_configuration_summary(self):
        """Print configuration summary"""
        monitoring_config = self.config.get("monitoring", {})
        watch_dirs = monitoring_config.get("watch_directories", [])

        self.logger.info("Configuration Summary:")
        self.logger.info(f"  Monitored Directories: {len(watch_dirs)}")
        for dir_config in watch_dirs:
            self.logger.info(
                f"    - {dir_config['path']} → {dir_config['collection']} "
                f"({', '.join(dir_config.get('file_patterns', []))})"
            )

        processing_config = self.config.get("processing", {})
        self.logger.info(f"  Concurrent Workers: {processing_config.get('max_concurrent_workers', 3)}")
        self.logger.info(f"  Chunk Size: {processing_config.get('chunk_size', 1000)}")

        validation_config = self.config.get("validation", {})
        if validation_config.get("enabled", True):
            self.logger.info(f"  Validation Schedule: {validation_config.get('schedule', 'Not configured')}")

        self.logger.info("")

    def _print_final_statistics(self):
        """Print final statistics"""
        if self.orchestrator:
            stats = self.orchestrator.get_stats()
            self.logger.info("")
            self.logger.info("Final Statistics:")
            self.logger.info(f"  Tasks Queued: {stats['tasks_queued']}")
            self.logger.info(f"  Tasks Completed: {stats['tasks_completed']}")
            self.logger.info(f"  Tasks Failed: {stats['tasks_failed']}")
            self.logger.info(f"  Total Chunks Created: {stats['total_chunks_created']}")
            self.logger.info(f"  Avg Processing Time: {stats['avg_processing_time']:.2f}s")
            self.logger.info("")

    def get_status(self) -> Dict:
        """Get status of all services"""
        status_dict = {
            "running": self.running,
            "uptime_seconds": (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time else 0
            ),
            "services": {
                name: status.value
                for name, status in self.status.items()
            }
        }

        # Add service-specific stats
        if self.orchestrator:
            status_dict["orchestrator_stats"] = self.orchestrator.get_stats()

        if self.file_monitor:
            status_dict["file_monitor_stats"] = self.file_monitor.get_stats()

        if self.scheduled_validator:
            status_dict["validator_stats"] = self.scheduled_validator.get_stats()

        if self.metrics_collector:
            status_dict["metrics"] = self.metrics_collector.get_all_metrics()

        return status_dict

    async def health_check(self) -> Dict:
        """Perform health check on all services"""
        health = {
            "healthy": True,
            "services": {}
        }

        for service_name, service_status in self.status.items():
            is_healthy = service_status == ServiceStatus.RUNNING
            health["services"][service_name] = {
                "status": service_status.value,
                "healthy": is_healthy
            }
            if not is_healthy:
                health["healthy"] = False

        return health

    async def run_until_stopped(self):
        """Run services until shutdown signal"""
        # Setup signal handlers
        loop = asyncio.get_event_loop()

        def signal_handler():
            self.logger.info("\nReceived shutdown signal...")
            self.shutdown_event.set()

        # Register signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        # Wait for shutdown event
        await self.shutdown_event.wait()

        # Cleanup
        await self.stop()


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

async def main():
    """Example usage of ServiceManager"""

    # Create service manager
    manager = ServiceManager(config_path="config/ingestion_config.yaml")

    # Start all services
    await manager.start()

    # Run until Ctrl+C
    try:
        await manager.run_until_stopped()
    except KeyboardInterrupt:
        print("\nShutdown requested...")

    # Services are automatically stopped by run_until_stopped()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run
    asyncio.run(main())
