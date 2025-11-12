"""
Ingestion Management CLI

Command-line interface for managing the automatic data ingestion system.

Commands:
- start: Start all ingestion services
- stop: Stop all services gracefully
- status: Show system status and statistics
- ingest: Manually trigger ingestion for a file
- validate: Run consistency validation
- rollback: Rollback a collection to a previous version
- stats: Show detailed statistics
- list: List files, collections, or versions

Usage:
    python scripts/manage_ingestion.py start
    python scripts/manage_ingestion.py status
    python scripts/manage_ingestion.py ingest --path data/academic/file.pdf --collection academic_knowledge
    python scripts/manage_ingestion.py validate --collection academic_knowledge
    python scripts/manage_ingestion.py rollback --collection academic_knowledge --version <version_id>

Author: Career Planning System
Created: 2025
"""

import asyncio
import click
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.service_manager import ServiceManager
from services.ingestion_registry import IngestionRegistry
from services.scheduled_validator import ScheduledValidator
from services.ingestion_orchestrator import IngestionOrchestrator, TaskPriority
from services.metrics_collector import MetricsCollector
from rag.version_manager import VersionManager


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("manage_ingestion")


@click.group()
def cli():
    """Career Planning Automatic Data Ingestion Management Tool"""
    pass


@cli.command()
@click.option(
    '--config',
    default='config/ingestion_config.yaml',
    help='Path to configuration file'
)
def start(config):
    """Start all ingestion services"""
    click.echo("=" * 60)
    click.echo("Starting Career Planning Ingestion System")
    click.echo("=" * 60)

    async def _start():
        manager = ServiceManager(config_path=config)
        await manager.start()
        await manager.run_until_stopped()

    try:
        asyncio.run(_start())
    except KeyboardInterrupt:
        click.echo("\nShutdown complete.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show system status"""
    click.echo("System Status:")
    click.echo("-" * 60)

    async def _status():
        # Initialize registry to check status
        registry = IngestionRegistry()
        await registry.initialize()

        # Get statistics
        stats = await registry.get_statistics()

        click.echo(f"\nRegistry Statistics:")
        click.echo(f"  Total Files: {stats['total_files']}")
        click.echo(f"  Total Chunks: {stats['total_chunks']}")
        click.echo(f"  Total Size: {stats['total_size_mb']} MB")
        click.echo(f"  Avg Ingestion Time: {stats['avg_ingestion_duration_seconds']:.2f}s")

        click.echo(f"\nStatus by Status:")
        for status, count in stats.get('status_counts', {}).items():
            click.echo(f"  {status}: {count}")

        # Load metrics if available
        metrics_file = Path("logs/metrics.json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                click.echo(f"\nLast Metrics Update: {metrics_data.get('last_updated', 'N/A')}")

        await registry.close()

    try:
        asyncio.run(_status())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--path', required=True, help='Path to file to ingest')
@click.option('--collection', required=True, help='Collection name')
@click.option('--priority', default='high', type=click.Choice(['low', 'normal', 'high']))
@click.option('--config', default='config/ingestion_config.yaml')
def ingest(path, collection, priority, config):
    """Manually trigger ingestion for a specific file"""
    click.echo(f"Ingesting: {path} → {collection}")

    file_path = Path(path)
    if not file_path.exists():
        click.echo(f"Error: File not found: {path}", err=True)
        sys.exit(1)

    async def _ingest():
        # Initialize components
        registry = IngestionRegistry()
        await registry.initialize()

        metrics_collector = MetricsCollector()
        metrics_collector.start()

        orchestrator = IngestionOrchestrator(
            config_path=config,
            registry=registry,
            metrics_collector=metrics_collector
        )

        # Start orchestrator
        orchestrator.start()

        # Map priority
        priority_map = {
            'low': TaskPriority.LOW,
            'normal': TaskPriority.NORMAL,
            'high': TaskPriority.HIGH
        }

        # Queue task
        await orchestrator.queue_manual_ingestion(
            str(file_path),
            collection,
            priority_map[priority]
        )

        click.echo(f"Task queued with priority: {priority}")
        click.echo("Processing...")

        # Wait for completion (with timeout)
        timeout = 300  # 5 minutes
        start_time = asyncio.get_event_loop().time()

        while True:
            await asyncio.sleep(1)

            stats = orchestrator.get_stats()

            if stats['queue_size'] == 0 and stats['in_progress'] == 0:
                break

            if asyncio.get_event_loop().time() - start_time > timeout:
                click.echo("Timeout waiting for ingestion to complete")
                break

        # Stop services
        await orchestrator.stop()
        await metrics_collector.stop()
        await registry.close()

        # Show final stats
        stats = orchestrator.get_stats()
        click.echo(f"\n✓ Ingestion complete:")
        click.echo(f"  Chunks created: {stats['total_chunks_created']}")
        click.echo(f"  Processing time: {stats['avg_processing_time']:.2f}s")

    try:
        asyncio.run(_ingest())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--collection', help='Collection to validate (default: all)')
@click.option('--auto-fix/--no-auto-fix', default=True, help='Automatically fix issues')
@click.option('--config', default='config/ingestion_config.yaml')
def validate(collection, auto_fix, config):
    """Run consistency validation"""
    click.echo("Running validation...")

    async def _validate():
        # Initialize components
        registry = IngestionRegistry()
        await registry.initialize()

        validator = ScheduledValidator(
            config_path=config,
            registry=registry
        )

        # Run validation
        collections = [collection] if collection else None
        report = await validator.run_validation(collections=collections, auto_fix=auto_fix)

        # Display report
        click.echo("\nValidation Report:")
        click.echo("-" * 60)
        click.echo(f"Status: {report.status}")
        click.echo(f"Collections: {', '.join(report.collections_validated)}")
        click.echo(f"Files checked: {report.total_files_checked}")
        click.echo(f"Issues found: {len(report.issues_found)}")
        click.echo(f"Issues fixed: {report.issues_fixed}")

        if report.issues_found:
            click.echo("\nIssues:")
            for issue in report.issues_found[:10]:  # Show first 10
                click.echo(f"  [{issue.severity}] {issue.issue_type.value}: {Path(issue.file_path).name}")

            if len(report.issues_found) > 10:
                click.echo(f"  ... and {len(report.issues_found) - 10} more")

        await registry.close()

    try:
        asyncio.run(_validate())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--collection', required=True, help='Collection name')
@click.option('--version', help='Version ID to rollback to (leave empty to list versions)')
@click.option('--backup/--no-backup', default=True, help='Create backup before rollback')
def rollback(collection, version, backup):
    """Rollback a collection to a previous version"""

    async def _rollback():
        version_manager = VersionManager()

        if not version:
            # List versions
            click.echo(f"\nAvailable versions for '{collection}':")
            click.echo("-" * 60)

            versions = await version_manager.list_versions(collection)

            if not versions:
                click.echo("No versions found")
                return

            for v in versions:
                click.echo(f"\nVersion: {v.version_id}")
                click.echo(f"  Timestamp: {v.timestamp}")
                click.echo(f"  Chunks: {v.chunk_count}")
                click.echo(f"  Documents: {v.document_count}")
                click.echo(f"  Size: {v.snapshot_size_bytes / 1024 / 1024:.2f} MB")
                if v.description:
                    click.echo(f"  Description: {v.description}")

            click.echo(f"\nTo rollback, run:")
            click.echo(f"  python scripts/manage_ingestion.py rollback --collection {collection} --version <version_id>")
            return

        # Perform rollback
        click.echo(f"Rolling back '{collection}' to version '{version}'...")

        if backup:
            click.echo("Creating backup of current state...")

        success = await version_manager.rollback(collection, version, create_backup=backup)

        if success:
            click.echo("✓ Rollback completed successfully")
        else:
            click.echo("✗ Rollback failed", err=True)
            sys.exit(1)

    try:
        asyncio.run(_rollback())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--collection', help='Filter by collection')
@click.option('--format', 'output_format', default='text', type=click.Choice(['text', 'json']))
def stats(collection, output_format):
    """Show detailed statistics"""

    async def _stats():
        registry = IngestionRegistry()
        await registry.initialize()

        stats_data = await registry.get_statistics(collection_name=collection)

        if output_format == 'json':
            click.echo(json.dumps(stats_data, indent=2))
        else:
            click.echo("\nIngestion Statistics:")
            click.echo("=" * 60)
            if collection:
                click.echo(f"Collection: {collection}")
            click.echo(f"Total Files: {stats_data['total_files']}")
            click.echo(f"Total Chunks: {stats_data['total_chunks']}")
            click.echo(f"Total Size: {stats_data['total_size_mb']} MB")
            click.echo(f"Avg Ingestion Duration: {stats_data['avg_ingestion_duration_seconds']:.2f}s")

            click.echo("\nStatus Breakdown:")
            for status, count in stats_data.get('status_counts', {}).items():
                click.echo(f"  {status}: {count}")

        await registry.close()

    try:
        asyncio.run(_stats())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('list_type', type=click.Choice(['files', 'collections', 'versions']))
@click.option('--collection', help='Collection name (for files and versions)')
@click.option('--limit', default=20, help='Maximum items to display')
def list(list_type, collection, limit):
    """List files, collections, or versions"""

    async def _list():
        if list_type == 'files':
            registry = IngestionRegistry()
            await registry.initialize()

            files = await registry.get_all_files(collection_name=collection)

            click.echo(f"\nFiles ({len(files)} total):")
            click.echo("-" * 80)

            for file_record in files[:limit]:
                click.echo(f"\n{Path(file_record.file_path).name}")
                click.echo(f"  Collection: {file_record.collection_name}")
                click.echo(f"  Type: {file_record.file_type}")
                click.echo(f"  Chunks: {file_record.chunk_count}")
                click.echo(f"  Status: {file_record.status}")
                click.echo(f"  Last Ingested: {file_record.last_ingested}")

            if len(files) > limit:
                click.echo(f"\n... and {len(files) - limit} more")

            await registry.close()

        elif list_type == 'versions':
            if not collection:
                click.echo("Error: --collection required for listing versions", err=True)
                sys.exit(1)

            version_manager = VersionManager()
            versions = await version_manager.list_versions(collection)

            click.echo(f"\nVersions for '{collection}' ({len(versions)} total):")
            click.echo("-" * 80)

            for v in versions[:limit]:
                click.echo(f"\n{v.version_id}")
                click.echo(f"  Timestamp: {v.timestamp}")
                click.echo(f"  Chunks: {v.chunk_count}")
                click.echo(f"  Size: {v.snapshot_size_bytes / 1024 / 1024:.2f} MB")

            if len(versions) > limit:
                click.echo(f"\n... and {len(versions) - limit} more")

        elif list_type == 'collections':
            # Get unique collections from registry
            registry = IngestionRegistry()
            await registry.initialize()

            files = await registry.get_all_files()
            collections = set(f.collection_name for f in files)

            click.echo(f"\nCollections ({len(collections)} total):")
            click.echo("-" * 60)

            for coll in sorted(collections):
                # Get stats for this collection
                stats = await registry.get_statistics(collection_name=coll)
                click.echo(f"\n{coll}:")
                click.echo(f"  Files: {stats['total_files']}")
                click.echo(f"  Chunks: {stats['total_chunks']}")
                click.echo(f"  Size: {stats['total_size_mb']} MB")

            await registry.close()

    try:
        asyncio.run(_list())
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def version():
    """Show version information"""
    click.echo("Career Planning Ingestion System")
    click.echo("Version: 1.0.0")
    click.echo("Author: Career Planning Team")


if __name__ == '__main__':
    cli()
