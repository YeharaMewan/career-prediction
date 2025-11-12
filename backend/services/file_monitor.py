"""
File Monitor Service

Real-time file system monitoring using Watchdog.
Detects file changes and triggers ingestion pipeline.

Features:
- Real-time file system event monitoring
- Debouncing to handle rapid changes
- Pattern-based filtering (*.pdf, *.docx, etc.)
- Event queue for batch processing
- Graceful shutdown support
- Integration with ingestion orchestrator

Author: Career Planning System
Created: 2025
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import yaml

from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileSystemEvent,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent
)


class FileEventType(Enum):
    """File system event types"""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileChangeEvent:
    """Represents a file change event"""
    event_type: FileEventType
    file_path: str
    collection_name: str
    timestamp: datetime
    old_path: Optional[str] = None  # For move events

    def __hash__(self):
        return hash((self.file_path, self.collection_name))

    def __eq__(self, other):
        return (
            isinstance(other, FileChangeEvent) and
            self.file_path == other.file_path and
            self.collection_name == other.collection_name
        )


class IngestionEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler for file system changes.
    Filters events and queues them for processing.
    """

    def __init__(
        self,
        collection_name: str,
        file_patterns: List[str],
        ignore_patterns: List[str],
        event_callback: Callable
    ):
        """
        Initialize event handler

        Args:
            collection_name: Vector DB collection name
            file_patterns: List of patterns to watch (e.g., ["*.pdf", "*.docx"])
            ignore_patterns: List of patterns to ignore (e.g., ["*.tmp"])
            event_callback: Async callback function for events
        """
        super().__init__()
        self.collection_name = collection_name
        self.file_patterns = file_patterns
        self.ignore_patterns = ignore_patterns
        self.event_callback = event_callback
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{collection_name}")

    def _matches_pattern(self, file_path: str) -> bool:
        """Check if file matches watch patterns and not ignore patterns"""
        path = Path(file_path)

        # Check ignore patterns first
        for pattern in self.ignore_patterns:
            if path.match(pattern):
                return False

        # Check if matches any watch pattern
        for pattern in self.file_patterns:
            if path.match(pattern):
                return True

        return False

    def _should_process(self, event: FileSystemEvent) -> bool:
        """Determine if event should be processed"""
        # Ignore directories
        if event.is_directory:
            return False

        # Check pattern match
        return self._matches_pattern(event.src_path)

    def on_created(self, event: FileCreatedEvent):
        """Handle file creation"""
        if self._should_process(event):
            self.logger.info(f"File created: {event.src_path}")
            change_event = FileChangeEvent(
                event_type=FileEventType.CREATED,
                file_path=event.src_path,
                collection_name=self.collection_name,
                timestamp=datetime.now()
            )
            # Schedule callback in event loop
            asyncio.create_task(self.event_callback(change_event))

    def on_modified(self, event: FileModifiedEvent):
        """Handle file modification"""
        if self._should_process(event):
            self.logger.info(f"File modified: {event.src_path}")
            change_event = FileChangeEvent(
                event_type=FileEventType.MODIFIED,
                file_path=event.src_path,
                collection_name=self.collection_name,
                timestamp=datetime.now()
            )
            asyncio.create_task(self.event_callback(change_event))

    def on_deleted(self, event: FileDeletedEvent):
        """Handle file deletion"""
        if self._should_process(event):
            self.logger.info(f"File deleted: {event.src_path}")
            change_event = FileChangeEvent(
                event_type=FileEventType.DELETED,
                file_path=event.src_path,
                collection_name=self.collection_name,
                timestamp=datetime.now()
            )
            asyncio.create_task(self.event_callback(change_event))

    def on_moved(self, event: FileMovedEvent):
        """Handle file move/rename"""
        if self._should_process(event):
            self.logger.info(f"File moved: {event.src_path} -> {event.dest_path}")
            change_event = FileChangeEvent(
                event_type=FileEventType.MOVED,
                file_path=event.dest_path,
                collection_name=self.collection_name,
                timestamp=datetime.now(),
                old_path=event.src_path
            )
            asyncio.create_task(self.event_callback(change_event))


class FileMonitor:
    """
    File system monitoring service with debouncing and event queuing.

    Features:
    - Real-time monitoring via Watchdog
    - Debouncing to handle rapid file changes
    - Event queue for batch processing
    - Multiple directory monitoring
    - Pattern-based filtering
    - Graceful shutdown
    """

    def __init__(
        self,
        config_path: str = "config/ingestion_config.yaml",
        debounce_seconds: int = 5,
        max_queue_size: int = 1000
    ):
        """
        Initialize File Monitor

        Args:
            config_path: Path to ingestion configuration
            debounce_seconds: Wait time after last change before processing
            max_queue_size: Maximum size of event queue
        """
        self.config_path = Path(config_path)
        self.debounce_seconds = debounce_seconds
        self.max_queue_size = max_queue_size

        # Load configuration
        self.config = self._load_config()

        # Event tracking
        self.pending_events: Dict[str, FileChangeEvent] = {}  # file_path -> event
        self.event_timestamps: Dict[str, datetime] = {}  # file_path -> last change time
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)

        # Watchdog observers
        self.observers: List[Observer] = []

        # State
        self.running = False
        self.debounce_task: Optional[asyncio.Task] = None

        # Callback for processed events
        self.event_callback: Optional[Callable] = None

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            self.logger.warning(f"Config not found: {self.config_path}, using defaults")
            return {"monitoring": {"enabled": True, "watch_directories": []}}

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def set_event_callback(self, callback: Callable):
        """
        Set callback function for processed events.

        Callback signature: async def callback(event: FileChangeEvent) -> None
        """
        self.event_callback = callback

    async def _handle_event(self, event: FileChangeEvent):
        """
        Handle incoming file system event with debouncing.

        Args:
            event: File change event
        """
        file_key = f"{event.file_path}:{event.collection_name}"

        # Update pending events
        self.pending_events[file_key] = event
        self.event_timestamps[file_key] = datetime.now()

        self.logger.debug(
            f"Event queued: {event.event_type.value} - {event.file_path}"
        )

    async def _debounce_worker(self):
        """
        Background worker that processes debounced events.
        Waits for debounce period after last change before triggering ingestion.
        """
        while self.running:
            try:
                await asyncio.sleep(1)  # Check every second

                now = datetime.now()
                ready_events = []

                # Find events ready for processing (no changes for debounce_seconds)
                for file_key, last_change in list(self.event_timestamps.items()):
                    if (now - last_change).total_seconds() >= self.debounce_seconds:
                        if file_key in self.pending_events:
                            ready_events.append((file_key, self.pending_events[file_key]))

                # Process ready events
                for file_key, event in ready_events:
                    # Remove from pending
                    del self.pending_events[file_key]
                    del self.event_timestamps[file_key]

                    # Add to processing queue
                    try:
                        await asyncio.wait_for(
                            self.event_queue.put(event),
                            timeout=5.0
                        )
                        self.logger.info(
                            f"Event ready for processing: {event.event_type.value} - "
                            f"{Path(event.file_path).name}"
                        )
                    except asyncio.TimeoutError:
                        self.logger.error("Event queue full, dropping event")

            except Exception as e:
                self.logger.error(f"Debounce worker error: {str(e)}")

    async def _event_processor(self):
        """
        Background worker that processes events from the queue.
        Calls the registered callback for each event.
        """
        while self.running:
            try:
                # Get event from queue (with timeout for graceful shutdown)
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Call user callback
                if self.event_callback:
                    try:
                        await self.event_callback(event)
                    except Exception as e:
                        self.logger.error(
                            f"Event callback error for {event.file_path}: {str(e)}"
                        )

                # Mark as done
                self.event_queue.task_done()

            except Exception as e:
                self.logger.error(f"Event processor error: {str(e)}")

    def start(self):
        """Start file monitoring"""
        if self.running:
            self.logger.warning("File monitor already running")
            return

        monitoring_config = self.config.get("monitoring", {})

        if not monitoring_config.get("enabled", True):
            self.logger.info("File monitoring is disabled in config")
            return

        watch_directories = monitoring_config.get("watch_directories", [])

        if not watch_directories:
            self.logger.warning("No directories configured for monitoring")
            return

        self.running = True

        # Start observers for each directory
        for dir_config in watch_directories:
            path = Path(dir_config["path"])
            collection_name = dir_config["collection"]
            file_patterns = dir_config.get("file_patterns", ["*.*"])
            recursive = dir_config.get("recursive", True)

            # Get ignore patterns from config
            ignore_patterns = monitoring_config.get("ignore_patterns", [
                "*.tmp", "*.swp", "*~", ".DS_Store", "Thumbs.db"
            ])

            # Create directory if it doesn't exist
            path.mkdir(parents=True, exist_ok=True)

            # Create event handler
            handler = IngestionEventHandler(
                collection_name=collection_name,
                file_patterns=file_patterns,
                ignore_patterns=ignore_patterns,
                event_callback=self._handle_event
            )

            # Create and start observer
            observer = Observer()
            observer.schedule(handler, str(path), recursive=recursive)
            observer.start()

            self.observers.append(observer)

            self.logger.info(
                f"Monitoring started: {path} -> {collection_name} "
                f"(patterns: {file_patterns})"
            )

        # Start debounce worker
        self.debounce_task = asyncio.create_task(self._debounce_worker())

        # Start event processor
        self.processor_task = asyncio.create_task(self._event_processor())

        self.logger.info(
            f"File monitor started (debounce: {self.debounce_seconds}s, "
            f"watching {len(self.observers)} directories)"
        )

    async def stop(self, timeout: int = 30):
        """
        Stop file monitoring gracefully.

        Args:
            timeout: Maximum seconds to wait for pending events
        """
        if not self.running:
            return

        self.logger.info("Stopping file monitor...")
        self.running = False

        # Stop watchdog observers
        for observer in self.observers:
            observer.stop()
            observer.join(timeout=5)

        self.observers.clear()

        # Wait for pending events to be processed
        if not self.event_queue.empty():
            self.logger.info(
                f"Waiting for {self.event_queue.qsize()} pending events..."
            )
            try:
                await asyncio.wait_for(
                    self.event_queue.join(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for pending events")

        # Cancel background tasks
        if self.debounce_task:
            self.debounce_task.cancel()
            try:
                await self.debounce_task
            except asyncio.CancelledError:
                pass

        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("File monitor stopped")

    def get_stats(self) -> Dict:
        """Get monitoring statistics"""
        return {
            "running": self.running,
            "observers_count": len(self.observers),
            "pending_events": len(self.pending_events),
            "queue_size": self.event_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "debounce_seconds": self.debounce_seconds
        }


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

async def example_callback(event: FileChangeEvent):
    """Example callback for file change events"""
    print(f"Processing: {event.event_type.value} - {event.file_path}")
    # Here you would trigger the ingestion orchestrator
    await asyncio.sleep(0.1)  # Simulate processing


async def main():
    """Example usage of FileMonitor"""

    # Create file monitor
    monitor = FileMonitor(
        config_path="config/ingestion_config.yaml",
        debounce_seconds=5
    )

    # Set callback
    monitor.set_event_callback(example_callback)

    # Start monitoring
    monitor.start()

    # Run for a while
    print("Monitoring files... (Press Ctrl+C to stop)")
    try:
        while True:
            await asyncio.sleep(5)
            stats = monitor.get_stats()
            print(f"Stats: {stats}")
    except KeyboardInterrupt:
        print("\nShutting down...")

    # Stop gracefully
    await monitor.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
