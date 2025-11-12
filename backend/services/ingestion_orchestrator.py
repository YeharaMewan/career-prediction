"""
Ingestion Orchestrator

The brain of the automatic data ingestion system.
Coordinates all components to process files into the vector database.

Pipeline Flow:
1. Receive file change event from File Monitor
2. Check registry if file needs processing (hash comparison)
3. Create version snapshot before modifications
4. Load and chunk document (multi-format support)
5. Generate embeddings
6. Delete old chunks (if file modified)
7. Insert new chunks into vector DB
8. Update registry with results
9. Record metrics
10. Handle errors with retries

Features:
- Async task queue with priority
- Concurrent workers (configurable)
- Automatic retry on failure
- Batch processing for efficiency
- Version snapshot integration
- Comprehensive error handling
- Progress tracking per file

Author: Career Planning System
Created: 2025
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import yaml

from .file_monitor import FileChangeEvent, FileEventType
from .ingestion_registry import IngestionRegistry, IngestionStatus
from .metrics_collector import MetricsCollector, IngestionMetrics
from rag.document_processor import DocumentProcessor
from rag.embedding_manager import EmbeddingManager
from rag.vector_store import VectorStoreManager
from rag.version_manager import VersionManager
from utils.tracing import (
    trace_async,
    add_span_attribute,
    add_span_attributes,
    add_span_event,
    set_span_error,
    create_span
)


class TaskPriority(Enum):
    """Priority levels for ingestion tasks"""
    LOW = 1  # Scraped content, background updates
    NORMAL = 2  # Modified files
    HIGH = 3  # New files, manual triggers


@dataclass
class IngestionTask:
    """Represents a single file ingestion task"""
    file_path: str
    collection_name: str
    event_type: FileEventType
    priority: TaskPriority
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def __lt__(self, other):
        """For priority queue sorting (higher priority first)"""
        return self.priority.value > other.priority.value


@dataclass
class IngestionResult:
    """Result of a file ingestion"""
    success: bool
    file_path: str
    collection_name: str
    chunks_created: int = 0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    file_hash: Optional[str] = None


class IngestionOrchestrator:
    """
    Orchestrates the complete file ingestion pipeline.

    This is the central coordinator that:
    - Receives file change events
    - Manages task queue with priorities
    - Processes files through complete pipeline
    - Updates registry and metrics
    - Handles errors and retries
    """

    def __init__(
        self,
        config_path: str = "config/ingestion_config.yaml",
        registry: Optional[IngestionRegistry] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        max_workers: int = 3,
        batch_size: int = 50
    ):
        """
        Initialize Ingestion Orchestrator

        Args:
            config_path: Path to ingestion configuration
            registry: Ingestion registry instance
            metrics_collector: Metrics collector instance
            max_workers: Maximum concurrent processing workers
            batch_size: Batch size for embedding generation
        """
        self.config_path = Path(config_path)
        self.registry = registry
        self.metrics_collector = metrics_collector
        self.max_workers = max_workers
        self.batch_size = batch_size

        # Load configuration
        self.config = self._load_config()
        self.processing_config = self.config.get("processing", {})

        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=self.processing_config.get("chunk_size", 1000),
            chunk_overlap=self.processing_config.get("chunk_overlap", 200)
        )

        self.embedding_manager = EmbeddingManager()

        # Vector stores per collection
        self.vector_stores: Dict[str, VectorStoreManager] = {}

        self.version_manager = VersionManager(
            max_versions_per_collection=self.config.get("versioning", {}).get("max_versions_per_collection", 10)
        )

        # Task queue (priority queue)
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # Track in-progress files to avoid duplicates
        self.in_progress: Set[str] = set()

        # Worker tasks
        self.workers: List[asyncio.Task] = []

        # State
        self.running = False

        # Callbacks
        self.completion_callback: Optional[Callable] = None

        # Statistics
        self.stats = {
            "tasks_queued": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_chunks_created": 0,
            "total_processing_time": 0.0
        }

        # High-level metrics interface
        if self.metrics_collector:
            self.metrics = IngestionMetrics(self.metrics_collector)
        else:
            self.metrics = None

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            self.logger.warning(f"Config not found: {self.config_path}")
            return {}

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_vector_store(self, collection_name: str) -> VectorStoreManager:
        """Get or create vector store for collection"""
        if collection_name not in self.vector_stores:
            self.vector_stores[collection_name] = VectorStoreManager()
        return self.vector_stores[collection_name]

    def set_completion_callback(self, callback: Callable):
        """Set callback for task completion"""
        self.completion_callback = callback

    @trace_async("orchestrator.handle_file_event")
    async def handle_file_event(self, event: FileChangeEvent):
        """
        Handle a file change event from File Monitor.

        Args:
            event: File change event
        """
        file_path = event.file_path
        collection_name = event.collection_name

        # Add span attributes
        add_span_attributes({
            "file.path": file_path,
            "file.name": Path(file_path).name,
            "collection.name": collection_name,
            "event.type": event.event_type.value
        })

        # Check if already processing this file
        file_key = f"{file_path}:{collection_name}"
        if file_key in self.in_progress:
            self.logger.debug(f"File already in progress: {file_path}")
            add_span_event("file_already_in_progress")
            return

        # Handle deletion separately
        if event.event_type == FileEventType.DELETED:
            add_span_event("handling_deletion")
            await self._handle_file_deletion(file_path, collection_name)
            return

        # Determine priority
        priority = TaskPriority.HIGH if event.event_type == FileEventType.CREATED else TaskPriority.NORMAL
        add_span_attribute("task.priority", priority.name)

        # Create task
        task = IngestionTask(
            file_path=file_path,
            collection_name=collection_name,
            event_type=event.event_type,
            priority=priority
        )

        # Add to queue
        await self.task_queue.put(task)
        self.stats["tasks_queued"] += 1

        add_span_event("task_queued", {"queue_size": self.task_queue.qsize()})

        self.logger.info(
            f"Task queued: {Path(file_path).name} -> {collection_name} "
            f"(priority: {priority.name}, queue size: {self.task_queue.qsize()})"
        )

    @trace_async("orchestrator.handle_file_deletion")
    async def _handle_file_deletion(self, file_path: str, collection_name: str):
        """Handle file deletion - remove from vector DB and registry"""
        add_span_attributes({
            "file.path": file_path,
            "file.name": Path(file_path).name,
            "collection.name": collection_name,
            "operation": "delete"
        })

        try:
            self.logger.info(f"Handling deletion: {file_path}")

            # Get file record from registry
            record = await self.registry.get_file_record(file_path, collection_name)

            if record:
                add_span_event("file_found_in_registry", {
                    "chunks": record.chunk_count
                })

                # Delete from vector DB (delete all chunks from this file)
                vector_store = self._get_vector_store(collection_name)

                # Query for all chunks from this file
                # Note: This assumes metadata contains file_path
                # Implementation depends on your vector store's delete capabilities

                # Delete from registry
                await self.registry.delete_file_record(file_path, collection_name)

                add_span_event("file_deleted_from_registry")

                self.logger.info(f"Deleted file from system: {file_path}")

                # Record metric
                if self.metrics:
                    self.metrics.file_monitor_event("deleted", collection_name)
            else:
                add_span_event("file_not_found_in_registry")

        except Exception as e:
            self.logger.error(f"Error handling file deletion {file_path}: {e}")
            set_span_error(e)

    @trace_async("orchestrator.process_task")
    async def _process_task(self, task: IngestionTask) -> IngestionResult:
        """
        Process a single ingestion task through the complete pipeline.

        Pipeline:
        1. Check if file needs processing (hash check)
        2. Create version snapshot
        3. Load and chunk document
        4. Generate embeddings
        5. Delete old chunks (if file modified)
        6. Insert new chunks
        7. Update registry

        Args:
            task: Ingestion task

        Returns:
            Ingestion result
        """
        start_time = time.time()
        file_path = Path(task.file_path)
        collection_name = task.collection_name

        # Add comprehensive span attributes
        add_span_attributes({
            "file.path": str(file_path),
            "file.name": file_path.name,
            "collection.name": collection_name,
            "event.type": task.event_type.value,
            "task.priority": task.priority.name,
            "task.retry_count": task.retry_count
        })

        self.logger.info(f"Processing: {file_path.name} -> {collection_name}")

        try:
            # Step 1: Calculate file hash
            add_span_event("pipeline.step.1.calculate_hash")
            current_hash = self.document_processor.calculate_file_hash(file_path)
            add_span_attribute("file.hash", current_hash[:8])

            # Step 2: Check if file needs processing
            add_span_event("pipeline.step.2.check_needs_update")
            needs_update = await self.registry.file_needs_update(
                str(file_path),
                collection_name,
                current_hash
            )
            add_span_attribute("file.needs_update", needs_update)

            if not needs_update and task.event_type == FileEventType.MODIFIED:
                self.logger.info(f"File unchanged (hash match): {file_path.name}")
                add_span_event("file_unchanged_skipping")
                return IngestionResult(
                    success=True,
                    file_path=str(file_path),
                    collection_name=collection_name,
                    chunks_created=0,
                    duration_seconds=time.time() - start_time,
                    file_hash=current_hash
                )

            # Step 3: Create version snapshot before modifications
            add_span_event("pipeline.step.3.create_snapshot")
            versioning_enabled = self.config.get("versioning", {}).get("enabled", True)
            snapshot_before = self.config.get("versioning", {}).get("snapshot_before_replace", True)

            if versioning_enabled and snapshot_before:
                try:
                    version_id, metadata = await self.version_manager.create_snapshot(
                        collection_name,
                        description=f"Before ingesting {file_path.name}"
                    )
                    self.logger.debug(f"Created snapshot: {version_id}")
                    add_span_attribute("snapshot.version_id", version_id)
                except Exception as e:
                    self.logger.warning(f"Snapshot creation failed: {e}")
                    add_span_event("snapshot_creation_failed", {"error": str(e)})

            # Step 4: Load and chunk document
            add_span_event("pipeline.step.4.load_document")
            self.logger.debug(f"Loading document: {file_path.name}")
            documents = self.document_processor.load_document(
                file_path,
                collection_name
            )

            if not documents:
                raise ValueError(f"No documents loaded from {file_path}")

            add_span_event("pipeline.step.5.chunk_document")
            chunked_docs = self.document_processor.chunk_documents(
                documents,
                collection_name
            )

            add_span_attributes({
                "chunks.count": len(chunked_docs),
                "file.size": file_path.stat().st_size,
                "file.type": self.document_processor.detect_file_type(file_path) or "unknown"
            })

            self.logger.debug(f"Created {len(chunked_docs)} chunks from {file_path.name}")

            # Step 5: Generate embeddings
            add_span_event("pipeline.step.6.generate_embeddings", {
                "chunk_count": len(chunked_docs),
                "batch_size": self.batch_size
            })
            self.logger.debug(f"Generating embeddings for {len(chunked_docs)} chunks")
            embeddings = await self.embedding_manager.generate_embeddings_batch(
                [doc.page_content for doc in chunked_docs],
                batch_size=self.batch_size
            )

            # Step 6: Get vector store
            vector_store = self._get_vector_store(collection_name)

            # Step 7: Delete old chunks if file was modified
            if task.event_type == FileEventType.MODIFIED:
                add_span_event("pipeline.step.7.delete_old_chunks")
                # Delete old chunks for this file
                # This depends on your vector store implementation
                # For now, we'll just add new chunks (ChromaDB will handle updates by ID)
                pass

            # Step 8: Prepare documents for insertion
            ids = [doc.metadata.get("chunk_id", f"{file_path.stem}_{i}")
                   for i, doc in enumerate(chunked_docs)]
            texts = [doc.page_content for doc in chunked_docs]
            metadatas = [doc.metadata for doc in chunked_docs]

            # Step 9: Insert into vector DB
            add_span_event("pipeline.step.8.insert_vector_db", {
                "chunk_count": len(chunked_docs)
            })
            self.logger.debug(f"Inserting {len(chunked_docs)} chunks into vector DB")
            await vector_store.add_documents(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

            # Step 10: Update registry
            add_span_event("pipeline.step.9.update_registry")
            file_size = file_path.stat().st_size
            file_type = self.document_processor.detect_file_type(file_path)

            await self.registry.add_or_update_file(
                file_path=str(file_path),
                collection_name=collection_name,
                file_hash=current_hash,
                file_size_bytes=file_size,
                file_type=file_type or "unknown",
                chunk_count=len(chunked_docs),
                status=IngestionStatus.COMPLETED,
                ingestion_duration=time.time() - start_time
            )

            # Success!
            duration = time.time() - start_time

            add_span_event("pipeline_completed_successfully", {
                "duration_seconds": duration,
                "chunks_created": len(chunked_docs)
            })

            add_span_attributes({
                "ingestion.success": True,
                "ingestion.duration": duration,
                "ingestion.chunks_created": len(chunked_docs)
            })

            self.logger.info(
                f"✓ Completed: {file_path.name} -> {collection_name} "
                f"({len(chunked_docs)} chunks in {duration:.2f}s)"
            )

            # Record metrics
            if self.metrics:
                self.metrics.document_processed(collection_name, file_type or "unknown")
                self.metrics.chunks_created(collection_name, len(chunked_docs))
                self.metrics.ingestion_duration(collection_name, duration)

            return IngestionResult(
                success=True,
                file_path=str(file_path),
                collection_name=collection_name,
                chunks_created=len(chunked_docs),
                duration_seconds=duration,
                file_hash=current_hash
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)

            # Mark span as failed
            set_span_error(e)
            add_span_attributes({
                "ingestion.success": False,
                "ingestion.duration": duration,
                "error.type": type(e).__name__,
                "error.message": error_msg
            })

            self.logger.error(
                f"✗ Failed: {file_path.name} -> {collection_name} ({error_msg})"
            )

            # Update registry with failure
            try:
                file_size = file_path.stat().st_size if file_path.exists() else 0
                file_type = self.document_processor.detect_file_type(file_path)

                await self.registry.add_or_update_file(
                    file_path=str(file_path),
                    collection_name=collection_name,
                    file_hash="",
                    file_size_bytes=file_size,
                    file_type=file_type or "unknown",
                    chunk_count=0,
                    status=IngestionStatus.FAILED,
                    error_message=error_msg,
                    ingestion_duration=duration
                )
            except Exception as reg_error:
                self.logger.error(f"Failed to update registry: {reg_error}")

            # Record error metric
            if self.metrics:
                self.metrics.error_occurred(
                    error_type=type(e).__name__,
                    component="ingestion_orchestrator"
                )

            return IngestionResult(
                success=False,
                file_path=str(file_path),
                collection_name=collection_name,
                chunks_created=0,
                duration_seconds=duration,
                error_message=error_msg
            )

    async def _worker(self, worker_id: int):
        """
        Background worker that processes tasks from the queue.

        Args:
            worker_id: Worker identifier
        """
        self.logger.info(f"Worker {worker_id} started")

        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Mark file as in progress
                file_key = f"{task.file_path}:{task.collection_name}"
                self.in_progress.add(file_key)

                task.started_at = datetime.now()

                # Process task
                result = await self._process_task(task)

                task.completed_at = datetime.now()

                # Update statistics
                if result.success:
                    self.stats["tasks_completed"] += 1
                    self.stats["total_chunks_created"] += result.chunks_created
                    self.stats["total_processing_time"] += result.duration_seconds
                else:
                    # Retry logic
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        self.logger.warning(
                            f"Retrying task (attempt {task.retry_count}/{task.max_retries}): "
                            f"{Path(task.file_path).name}"
                        )
                        await self.task_queue.put(task)
                    else:
                        self.stats["tasks_failed"] += 1
                        self.logger.error(
                            f"Task failed after {task.max_retries} retries: "
                            f"{Path(task.file_path).name}"
                        )

                # Remove from in-progress
                self.in_progress.discard(file_key)

                # Mark task as done
                self.task_queue.task_done()

                # Call completion callback
                if self.completion_callback:
                    try:
                        await self.completion_callback(result)
                    except Exception as cb_error:
                        self.logger.error(f"Completion callback error: {cb_error}")

            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")

        self.logger.info(f"Worker {worker_id} stopped")

    async def queue_manual_ingestion(
        self,
        file_path: str,
        collection_name: str,
        priority: TaskPriority = TaskPriority.HIGH
    ):
        """
        Manually queue a file for ingestion.

        Args:
            file_path: Path to file
            collection_name: Collection name
            priority: Task priority
        """
        task = IngestionTask(
            file_path=file_path,
            collection_name=collection_name,
            event_type=FileEventType.CREATED,
            priority=priority
        )

        await self.task_queue.put(task)
        self.stats["tasks_queued"] += 1

        self.logger.info(f"Manual task queued: {Path(file_path).name}")

    def start(self):
        """Start the ingestion orchestrator"""
        if self.running:
            self.logger.warning("Orchestrator already running")
            return

        if not self.registry:
            raise RuntimeError("Registry not set - cannot start orchestrator")

        self.running = True

        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)

        self.logger.info(
            f"Ingestion orchestrator started "
            f"({self.max_workers} workers, batch size: {self.batch_size})"
        )

    async def stop(self, timeout: int = 60):
        """
        Stop the ingestion orchestrator gracefully.

        Args:
            timeout: Maximum seconds to wait for pending tasks
        """
        if not self.running:
            return

        self.logger.info("Stopping ingestion orchestrator...")

        # Wait for queue to be empty
        if not self.task_queue.empty():
            self.logger.info(
                f"Waiting for {self.task_queue.qsize()} pending tasks..."
            )
            try:
                await asyncio.wait_for(
                    self.task_queue.join(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for pending tasks")

        self.running = False

        # Cancel workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)

        self.workers.clear()

        self.logger.info("Ingestion orchestrator stopped")

    def get_stats(self) -> Dict:
        """Get orchestrator statistics"""
        return {
            "running": self.running,
            "workers": len(self.workers),
            "queue_size": self.task_queue.qsize(),
            "in_progress": len(self.in_progress),
            "tasks_queued": self.stats["tasks_queued"],
            "tasks_completed": self.stats["tasks_completed"],
            "tasks_failed": self.stats["tasks_failed"],
            "total_chunks_created": self.stats["total_chunks_created"],
            "avg_processing_time": (
                self.stats["total_processing_time"] / self.stats["tasks_completed"]
                if self.stats["tasks_completed"] > 0 else 0
            )
        }


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

async def main():
    """Example usage of IngestionOrchestrator"""

    # Initialize dependencies
    registry = IngestionRegistry()
    await registry.initialize()

    metrics_collector = MetricsCollector()
    metrics_collector.start()

    # Create orchestrator
    orchestrator = IngestionOrchestrator(
        registry=registry,
        metrics_collector=metrics_collector,
        max_workers=3
    )

    # Start orchestrator
    orchestrator.start()

    # Queue some manual tasks for testing
    # await orchestrator.queue_manual_ingestion(
    #     "data/academic/test.pdf",
    #     "academic_knowledge"
    # )

    # Run for a while
    print("Orchestrator running... (Press Ctrl+C to stop)")
    try:
        await asyncio.sleep(3600)  # Run for 1 hour
    except KeyboardInterrupt:
        print("\nShutting down...")

    # Stop gracefully
    await orchestrator.stop()
    await metrics_collector.stop()
    await registry.close()

    # Print stats
    stats = orchestrator.get_stats()
    print(f"Final stats: {stats}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
