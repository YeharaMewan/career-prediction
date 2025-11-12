"""
Ingestion Registry

SQLite-based tracking system for ingested files.
Tracks file hashes, ingestion timestamps, status, and history.

Features:
- File ingestion tracking with SHA-256 hashes
- Ingestion history with full audit trail
- Change detection via hash comparison
- Orphan detection (files deleted but still in vector DB)
- Async operations with aiosqlite
- Automatic cleanup of old history entries

Author: Career Planning System
Created: 2025
"""

import os
import json
import logging
import aiosqlite
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum


class IngestionStatus(Enum):
    """Status of file ingestion"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FileRecord:
    """Record of an ingested file"""
    id: Optional[int] = None
    file_path: str = ""
    collection_name: str = ""
    file_hash: str = ""
    file_size_bytes: int = 0
    file_type: str = ""
    last_ingested: Optional[datetime] = None
    chunk_count: int = 0
    status: str = IngestionStatus.COMPLETED.value
    error_message: Optional[str] = None
    ingestion_duration_seconds: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary with ISO format dates"""
        data = asdict(self)
        if self.last_ingested:
            data['last_ingested'] = self.last_ingested.isoformat()
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_db_row(cls, row: Tuple) -> 'FileRecord':
        """Create FileRecord from database row"""
        return cls(
            id=row[0],
            file_path=row[1],
            collection_name=row[2],
            file_hash=row[3],
            file_size_bytes=row[4],
            file_type=row[5],
            last_ingested=datetime.fromisoformat(row[6]) if row[6] else None,
            chunk_count=row[7],
            status=row[8],
            error_message=row[9],
            ingestion_duration_seconds=row[10],
            created_at=datetime.fromisoformat(row[11]) if row[11] else None,
            updated_at=datetime.fromisoformat(row[12]) if row[12] else None
        )


@dataclass
class HistoryEntry:
    """Single history entry for a file ingestion"""
    id: Optional[int] = None
    file_path: str = ""
    collection_name: str = ""
    file_hash: str = ""
    chunk_count: int = 0
    status: str = IngestionStatus.COMPLETED.value
    error_message: Optional[str] = None
    ingestion_duration_seconds: Optional[float] = None
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        return data


class IngestionRegistry:
    """
    SQLite-based registry for tracking file ingestion.

    Features:
    - Track file ingestion with SHA-256 hashes
    - Maintain full history of ingestions
    - Detect file changes via hash comparison
    - Identify orphaned entries
    - Async database operations
    - Automatic cleanup of old history
    """

    # Database schema version
    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str = "knowledge_base/ingestion_registry.db",
        track_history: bool = True,
        max_history_per_file: int = 20,
        history_retention_days: int = 90
    ):
        """
        Initialize Ingestion Registry

        Args:
            db_path: Path to SQLite database file
            track_history: Whether to maintain ingestion history
            max_history_per_file: Maximum history entries per file
            history_retention_days: Delete history older than this
        """
        self.db_path = Path(db_path)
        self.track_history = track_history
        self.max_history_per_file = max_history_per_file
        self.history_retention_days = history_retention_days

        # Create directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Connection will be created per-operation (async pattern)
        self._initialized = False

    async def initialize(self):
        """Initialize database schema"""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            # Enable foreign keys
            await db.execute("PRAGMA foreign_keys = ON")

            # Create files table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    file_type TEXT NOT NULL,
                    last_ingested TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    ingestion_duration_seconds REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(file_path, collection_name)
                )
            """)

            # Create history table
            if self.track_history:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT NOT NULL,
                        collection_name TEXT NOT NULL,
                        file_hash TEXT NOT NULL,
                        chunk_count INTEGER DEFAULT 0,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        ingestion_duration_seconds REAL,
                        timestamp TEXT NOT NULL
                    )
                """)

            # Create indexes for performance
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_path
                ON files(file_path)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_collection
                ON files(collection_name)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_status
                ON files(status)
            """)

            if self.track_history:
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_history_path
                    ON history(file_path)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_history_timestamp
                    ON history(timestamp)
                """)

            # Create metadata table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # Store schema version
            await db.execute("""
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('schema_version', ?)
            """, (str(self.SCHEMA_VERSION),))

            await db.commit()

        self._initialized = True
        self.logger.info(f"Ingestion registry initialized: {self.db_path}")

    async def add_or_update_file(
        self,
        file_path: str,
        collection_name: str,
        file_hash: str,
        file_size_bytes: int,
        file_type: str,
        chunk_count: int = 0,
        status: IngestionStatus = IngestionStatus.COMPLETED,
        error_message: Optional[str] = None,
        ingestion_duration: Optional[float] = None
    ) -> int:
        """
        Add or update a file record in the registry.

        Args:
            file_path: Path to file
            collection_name: Vector DB collection name
            file_hash: SHA-256 hash of file
            file_size_bytes: File size in bytes
            file_type: File type (pdf, docx, etc.)
            chunk_count: Number of chunks created
            status: Ingestion status
            error_message: Error message if failed
            ingestion_duration: Duration in seconds

        Returns:
            Record ID
        """
        if not self._initialized:
            await self.initialize()

        now = datetime.now().isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            # Check if record exists
            cursor = await db.execute("""
                SELECT id FROM files
                WHERE file_path = ? AND collection_name = ?
            """, (file_path, collection_name))

            existing = await cursor.fetchone()

            if existing:
                # Update existing record
                await db.execute("""
                    UPDATE files SET
                        file_hash = ?,
                        file_size_bytes = ?,
                        file_type = ?,
                        last_ingested = ?,
                        chunk_count = ?,
                        status = ?,
                        error_message = ?,
                        ingestion_duration_seconds = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    file_hash,
                    file_size_bytes,
                    file_type,
                    now,
                    chunk_count,
                    status.value,
                    error_message,
                    ingestion_duration,
                    now,
                    existing[0]
                ))

                record_id = existing[0]
            else:
                # Insert new record
                cursor = await db.execute("""
                    INSERT INTO files (
                        file_path, collection_name, file_hash, file_size_bytes,
                        file_type, last_ingested, chunk_count, status,
                        error_message, ingestion_duration_seconds, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_path,
                    collection_name,
                    file_hash,
                    file_size_bytes,
                    file_type,
                    now,
                    chunk_count,
                    status.value,
                    error_message,
                    ingestion_duration,
                    now,
                    now
                ))

                record_id = cursor.lastrowid

            # Add to history
            if self.track_history and status in [IngestionStatus.COMPLETED, IngestionStatus.FAILED]:
                await self._add_history_entry(
                    db,
                    file_path,
                    collection_name,
                    file_hash,
                    chunk_count,
                    status,
                    error_message,
                    ingestion_duration
                )

            await db.commit()

        self.logger.debug(f"Registry updated: {file_path} -> {collection_name}")
        return record_id

    async def _add_history_entry(
        self,
        db: aiosqlite.Connection,
        file_path: str,
        collection_name: str,
        file_hash: str,
        chunk_count: int,
        status: IngestionStatus,
        error_message: Optional[str],
        ingestion_duration: Optional[float]
    ):
        """Add entry to history table"""
        await db.execute("""
            INSERT INTO history (
                file_path, collection_name, file_hash, chunk_count,
                status, error_message, ingestion_duration_seconds, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_path,
            collection_name,
            file_hash,
            chunk_count,
            status.value,
            error_message,
            ingestion_duration,
            datetime.now().isoformat()
        ))

        # Cleanup old history entries for this file
        await db.execute("""
            DELETE FROM history
            WHERE file_path = ? AND collection_name = ?
            AND id NOT IN (
                SELECT id FROM history
                WHERE file_path = ? AND collection_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            )
        """, (
            file_path,
            collection_name,
            file_path,
            collection_name,
            self.max_history_per_file
        ))

    async def get_file_record(
        self,
        file_path: str,
        collection_name: str
    ) -> Optional[FileRecord]:
        """Get file record from registry"""
        if not self._initialized:
            await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT * FROM files
                WHERE file_path = ? AND collection_name = ?
            """, (file_path, collection_name))

            row = await cursor.fetchone()

        if row:
            return FileRecord.from_db_row(row)
        return None

    async def file_needs_update(
        self,
        file_path: str,
        collection_name: str,
        current_hash: str
    ) -> bool:
        """
        Check if file needs re-ingestion based on hash.

        Returns:
            True if file is new or hash has changed
        """
        record = await self.get_file_record(file_path, collection_name)

        if record is None:
            return True  # New file

        return record.file_hash != current_hash  # Hash changed

    async def get_all_files(
        self,
        collection_name: Optional[str] = None,
        status: Optional[IngestionStatus] = None
    ) -> List[FileRecord]:
        """
        Get all file records, optionally filtered by collection and/or status.

        Args:
            collection_name: Optional collection name filter
            status: Optional status filter

        Returns:
            List of FileRecord objects
        """
        if not self._initialized:
            await self.initialize()

        query = "SELECT * FROM files WHERE 1=1"
        params = []

        if collection_name:
            query += " AND collection_name = ?"
            params.append(collection_name)

        if status:
            query += " AND status = ?"
            params.append(status.value)

        query += " ORDER BY last_ingested DESC"

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        return [FileRecord.from_db_row(row) for row in rows]

    async def get_history(
        self,
        file_path: str,
        collection_name: str,
        limit: int = 20
    ) -> List[HistoryEntry]:
        """
        Get ingestion history for a file.

        Args:
            file_path: Path to file
            collection_name: Collection name
            limit: Maximum entries to return

        Returns:
            List of HistoryEntry objects, newest first
        """
        if not self._initialized or not self.track_history:
            return []

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT * FROM history
                WHERE file_path = ? AND collection_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (file_path, collection_name, limit))

            rows = await cursor.fetchall()

        return [
            HistoryEntry(
                id=row[0],
                file_path=row[1],
                collection_name=row[2],
                file_hash=row[3],
                chunk_count=row[4],
                status=row[5],
                error_message=row[6],
                ingestion_duration_seconds=row[7],
                timestamp=datetime.fromisoformat(row[8]) if row[8] else None
            )
            for row in rows
        ]

    async def delete_file_record(
        self,
        file_path: str,
        collection_name: str
    ) -> bool:
        """
        Delete a file record from registry.

        Args:
            file_path: Path to file
            collection_name: Collection name

        Returns:
            True if deleted, False if not found
        """
        if not self._initialized:
            await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                DELETE FROM files
                WHERE file_path = ? AND collection_name = ?
            """, (file_path, collection_name))

            await db.commit()

            deleted = cursor.rowcount > 0

        if deleted:
            self.logger.debug(f"Registry record deleted: {file_path}")

        return deleted

    async def get_statistics(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about ingested files.

        Args:
            collection_name: Optional collection name filter

        Returns:
            Dictionary with statistics
        """
        if not self._initialized:
            await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            # Base query conditions
            where_clause = ""
            params = []
            if collection_name:
                where_clause = "WHERE collection_name = ?"
                params.append(collection_name)

            # Total files
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM files {where_clause}",
                params
            )
            total_files = (await cursor.fetchone())[0]

            # Files by status
            cursor = await db.execute(
                f"SELECT status, COUNT(*) FROM files {where_clause} GROUP BY status",
                params
            )
            status_counts = {row[0]: row[1] for row in await cursor.fetchall()}

            # Total chunks
            cursor = await db.execute(
                f"SELECT SUM(chunk_count) FROM files {where_clause}",
                params
            )
            total_chunks = (await cursor.fetchone())[0] or 0

            # Total size
            cursor = await db.execute(
                f"SELECT SUM(file_size_bytes) FROM files {where_clause}",
                params
            )
            total_size_bytes = (await cursor.fetchone())[0] or 0

            # Average ingestion time
            cursor = await db.execute(
                f"""SELECT AVG(ingestion_duration_seconds)
                    FROM files
                    {where_clause}
                    {'AND' if where_clause else 'WHERE'} ingestion_duration_seconds IS NOT NULL""",
                params
            )
            avg_duration = (await cursor.fetchone())[0] or 0

        return {
            "total_files": total_files,
            "status_counts": status_counts,
            "total_chunks": total_chunks,
            "total_size_bytes": total_size_bytes,
            "total_size_mb": round(total_size_bytes / 1024 / 1024, 2),
            "avg_ingestion_duration_seconds": round(avg_duration, 2),
            "collection_name": collection_name
        }

    async def cleanup_old_history(self):
        """Delete history entries older than retention period"""
        if not self._initialized or not self.track_history:
            return

        cutoff_date = (
            datetime.now() - timedelta(days=self.history_retention_days)
        ).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                DELETE FROM history
                WHERE timestamp < ?
            """, (cutoff_date,))

            await db.commit()

            deleted = cursor.rowcount

        if deleted > 0:
            self.logger.info(f"Cleaned up {deleted} old history entries")

    async def find_orphaned_records(
        self,
        collection_name: str
    ) -> List[FileRecord]:
        """
        Find records in registry where files no longer exist on disk.

        Args:
            collection_name: Collection to check

        Returns:
            List of orphaned FileRecord objects
        """
        if not self._initialized:
            await self.initialize()

        records = await self.get_all_files(collection_name=collection_name)

        orphaned = []
        for record in records:
            if not Path(record.file_path).exists():
                orphaned.append(record)

        return orphaned

    async def close(self):
        """Close database connection and cleanup"""
        # Connection is managed per-operation, nothing to close
        self.logger.info("Registry closed")


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

async def main():
    """Example usage of IngestionRegistry"""
    import time

    # Initialize registry
    registry = IngestionRegistry(
        db_path="knowledge_base/ingestion_registry.db",
        track_history=True
    )

    await registry.initialize()

    # Add a file record
    file_path = "data/academic/test.pdf"
    collection = "academic_knowledge"
    file_hash = "abc123def456"

    start_time = time.time()

    record_id = await registry.add_or_update_file(
        file_path=file_path,
        collection_name=collection,
        file_hash=file_hash,
        file_size_bytes=1024000,
        file_type="pdf",
        chunk_count=50,
        status=IngestionStatus.COMPLETED,
        ingestion_duration=3.5
    )

    print(f"Added record ID: {record_id}")

    # Check if file needs update
    needs_update = await registry.file_needs_update(
        file_path, collection, "different_hash"
    )
    print(f"Needs update: {needs_update}")

    # Get file record
    record = await registry.get_file_record(file_path, collection)
    if record:
        print(f"Record: {record.to_dict()}")

    # Get statistics
    stats = await registry.get_statistics(collection)
    print(f"Statistics: {stats}")

    # Get history
    history = await registry.get_history(file_path, collection)
    print(f"History entries: {len(history)}")

    await registry.close()


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
