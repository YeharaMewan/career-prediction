"""
Version Manager for Vector Database Collections

Provides versioning, snapshot, backup, and rollback capabilities for ChromaDB collections.
Supports automatic version retention policies and compression.

Author: Career Planning System
Created: 2025
"""

import os
import json
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import asyncio
import aiofiles

import chromadb
from chromadb.config import Settings


@dataclass
class VersionMetadata:
    """Metadata for a collection version snapshot"""
    version_id: str
    collection_name: str
    timestamp: datetime
    document_count: int
    chunk_count: int
    file_hashes: Dict[str, str]  # file_path -> hash
    snapshot_size_bytes: int
    compressed: bool
    description: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'VersionMetadata':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class VersionManager:
    """
    Manages versions, snapshots, and backups for vector database collections.

    Features:
    - Create snapshots before modifications
    - Rollback to previous versions
    - Automatic version retention policies
    - Compression support
    - Version history tracking
    """

    def __init__(
        self,
        chroma_persist_dir: str = "knowledge_base/chroma",
        snapshot_dir: str = "knowledge_base/snapshots",
        max_versions_per_collection: int = 10,
        max_age_days: int = 30,
        min_versions_to_keep: int = 3,
        compress_snapshots: bool = True
    ):
        """
        Initialize Version Manager

        Args:
            chroma_persist_dir: ChromaDB persistence directory
            snapshot_dir: Directory for storing snapshots
            max_versions_per_collection: Maximum versions to keep per collection
            max_age_days: Delete versions older than this
            min_versions_to_keep: Minimum versions to keep regardless of age
            compress_snapshots: Enable compression for snapshots
        """
        self.chroma_persist_dir = Path(chroma_persist_dir)
        self.snapshot_dir = Path(snapshot_dir)
        self.max_versions = max_versions_per_collection
        self.max_age_days = max_age_days
        self.min_versions = min_versions_to_keep
        self.compress_snapshots = compress_snapshots

        # Create directories
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

    def _generate_version_id(self, collection_name: str) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{collection_name}_{timestamp}"

    def _get_collection_snapshot_dir(self, collection_name: str) -> Path:
        """Get snapshot directory for a collection"""
        coll_dir = self.snapshot_dir / collection_name
        coll_dir.mkdir(parents=True, exist_ok=True)
        return coll_dir

    def _get_version_dir(self, version_id: str, collection_name: str) -> Path:
        """Get directory for a specific version"""
        return self._get_collection_snapshot_dir(collection_name) / version_id

    async def create_snapshot(
        self,
        collection_name: str,
        description: Optional[str] = None,
        file_hashes: Optional[Dict[str, str]] = None
    ) -> Tuple[str, VersionMetadata]:
        """
        Create a snapshot of a collection

        Args:
            collection_name: Name of the collection to snapshot
            description: Optional description for this version
            file_hashes: Optional dict of file_path -> hash for tracking

        Returns:
            Tuple of (version_id, VersionMetadata)
        """
        self.logger.info(f"Creating snapshot for collection: {collection_name}")

        try:
            # Generate version ID
            version_id = self._generate_version_id(collection_name)
            version_dir = self._get_version_dir(version_id, collection_name)
            version_dir.mkdir(parents=True, exist_ok=True)

            # Get collection
            collection = self.client.get_collection(name=collection_name)

            # Get all data from collection
            results = collection.get(include=["embeddings", "documents", "metadatas"])

            document_count = len(set(
                meta.get("source", "") for meta in results["metadatas"]
            ))
            chunk_count = len(results["ids"])

            # Save collection data
            collection_data = {
                "ids": results["ids"],
                "embeddings": results["embeddings"],
                "documents": results["documents"],
                "metadatas": results["metadatas"]
            }

            data_file = version_dir / "collection_data.json"
            async with aiofiles.open(data_file, 'w') as f:
                await f.write(json.dumps(collection_data, indent=2))

            # Calculate snapshot size
            snapshot_size = sum(
                f.stat().st_size for f in version_dir.rglob('*') if f.is_file()
            )

            # Compress if enabled
            compressed = False
            if self.compress_snapshots:
                await self._compress_snapshot(version_dir, version_id, collection_name)
                compressed = True
                # Recalculate size after compression
                archive_path = version_dir.parent / f"{version_id}.tar.gz"
                if archive_path.exists():
                    snapshot_size = archive_path.stat().st_size

            # Create metadata
            metadata = VersionMetadata(
                version_id=version_id,
                collection_name=collection_name,
                timestamp=datetime.now(),
                document_count=document_count,
                chunk_count=chunk_count,
                file_hashes=file_hashes or {},
                snapshot_size_bytes=snapshot_size,
                compressed=compressed,
                description=description
            )

            # Save metadata
            await self._save_metadata(metadata)

            self.logger.info(
                f"Snapshot created: {version_id} "
                f"({chunk_count} chunks, {snapshot_size / 1024 / 1024:.2f} MB)"
            )

            # Cleanup old versions
            await self._cleanup_old_versions(collection_name)

            return version_id, metadata

        except Exception as e:
            self.logger.error(f"Failed to create snapshot: {str(e)}")
            raise

    async def _compress_snapshot(
        self,
        version_dir: Path,
        version_id: str,
        collection_name: str
    ):
        """Compress snapshot directory to tar.gz"""
        archive_path = version_dir.parent / f"{version_id}.tar.gz"

        def _compress():
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(version_dir, arcname=version_id)

        # Run compression in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(None, _compress)

        # Remove uncompressed directory
        shutil.rmtree(version_dir)

        self.logger.debug(f"Compressed snapshot: {archive_path}")

    async def _decompress_snapshot(self, archive_path: Path, extract_to: Path):
        """Decompress tar.gz snapshot"""
        def _decompress():
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_to)

        await asyncio.get_event_loop().run_in_executor(None, _decompress)
        self.logger.debug(f"Decompressed snapshot: {archive_path}")

    async def _save_metadata(self, metadata: VersionMetadata):
        """Save version metadata to JSON file"""
        version_dir = self._get_version_dir(
            metadata.version_id,
            metadata.collection_name
        )

        # If compressed, metadata goes next to archive
        if metadata.compressed:
            metadata_file = (
                version_dir.parent / f"{metadata.version_id}_metadata.json"
            )
        else:
            metadata_file = version_dir / "metadata.json"

        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata.to_dict(), indent=2))

    async def get_metadata(
        self,
        version_id: str,
        collection_name: str
    ) -> Optional[VersionMetadata]:
        """Get metadata for a specific version"""
        # Try compressed metadata first
        metadata_file = (
            self._get_collection_snapshot_dir(collection_name) /
            f"{version_id}_metadata.json"
        )

        # Try uncompressed metadata
        if not metadata_file.exists():
            metadata_file = (
                self._get_version_dir(version_id, collection_name) /
                "metadata.json"
            )

        if not metadata_file.exists():
            return None

        async with aiofiles.open(metadata_file, 'r') as f:
            data = json.loads(await f.read())

        return VersionMetadata.from_dict(data)

    async def list_versions(
        self,
        collection_name: str
    ) -> List[VersionMetadata]:
        """List all versions for a collection, sorted by timestamp (newest first)"""
        collection_dir = self._get_collection_snapshot_dir(collection_name)

        versions = []

        # Check compressed snapshots
        for metadata_file in collection_dir.glob("*_metadata.json"):
            async with aiofiles.open(metadata_file, 'r') as f:
                data = json.loads(await f.read())
            versions.append(VersionMetadata.from_dict(data))

        # Check uncompressed snapshots
        for version_dir in collection_dir.iterdir():
            if version_dir.is_dir():
                metadata_file = version_dir / "metadata.json"
                if metadata_file.exists():
                    async with aiofiles.open(metadata_file, 'r') as f:
                        data = json.loads(await f.read())
                    versions.append(VersionMetadata.from_dict(data))

        # Sort by timestamp, newest first
        versions.sort(key=lambda v: v.timestamp, reverse=True)

        return versions

    async def rollback(
        self,
        collection_name: str,
        version_id: str,
        create_backup: bool = True
    ) -> bool:
        """
        Rollback a collection to a specific version

        Args:
            collection_name: Name of the collection
            version_id: Version ID to rollback to
            create_backup: Create backup of current state before rollback

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(
            f"Rolling back collection '{collection_name}' to version '{version_id}'"
        )

        try:
            # Create backup of current state if requested
            if create_backup:
                await self.create_snapshot(
                    collection_name,
                    description=f"Backup before rollback to {version_id}"
                )

            # Get metadata for target version
            metadata = await self.get_metadata(version_id, collection_name)
            if not metadata:
                raise ValueError(f"Version {version_id} not found")

            # Load snapshot data
            if metadata.compressed:
                # Decompress first
                archive_path = (
                    self._get_collection_snapshot_dir(collection_name) /
                    f"{version_id}.tar.gz"
                )
                temp_dir = self._get_collection_snapshot_dir(collection_name) / "temp"
                temp_dir.mkdir(exist_ok=True)

                await self._decompress_snapshot(archive_path, temp_dir)
                data_file = temp_dir / version_id / "collection_data.json"
            else:
                data_file = (
                    self._get_version_dir(version_id, collection_name) /
                    "collection_data.json"
                )

            # Load collection data
            async with aiofiles.open(data_file, 'r') as f:
                collection_data = json.loads(await f.read())

            # Delete current collection
            try:
                self.client.delete_collection(name=collection_name)
            except Exception as e:
                self.logger.warning(f"Collection deletion warning: {str(e)}")

            # Recreate collection
            collection = self.client.create_collection(name=collection_name)

            # Restore data in batches (ChromaDB has limits)
            batch_size = 1000
            total_items = len(collection_data["ids"])

            for i in range(0, total_items, batch_size):
                end_idx = min(i + batch_size, total_items)

                collection.add(
                    ids=collection_data["ids"][i:end_idx],
                    embeddings=collection_data["embeddings"][i:end_idx],
                    documents=collection_data["documents"][i:end_idx],
                    metadatas=collection_data["metadatas"][i:end_idx]
                )

            # Cleanup temp directory if decompressed
            if metadata.compressed:
                shutil.rmtree(temp_dir, ignore_errors=True)

            self.logger.info(
                f"Successfully rolled back to version {version_id} "
                f"({total_items} items restored)"
            )

            return True

        except Exception as e:
            self.logger.error(f"Rollback failed: {str(e)}")
            return False

    async def delete_version(self, version_id: str, collection_name: str):
        """Delete a specific version snapshot"""
        self.logger.info(f"Deleting version: {version_id}")

        # Delete compressed snapshot
        archive_path = (
            self._get_collection_snapshot_dir(collection_name) /
            f"{version_id}.tar.gz"
        )
        if archive_path.exists():
            archive_path.unlink()

        # Delete compressed metadata
        metadata_file = (
            self._get_collection_snapshot_dir(collection_name) /
            f"{version_id}_metadata.json"
        )
        if metadata_file.exists():
            metadata_file.unlink()

        # Delete uncompressed snapshot
        version_dir = self._get_version_dir(version_id, collection_name)
        if version_dir.exists():
            shutil.rmtree(version_dir)

        self.logger.debug(f"Deleted version: {version_id}")

    async def _cleanup_old_versions(self, collection_name: str):
        """Cleanup old versions based on retention policy"""
        versions = await self.list_versions(collection_name)

        if len(versions) <= self.min_versions:
            return  # Keep minimum versions

        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=self.max_age_days)

        # Determine versions to delete
        to_delete = []
        keep_count = 0

        for version in versions:
            # Always keep minimum versions (newest)
            if keep_count < self.min_versions:
                keep_count += 1
                continue

            # Delete if too old or exceeds max count
            if version.timestamp < cutoff_date or keep_count >= self.max_versions:
                to_delete.append(version)
            else:
                keep_count += 1

        # Delete old versions
        for version in to_delete:
            await self.delete_version(version.version_id, collection_name)

        if to_delete:
            self.logger.info(
                f"Cleaned up {len(to_delete)} old versions for '{collection_name}'"
            )

    async def get_disk_usage(self, collection_name: Optional[str] = None) -> Dict:
        """
        Get disk usage statistics for snapshots

        Args:
            collection_name: Optional collection name, or None for all

        Returns:
            Dict with usage statistics
        """
        if collection_name:
            snapshot_dir = self._get_collection_snapshot_dir(collection_name)
            dirs_to_check = [snapshot_dir]
        else:
            dirs_to_check = [
                d for d in self.snapshot_dir.iterdir() if d.is_dir()
            ]

        total_size = 0
        version_count = 0

        for dir_path in dirs_to_check:
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    if file_path.suffix == '.gz':
                        version_count += 1
                    elif file_path.name == 'metadata.json':
                        version_count += 1

        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "version_count": version_count,
            "collections": len(dirs_to_check) if not collection_name else 1
        }

    async def verify_snapshot(
        self,
        version_id: str,
        collection_name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify integrity of a snapshot

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            metadata = await self.get_metadata(version_id, collection_name)
            if not metadata:
                return False, "Metadata not found"

            # Check if snapshot file exists
            if metadata.compressed:
                archive_path = (
                    self._get_collection_snapshot_dir(collection_name) /
                    f"{version_id}.tar.gz"
                )
                if not archive_path.exists():
                    return False, "Compressed snapshot file not found"

                # Verify archive can be opened
                try:
                    with tarfile.open(archive_path, "r:gz") as tar:
                        tar.getmembers()
                except Exception as e:
                    return False, f"Corrupted archive: {str(e)}"
            else:
                data_file = (
                    self._get_version_dir(version_id, collection_name) /
                    "collection_data.json"
                )
                if not data_file.exists():
                    return False, "Snapshot data file not found"

                # Verify JSON can be loaded
                try:
                    async with aiofiles.open(data_file, 'r') as f:
                        json.loads(await f.read())
                except Exception as e:
                    return False, f"Corrupted data file: {str(e)}"

            return True, None

        except Exception as e:
            return False, f"Verification error: {str(e)}"


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

async def main():
    """Example usage of VersionManager"""

    # Initialize version manager
    manager = VersionManager(
        chroma_persist_dir="knowledge_base/chroma",
        snapshot_dir="knowledge_base/snapshots",
        max_versions_per_collection=10,
        compress_snapshots=True
    )

    collection_name = "academic_knowledge"

    # Create a snapshot
    version_id, metadata = await manager.create_snapshot(
        collection_name,
        description="Before major update",
        file_hashes={"file1.pdf": "abc123", "file2.pdf": "def456"}
    )

    print(f"Created snapshot: {version_id}")
    print(f"Chunks: {metadata.chunk_count}")
    print(f"Size: {metadata.snapshot_size_bytes / 1024 / 1024:.2f} MB")

    # List all versions
    versions = await manager.list_versions(collection_name)
    print(f"\nAvailable versions: {len(versions)}")
    for v in versions[:5]:  # Show latest 5
        print(f"  - {v.version_id}: {v.timestamp} ({v.chunk_count} chunks)")

    # Get disk usage
    usage = await manager.get_disk_usage(collection_name)
    print(f"\nDisk usage: {usage['total_size_mb']} MB ({usage['version_count']} versions)")

    # Verify a snapshot
    is_valid, error = await manager.verify_snapshot(version_id, collection_name)
    print(f"\nSnapshot verification: {'✓ Valid' if is_valid else f'✗ Invalid: {error}'}")

    # Rollback example (commented out)
    # success = await manager.rollback(collection_name, version_id)
    # print(f"Rollback: {'✓ Success' if success else '✗ Failed'}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run example
    asyncio.run(main())
