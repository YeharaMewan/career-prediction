"""
Reset ChromaDB - Fix corrupted metadata issue

This script deletes corrupted ChromaDB directories and allows fresh ingestion.
Use when encountering KeyError: '_type' or other metadata errors.
"""

import shutil
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import CHROMA_PERSIST_DIR_OPENAI, CHROMA_PERSIST_DIR_GEMINI

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def reset_chromadb():
    """Delete corrupted ChromaDB directories."""

    directories = [
        CHROMA_PERSIST_DIR_OPENAI,
        CHROMA_PERSIST_DIR_GEMINI
    ]

    logger.info("="*80)
    logger.info("CHROMADB RESET - Fixing Corrupted Metadata")
    logger.info("="*80)

    deleted_count = 0

    for directory in directories:
        if directory.exists():
            try:
                logger.info(f"Deleting: {directory}")
                shutil.rmtree(directory)
                logger.info(f"✓ Deleted successfully")
                deleted_count += 1
            except Exception as e:
                logger.error(f"✗ Failed to delete {directory}: {e}")
        else:
            logger.info(f"Skipping (not found): {directory}")

    logger.info("")
    logger.info(f"Reset complete: {deleted_count} directories deleted")
    logger.info("You can now run: python scripts/ingest_knowledge.py --provider openai --collection all")
    logger.info("="*80)


if __name__ == "__main__":
    try:
        reset_chromadb()
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        sys.exit(1)
