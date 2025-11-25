"""
Migration Script: Create Gemini Vector Database from Existing Documents

This script re-embeds all existing documents with Gemini embeddings and
populates the Gemini vector database (chroma_gemini/) to enable fallback support.

Usage:
    python scripts/migrate_to_dual_embeddings.py
    python scripts/migrate_to_dual_embeddings.py --collection academic
    python scripts/migrate_to_dual_embeddings.py --dry-run  # Estimate cost only
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import ACADEMIC_DATA_DIR, SKILL_DATA_DIR
from scripts.ingest_knowledge import KnowledgeIngestionPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def estimate_migration_cost():
    """
    Estimate the cost of migrating to Gemini embeddings.

    Gemini text-embedding-004 is essentially free (first 1M tokens free).
    This function counts documents and provides an estimate.
    """
    from rag.document_processor import DocumentProcessor

    logger.info("\n" + "="*80)
    logger.info("MIGRATION COST ESTIMATION")
    logger.info("="*80 + "\n")

    doc_processor = DocumentProcessor()

    # Process academic documents
    logger.info("Analyzing academic documents...")
    academic_docs = doc_processor.process_directory(ACADEMIC_DATA_DIR, "academic")
    academic_count = len(academic_docs) if academic_docs else 0
    academic_tokens = sum(len(doc.page_content.split()) for doc in academic_docs) if academic_docs else 0

    # Process skill documents
    logger.info("Analyzing skill documents...")
    skill_docs = doc_processor.process_directory(SKILL_DATA_DIR, "skill")
    skill_count = len(skill_docs) if skill_docs else 0
    skill_tokens = sum(len(doc.page_content.split()) for doc in skill_docs) if skill_docs else 0

    # Calculate totals
    total_docs = academic_count + skill_count
    total_tokens = academic_tokens + skill_tokens

    # Gemini pricing (first 1M tokens free, then negligible cost)
    gemini_cost_per_million = 0.00001  # Conservative estimate
    estimated_cost = (total_tokens / 1_000_000) * gemini_cost_per_million

    # Display results
    logger.info("\n" + "-"*80)
    logger.info("MIGRATION SUMMARY")
    logger.info("-"*80)
    logger.info(f"Academic documents: {academic_count} chunks ({academic_tokens:,} tokens)")
    logger.info(f"Skill documents: {skill_count} chunks ({skill_tokens:,} tokens)")
    logger.info(f"Total documents: {total_docs} chunks")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Estimated Gemini cost: ${estimated_cost:.6f} (essentially FREE)")
    logger.info(f"Estimated time: ~{total_docs // 50 + 1} batches × 0.5s = ~{(total_docs // 50 + 1) * 0.5:.0f}s")
    logger.info("-"*80 + "\n")

    return {
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
    }


def migrate_to_gemini(collection: str = "all", reset: bool = False):
    """
    Migrate existing documents to Gemini vector database.

    Args:
        collection: "academic", "skill", or "all"
        reset: Whether to reset Gemini collection before migration
    """
    logger.info("\n" + "="*80)
    logger.info("MIGRATING TO GEMINI EMBEDDINGS")
    logger.info("="*80 + "\n")

    logger.info("Creating Gemini vector database from existing documents...")
    logger.info("This will embed all documents with Gemini's text-embedding-004 model.\n")

    # Initialize Gemini pipeline
    logger.info(">>> INITIALIZING GEMINI PIPELINE <<<")
    pipeline_gemini = KnowledgeIngestionPipeline(provider="gemini")

    # Ingest based on collection argument
    if collection == "all":
        logger.info(">>> INGESTING ALL COLLECTIONS TO GEMINI DATABASE <<<\n")
        results = pipeline_gemini.ingest_all(reset=reset)

        # Summary
        if "summary" in results:
            logger.info("\n" + "="*80)
            logger.info("MIGRATION COMPLETE")
            logger.info("="*80)
            logger.info(f"Total documents migrated: {results['summary']['total_in_database']}")
            logger.info(f"Total cost: ${results['summary']['total_cost_usd']:.6f}")
            logger.info(f"Total time: {results['summary']['total_time']:.1f} seconds")
            logger.info("="*80 + "\n")

    elif collection == "academic":
        logger.info(">>> INGESTING ACADEMIC COLLECTION TO GEMINI DATABASE <<<\n")
        results = pipeline_gemini.ingest_collection(
            ACADEMIC_DATA_DIR, "academic", reset=reset
        )

    elif collection == "skill":
        logger.info(">>> INGESTING SKILL COLLECTION TO GEMINI DATABASE <<<\n")
        results = pipeline_gemini.ingest_collection(
            SKILL_DATA_DIR, "skill", reset=reset
        )

    logger.info("\n✅ Migration completed successfully!")
    logger.info("Gemini vector database is now available for fallback support.\n")

    return results


def verify_migration():
    """
    Verify that both OpenAI and Gemini databases exist and contain data.
    """
    from rag.vector_store import VectorStoreManager

    logger.info("\n" + "="*80)
    logger.info("VERIFYING DUAL DATABASES")
    logger.info("="*80 + "\n")

    # Check OpenAI database
    logger.info(">>> OPENAI DATABASE <<<")
    try:
        openai_store = VectorStoreManager(provider="openai")
        openai_stats = openai_store.get_all_stats()
        logger.info(f"Persist directory: {openai_stats['persist_directory']}")
        logger.info(f"Total documents: {openai_stats['total_documents']}")
        for coll_name, coll_stats in openai_stats["collections"].items():
            if "count" in coll_stats:
                logger.info(f"  {coll_name}: {coll_stats['count']} documents")
    except Exception as e:
        logger.error(f"OpenAI database error: {e}")

    # Check Gemini database
    logger.info("\n>>> GEMINI DATABASE <<<")
    try:
        gemini_store = VectorStoreManager(provider="gemini")
        gemini_stats = gemini_store.get_all_stats()
        logger.info(f"Persist directory: {gemini_stats['persist_directory']}")
        logger.info(f"Total documents: {gemini_stats['total_documents']}")
        for coll_name, coll_stats in gemini_stats["collections"].items():
            if "count" in coll_stats:
                logger.info(f"  {coll_name}: {coll_stats['count']} documents")
    except Exception as e:
        logger.error(f"Gemini database error: {e}")

    logger.info("\n" + "="*80 + "\n")


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate existing documents to Gemini embeddings for fallback support"
    )
    parser.add_argument(
        "--collection",
        choices=["academic", "skill", "all"],
        default="all",
        help="Which collection to migrate (default: all)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset Gemini collection before migration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate cost only, don't perform migration",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify both databases exist and contain data",
    )

    args = parser.parse_args()

    try:
        if args.verify:
            verify_migration()
            return

        if args.dry_run:
            estimate_migration_cost()
            logger.info("This was a DRY RUN. No data was migrated.")
            logger.info("Run without --dry-run to perform actual migration.\n")
            return

        # Estimate cost first
        estimation = estimate_migration_cost()

        # Confirm migration
        logger.info("⚠️  This will create Gemini embeddings for all documents.")
        logger.info("Press Ctrl+C to cancel, or press Enter to continue...")
        input()

        # Perform migration
        migrate_to_gemini(collection=args.collection, reset=args.reset)

        # Verify migration
        logger.info("Verifying migration...")
        verify_migration()

        logger.info("✅ All done! Your system now supports automatic LLM + RAG fallback.\n")

    except KeyboardInterrupt:
        logger.warning("\n\nMigration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
