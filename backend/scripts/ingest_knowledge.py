"""
Knowledge Base Ingestion Pipeline

Loads PDFs from data directories, processes them, generates embeddings,
and stores them in ChromaDB collections.

Usage:
    python scripts/ingest_knowledge.py --collection academic
    python scripts/ingest_knowledge.py --collection skill
    python scripts/ingest_knowledge.py --collection all
    python scripts/ingest_knowledge.py --reset  # Reset and rebuild all collections
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.document_processor import DocumentProcessor
from rag.embedding_manager import EmbeddingManager
from rag.vector_store import VectorStoreManager
from rag.config import ACADEMIC_DATA_DIR, SKILL_DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KnowledgeIngestionPipeline:
    """
    Complete pipeline for ingesting documents into RAG system.
    Supports dual provider ingestion (OpenAI and Gemini).
    """

    def __init__(self, provider: str = "fallback"):
        """
        Initialize all components with specified provider.

        Args:
            provider: Embedding provider ("openai", "gemini", or "fallback")
        """
        logger.info(f"Initializing Knowledge Ingestion Pipeline (provider: {provider})...")

        self.provider = provider
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager(provider=provider)
        self.vector_store = VectorStoreManager(provider=provider)

        # Initialize collections
        self.vector_store.initialize_collections()

        logger.info(
            f"Pipeline initialized successfully "
            f"[provider: {self.embedding_manager.get_active_provider()}]"
        )

    def ingest_collection(
        self, data_dir: Path, collection_type: str, reset: bool = False
    ) -> dict:
        """
        Ingest documents from a directory into a collection.

        Args:
            data_dir: Directory containing PDFs
            collection_type: "academic" or "skill"
            reset: Whether to reset collection before ingesting

        Returns:
            Dictionary with ingestion statistics
        """
        start_time = time.time()

        logger.info(f"\n{'='*80}")
        logger.info(f"INGESTING {collection_type.upper()} COLLECTION")
        logger.info(f"{'='*80}\n")

        # Reset collection if requested
        if reset:
            logger.info(f"Resetting {collection_type} collection...")
            try:
                self.vector_store.delete_collection(collection_type)
                self.vector_store.get_collection(collection_type)  # Recreate
                logger.info(f"Collection reset complete")
            except Exception as e:
                logger.warning(f"Could not reset collection: {e}")

        # Step 1: Process documents
        logger.info(f"Step 1: Processing PDFs from {data_dir}")
        documents = self.doc_processor.process_directory(data_dir, collection_type)

        if not documents:
            logger.warning(f"No documents found in {data_dir}")
            logger.warning(f"Please add PDF files to {data_dir} and try again")
            return {
                "status": "no_documents",
                "collection_type": collection_type,
                "documents_processed": 0,
                "time_elapsed": time.time() - start_time,
            }

        # Get document statistics
        doc_stats = self.doc_processor.get_stats(documents)
        logger.info(f"Processed {doc_stats['total_chunks']} chunks from {doc_stats['unique_sources']} files")
        logger.info(f"Average chunk size: {doc_stats['avg_chunk_size']} characters")

        # Step 2: Generate embeddings
        logger.info(f"\nStep 2: Generating embeddings...")
        docs, embeddings = self.embedding_manager.embed_documents(documents)

        if not embeddings:
            logger.error("Failed to generate embeddings")
            return {
                "status": "embedding_failed",
                "collection_type": collection_type,
                "documents_processed": 0,
                "time_elapsed": time.time() - start_time,
            }

        # Get embedding statistics
        embed_stats = self.embedding_manager.get_stats()
        logger.info(f"Generated {len(embeddings)} embeddings")
        logger.info(f"Embedding cost: ${embed_stats['estimated_cost_usd']:.4f}")

        # Step 3: Store in vector database
        logger.info(f"\nStep 3: Storing in ChromaDB...")
        count = self.vector_store.add_documents(docs, embeddings, collection_type)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Get final collection statistics
        coll_stats = self.vector_store.get_collection_stats(collection_type)

        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"INGESTION COMPLETE - {collection_type.upper()}")
        logger.info(f"{'='*80}")
        logger.info(f"Documents added: {count}")
        logger.info(f"Total in collection: {coll_stats['count']}")
        logger.info(f"Sources: {', '.join(doc_stats['sources'])}")
        logger.info(f"Embedding cost: ${embed_stats['estimated_cost_usd']:.4f}")
        logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
        logger.info(f"{'='*80}\n")

        return {
            "status": "success",
            "collection_type": collection_type,
            "documents_added": count,
            "total_in_collection": coll_stats["count"],
            "sources": doc_stats["sources"],
            "embedding_cost_usd": embed_stats["estimated_cost_usd"],
            "time_elapsed": elapsed_time,
        }

    def ingest_all(self, reset: bool = False) -> dict:
        """
        Ingest both academic and skill collections.

        Args:
            reset: Whether to reset collections before ingesting

        Returns:
            Dictionary with combined statistics
        """
        logger.info("\n" + "="*80)
        logger.info("INGESTING ALL COLLECTIONS")
        logger.info("="*80 + "\n")

        results = {}

        # Ingest academic collection
        results["academic"] = self.ingest_collection(
            ACADEMIC_DATA_DIR, "academic", reset
        )

        # Ingest skill collection
        results["skill"] = self.ingest_collection(SKILL_DATA_DIR, "skill", reset)

        # Combined summary
        logger.info("\n" + "="*80)
        logger.info("COMPLETE INGESTION SUMMARY")
        logger.info("="*80)

        total_added = 0
        total_cost = 0.0
        total_time = 0.0

        for coll_type, stats in results.items():
            if stats["status"] == "success":
                total_added += stats["documents_added"]
                total_cost += stats["embedding_cost_usd"]
                total_time += stats["time_elapsed"]

        logger.info(f"Total documents added: {total_added}")
        logger.info(f"Total embedding cost: ${total_cost:.4f}")
        logger.info(f"Total time: {total_time:.2f} seconds")

        # Get final vector store statistics
        all_stats = self.vector_store.get_all_stats()
        logger.info(f"Total documents in database: {all_stats['total_documents']}")
        logger.info("="*80 + "\n")

        results["summary"] = {
            "total_documents_added": total_added,
            "total_cost_usd": total_cost,
            "total_time": total_time,
            "total_in_database": all_stats["total_documents"],
        }

        return results

    def check_status(self):
        """Check status of all collections."""
        logger.info("\n" + "="*80)
        logger.info("VECTOR STORE STATUS")
        logger.info("="*80 + "\n")

        stats = self.vector_store.get_all_stats()

        logger.info(f"Persist Directory: {stats['persist_directory']}")
        logger.info(f"Total Documents: {stats['total_documents']}\n")

        for coll_name, coll_stats in stats["collections"].items():
            if "count" in coll_stats:
                logger.info(f"{coll_name.upper()} Collection:")
                logger.info(f"  Documents: {coll_stats['count']}")
                if "sample_metadata" in coll_stats:
                    logger.info(f"  Sample source: {coll_stats['sample_metadata'].get('source_file', 'N/A')}")
            else:
                logger.info(f"{coll_name.upper()} Collection: {coll_stats.get('error', 'Unknown error')}")

        logger.info("\n" + "="*80 + "\n")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into RAG knowledge base with multi-provider support"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "fallback", "all"],
        default="fallback",
        help="Embedding provider (openai, gemini, fallback, or all for dual ingestion)",
    )
    parser.add_argument(
        "--collection",
        choices=["academic", "skill", "all"],
        default="all",
        help="Which collection to ingest (default: all)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset collection before ingesting",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check status of collections without ingesting",
    )

    args = parser.parse_args()

    try:
        # Handle dual provider ingestion
        if args.provider == "all":
            logger.info("\n" + "="*80)
            logger.info("DUAL PROVIDER INGESTION (OpenAI + Gemini)")
            logger.info("="*80 + "\n")

            # Ingest with OpenAI
            logger.info("\n>>> INGESTING TO OPENAI DATABASE <<<")
            pipeline_openai = KnowledgeIngestionPipeline(provider="openai")
            if args.collection == "all":
                results_openai = pipeline_openai.ingest_all(reset=args.reset)
            elif args.collection == "academic":
                results_openai = pipeline_openai.ingest_collection(
                    ACADEMIC_DATA_DIR, "academic", reset=args.reset
                )
            elif args.collection == "skill":
                results_openai = pipeline_openai.ingest_collection(
                    SKILL_DATA_DIR, "skill", reset=args.reset
                )

            # Ingest with Gemini
            logger.info("\n>>> INGESTING TO GEMINI DATABASE <<<")
            pipeline_gemini = KnowledgeIngestionPipeline(provider="gemini")
            if args.collection == "all":
                results_gemini = pipeline_gemini.ingest_all(reset=args.reset)
            elif args.collection == "academic":
                results_gemini = pipeline_gemini.ingest_collection(
                    ACADEMIC_DATA_DIR, "academic", reset=args.reset
                )
            elif args.collection == "skill":
                results_gemini = pipeline_gemini.ingest_collection(
                    SKILL_DATA_DIR, "skill", reset=args.reset
                )

            # Combined summary
            logger.info("\n" + "="*80)
            logger.info("DUAL INGESTION COMPLETE")
            logger.info("="*80)
            logger.info("OpenAI Database:")
            if isinstance(results_openai, dict) and "summary" in results_openai:
                logger.info(f"  Total documents: {results_openai['summary']['total_in_database']}")
                logger.info(f"  Total cost: ${results_openai['summary']['total_cost_usd']:.4f}")
            logger.info("\nGemini Database:")
            if isinstance(results_gemini, dict) and "summary" in results_gemini:
                logger.info(f"  Total documents: {results_gemini['summary']['total_in_database']}")
                logger.info(f"  Total cost: ${results_gemini['summary']['total_cost_usd']:.6f}")
            logger.info("="*80 + "\n")

        else:
            # Single provider ingestion
            pipeline = KnowledgeIngestionPipeline(provider=args.provider)

            # Check status only
            if args.status:
                pipeline.check_status()
                return

            # Ingest based on arguments
            if args.collection == "all":
                results = pipeline.ingest_all(reset=args.reset)
            elif args.collection == "academic":
                results = pipeline.ingest_collection(
                    ACADEMIC_DATA_DIR, "academic", reset=args.reset
                )
            elif args.collection == "skill":
                results = pipeline.ingest_collection(
                    SKILL_DATA_DIR, "skill", reset=args.reset
                )

        logger.info("✓ Ingestion completed successfully")

    except KeyboardInterrupt:
        logger.warning("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
