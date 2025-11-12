"""
Embedding Manager

Manages OpenAI embedding generation for documents and queries.
"""

import logging
from typing import List, Dict, Any
import time

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from .config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    OPENAI_API_KEY,
    EMBEDDING_BATCH_SIZE,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding generation using OpenAI's embedding API.

    Features:
    - Batch processing for efficiency
    - Error handling and retries
    - Cost tracking
    - Rate limiting awareness
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ):
        """
        Initialize embedding manager.

        Args:
            model: OpenAI embedding model name
            batch_size: Number of texts to embed per API call
        """
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in .env file."
            )

        self.model = model
        self.batch_size = batch_size
        self.dimensions = EMBEDDING_DIMENSIONS

        # Initialize OpenAI Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=OPENAI_API_KEY,
        )

        # Track usage
        self.total_tokens = 0
        self.total_requests = 0

        logger.info(
            f"EmbeddingManager initialized: model={model}, dimensions={self.dimensions}"
        )

    def embed_documents(
        self, documents: List[Document]
    ) -> tuple[List[Document], List[List[float]]]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of LangChain Document objects

        Returns:
            Tuple of (documents, embeddings)
        """
        if not documents:
            logger.warning("No documents to embed")
            return [], []

        logger.info(f"Generating embeddings for {len(documents)} documents...")

        # Extract texts from documents
        texts = [doc.page_content for doc in documents]

        # Generate embeddings in batches
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            try:
                logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)"
                )

                # Generate embeddings
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)

                # Update tracking
                self.total_requests += 1
                self.total_tokens += sum(len(text.split()) for text in batch_texts)

                # Brief pause to respect rate limits
                if batch_num < total_batches:
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error embedding batch {batch_num}: {e}")
                # Retry with smaller batch or skip
                logger.warning(f"Skipping batch {batch_num} due to error")
                # Add None placeholders for failed embeddings
                all_embeddings.extend([None] * len(batch_texts))
                continue

        # Filter out failed embeddings
        valid_docs = []
        valid_embeddings = []
        for doc, embedding in zip(documents, all_embeddings):
            if embedding is not None:
                valid_docs.append(doc)
                valid_embeddings.append(embedding)
            else:
                logger.warning(
                    f"Skipping document due to embedding failure: {doc.metadata.get('source_file', 'unknown')}"
                )

        logger.info(
            f"Successfully generated {len(valid_embeddings)} embeddings ({len(documents) - len(valid_embeddings)} failed)"
        )

        return valid_docs, valid_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query text.

        Args:
            query: Query string

        Returns:
            Embedding vector
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            embedding = self.embeddings.embed_query(query)
            self.total_requests += 1
            self.total_tokens += len(query.split())
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def estimate_cost(self, num_tokens: int = None) -> Dict[str, float]:
        """
        Estimate embedding cost.

        text-embedding-3-small: $0.02 per 1M tokens
        text-embedding-3-large: $0.13 per 1M tokens

        Args:
            num_tokens: Number of tokens (uses tracked total if None)

        Returns:
            Dictionary with cost estimates
        """
        tokens = num_tokens or self.total_tokens

        # Pricing per million tokens
        pricing = {
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13,
            "text-embedding-ada-002": 0.10,
        }

        cost_per_million = pricing.get(self.model, 0.02)
        estimated_cost = (tokens / 1_000_000) * cost_per_million

        return {
            "tokens": tokens,
            "cost_usd": round(estimated_cost, 4),
            "model": self.model,
            "requests": self.total_requests,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding generation statistics.

        Returns:
            Dictionary with statistics
        """
        cost_info = self.estimate_cost()

        stats = {
            "model": self.model,
            "dimensions": self.dimensions,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": cost_info["cost_usd"],
            "batch_size": self.batch_size,
        }

        return stats

    def reset_stats(self):
        """Reset usage tracking statistics."""
        self.total_tokens = 0
        self.total_requests = 0
        logger.info("Embedding statistics reset")


# Example usage
if __name__ == "__main__":
    from .document_processor import DocumentProcessor
    from .config import ACADEMIC_DATA_DIR

    # Initialize managers
    processor = DocumentProcessor()
    embedding_manager = EmbeddingManager()

    # Process documents
    print("\n=== Processing Academic Documents ===")
    documents = processor.process_directory(ACADEMIC_DATA_DIR, "academic")

    if documents:
        print(f"Loaded {len(documents)} document chunks")

        # Generate embeddings
        print("\n=== Generating Embeddings ===")
        docs, embeddings = embedding_manager.embed_documents(documents)

        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")

        # Get statistics
        stats = embedding_manager.get_stats()
        print(f"\n=== Embedding Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Test query embedding
        print("\n=== Test Query Embedding ===")
        query = "What universities offer computer science degrees?"
        query_embedding = embedding_manager.embed_query(query)
        print(f"Query: {query}")
        print(f"Embedding dimensions: {len(query_embedding)}")
    else:
        print("No documents found in academic directory")
