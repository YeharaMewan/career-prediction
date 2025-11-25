"""
Embedding Manager

Manages embedding generation for documents and queries with automatic fallback support.
Supports OpenAI, Gemini, and automatic fallback between providers.
"""

import logging
from typing import List, Dict, Any, Optional
import time

from langchain_core.documents import Document

from .config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_PROVIDER,
    ENABLE_EMBEDDING_FALLBACK,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_DIMENSIONS,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_EMBEDDING_DIMENSIONS,
)
from .embedding_strategy import (
    create_embedding_strategy,
    BaseEmbeddingStrategy,
    FallbackEmbeddingStrategy,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding generation with automatic fallback support.

    Features:
    - Multi-provider support (OpenAI, Gemini)
    - Automatic fallback when primary provider fails
    - Batch processing for efficiency
    - Error handling and retries
    - Cost tracking per provider
    - Rate limiting awareness
    """

    def __init__(
        self,
        provider: str = EMBEDDING_PROVIDER,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        enable_fallback: bool = ENABLE_EMBEDDING_FALLBACK,
    ):
        """
        Initialize embedding manager with provider strategy.

        Args:
            provider: Embedding provider ("openai", "gemini", or "fallback")
            model: Model name (uses config defaults if None)
            dimensions: Embedding dimensions (uses config defaults if None)
            batch_size: Number of texts to embed per API call
            enable_fallback: Whether to enable automatic fallback
        """
        self.provider = provider
        self.batch_size = batch_size
        self.enable_fallback = enable_fallback

        # Create embedding strategy
        try:
            self.strategy: BaseEmbeddingStrategy = create_embedding_strategy(
                provider=provider,
                model=model,
                dimensions=dimensions,
                enable_fallback=enable_fallback,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.error(f"Failed to create embedding strategy: {e}")
            raise

        # Get embeddings instance from strategy
        self.embeddings = self.strategy.get_embeddings()

        # Get active configuration
        self.model = self.strategy.model
        self.dimensions = self.strategy.dimensions
        self.active_provider = self.strategy.provider_name

        # Track usage
        self.total_tokens = 0
        self.total_requests = 0
        self.fallback_occurred = False

        logger.info(
            f"EmbeddingManager initialized: provider={self.active_provider}, "
            f"model={self.model}, dimensions={self.dimensions}"
        )

    def embed_documents(
        self, documents: List[Document]
    ) -> tuple[List[Document], List[List[float]]]:
        """
        Generate embeddings for a list of documents with automatic fallback.

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

                # Try fallback if available and not already fallen back
                if self._try_fallback():
                    logger.info(f"Retrying batch {batch_num} with fallback provider...")
                    try:
                        batch_embeddings = self.embeddings.embed_documents(batch_texts)
                        all_embeddings.extend(batch_embeddings)
                        self.total_requests += 1
                        self.total_tokens += sum(len(text.split()) for text in batch_texts)
                        continue
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")

                # If fallback failed or unavailable, skip batch
                logger.warning(f"Skipping batch {batch_num} due to error")
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
        Generate embedding for a single query text with automatic fallback.

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

            # Try fallback if available
            if self._try_fallback():
                logger.info("Retrying query embedding with fallback provider...")
                try:
                    embedding = self.embeddings.embed_query(query)
                    self.total_requests += 1
                    self.total_tokens += len(query.split())
                    return embedding
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise fallback_error
            else:
                raise

    def _try_fallback(self) -> bool:
        """
        Try to trigger fallback to secondary provider.

        Returns:
            True if fallback was triggered, False otherwise
        """
        if self.fallback_occurred:
            # Already using fallback
            return False

        if not self.enable_fallback:
            # Fallback disabled
            return False

        if not isinstance(self.strategy, FallbackEmbeddingStrategy):
            # Not using fallback strategy
            return False

        # Trigger fallback
        success = self.strategy.trigger_fallback()
        if success:
            self.fallback_occurred = True
            self.embeddings = self.strategy.get_embeddings()
            self.active_provider = self.strategy.get_active_provider()
            self.dimensions = self.strategy.get_active_dimensions()
            logger.warning(f"Switched to fallback provider: {self.active_provider}")
            return True

        return False

    def get_active_provider(self) -> str:
        """Get the currently active embedding provider"""
        if isinstance(self.strategy, FallbackEmbeddingStrategy):
            return self.strategy.get_active_provider()
        return self.active_provider

    def estimate_cost(self, num_tokens: int = None) -> Dict[str, float]:
        """
        Estimate embedding cost using active provider pricing.

        Args:
            num_tokens: Number of tokens (uses tracked total if None)

        Returns:
            Dictionary with cost estimates
        """
        tokens = num_tokens or self.total_tokens

        # Get cost per token from active strategy
        cost_per_token = self.strategy.get_cost_per_token()
        estimated_cost = self.strategy.calculate_embedding_cost(tokens)

        return {
            "tokens": tokens,
            "cost_usd": round(estimated_cost, 6),
            "model": self.model,
            "provider": self.get_active_provider(),
            "requests": self.total_requests,
            "fallback_occurred": self.fallback_occurred,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding generation statistics including provider info.

        Returns:
            Dictionary with statistics
        """
        cost_info = self.estimate_cost()

        stats = {
            "provider": self.get_active_provider(),
            "model": self.model,
            "dimensions": self.dimensions,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": cost_info["cost_usd"],
            "batch_size": self.batch_size,
            "fallback_occurred": self.fallback_occurred,
            "fallback_enabled": self.enable_fallback,
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
