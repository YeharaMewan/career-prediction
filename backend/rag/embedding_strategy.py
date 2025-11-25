"""
Embedding Provider Strategy Pattern Implementation

This module implements the Strategy pattern for embedding providers, allowing
flexible switching between OpenAI, Gemini, and other embedding providers with
automatic fallback support.
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from datetime import datetime

# Embedding pricing is defined within this module
# (OpenAI: $0.02/1M tokens, Gemini: essentially free)

# OpenAI embeddings
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIEmbeddings = None

# Gemini embeddings
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    GoogleGenerativeAIEmbeddings = None

logger = logging.getLogger(__name__)


class BaseEmbeddingStrategy(ABC):
    """
    Abstract base class for embedding provider strategies.
    """

    def __init__(
        self,
        model: str,
        dimensions: Optional[int] = None,
        batch_size: int = 50,
        **kwargs
    ):
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.provider_name = "base"
        self.logger = logging.getLogger(f"embedding_strategy.{self.provider_name}")

    @abstractmethod
    def get_embeddings(self):
        """
        Get the configured embeddings instance.
        Must be implemented by all concrete strategies.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this embedding provider is available (API key configured, etc.)
        """
        pass

    @abstractmethod
    def get_cost_per_token(self) -> float:
        """
        Get the cost per token for this embedding provider.
        """
        pass

    def calculate_embedding_cost(self, num_tokens: int) -> float:
        """Calculate embedding cost for given number of tokens"""
        cost_per_token = self.get_cost_per_token()
        return (num_tokens * cost_per_token) / 1_000_000  # Cost is per 1M tokens

    def log_usage(self, success: bool, num_tokens: int = 0, error: Optional[str] = None):
        """Log embedding usage for monitoring"""
        status = "✅ Success" if success else "❌ Failed"
        cost = self.calculate_embedding_cost(num_tokens) if success else 0.0
        self.logger.info(
            f"{status} - Provider: {self.provider_name}, Model: {self.model}, "
            f"Tokens: {num_tokens}, Cost: ${cost:.6f}"
        )
        if error:
            self.logger.error(f"Error: {error}")


class OpenAIEmbeddingStrategy(BaseEmbeddingStrategy):
    """
    Strategy for OpenAI embedding provider.
    Supports: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 50,
        **kwargs
    ):
        super().__init__(model, dimensions, batch_size, **kwargs)
        self.provider_name = "openai"

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured"""
        if not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI SDK not installed. Install: pip install langchain-openai")
            return False

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.logger.warning("OpenAI API key not found")
            return False

        # Basic validation
        if not (api_key.startswith("sk-") and len(api_key) > 20):
            self.logger.warning("OpenAI API key format invalid")
            return False

        return True

    def get_cost_per_token(self) -> float:
        """Get cost per token for OpenAI embeddings"""
        # OpenAI embedding pricing (per 1M tokens)
        pricing = {
            "text-embedding-3-small": 0.02,  # $0.02 per 1M tokens
            "text-embedding-3-large": 0.13,  # $0.13 per 1M tokens
            "text-embedding-ada-002": 0.10,  # $0.10 per 1M tokens
        }
        return pricing.get(self.model, 0.02)  # Default to 3-small pricing

    def get_embeddings(self):
        """Get configured OpenAI Embeddings"""
        if not self.is_available():
            raise ValueError("OpenAI API key not configured or invalid")

        self.logger.info(f"Initializing OpenAI embeddings: {self.model} ({self.dimensions} dimensions)")

        return OpenAIEmbeddings(
            model=self.model,
            dimensions=self.dimensions,
            **self.kwargs
        )


class GeminiEmbeddingStrategy(BaseEmbeddingStrategy):
    """
    Strategy for Google Gemini embedding provider.
    Supports: models/text-embedding-004, models/embedding-001
    """

    def __init__(
        self,
        model: str = "models/text-embedding-004",
        dimensions: int = 768,
        batch_size: int = 50,
        **kwargs
    ):
        super().__init__(model, dimensions, batch_size, **kwargs)
        self.provider_name = "gemini"

    def is_available(self) -> bool:
        """Check if Gemini is available"""
        if not GEMINI_AVAILABLE:
            self.logger.warning("Gemini SDK not installed. Install: pip install langchain-google-genai")
            return False

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            self.logger.warning("Google API key not found")
            return False

        return True

    def get_cost_per_token(self) -> float:
        """Get cost per token for Gemini embeddings"""
        # Gemini embedding pricing (per 1M tokens)
        # text-embedding-004 is FREE for first 1M tokens, then very cheap
        return 0.00001  # Essentially free, conservative estimate

    def get_embeddings(self):
        """Get configured Gemini Embeddings"""
        if not self.is_available():
            raise ValueError("Google API key not configured or Gemini SDK not installed")

        self.logger.info(f"Initializing Gemini embeddings: {self.model} ({self.dimensions} dimensions)")

        # Note: Gemini embeddings don't support dimensions parameter
        return GoogleGenerativeAIEmbeddings(
            model=self.model,
            task_type="retrieval_document",  # For document embeddings
            **self.kwargs
        )


class FallbackEmbeddingStrategy(BaseEmbeddingStrategy):
    """
    Fallback strategy that tries OpenAI first, then falls back to Gemini.

    This provides automatic resilience when the primary provider is unavailable.
    """

    def __init__(
        self,
        primary_model: str = "text-embedding-3-small",
        fallback_model: str = "models/text-embedding-004",
        primary_dimensions: int = 1536,
        fallback_dimensions: int = 768,
        batch_size: int = 50,
        **kwargs
    ):
        super().__init__(primary_model, primary_dimensions, batch_size, **kwargs)
        self.provider_name = "fallback"

        # Primary strategy (OpenAI)
        self.primary_strategy = OpenAIEmbeddingStrategy(
            model=primary_model,
            dimensions=primary_dimensions,
            batch_size=batch_size,
            **kwargs
        )

        # Fallback strategy (Gemini)
        self.fallback_strategy = GeminiEmbeddingStrategy(
            model=fallback_model,
            dimensions=fallback_dimensions,
            batch_size=batch_size,
            **kwargs
        )

        # Track which strategy is currently active
        self.active_strategy: BaseEmbeddingStrategy = None
        self.fallback_occurred = False

        # Determine active strategy at initialization
        self._determine_active_strategy()

    def _determine_active_strategy(self):
        """
        Determine which strategy to use at initialization.
        This is the initialization-time fallback.
        """
        if self.primary_strategy.is_available():
            self.active_strategy = self.primary_strategy
            self.logger.info("Using PRIMARY embedding provider: OpenAI")
        elif self.fallback_strategy.is_available():
            self.active_strategy = self.fallback_strategy
            self.fallback_occurred = True
            self.logger.warning("PRIMARY unavailable. Using FALLBACK embedding provider: Gemini")
        else:
            self.logger.error("CRITICAL: Neither OpenAI nor Gemini embeddings available!")
            raise ValueError("No embedding provider available. Check API keys.")

    def is_available(self) -> bool:
        """Check if at least one provider is available"""
        return self.primary_strategy.is_available() or self.fallback_strategy.is_available()

    def get_cost_per_token(self) -> float:
        """Get cost per token for active strategy"""
        return self.active_strategy.get_cost_per_token()

    def get_embeddings(self):
        """Get embeddings with automatic fallback"""
        if not self.active_strategy:
            raise ValueError("No embedding provider available")

        return self.active_strategy.get_embeddings()

    def get_active_provider(self) -> str:
        """Get the name of the currently active provider"""
        return self.active_strategy.provider_name if self.active_strategy else "none"

    def get_active_dimensions(self) -> int:
        """Get the dimensions of the currently active provider"""
        return self.active_strategy.dimensions if self.active_strategy else 0

    def trigger_fallback(self):
        """
        Manually trigger fallback to secondary provider.
        This is used for runtime fallback when primary provider fails during operation.
        """
        if self.fallback_occurred:
            self.logger.warning("Already using fallback provider. Cannot fallback further.")
            return False

        if not self.fallback_strategy.is_available():
            self.logger.error("Fallback provider not available!")
            return False

        self.logger.warning("FALLBACK #1: Switching from OpenAI to Gemini embeddings")
        self.active_strategy = self.fallback_strategy
        self.fallback_occurred = True
        return True


def create_embedding_strategy(
    provider: str = "fallback",
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
    enable_fallback: bool = True,
    **kwargs
) -> BaseEmbeddingStrategy:
    """
    Factory function to create embedding strategies.

    Args:
        provider: "openai", "gemini", or "fallback"
        model: Model name (uses defaults if not specified)
        dimensions: Embedding dimensions (uses defaults if not specified)
        enable_fallback: Whether to enable automatic fallback (only for fallback provider)
        **kwargs: Additional arguments passed to embedding provider

    Returns:
        BaseEmbeddingStrategy instance

    Examples:
        # OpenAI only
        strategy = create_embedding_strategy("openai")

        # Gemini only
        strategy = create_embedding_strategy("gemini")

        # Automatic fallback (recommended)
        strategy = create_embedding_strategy("fallback")
    """
    logger.info(f"Creating embedding strategy: {provider}")

    if provider == "openai":
        return OpenAIEmbeddingStrategy(
            model=model or "text-embedding-3-small",
            dimensions=dimensions or 1536,
            **kwargs
        )
    elif provider == "gemini":
        return GeminiEmbeddingStrategy(
            model=model or "models/text-embedding-004",
            dimensions=dimensions or 768,
            **kwargs
        )
    elif provider == "fallback":
        if enable_fallback:
            return FallbackEmbeddingStrategy(
                primary_model=model or "text-embedding-3-small",
                fallback_model="models/text-embedding-004",
                primary_dimensions=dimensions or 1536,
                fallback_dimensions=768,
                **kwargs
            )
        else:
            # Fallback disabled, use OpenAI only
            return OpenAIEmbeddingStrategy(
                model=model or "text-embedding-3-small",
                dimensions=dimensions or 1536,
                **kwargs
            )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Use 'openai', 'gemini', or 'fallback'")
