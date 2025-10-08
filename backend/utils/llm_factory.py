"""
LLM Factory for creating configured LLM instances.

This factory creates the appropriate LLM strategy based on environment
configuration, supporting OpenAI, Gemini, and automatic fallback.
"""
import os
import logging
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from .llm_strategy import (
    BaseLLMStrategy,
    OpenAIStrategy,
    GeminiStrategy,
    FallbackLLMStrategy,
    LLMStrategyWrapper
)

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory class for creating LLM strategy instances.

    Usage:
        # Get default LLM (based on environment config)
        llm = LLMFactory.create_llm()

        # Get specific provider
        llm = LLMFactory.create_llm(provider="openai")

        # Get with fallback
        llm = LLMFactory.create_llm(provider="fallback")
    """

    # Default model mappings for each provider
    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "gemini": "gemini-1.5-pro"
    }

    @classmethod
    def create_strategy(
        cls,
        provider: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        timeout: int = 60,
        max_retries: int = 2,
        **kwargs
    ) -> BaseLLMStrategy:
        """
        Create an LLM strategy based on provider name.

        Args:
            provider: Provider name ("openai", "gemini", "fallback")
            model: Model name (uses default if not specified)
            temperature: Temperature for generation
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
            **kwargs: Additional provider-specific parameters

        Returns:
            BaseLLMStrategy instance

        Raises:
            ValueError: If provider is unknown
        """
        provider = provider.lower()

        # Use default model if not specified
        if not model and provider in cls.DEFAULT_MODELS:
            model = cls.DEFAULT_MODELS[provider]

        if provider == "openai":
            return OpenAIStrategy(
                model=model or "gpt-4o",
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs
            )

        elif provider == "gemini":
            return GeminiStrategy(
                model=model or "gemini-1.5-pro",
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs
            )

        elif provider == "fallback":
            # Create fallback strategy with OpenAI primary and Gemini fallback
            primary = OpenAIStrategy(
                model=model or "gpt-4o",
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs
            )

            fallback = GeminiStrategy(
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs
            )

            return FallbackLLMStrategy(
                primary_strategy=primary,
                fallback_strategies=[fallback]
            )

        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported providers: openai, gemini, fallback"
            )

    @classmethod
    def create_llm(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        enable_fallback: Optional[bool] = None,
        **kwargs
    ) -> LLMStrategyWrapper:
        """
        Create a configured LLM with automatic fallback support.

        Reads configuration from environment variables:
        - LLM_PROVIDER: Provider to use (default: "openai")
        - OPENAI_MODEL: OpenAI model name (default: "gpt-4o")
        - GEMINI_MODEL: Gemini model name (default: "gemini-1.5-pro")
        - LLM_TEMPERATURE: Temperature (default: 0.1)
        - LLM_TIMEOUT: Timeout in seconds (default: 60)
        - LLM_MAX_RETRIES: Max retries (default: 2)
        - ENABLE_FALLBACK: Enable automatic fallback (default: true)

        Args:
            provider: Override provider from env
            model: Override model from env
            temperature: Override temperature from env
            timeout: Override timeout from env
            max_retries: Override max_retries from env
            enable_fallback: Override fallback setting from env
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMStrategyWrapper instance with configured LLM
        """
        # Load environment variables
        load_dotenv()

        # Read configuration with defaults
        provider = provider or os.getenv("LLM_PROVIDER", "openai")
        temperature = temperature if temperature is not None else float(
            os.getenv("LLM_TEMPERATURE", "0.1")
        )
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        max_retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", "2"))
        enable_fallback = enable_fallback if enable_fallback is not None else (
            os.getenv("ENABLE_FALLBACK", "true").lower() == "true"
        )

        # Determine model based on provider
        if not model:
            if provider == "openai":
                model = os.getenv("OPENAI_MODEL", "gpt-4o")
            elif provider == "gemini":
                model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

        # If fallback is enabled and provider is not already "fallback"
        if enable_fallback and provider != "fallback":
            logger.info(f"🔄 Fallback enabled: {provider} → gemini")
            provider = "fallback"

        # Create strategy
        logger.info(
            f"Creating LLM - Provider: {provider}, Model: {model}, "
            f"Temperature: {temperature}, Timeout: {timeout}s"
        )

        strategy = cls.create_strategy(
            provider=provider,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )

        # Wrap in monitoring wrapper
        return LLMStrategyWrapper(strategy)

    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """
        Check which LLM providers are available.

        Returns:
            Dict mapping provider name to availability status
        """
        providers = {}

        try:
            openai_strategy = OpenAIStrategy()
            providers["openai"] = openai_strategy.is_available()
        except Exception as e:
            logger.warning(f"OpenAI check failed: {e}")
            providers["openai"] = False

        try:
            gemini_strategy = GeminiStrategy()
            providers["gemini"] = gemini_strategy.is_available()
        except Exception as e:
            logger.warning(f"Gemini check failed: {e}")
            providers["gemini"] = False

        providers["fallback"] = providers["openai"] or providers["gemini"]

        return providers

    @classmethod
    def validate_configuration(cls) -> Dict[str, Any]:
        """
        Validate LLM configuration and return status report.

        Returns:
            Dict with configuration status and available providers
        """
        load_dotenv()

        config = {
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "enable_fallback": os.getenv("ENABLE_FALLBACK", "true").lower() == "true",
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "gemini_configured": bool(os.getenv("GOOGLE_API_KEY")),
            "available_providers": cls.get_available_providers()
        }

        # Validate configuration
        if config["enable_fallback"]:
            if not (config["openai_configured"] or config["gemini_configured"]):
                config["status"] = "error"
                config["message"] = "Fallback enabled but no providers configured"
            else:
                config["status"] = "ok"
                config["message"] = "Fallback configuration valid"
        else:
            primary_provider = config["provider"]
            if primary_provider == "openai" and not config["openai_configured"]:
                config["status"] = "error"
                config["message"] = "OpenAI selected but not configured"
            elif primary_provider == "gemini" and not config["gemini_configured"]:
                config["status"] = "error"
                config["message"] = "Gemini selected but not configured"
            else:
                config["status"] = "ok"
                config["message"] = f"Provider {primary_provider} configured"

        return config


# Convenience function for quick LLM creation
def create_llm(**kwargs) -> LLMStrategyWrapper:
    """
    Convenience function to create LLM with default settings.

    Usage:
        from utils.llm_factory import create_llm

        llm = create_llm()
        response = llm.invoke("Hello, world!")
    """
    return LLMFactory.create_llm(**kwargs)


# For backward compatibility
def get_llm(**kwargs):
    """Alias for create_llm()"""
    return create_llm(**kwargs)
