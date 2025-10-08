"""
LLM Provider Strategy Pattern Implementation

This module implements the Strategy pattern for LLM providers, allowing
flexible switching between OpenAI, Gemini, and other LLM providers with
automatic fallback support.
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

# Gemini imports (will be conditional based on availability)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    ChatGoogleGenerativeAI = None

logger = logging.getLogger(__name__)


class BaseLLMStrategy(ABC):
    """
    Abstract base class for LLM provider strategies.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        timeout: int = 60,
        max_retries: int = 2,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.kwargs = kwargs
        self.provider_name = "base"
        self.logger = logging.getLogger(f"llm_strategy.{self.provider_name}")

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """
        Get the configured LLM instance.
        Must be implemented by all concrete strategies.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this LLM provider is available (API key configured, etc.)
        """
        pass

    def log_usage(self, success: bool, error: Optional[str] = None):
        """Log LLM usage for monitoring"""
        status = "✅ Success" if success else "❌ Failed"
        self.logger.info(f"{status} - Provider: {self.provider_name}, Model: {self.model}")
        if error:
            self.logger.error(f"Error: {error}")


class OpenAIStrategy(BaseLLMStrategy):
    """
    Strategy for OpenAI LLM provider.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        timeout: int = 60,
        max_retries: int = 2,
        **kwargs
    ):
        super().__init__(model, temperature, timeout, max_retries, **kwargs)
        self.provider_name = "openai"

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.logger.warning("OpenAI API key not found")
            return False

        # Basic validation
        if not (api_key.startswith("sk-") and len(api_key) > 20):
            self.logger.warning("OpenAI API key format invalid")
            return False

        return True

    def get_llm(self) -> BaseChatModel:
        """Get configured OpenAI ChatModel"""
        if not self.is_available():
            raise ValueError("OpenAI API key not configured or invalid")

        self.logger.info(f"Initializing OpenAI model: {self.model}")

        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=self.max_retries,
            **self.kwargs
        )


class GeminiStrategy(BaseLLMStrategy):
    """
    Strategy for Google Gemini LLM provider.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.1,
        timeout: int = 60,
        max_retries: int = 2,
        **kwargs
    ):
        super().__init__(model, temperature, timeout, max_retries, **kwargs)
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

    def get_llm(self) -> BaseChatModel:
        """Get configured Gemini ChatModel"""
        if not self.is_available():
            raise ValueError("Gemini not available. Check API key and dependencies.")

        self.logger.info(f"Initializing Gemini model: {self.model}")

        return ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=self.max_retries,
            **self.kwargs
        )


class FallbackLLMStrategy(BaseLLMStrategy):
    """
    Strategy that tries multiple LLM providers in order with automatic fallback.

    Example:
        - Try OpenAI first
        - If OpenAI fails, automatically fallback to Gemini
        - If Gemini fails, raise error
    """

    def __init__(
        self,
        primary_strategy: BaseLLMStrategy,
        fallback_strategies: List[BaseLLMStrategy],
        **kwargs
    ):
        super().__init__(
            model=primary_strategy.model,
            temperature=primary_strategy.temperature,
            timeout=primary_strategy.timeout,
            **kwargs
        )
        self.provider_name = "fallback"
        self.primary_strategy = primary_strategy
        self.fallback_strategies = fallback_strategies
        self.current_strategy: Optional[BaseLLMStrategy] = None
        self.fallback_count = 0

    def is_available(self) -> bool:
        """Check if at least one provider is available"""
        if self.primary_strategy.is_available():
            return True

        for strategy in self.fallback_strategies:
            if strategy.is_available():
                return True

        return False

    def get_llm(self) -> BaseChatModel:
        """
        Get LLM with fallback logic.
        Tries primary first, then fallbacks in order.
        """
        # Try primary strategy
        if self.primary_strategy.is_available():
            try:
                self.logger.info(f"Using primary strategy: {self.primary_strategy.provider_name}")
                llm = self.primary_strategy.get_llm()
                self.current_strategy = self.primary_strategy
                self.fallback_count = 0
                return llm
            except Exception as e:
                self.logger.warning(
                    f"Primary strategy {self.primary_strategy.provider_name} failed: {str(e)}"
                )
        else:
            self.logger.warning(
                f"Primary strategy {self.primary_strategy.provider_name} not available"
            )

        # Try fallback strategies
        for idx, strategy in enumerate(self.fallback_strategies):
            if strategy.is_available():
                try:
                    self.fallback_count += 1
                    self.logger.warning(
                        f"🔄 FALLBACK #{self.fallback_count}: Trying {strategy.provider_name}"
                    )
                    llm = strategy.get_llm()
                    self.current_strategy = strategy
                    return llm
                except Exception as e:
                    self.logger.error(
                        f"Fallback strategy {strategy.provider_name} failed: {str(e)}"
                    )
                    continue
            else:
                self.logger.warning(f"Fallback strategy {strategy.provider_name} not available")

        # All strategies failed
        raise RuntimeError(
            "All LLM providers failed. Check your API keys and network connection."
        )

    def get_current_provider(self) -> str:
        """Get the name of the currently active provider"""
        if self.current_strategy:
            return self.current_strategy.provider_name
        return "none"

    def get_stats(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        return {
            "primary_provider": self.primary_strategy.provider_name,
            "current_provider": self.get_current_provider(),
            "fallback_count": self.fallback_count,
            "primary_available": self.primary_strategy.is_available(),
            "fallback_providers": [
                {
                    "name": s.provider_name,
                    "available": s.is_available()
                }
                for s in self.fallback_strategies
            ]
        }


class LLMStrategyWrapper:
    """
    Wrapper that provides a unified interface and additional monitoring with runtime fallback.
    """

    def __init__(self, strategy: BaseLLMStrategy):
        self.strategy = strategy
        self.llm = strategy.get_llm()
        self.usage_count = 0
        self.error_count = 0
        self.last_used = None
        self.logger = logging.getLogger(f"llm_wrapper.{strategy.provider_name}")

    def invoke(self, messages, **kwargs):
        """Invoke the LLM with monitoring and runtime fallback support"""
        try:
            self.usage_count += 1
            self.last_used = datetime.now()

            result = self.llm.invoke(messages, **kwargs)
            self.strategy.log_usage(success=True)

            return result
        except Exception as e:
            self.error_count += 1
            self.strategy.log_usage(success=False, error=str(e))

            # If this is a fallback strategy and primary failed, try fallback providers
            if isinstance(self.strategy, FallbackLLMStrategy):
                self.logger.warning(f"Primary provider failed: {str(e)}")

                # Try each fallback strategy
                for idx, fallback_strategy in enumerate(self.strategy.fallback_strategies):
                    if fallback_strategy.is_available():
                        try:
                            self.strategy.fallback_count += 1
                            self.logger.warning(
                                f"🔄 RUNTIME FALLBACK #{self.strategy.fallback_count}: "
                                f"Trying {fallback_strategy.provider_name}"
                            )

                            # Create new LLM from fallback strategy
                            fallback_llm = fallback_strategy.get_llm()

                            # Try invoking with fallback
                            result = fallback_llm.invoke(messages, **kwargs)

                            # Update to use this fallback provider
                            self.llm = fallback_llm
                            self.strategy.current_strategy = fallback_strategy
                            fallback_strategy.log_usage(success=True)

                            self.logger.info(
                                f"✅ Fallback successful! Now using {fallback_strategy.provider_name}"
                            )

                            return result

                        except Exception as fallback_error:
                            self.logger.error(
                                f"Fallback {fallback_strategy.provider_name} also failed: {str(fallback_error)}"
                            )
                            continue
                    else:
                        self.logger.warning(
                            f"Fallback {fallback_strategy.provider_name} not available"
                        )

                # All fallback attempts failed
                self.logger.error("All fallback providers failed")

            # Re-raise the original exception if no fallback succeeded
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = {
            "provider": self.strategy.provider_name,
            "model": self.strategy.model,
            "usage_count": self.usage_count,
            "error_count": self.error_count,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }

        # Add fallback stats if applicable
        if isinstance(self.strategy, FallbackLLMStrategy):
            stats.update(self.strategy.get_stats())

        return stats
