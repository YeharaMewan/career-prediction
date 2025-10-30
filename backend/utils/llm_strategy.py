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

# Import pricing calculator
from .llm_pricing import calculate_cost, calculate_detailed_cost

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

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.last_request_tokens = None
        self.last_request_cost = None

    def invoke(self, messages, **kwargs):
        """Invoke the LLM with monitoring and runtime fallback support"""
        try:
            self.usage_count += 1
            self.last_used = datetime.now()

            result = self.llm.invoke(messages, **kwargs)
            self.strategy.log_usage(success=True)

            # Extract token usage from response
            self._extract_and_track_tokens(result)

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

                            # Extract token usage from fallback response
                            self._extract_and_track_tokens(result)

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

    def _extract_and_track_tokens(self, result) -> Optional[Dict[str, Any]]:
        """
        Extract token usage from LLM response and track costs.

        Args:
            result: The LLM response object

        Returns:
            Dictionary with token and cost information, or None if unavailable
        """
        try:
            # Try to extract usage_metadata (new LangChain format)
            usage_metadata = None

            if hasattr(result, 'usage_metadata'):
                usage_metadata = result.usage_metadata
            elif hasattr(result, 'response_metadata'):
                # Try extracting from response_metadata (OpenAI format)
                response_metadata = result.response_metadata
                if isinstance(response_metadata, dict) and 'token_usage' in response_metadata:
                    token_usage = response_metadata['token_usage']
                    usage_metadata = {
                        'input_tokens': token_usage.get('prompt_tokens', 0),
                        'output_tokens': token_usage.get('completion_tokens', 0),
                        'total_tokens': token_usage.get('total_tokens', 0)
                    }

            if usage_metadata:
                # Extract token counts
                input_tokens = usage_metadata.get('input_tokens', 0)
                output_tokens = usage_metadata.get('output_tokens', 0)
                total_tokens = usage_metadata.get('total_tokens', input_tokens + output_tokens)

                # Get model name
                model_name = self.strategy.model

                # Calculate cost
                cost_usd, pricing_found = calculate_cost(model_name, input_tokens, output_tokens)

                # Update cumulative totals
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_tokens += total_tokens
                self.total_cost_usd += cost_usd

                # Store last request info
                self.last_request_tokens = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens
                }
                self.last_request_cost = cost_usd

                # Log token usage
                self.logger.info(
                    f"📊 Token Usage - Input: {input_tokens}, Output: {output_tokens}, "
                    f"Total: {total_tokens}, Cost: ${cost_usd:.6f}"
                )

                # Create dedicated LLM span with token data for easy visibility in Jaeger
                try:
                    from opentelemetry import trace
                    tracer = trace.get_tracer(__name__)

                    # Create a dedicated child span for this LLM call
                    span_name = f"LLM Call: {model_name}"
                    with tracer.start_as_current_span(span_name) as llm_span:
                        # Add comprehensive token and cost attributes
                        llm_span.set_attribute("llm.model", model_name)
                        llm_span.set_attribute("llm.provider", self.strategy.provider_name)
                        llm_span.set_attribute("llm.input_tokens", input_tokens)
                        llm_span.set_attribute("llm.output_tokens", output_tokens)
                        llm_span.set_attribute("llm.total_tokens", total_tokens)
                        llm_span.set_attribute("llm.cost_usd", cost_usd)
                        llm_span.set_attribute("llm.pricing_found", pricing_found)

                        # Add formatted cost for easy reading
                        llm_span.set_attribute("llm.cost_formatted", f"${cost_usd:.6f}")

                        self.logger.debug(f"✓ Created LLM span '{span_name}' with token data")
                except ImportError:
                    # OpenTelemetry not available
                    pass
                except Exception as otel_error:
                    self.logger.debug(f"Could not create LLM span: {otel_error}")

                return {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'cost_usd': cost_usd,
                    'pricing_found': pricing_found,
                    'model': model_name
                }

        except Exception as e:
            self.logger.warning(f"Failed to extract token usage: {e}")

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics including token usage and costs"""
        stats = {
            "provider": self.strategy.provider_name,
            "model": self.strategy.model,
            "usage_count": self.usage_count,
            "error_count": self.error_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,

            # Token usage statistics
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),

            # Last request info
            "last_request_tokens": self.last_request_tokens,
            "last_request_cost_usd": round(self.last_request_cost, 6) if self.last_request_cost else None
        }

        # Add fallback stats if applicable
        if isinstance(self.strategy, FallbackLLMStrategy):
            stats.update(self.strategy.get_stats())

        return stats
