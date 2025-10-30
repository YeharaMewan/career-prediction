"""
LLM Pricing Configuration and Cost Calculation

This module provides pricing information for various LLM providers and models,
and functions to calculate costs based on token usage.

Pricing is per 1M tokens (as of 2025) - converted to per-token rates internally.
"""
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing information for a specific model"""
    input_price_per_1m: float  # USD per 1M input tokens
    output_price_per_1m: float  # USD per 1M output tokens

    @property
    def input_price_per_token(self) -> float:
        """Get price per single input token"""
        return self.input_price_per_1m / 1_000_000

    @property
    def output_price_per_token(self) -> float:
        """Get price per single output token"""
        return self.output_price_per_1m / 1_000_000


# Pricing data for various models (updated January 2025)
# Source: Official provider pricing pages
MODEL_PRICING: Dict[str, ModelPricing] = {
    # OpenAI Models
    "gpt-4o": ModelPricing(
        input_price_per_1m=2.50,   # $2.50 per 1M input tokens
        output_price_per_1m=10.00  # $10.00 per 1M output tokens
    ),
    "gpt-4o-mini": ModelPricing(
        input_price_per_1m=0.150,  # $0.15 per 1M input tokens
        output_price_per_1m=0.600  # $0.60 per 1M output tokens
    ),
    "gpt-4-turbo": ModelPricing(
        input_price_per_1m=10.00,  # $10.00 per 1M input tokens
        output_price_per_1m=30.00  # $30.00 per 1M output tokens
    ),
    "gpt-4": ModelPricing(
        input_price_per_1m=30.00,  # $30.00 per 1M input tokens
        output_price_per_1m=60.00  # $60.00 per 1M output tokens
    ),
    "gpt-3.5-turbo": ModelPricing(
        input_price_per_1m=0.50,   # $0.50 per 1M input tokens
        output_price_per_1m=1.50   # $1.50 per 1M output tokens
    ),

    # Google Gemini Models
    "gemini-1.5-pro": ModelPricing(
        input_price_per_1m=1.25,   # $1.25 per 1M input tokens
        output_price_per_1m=5.00   # $5.00 per 1M output tokens
    ),
    "gemini-1.5-flash": ModelPricing(
        input_price_per_1m=0.075,  # $0.075 per 1M input tokens
        output_price_per_1m=0.30   # $0.30 per 1M output tokens
    ),
    "gemini-pro": ModelPricing(
        input_price_per_1m=0.50,   # $0.50 per 1M input tokens
        output_price_per_1m=1.50   # $1.50 per 1M output tokens
    ),
}


def get_model_pricing(model_name: str) -> Optional[ModelPricing]:
    """
    Get pricing information for a specific model.

    Args:
        model_name: Name of the model (e.g., "gpt-4o", "gemini-1.5-pro")

    Returns:
        ModelPricing object or None if model not found
    """
    # Try exact match first
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    # Try case-insensitive match
    model_name_lower = model_name.lower()
    for key, pricing in MODEL_PRICING.items():
        if key.lower() == model_name_lower:
            return pricing

    # Try partial match (for versioned models like "gpt-4o-2024-08-06")
    for key, pricing in MODEL_PRICING.items():
        if model_name_lower.startswith(key.lower()):
            return pricing

    return None


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int
) -> Tuple[float, bool]:
    """
    Calculate the cost for an LLM API call.

    Args:
        model_name: Name of the model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Tuple of (cost_in_usd, pricing_found)
        - cost_in_usd: The calculated cost in USD
        - pricing_found: True if pricing data was found, False if estimated
    """
    pricing = get_model_pricing(model_name)

    if pricing is None:
        # Use fallback pricing if model not found
        # Estimate based on GPT-4o pricing as reasonable default
        pricing = MODEL_PRICING["gpt-4o"]
        pricing_found = False
    else:
        pricing_found = True

    # Calculate costs
    input_cost = input_tokens * pricing.input_price_per_token
    output_cost = output_tokens * pricing.output_price_per_token
    total_cost = input_cost + output_cost

    return total_cost, pricing_found


def calculate_detailed_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int
) -> Dict[str, any]:
    """
    Calculate detailed cost breakdown for an LLM API call.

    Args:
        model_name: Name of the model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Dictionary with detailed cost breakdown
    """
    pricing = get_model_pricing(model_name)
    pricing_found = pricing is not None

    if pricing is None:
        pricing = MODEL_PRICING["gpt-4o"]
        model_used_for_pricing = "gpt-4o (fallback)"
    else:
        model_used_for_pricing = model_name

    # Calculate costs
    input_cost = input_tokens * pricing.input_price_per_token
    output_cost = output_tokens * pricing.output_price_per_token
    total_cost = input_cost + output_cost
    total_tokens = input_tokens + output_tokens

    return {
        "model": model_name,
        "model_used_for_pricing": model_used_for_pricing,
        "pricing_found": pricing_found,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "input_price_per_1m": pricing.input_price_per_1m,
        "output_price_per_1m": pricing.output_price_per_1m
    }


def format_cost(cost_usd: float) -> str:
    """
    Format cost in a human-readable way.

    Args:
        cost_usd: Cost in USD

    Returns:
        Formatted string (e.g., "$0.0023" or "< $0.0001")
    """
    if cost_usd < 0.0001:
        return "< $0.0001"
    elif cost_usd < 0.01:
        return f"${cost_usd:.6f}"
    elif cost_usd < 1.0:
        return f"${cost_usd:.4f}"
    else:
        return f"${cost_usd:.2f}"


def estimate_cost_for_request(
    model_name: str,
    prompt_length: int,
    expected_response_length: int = 500
) -> float:
    """
    Estimate cost for a request based on character counts.
    Uses rough approximation: ~4 characters per token.

    Args:
        model_name: Name of the model
        prompt_length: Length of prompt in characters
        expected_response_length: Expected response length in characters

    Returns:
        Estimated cost in USD
    """
    # Rough approximation: 4 characters per token
    estimated_input_tokens = prompt_length // 4
    estimated_output_tokens = expected_response_length // 4

    cost, _ = calculate_cost(model_name, estimated_input_tokens, estimated_output_tokens)
    return cost


# Example usage
if __name__ == "__main__":
    # Example 1: Calculate cost for a GPT-4o call
    model = "gpt-4o"
    input_tok = 1000
    output_tok = 500

    cost, found = calculate_cost(model, input_tok, output_tok)
    print(f"\nExample 1: {model}")
    print(f"Input: {input_tok} tokens, Output: {output_tok} tokens")
    print(f"Cost: {format_cost(cost)} (pricing found: {found})")

    # Example 2: Detailed cost breakdown
    detailed = calculate_detailed_cost("gpt-4o", 1000, 500)
    print(f"\nExample 2: Detailed breakdown")
    for key, value in detailed.items():
        print(f"  {key}: {value}")

    # Example 3: Unknown model (uses fallback)
    cost, found = calculate_cost("unknown-model", 1000, 500)
    print(f"\nExample 3: Unknown model")
    print(f"Cost: {format_cost(cost)} (pricing found: {found})")
