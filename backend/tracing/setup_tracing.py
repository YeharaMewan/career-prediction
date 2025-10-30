"""
Manual OpenTelemetry Tracing Setup

This module configures OpenTelemetry with Jaeger backend using the modern OTLP protocol.
It provides reusable functions for setting up distributed tracing in the Career Planning System.

Compatible with:
- OpenTelemetry v1.38.0 (October 2025)
- Jaeger v2.10.0 (recommended) or v1.74.0

Usage:
    from tracing.setup_tracing import setup_tracing, get_tracer, add_llm_token_attributes

    # Initialize tracing
    setup_tracing(service_name="my-service")

    # Get a tracer
    tracer = get_tracer(__name__)

    # Use the tracer with token tracking
    with tracer.start_as_current_span("llm_call") as span:
        result = llm.invoke(messages)
        add_llm_token_attributes(span, result)
"""

from typing import Optional, Dict, Any
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME


def setup_tracing(
    service_name: str = "career-planning-system",
    otlp_endpoint: str = "http://localhost:4317",
    insecure: bool = True
) -> TracerProvider:
    """
    Configure OpenTelemetry tracing with Jaeger backend.

    Args:
        service_name: Name of the service for identification in Jaeger
        otlp_endpoint: Jaeger OTLP gRPC endpoint (default: localhost:4317)
        insecure: Whether to use insecure connection (default: True for local dev)

    Returns:
        TracerProvider: Configured tracer provider instance
    """
    # Step 1: Configure the Resource with service name
    # This identifies your service in the Jaeger UI
    resource = Resource.create({
        SERVICE_NAME: service_name
    })

    # Step 2: Configure the TracerProvider with the resource
    provider = TracerProvider(resource=resource)

    # Step 3: Configure the OTLP Exporter for Jaeger
    # Uses gRPC protocol to send traces to Jaeger at port 4317
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        insecure=insecure
    )

    # Step 4: Configure the BatchSpanProcessor
    # Batches spans before sending to reduce network overhead
    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)

    # Step 5: Set the global tracer provider
    trace.set_tracer_provider(provider)

    print(f"OpenTelemetry tracing configured for service: {service_name}")
    print(f"Sending traces to Jaeger at: {otlp_endpoint}")

    return provider


def get_tracer(name: str = __name__) -> trace.Tracer:
    """
    Get a tracer instance for creating spans.

    Args:
        name: Name of the tracer (typically __name__ of the calling module)

    Returns:
        Tracer: A tracer instance for creating spans
    """
    return trace.get_tracer(name)


def shutdown_tracing():
    """
    Gracefully shutdown tracing and flush any remaining spans.
    Call this before application exit.
    """
    provider = trace.get_tracer_provider()
    if hasattr(provider, 'shutdown'):
        provider.shutdown()
        print("OpenTelemetry tracing shutdown complete")


# ============================================================================
# Token and Cost Tracking Utilities
# ============================================================================

def add_llm_token_attributes(
    span: trace.Span,
    token_data: Dict[str, Any]
) -> None:
    """
    Add LLM token usage and cost attributes to an OpenTelemetry span.

    This function adds standardized attributes for token usage and costs,
    making them visible in Jaeger UI for monitoring and analysis.

    Args:
        span: The OpenTelemetry span to add attributes to
        token_data: Dictionary with token and cost information containing:
            - input_tokens: Number of input tokens
            - output_tokens: Number of output tokens
            - total_tokens: Total tokens used
            - cost_usd: Cost in USD
            - model: Model name (optional)
            - pricing_found: Whether pricing data was found (optional)

    Example:
        with tracer.start_as_current_span("llm_call") as span:
            result = llm.invoke(messages)
            token_data = {
                'input_tokens': 100,
                'output_tokens': 50,
                'total_tokens': 150,
                'cost_usd': 0.0023,
                'model': 'gpt-4o'
            }
            add_llm_token_attributes(span, token_data)
    """
    if not token_data:
        return

    # Add token usage attributes
    if 'input_tokens' in token_data:
        span.set_attribute("llm.input_tokens", token_data['input_tokens'])

    if 'output_tokens' in token_data:
        span.set_attribute("llm.output_tokens", token_data['output_tokens'])

    if 'total_tokens' in token_data:
        span.set_attribute("llm.total_tokens", token_data['total_tokens'])

    # Add cost attribute
    if 'cost_usd' in token_data:
        span.set_attribute("llm.cost_usd", token_data['cost_usd'])

    # Add model information
    if 'model' in token_data:
        span.set_attribute("llm.model", token_data['model'])

    # Add pricing status
    if 'pricing_found' in token_data:
        span.set_attribute("llm.pricing_found", token_data['pricing_found'])


def add_llm_request_attributes(
    span: trace.Span,
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> None:
    """
    Add LLM request configuration attributes to a span.

    Args:
        span: The OpenTelemetry span to add attributes to
        model: Model name
        temperature: Temperature setting
        max_tokens: Maximum tokens setting
        **kwargs: Additional model parameters
    """
    span.set_attribute("llm.model", model)

    if temperature is not None:
        span.set_attribute("llm.temperature", temperature)

    if max_tokens is not None:
        span.set_attribute("llm.max_tokens", max_tokens)

    # Add any additional parameters
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(f"llm.{key}", value)


def extract_and_add_token_attributes_from_response(
    span: trace.Span,
    llm_response,
    model_name: str
) -> Optional[Dict[str, Any]]:
    """
    Extract token usage from LLM response and add to span automatically.

    This is a convenience function that extracts token data and adds it to the span.

    Args:
        span: The OpenTelemetry span to add attributes to
        llm_response: The LLM response object
        model_name: Name of the model used

    Returns:
        Dictionary with extracted token data, or None if extraction failed
    """
    try:
        # Import pricing calculator
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from utils.llm_pricing import calculate_cost

        # Extract usage metadata
        usage_metadata = None

        if hasattr(llm_response, 'usage_metadata'):
            usage_metadata = llm_response.usage_metadata
        elif hasattr(llm_response, 'response_metadata'):
            response_metadata = llm_response.response_metadata
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

            # Calculate cost
            cost_usd, pricing_found = calculate_cost(model_name, input_tokens, output_tokens)

            # Create token data
            token_data = {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'cost_usd': cost_usd,
                'model': model_name,
                'pricing_found': pricing_found
            }

            # Add to span
            add_llm_token_attributes(span, token_data)

            return token_data

    except Exception as e:
        print(f"Warning: Failed to extract token attributes: {e}")

    return None
