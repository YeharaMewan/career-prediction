"""
OpenTelemetry Distributed Tracing Infrastructure

Provides distributed tracing capabilities for the ingestion system using OpenTelemetry.

Features:
- Automatic span creation with decorators
- Trace context propagation
- Integration with Jaeger backend
- Span attributes and events
- Error tracking
- Configurable sampling

Usage:
    from utils.tracing import trace_async, get_tracer, add_span_event

    @trace_async("process_file", attributes={"file.type": "pdf"})
    async def process_file(file_path):
        # Automatically creates a span
        add_span_event("file_loaded", {"size": 1024})
        # Your code here
        return result

Author: Career Planning System
Created: 2025
"""

import os
import logging
import functools
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
import yaml
from pathlib import Path

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


# Global tracer instance
_tracer: Optional[trace.Tracer] = None
_tracer_provider: Optional[TracerProvider] = None
_tracing_enabled: bool = False
_config: Dict = {}

# Setup logging
logger = logging.getLogger(__name__)


def load_tracing_config(config_path: str = "config/ingestion_config.yaml") -> Dict:
    """
    Load tracing configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Tracing configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {
            "enabled": os.getenv("ENABLE_TRACING", "false").lower() == "true",
            "service_name": os.getenv("OTEL_SERVICE_NAME", "career-planning-ingestion"),
            "sampling_rate": 1.0
        }

    with open(config_file, 'r') as f:
        full_config = yaml.safe_load(f)

    tracing_config = full_config.get("tracing", {})

    # Merge with environment variables (env vars take precedence)
    tracing_config["enabled"] = (
        os.getenv("ENABLE_TRACING", str(tracing_config.get("enabled", False))).lower() == "true"
    )
    tracing_config["service_name"] = os.getenv(
        "OTEL_SERVICE_NAME",
        tracing_config.get("service_name", "career-planning-ingestion")
    )

    return tracing_config


def initialize_tracing(config_path: str = "config/ingestion_config.yaml") -> bool:
    """
    Initialize OpenTelemetry tracing with Jaeger exporter.

    Args:
        config_path: Path to configuration file

    Returns:
        True if tracing initialized successfully, False otherwise
    """
    global _tracer, _tracer_provider, _tracing_enabled, _config

    # Load configuration
    _config = load_tracing_config(config_path)

    if not _config.get("enabled", False):
        logger.info("OpenTelemetry tracing is disabled")
        _tracing_enabled = False
        return False

    try:
        # Create resource with service name
        resource = Resource(attributes={
            SERVICE_NAME: _config.get("service_name", "career-planning-ingestion")
        })

        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)

        # Get OTLP endpoint from environment or use default
        otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://localhost:4317"
        )

        # Create OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True  # Use insecure for local Jaeger
        )

        # Create batch span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        _tracer_provider.add_span_processor(span_processor)

        # Set global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        # Get tracer instance
        _tracer = trace.get_tracer(
            __name__,
            tracer_provider=_tracer_provider
        )

        _tracing_enabled = True

        logger.info(
            f"OpenTelemetry tracing initialized: "
            f"service={_config.get('service_name')}, "
            f"endpoint={otlp_endpoint}"
        )

        return True

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")
        _tracing_enabled = False
        return False


def get_tracer() -> trace.Tracer:
    """
    Get the global tracer instance.

    Returns:
        OpenTelemetry tracer (or no-op tracer if tracing disabled)
    """
    global _tracer

    if _tracer is None:
        # Initialize on first use
        initialize_tracing()

    if _tracer is None:
        # Return no-op tracer if initialization failed
        return trace.get_tracer(__name__)

    return _tracer


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled"""
    return _tracing_enabled


def get_current_span() -> Optional[Span]:
    """
    Get the current active span.

    Returns:
        Current span or None if no active span
    """
    if not _tracing_enabled:
        return None

    return trace.get_current_span()


def add_span_attribute(key: str, value: Any):
    """
    Add an attribute to the current span.

    Args:
        key: Attribute key
        value: Attribute value (str, int, float, bool)
    """
    if not _tracing_enabled:
        return

    span = get_current_span()
    if span and span.is_recording():
        span.set_attribute(key, value)


def add_span_attributes(attributes: Dict[str, Any]):
    """
    Add multiple attributes to the current span.

    Args:
        attributes: Dictionary of attributes
    """
    if not _tracing_enabled or not attributes:
        return

    span = get_current_span()
    if span and span.is_recording():
        for key, value in attributes.items():
            if value is not None:  # Skip None values
                span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Add an event to the current span.

    Args:
        name: Event name
        attributes: Optional event attributes
    """
    if not _tracing_enabled:
        return

    span = get_current_span()
    if span and span.is_recording():
        span.add_event(name, attributes=attributes or {})


def set_span_error(error: Exception, record_exception: bool = True):
    """
    Mark the current span as failed and record error details.

    Args:
        error: Exception that occurred
        record_exception: Whether to record full exception details
    """
    if not _tracing_enabled:
        return

    span = get_current_span()
    if span and span.is_recording():
        span.set_status(Status(StatusCode.ERROR, str(error)))
        if record_exception:
            span.record_exception(error)


def set_span_status(status_code: StatusCode, description: Optional[str] = None):
    """
    Set the status of the current span.

    Args:
        status_code: Status code (OK or ERROR)
        description: Optional status description
    """
    if not _tracing_enabled:
        return

    span = get_current_span()
    if span and span.is_recording():
        span.set_status(Status(status_code, description))


@contextmanager
def create_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL
):
    """
    Context manager for creating a span.

    Args:
        name: Span name
        attributes: Optional span attributes
        kind: Span kind (INTERNAL, CLIENT, SERVER, PRODUCER, CONSUMER)

    Usage:
        with create_span("process_file", {"file.path": "/path/to/file"}):
            # Your code here
            pass
    """
    if not _tracing_enabled:
        yield None
        return

    tracer = get_tracer()

    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            add_span_attributes(attributes)
        yield span


def trace_sync(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL
) -> Callable:
    """
    Decorator for tracing synchronous functions.

    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add to span
        kind: Span kind

    Usage:
        @trace_sync("my_function", attributes={"component": "processor"})
        def my_function(arg1, arg2):
            # Automatically traced
            return result
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _tracing_enabled:
                return func(*args, **kwargs)

            tracer = get_tracer()

            with tracer.start_as_current_span(span_name, kind=kind) as span:
                # Add static attributes
                if attributes:
                    add_span_attributes(attributes)

                # Add function info
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    set_span_error(e)
                    raise

        return wrapper
    return decorator


def trace_async(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL
) -> Callable:
    """
    Decorator for tracing asynchronous functions.

    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add to span
        kind: Span kind

    Usage:
        @trace_async("my_async_function", attributes={"component": "processor"})
        async def my_async_function(arg1, arg2):
            # Automatically traced
            return result
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not _tracing_enabled:
                return await func(*args, **kwargs)

            tracer = get_tracer()

            with tracer.start_as_current_span(span_name, kind=kind) as span:
                # Add static attributes
                if attributes:
                    add_span_attributes(attributes)

                # Add function info
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    set_span_error(e)
                    raise

        return wrapper
    return decorator


def inject_trace_context(carrier: Dict[str, str]):
    """
    Inject trace context into a carrier dictionary (for cross-service propagation).

    Args:
        carrier: Dictionary to inject context into
    """
    if not _tracing_enabled:
        return

    propagator = TraceContextTextMapPropagator()
    propagator.inject(carrier)


def extract_trace_context(carrier: Dict[str, str]) -> Optional[trace.SpanContext]:
    """
    Extract trace context from a carrier dictionary.

    Args:
        carrier: Dictionary containing trace context

    Returns:
        Extracted span context or None
    """
    if not _tracing_enabled:
        return None

    propagator = TraceContextTextMapPropagator()
    context = propagator.extract(carrier)
    return trace.get_current_span(context).get_span_context()


def shutdown_tracing():
    """Shutdown tracing and flush all pending spans"""
    global _tracer_provider

    if _tracer_provider:
        _tracer_provider.shutdown()
        logger.info("OpenTelemetry tracing shutdown complete")


# ==============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON USE CASES
# ==============================================================================

def trace_file_operation(
    operation: str,
    file_path: str,
    collection: str,
    additional_attrs: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Decorator for file operations with standard attributes.

    Args:
        operation: Operation name (e.g., "load", "process", "ingest")
        file_path: Path to file
        collection: Collection name
        additional_attrs: Additional custom attributes
    """
    attrs = {
        "operation": operation,
        "file.path": file_path,
        "file.name": Path(file_path).name,
        "collection.name": collection
    }

    if additional_attrs:
        attrs.update(additional_attrs)

    return trace_async(
        name=f"file.{operation}",
        attributes=attrs
    )


def trace_db_operation(
    operation: str,
    collection: str,
    additional_attrs: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Decorator for vector database operations.

    Args:
        operation: Operation name (e.g., "add", "query", "delete")
        collection: Collection name
        additional_attrs: Additional custom attributes
    """
    attrs = {
        "operation": operation,
        "collection.name": collection,
        "db.system": "chromadb"
    }

    if additional_attrs:
        attrs.update(additional_attrs)

    return trace_async(
        name=f"vectordb.{operation}",
        attributes=attrs,
        kind=trace.SpanKind.CLIENT
    )


# Auto-initialize on import (if not already initialized)
if _tracer is None and os.getenv("ENABLE_TRACING", "false").lower() == "true":
    initialize_tracing()
