"""
OpenTelemetry Tracing Module for Career Planning System

This module provides distributed tracing capabilities using OpenTelemetry and Jaeger.
"""

from .setup_tracing import setup_tracing, get_tracer

__all__ = ["setup_tracing", "get_tracer"]
