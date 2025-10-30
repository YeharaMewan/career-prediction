#!/bin/bash
# Entrypoint script for Career Planning Backend
# Supports optional OpenTelemetry auto-instrumentation

set -e

echo "========================================"
echo "Career Planning System - Backend"
echo "========================================"

# Check if tracing is enabled
if [ "${ENABLE_TRACING}" = "true" ]; then
    echo ""
    echo "🔍 OpenTelemetry Auto-Instrumentation ENABLED"
    echo "   Service: ${OTEL_SERVICE_NAME:-career-planning-system}"
    echo "   Endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT:-http://jaeger:4317}"
    echo "   Jaeger UI: http://localhost:16686"
    echo ""

    # Run with OpenTelemetry auto-instrumentation
    exec opentelemetry-instrument "$@"
else
    echo ""
    echo "ℹ️  OpenTelemetry Auto-Instrumentation DISABLED"
    echo "   Set ENABLE_TRACING=true to enable distributed tracing"
    echo "   See backend/tracing/README.md for more information"
    echo ""

    # Run normally without instrumentation
    exec "$@"
fi
