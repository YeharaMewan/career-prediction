# Docker Setup with OpenTelemetry Tracing

This guide explains how to use Docker Compose to run the Career Planning System with distributed tracing enabled.

## Quick Start

### 1. Enable Tracing in Environment Variables

Edit your `backend/.env` file and set:

```bash
ENABLE_TRACING=true
```

Or set it when running docker-compose:

```bash
ENABLE_TRACING=true docker-compose up
```

### 2. Start All Services

```bash
docker-compose up -d
```

This will start:
- **Backend** (Career Planning System) on port 8000
- **Frontend** (React UI) on port 3000
- **Jaeger** (Tracing Backend) on port 16686

### 3. Access Services

- **Application**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Jaeger UI**: http://localhost:16686

## Configuration Options

### Environment Variables

The following environment variables control tracing behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_TRACING` | `false` | Enable/disable auto-instrumentation |
| `OTEL_SERVICE_NAME` | `career-planning-system` | Service name in Jaeger UI |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://jaeger:4317` | Jaeger OTLP endpoint |
| `OTEL_PYTHON_LANGCHAIN_CAPTURE_INPUTS` | `true` | Capture LangChain inputs |
| `OTEL_PYTHON_LANGCHAIN_CAPTURE_OUTPUTS` | `true` | Capture LangChain outputs |

### Docker Compose Override

Create a `docker-compose.override.yml` file to customize settings:

```yaml
version: "3.8"

services:
  backend:
    environment:
      - ENABLE_TRACING=true
      - OTEL_SERVICE_NAME=my-custom-service-name
```

## How Auto-Instrumentation Works

When `ENABLE_TRACING=true`:

1. The `entrypoint.sh` script wraps the application with `opentelemetry-instrument`
2. OpenTelemetry automatically instruments:
   - LangGraph workflows
   - LangChain chains
   - LLM calls (OpenAI, Google Gemini)
   - Tool invocations
   - Agent workflows
3. Traces are sent to Jaeger at `http://jaeger:4317`
4. View traces in Jaeger UI at http://localhost:16686

## Service Architecture

```
┌─────────────────┐
│   Frontend      │  Port 3000
│   (React/Vite)  │
└────────┬────────┘
         │
         │ API Calls
         ▼
┌─────────────────┐       ┌─────────────────┐
│   Backend       │──────▶│    Jaeger       │
│   (Python)      │ OTLP  │    (v2.10.0)    │
│   Port 8000     │ 4317  │                 │
└─────────────────┘       └─────────────────┘
                                   │
                                   │ UI
                                   ▼
                           Port 16686 (Browser)
```

## Usage Examples

### Example 1: Run with Tracing Enabled

```bash
# Set environment variable
export ENABLE_TRACING=true

# Start services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Expected output:
# 🔍 OpenTelemetry Auto-Instrumentation ENABLED
#    Service: career-planning-system
#    Endpoint: http://jaeger:4317
```

### Example 2: Run without Tracing

```bash
# Don't set ENABLE_TRACING or set it to false
docker-compose up -d

# View logs
docker-compose logs -f backend

# Expected output:
# ℹ️  OpenTelemetry Auto-Instrumentation DISABLED
#    Set ENABLE_TRACING=true to enable distributed tracing
```

### Example 3: Custom Service Name

```bash
# Create docker-compose.override.yml
cat > docker-compose.override.yml << 'EOF'
version: "3.8"
services:
  backend:
    environment:
      - ENABLE_TRACING=true
      - OTEL_SERVICE_NAME=career-planner-prod
EOF

# Start services
docker-compose up -d
```

### Example 4: Run Only Jaeger

```bash
# Start only Jaeger service
docker-compose up -d jaeger

# Access Jaeger UI
open http://localhost:16686
```

## Viewing Traces in Jaeger

1. Open Jaeger UI: http://localhost:16686
2. Select service: `career-planning-system` (or your custom name)
3. Click "Find Traces"
4. Click on any trace to see:
   - Span timeline
   - Agent workflow execution
   - LLM call details
   - Tool invocations
   - Custom attributes

## Troubleshooting

### No Traces Appearing

**Check 1: Is tracing enabled?**
```bash
docker-compose exec backend env | grep ENABLE_TRACING
# Should output: ENABLE_TRACING=true
```

**Check 2: Is Jaeger running?**
```bash
docker-compose ps jaeger
# Should show: Up
```

**Check 3: Can backend reach Jaeger?**
```bash
docker-compose exec backend ping -c 3 jaeger
# Should show successful pings
```

**Check 4: View backend logs**
```bash
docker-compose logs backend | grep -i otel
# Should show OpenTelemetry initialization messages
```

### Backend Won't Start

**Check 1: View logs**
```bash
docker-compose logs backend
```

**Check 2: Check health status**
```bash
docker-compose ps
```

**Check 3: Verify dependencies are installed**
```bash
docker-compose exec backend pip list | grep opentelemetry
```

### Jaeger UI Not Accessible

**Check 1: Verify port is exposed**
```bash
docker-compose port jaeger 16686
# Should output: 0.0.0.0:16686
```

**Check 2: Check Jaeger health**
```bash
curl http://localhost:16686
# Should return HTML
```

## Development Workflow

### Rebuild After Code Changes

```bash
# Rebuild and restart backend
docker-compose up -d --build backend
```

### View Live Logs

```bash
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Jaeger only
docker-compose logs -f jaeger
```

### Stop All Services

```bash
docker-compose down
```

### Remove All Data and Rebuild

```bash
# Stop and remove containers, networks, volumes
docker-compose down -v

# Rebuild images from scratch
docker-compose build --no-cache

# Start services
docker-compose up -d
```

## Production Considerations

### 1. Disable Input/Output Capture

For production, consider disabling verbose capture to reduce overhead:

```yaml
environment:
  - OTEL_PYTHON_LANGCHAIN_CAPTURE_INPUTS=false
  - OTEL_PYTHON_LANGCHAIN_CAPTURE_OUTPUTS=false
```

### 2. Use External Jaeger

Replace the Jaeger container with an external Jaeger instance:

```yaml
backend:
  environment:
    - OTEL_EXPORTER_OTLP_ENDPOINT=http://production-jaeger:4317
```

Remove the `jaeger` service from docker-compose.yml.

### 3. Add Resource Attributes

```yaml
backend:
  environment:
    - OTEL_RESOURCE_ATTRIBUTES=deployment.environment=production,service.version=1.0.0
```

### 4. Configure Sampling

```yaml
backend:
  environment:
    - OTEL_TRACES_SAMPLER=parentbased_traceidratio
    - OTEL_TRACES_SAMPLER_ARG=0.1  # Sample 10% of traces
```

## Network Architecture

All services are connected via the `career-planning-network` bridge network:

- Backend can reach Jaeger at: `http://jaeger:4317`
- Frontend reaches backend at: `http://backend:8000` (internal) or `http://localhost:8000` (from browser)
- Jaeger UI accessible at: `http://localhost:16686` (from host)

## Health Checks

All services include health checks:

- **Backend**: `curl -f http://localhost:8000/health`
- **Jaeger**: `wget --spider http://localhost:16686`
- **Frontend**: `wget --spider http://localhost:80/`

Health checks ensure:
- Services start in correct order
- `depends_on` conditions are met
- Container restarts on failures

## Additional Resources

- [Main Tracing Guide](README.md)
- [Auto-Instrumentation Guide](auto_instrumentation_guide.md)
- [Example Scripts](example_traced_agent.py)
- [OpenTelemetry Docs](https://opentelemetry.io/docs/languages/python/)
- [Jaeger Docs](https://www.jaegertracing.io/docs/)

## Summary Commands

```bash
# Quick start with tracing enabled
ENABLE_TRACING=true docker-compose up -d

# View all logs
docker-compose logs -f

# Open Jaeger UI (after services start)
open http://localhost:16686

# Stop all services
docker-compose down
```

Happy tracing with Docker!
