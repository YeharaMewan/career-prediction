# Automatic Instrumentation for LangChain/LangGraph

If your project uses **LangChain** or **LangGraph** frameworks, you can enable distributed tracing **without modifying any code** using OpenTelemetry's automatic instrumentation.

## What is Automatic Instrumentation?

Automatic instrumentation automatically captures traces for:
- LangChain chain executions
- LangGraph node executions
- LLM calls (OpenAI, Google Gemini, etc.)
- Tool invocations
- Agent workflows
- State transitions

## Setup Instructions

### Prerequisites

Ensure Jaeger v2 is running:
```bash
docker run -d --name jaeger -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/jaeger:2.10.0
```

### 1. Install Required Dependencies (Latest Versions - October 2025)

Install the core OpenTelemetry packages and LangChain instrumentation:

```bash
# Install all required packages with specific versions
pip install opentelemetry-api==1.38.0 \
            opentelemetry-sdk==1.38.0 \
            opentelemetry-exporter-otlp-proto-grpc==1.38.0 \
            opentelemetry-instrumentation==0.59b0 \
            opentelemetry-instrumentation-langchain==0.47.3
```

Or simply install from requirements.txt (already updated):

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Environment Variables

Before running your application, set these environment variables in your terminal:

```bash
# Linux/macOS
export OTEL_SERVICE_NAME="career-planning-system"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Windows PowerShell
$env:OTEL_SERVICE_NAME="career-planning-system"
$env:OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Windows CMD
set OTEL_SERVICE_NAME=career-planning-system
set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

### 3. Run Your Application with Auto-Instrumentation

Use the `opentelemetry-instrument` command to automatically instrument your application:

```bash
opentelemetry-instrument python main.py
```

Or for the API server:

```bash
opentelemetry-instrument python main.py --port 8000
```

### 4. Advanced Configuration

You can configure additional settings via environment variables:

```bash
# Service name (required)
export OTEL_SERVICE_NAME="career-planning-system"

# OTLP endpoint (required)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Enable/disable specific instrumentations
export OTEL_PYTHON_LANGCHAIN_CAPTURE_INPUTS=true
export OTEL_PYTHON_LANGCHAIN_CAPTURE_OUTPUTS=true

# Set resource attributes
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=production,service.version=1.0.0"

# Configure sampling (optional - defaults to always on)
export OTEL_TRACES_SAMPLER="parentbased_always_on"
```

## Example: Running the Career Planning System

### Terminal Session Example

```bash
# Step 1: Ensure Jaeger is running
docker ps | grep jaegertracing

# Step 2: Set environment variables
export OTEL_SERVICE_NAME="career-planning-system"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_PYTHON_LANGCHAIN_CAPTURE_INPUTS=true
export OTEL_PYTHON_LANGCHAIN_CAPTURE_OUTPUTS=true

# Step 3: Run the application with auto-instrumentation
opentelemetry-instrument python main.py --interactive

# Or run the API server
opentelemetry-instrument uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Using a .env File

Alternatively, create a `.env.tracing` file:

```env
OTEL_SERVICE_NAME=career-planning-system
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_PYTHON_LANGCHAIN_CAPTURE_INPUTS=true
OTEL_PYTHON_LANGCHAIN_CAPTURE_OUTPUTS=true
OTEL_RESOURCE_ATTRIBUTES=deployment.environment=development
```

Then load it before running:

```bash
# Linux/macOS
source .env.tracing
opentelemetry-instrument python main.py

# Windows - use a script to load .env file
```

## What Gets Traced Automatically?

With automatic instrumentation enabled, you'll see traces for:

1. **LangGraph Workflows**
   - Node executions
   - State transitions
   - Conditional edges
   - Tool calls

2. **LangChain Components**
   - Chain executions
   - LLM calls
   - Retriever queries
   - Tool invocations

3. **LLM Interactions**
   - OpenAI API calls
   - Google Gemini API calls
   - Token usage
   - Latency

4. **Agent Workflows**
   - Main supervisor execution
   - Career planning supervisor
   - Worker agents (profiler, academic, skill, trends)
   - Handoffs between agents

## Viewing Traces

After running your application with automatic instrumentation:

1. Open Jaeger UI: http://localhost:16686
2. Select service: `career-planning-system`
3. Click "Find Traces"
4. Explore the automatically captured traces

You'll see:
- Complete workflow traces
- LLM call durations
- Agent handoff sequences
- Tool execution times
- Error traces (if any)

## Comparison: Manual vs Automatic Instrumentation

| Feature | Manual Instrumentation | Automatic Instrumentation |
|---------|----------------------|--------------------------|
| Setup Complexity | High - requires code changes | Low - just environment variables |
| Code Changes | Required | None |
| Framework Support | Any Python code | LangChain/LangGraph specific |
| Custom Spans | Full control | Limited to framework spans |
| Custom Attributes | Full control | Limited |
| Best For | Custom applications, fine-grained control | LangChain/LangGraph projects, quick setup |

## Troubleshooting

### No traces appearing in Jaeger

1. Verify Jaeger is running: `docker ps | grep jaeger`
2. Check environment variables are set: `echo $OTEL_SERVICE_NAME`
3. Ensure you're using `opentelemetry-instrument` command
4. Check for errors in console output

### Traces are incomplete

1. Ensure all required packages are installed
2. Check that `OTEL_PYTHON_LANGCHAIN_CAPTURE_INPUTS=true`
3. Verify network connectivity to localhost:4317

### Performance impact

- Automatic instrumentation has minimal overhead
- To reduce overhead, disable input/output capture:
  ```bash
  export OTEL_PYTHON_LANGCHAIN_CAPTURE_INPUTS=false
  export OTEL_PYTHON_LANGCHAIN_CAPTURE_OUTPUTS=false
  ```

## Next Steps

- Review traces in Jaeger UI to understand your application's behavior
- Identify performance bottlenecks
- Monitor LLM call durations and costs
- Debug agent workflow issues
- Optimize slow operations

For manual instrumentation with more control, see the main README.md.
