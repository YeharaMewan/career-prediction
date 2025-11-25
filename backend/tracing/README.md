# Distributed Tracing with OpenTelemetry and Jaeger

This guide provides a complete, step-by-step implementation of distributed tracing for the Career Planning System using **OpenTelemetry (OTel)** and **Jaeger**.

The solution uses the modern **OpenTelemetry-native approach** with the **OTLP (OpenTelemetry Protocol)** for communicating with Jaeger.

---

## Part 1: Jaeger Backend Setup

### Docker Command to Start Jaeger

**Recommended: Jaeger v2 (Latest - October 2025)**

Jaeger v2 is the latest version with OpenTelemetry natively integrated in the core. Run this command to start Jaeger v2:

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/jaeger:2.10.0
```

Or use the latest tag:

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/jaeger:latest
```

**Alternative: Jaeger v1 (Legacy - EOL December 31, 2025)**

If you need Jaeger v1 for compatibility:

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:1.74.0
```

**Note:** Jaeger v1 reaches end-of-life on December 31, 2025. Use Jaeger v2 for new projects.

### Port Explanations

| Port | Protocol | Purpose |
|------|----------|---------|
| **16686** | HTTP | Jaeger UI - Web interface for viewing traces |
| **4317** | gRPC | OTLP gRPC Receiver - Receives traces via OTLP/gRPC protocol |
| **4318** | HTTP | OTLP HTTP Receiver - Receives traces via OTLP/HTTP protocol |

**Why these ports?**
- **Port 16686**: Access the Jaeger UI in your browser at `http://localhost:16686` to visualize traces
- **Port 4317**: OpenTelemetry sends traces here using the gRPC protocol (recommended for performance)
- **Port 4318**: Alternative HTTP endpoint if you prefer HTTP over gRPC

### Verify Jaeger is Running

```bash
# Check if container is running
docker ps | grep jaeger

# View Jaeger logs
docker logs jaeger

# Access Jaeger UI
# Open http://localhost:16686 in your browser
```

### Stop and Remove Jaeger

```bash
docker stop jaeger
docker rm jaeger
```

---

## Part 2: Python Project Dependencies

### Required OpenTelemetry Libraries (Latest Versions - October 2025)

Install the following packages using pip:

```bash
# Core OpenTelemetry API and SDK (v1.38.0)
pip install opentelemetry-api==1.38.0
pip install opentelemetry-sdk==1.38.0

# OTLP gRPC Exporter for sending traces to Jaeger
pip install opentelemetry-exporter-otlp-proto-grpc==1.38.0

# Optional: Instrumentation libraries for automatic tracing
pip install opentelemetry-instrumentation==0.59b0
pip install opentelemetry-instrumentation-langchain==0.47.3
```

### Single Command Installation

```bash
pip install opentelemetry-api==1.38.0 opentelemetry-sdk==1.38.0 opentelemetry-exporter-otlp-proto-grpc==1.38.0
```

Or install from the project's requirements.txt (already updated with latest versions):

```bash
cd backend
pip install -r requirements.txt
```

### Verify Installation

```bash
pip list | grep opentelemetry
```

Expected output:
```
opentelemetry-api                      1.38.0
opentelemetry-exporter-otlp-proto-grpc 1.38.0
opentelemetry-instrumentation          0.59b0
opentelemetry-instrumentation-langchain 0.47.3
opentelemetry-sdk                      1.38.0
```

---

## Part 3: Manual Python Implementation

### Complete Example Script

The `example_traced_agent.py` file in this directory provides a complete, working example of manual OpenTelemetry instrumentation.

### Key Components Explained

#### 1. Import Required Modules

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
```

#### 2. Configure the Resource

```python
# Identifies your service in Jaeger
resource = Resource.create({
    SERVICE_NAME: "my-ai-agent-project"
})
```

The `SERVICE_NAME` is how your application appears in the Jaeger UI. Choose a descriptive name.

#### 3. Configure the TracerProvider

```python
provider = TracerProvider(resource=resource)
```

The `TracerProvider` is the core component that manages trace creation.

#### 4. Configure the OTLP Exporter

```python
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    insecure=True  # Use for local development
)
```

This exporter sends traces to Jaeger using gRPC on port 4317 (OTLP protocol).

#### 5. Configure the BatchSpanProcessor

```python
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)
```

The `BatchSpanProcessor` batches multiple spans before sending them to Jaeger, improving performance.

#### 6. Set Global Tracer Provider

```python
trace.set_tracer_provider(provider)
```

This makes the tracer available globally throughout your application.

#### 7. Get a Tracer Instance

```python
tracer = trace.get_tracer(__name__)
```

Use `__name__` to identify which module created the spans.

### Creating Spans: Two Methods

#### Method 1: Context Manager (Recommended)

```python
def call_llm_model(user_query: str) -> str:
    """Example function with manual span creation"""
    with tracer.start_as_current_span("call_llm_model") as span:
        # Add custom attributes for better observability
        span.set_attribute("user.query", user_query)
        span.set_attribute("model.name", "gpt-4-mini")
        span.set_attribute("model.temperature", 0.7)

        # Your function logic here
        response = "AI response..."

        span.set_attribute("response.length", len(response))
        return response
```

#### Method 2: Nested Spans for Complex Operations

```python
def use_database_tool(query: str) -> dict:
    """Example with nested spans"""
    with tracer.start_as_current_span("use_database_tool") as span:
        span.set_attribute("db.query", query)

        # Create a nested span for sub-operation
        with tracer.start_as_current_span("db.execute_query") as db_span:
            db_span.set_attribute("db.operation", "SELECT")
            db_span.set_attribute("db.rows_returned", 42)

            # Execute query logic here
            result = {"status": "success"}

        return result
```

### Running the Example

```bash
# 1. Ensure Jaeger is running
docker ps | grep jaeger

# 2. Run the example script
cd backend/tracing
python example_traced_agent.py

# 3. View traces in Jaeger UI
# Open http://localhost:16686
# Select service: my-ai-agent-project
# Click "Find Traces"
```

### What You'll See in Jaeger

After running the example, you'll see:
- Multiple traces for different workflows
- Nested spans showing parent-child relationships
- Custom attributes for each span (query text, tokens used, etc.)
- Timing information for each operation
- Complete workflow visualization

---

## Part 4: Alternative Method - Automatic Instrumentation for LangChain/LangGraph

For projects using **LangChain** or **LangGraph** frameworks, you can enable tracing **without any code changes**.

### Prerequisites

```bash
pip install opentelemetry-instrumentation-langchain
```

### Setup: Environment Variables Only

Set the following environment variables before running your application:

#### Linux/macOS
```bash
export OTEL_SERVICE_NAME="my-ai-agent-project"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
```

#### Windows PowerShell
```powershell
$env:OTEL_SERVICE_NAME="my-ai-agent-project"
$env:OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
```

#### Windows CMD
```cmd
set OTEL_SERVICE_NAME=my-ai-agent-project
set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

### Running with Automatic Instrumentation

Use the `opentelemetry-instrument` command wrapper:

```bash
# Run your main application
opentelemetry-instrument python main.py

# Or run the API server
opentelemetry-instrument python main.py --port 8000

# Or run in interactive mode
opentelemetry-instrument python main.py --interactive
```

### What Gets Traced Automatically?

- **LangGraph Workflows**: Node executions, state transitions, edges
- **LangChain Chains**: Chain executions, prompts, LLM calls
- **LLM Interactions**: OpenAI, Google Gemini, other providers
- **Agent Workflows**: Supervisor and worker agent executions
- **Tool Calls**: Web search, database queries, custom tools

### No Code Changes Required!

The automatic instrumentation method:
- Requires NO modifications to your existing code
- Just set environment variables and use the `opentelemetry-instrument` wrapper
- Automatically captures traces for LangChain/LangGraph components
- Works with existing Career Planning System without changes

### Advanced Configuration

Optional environment variables for fine-tuning:

```bash
# Capture LLM inputs and outputs (may be verbose)
export OTEL_PYTHON_LANGCHAIN_CAPTURE_INPUTS=true
export OTEL_PYTHON_LANGCHAIN_CAPTURE_OUTPUTS=true

# Add custom resource attributes
export OTEL_RESOURCE_ATTRIBUTES="deployment.environment=production,service.version=1.0.0"

# Configure sampling
export OTEL_TRACES_SAMPLER="parentbased_always_on"
```

For more details, see `auto_instrumentation_guide.md`.

---

## Verifying Results

### Step 1: Check Jaeger is Running

```bash
docker ps | grep jaeger
```

### Step 2: Run Your Application

Choose one of:
- Manual instrumentation: `python example_traced_agent.py`
- Automatic instrumentation: `opentelemetry-instrument python main.py`

### Step 3: Access Jaeger UI

1. Open your browser to: **http://localhost:16686**
2. In the "Service" dropdown, select your service (e.g., `my-ai-agent-project` or `career-planning-system`)
3. Click "Find Traces" button
4. You should see a list of traces with timing information

### Step 4: Explore Traces

Click on any trace to see:
- **Span Timeline**: Visual representation of operation durations
- **Span Details**: Custom attributes, tags, and metadata
- **Nested Spans**: Parent-child relationships between operations
- **Error Information**: If any spans encountered errors

### Expected Results

For the example script, you should see traces like:
- `agent_workflow` (parent span)
  - `step_1_analyze_input`
    - `call_llm_model`
  - `step_2_retrieve_data`
    - `use_database_tool`
      - `db.execute_query`
      - `db.process_results`
  - `step_3_web_search`
    - `perform_web_search`
  - `step_4_generate_recommendation`
    - `call_llm_model`

---

## Integration with Career Planning System

### Option 1: Manual Integration

To integrate tracing into the existing Career Planning System:

```python
# In your main.py or agent files
from tracing.setup_tracing import setup_tracing, get_tracer

# At application startup
setup_tracing(service_name="career-planning-system")
tracer = get_tracer(__name__)

# Wrap your agent functions
def process_career_query(query: str):
    with tracer.start_as_current_span("process_career_query") as span:
        span.set_attribute("user.query", query)
        # Your existing code here
        return result
```

### Option 2: Automatic Integration (Recommended)

For the LangGraph-based Career Planning System, use automatic instrumentation:

```bash
# Set environment variables
export OTEL_SERVICE_NAME="career-planning-system"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_PYTHON_LANGCHAIN_CAPTURE_INPUTS=true

# Run with auto-instrumentation
opentelemetry-instrument python main.py --interactive
```

This automatically traces:
- Main Supervisor agent
- Career Planning Supervisor
- Worker agents (User Profiler, Academic Pathway, Skill Development, Future Trends)
- LLM calls to OpenAI/Gemini
- Tool invocations (web search, etc.)

---

## Project Structure

```
backend/tracing/
├── __init__.py                      # Package initialization
├── setup_tracing.py                 # Reusable tracing setup module
├── example_traced_agent.py          # Complete manual instrumentation example
├── auto_instrumentation_guide.md    # Detailed auto-instrumentation guide
└── README.md                        # This file
```

---

## Troubleshooting

### Issue: No traces appearing in Jaeger

**Solutions:**
1. Verify Jaeger is running: `docker ps | grep jaeger`
2. Check endpoint is correct: `http://localhost:4317` (not 16686!)
3. Ensure `provider.shutdown()` is called before exit to flush spans
4. Check for firewall/network issues

### Issue: "Connection refused" error

**Solutions:**
1. Verify Jaeger container is running
2. Check port 4317 is exposed: `docker port jaeger`
3. Try using `127.0.0.1` instead of `localhost`

### Issue: Spans appear but missing attributes

**Solutions:**
1. Ensure you're calling `span.set_attribute()` correctly
2. Check attribute values are JSON-serializable
3. Verify span is not closed before setting attributes

### Issue: Performance degradation

**Solutions:**
1. Use `BatchSpanProcessor` (default in examples)
2. Reduce sampling rate if needed
3. Disable input/output capture for automatic instrumentation
4. Adjust batch size: `BatchSpanProcessor(otlp_exporter, max_export_batch_size=512)`

---

## Best Practices

1. **Use Descriptive Span Names**: Use names that clearly describe the operation (e.g., `call_llm_model` not `func1`)

2. **Add Meaningful Attributes**: Include relevant context like user queries, model names, token counts, etc.

3. **Use Nested Spans**: Break complex operations into smaller spans for better granularity

4. **Handle Errors Gracefully**: Use `span.record_exception(e)` to capture errors in traces

5. **Shutdown Properly**: Always call `provider.shutdown()` before application exit to ensure all spans are sent

6. **Choose Service Names Carefully**: Use consistent, descriptive service names across your system

7. **Don't Over-Instrument**: Focus on critical paths and operations that need monitoring

---

## Additional Resources

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/languages/python/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [OTLP Specification](https://opentelemetry.io/docs/specs/otlp/)
- [LangChain Observability](https://python.langchain.com/docs/guides/productionization/observability/)

---

## Next Steps

1. Start Jaeger v2: `docker run -d --name jaeger -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/jaeger:2.10.0`

2. Install dependencies: `pip install opentelemetry-api==1.38.0 opentelemetry-sdk==1.38.0 opentelemetry-exporter-otlp-proto-grpc==1.38.0`

3. Navigate to tracing folder: `cd backend/tracing`

4. Run example: `python example_traced_agent.py`

5. View traces: Open http://localhost:16686 in your browser

6. Integrate into your application using either manual or automatic instrumentation

Happy tracing!
