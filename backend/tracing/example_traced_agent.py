"""
Example: Manual OpenTelemetry Instrumentation for AI Agent

This script demonstrates how to manually instrument an AI agent application
with OpenTelemetry tracing and send traces to Jaeger using the OTLP protocol.

Compatible with:
- OpenTelemetry v1.38.0 (October 2025)
- Jaeger v2.10.0 (recommended) or v1.74.0

Before running this script:
1. Start Jaeger v2: docker run -d --name jaeger -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/jaeger:2.10.0
2. Install dependencies: pip install opentelemetry-api==1.38.0 opentelemetry-sdk==1.38.0 opentelemetry-exporter-otlp-proto-grpc==1.38.0
3. Run this script: python example_traced_agent.py
4. View traces at http://localhost:16686
"""

import time
import random
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME


# ============================================================================
# STEP 1: Configure the Resource with service name
# ============================================================================
resource = Resource.create({
    SERVICE_NAME: "my-ai-agent-project"  # This name appears in Jaeger UI
})


# ============================================================================
# STEP 2: Configure the TracerProvider
# ============================================================================
provider = TracerProvider(resource=resource)


# ============================================================================
# STEP 3: Configure the OTLP Exporter to send data to Jaeger
# Port 4317 is the OTLP gRPC receiver in Jaeger
# ============================================================================
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    insecure=True  # Use insecure for local development
)


# ============================================================================
# STEP 4: Configure the BatchSpanProcessor
# Batches spans before sending to improve performance
# ============================================================================
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)


# ============================================================================
# STEP 5: Set the global tracer provider
# ============================================================================
trace.set_tracer_provider(provider)


# ============================================================================
# STEP 6: Get a tracer instance
# Use __name__ to identify which module created the span
# ============================================================================
tracer = trace.get_tracer(__name__)


# ============================================================================
# Example Functions: Simulating AI Agent Tasks
# ============================================================================

def call_llm_model(user_query: str) -> str:
    """
    Simulate calling an LLM model (e.g., OpenAI GPT, Google Gemini).
    This function is automatically traced using the decorator.
    """
    with tracer.start_as_current_span("call_llm_model") as span:
        # Add custom attributes to the span for better observability
        span.set_attribute("user.query", user_query)
        span.set_attribute("model.name", "gpt-4-mini")
        span.set_attribute("model.temperature", 0.7)

        # Simulate LLM processing time
        processing_time = random.uniform(0.5, 2.0)
        time.sleep(processing_time)

        # Simulate token usage
        tokens_used = random.randint(100, 500)
        span.set_attribute("model.tokens_used", tokens_used)
        span.set_attribute("model.processing_time_sec", processing_time)

        response = f"AI Response to: {user_query}"
        span.set_attribute("response.length", len(response))

        print(f"  [LLM] Processed query: '{user_query}' (tokens: {tokens_used})")
        return response


def use_database_tool(query: str) -> dict:
    """
    Simulate using a database tool to retrieve information.
    Demonstrates nested spans and custom attributes.
    """
    with tracer.start_as_current_span("use_database_tool") as span:
        # Add attributes to the parent span
        span.set_attribute("db.query", query)
        span.set_attribute("db.type", "postgresql")

        # Create a nested span for the actual database query
        with tracer.start_as_current_span("db.execute_query") as db_span:
            db_span.set_attribute("db.operation", "SELECT")
            db_span.set_attribute("db.table", "career_data")

            # Simulate database query execution
            query_time = random.uniform(0.1, 0.5)
            time.sleep(query_time)

            rows_returned = random.randint(1, 50)
            db_span.set_attribute("db.rows_returned", rows_returned)
            db_span.set_attribute("db.query_time_sec", query_time)

            print(f"  [DB] Executed query: '{query}' (rows: {rows_returned})")

        # Create another nested span for data processing
        with tracer.start_as_current_span("db.process_results") as process_span:
            process_span.set_attribute("processing.type", "data_transformation")

            # Simulate data processing
            time.sleep(0.1)

            result = {
                "query": query,
                "rows": rows_returned,
                "status": "success"
            }
            process_span.set_attribute("result.status", result["status"])

            print(f"  [DB] Processed {rows_returned} rows")

        return result


def perform_web_search(search_term: str) -> list:
    """
    Simulate performing a web search using an external API.
    Demonstrates error handling and span attributes.
    """
    with tracer.start_as_current_span("perform_web_search") as span:
        span.set_attribute("search.term", search_term)
        span.set_attribute("search.engine", "tavily")

        try:
            # Simulate API call
            time.sleep(random.uniform(0.3, 0.8))

            results_count = random.randint(5, 20)
            span.set_attribute("search.results_count", results_count)
            span.set_attribute("search.status", "success")

            results = [f"Result {i} for {search_term}" for i in range(results_count)]

            print(f"  [SEARCH] Found {results_count} results for '{search_term}'")
            return results

        except Exception as e:
            # Record exceptions in the span
            span.set_attribute("search.status", "error")
            span.set_attribute("error.message", str(e))
            span.record_exception(e)
            raise


def agent_workflow(user_input: str):
    """
    Main agent workflow that orchestrates multiple tasks.
    This creates a parent span that contains all sub-operations.
    """
    with tracer.start_as_current_span("agent_workflow") as span:
        span.set_attribute("workflow.type", "career_planning")
        span.set_attribute("workflow.user_input", user_input)

        print(f"\nStarting Agent Workflow for: '{user_input}'")
        print("=" * 60)

        # Step 1: Analyze user input with LLM
        with tracer.start_as_current_span("step_1_analyze_input"):
            print("\n[Step 1] Analyzing user input...")
            llm_response = call_llm_model(user_input)

        # Step 2: Retrieve relevant data from database
        with tracer.start_as_current_span("step_2_retrieve_data"):
            print("\n[Step 2] Retrieving relevant data...")
            db_result = use_database_tool("SELECT * FROM careers WHERE field = 'AI'")

        # Step 3: Perform web search for latest information
        with tracer.start_as_current_span("step_3_web_search"):
            print("\n[Step 3] Searching for latest information...")
            search_results = perform_web_search("AI career trends 2025")

        # Step 4: Generate final recommendation with LLM
        with tracer.start_as_current_span("step_4_generate_recommendation"):
            print("\n[Step 4] Generating final recommendation...")
            final_query = f"Based on the data, recommend careers for: {user_input}"
            final_response = call_llm_model(final_query)

        span.set_attribute("workflow.status", "completed")
        span.set_attribute("workflow.steps_completed", 4)

        print("\n" + "=" * 60)
        print(f"Agent Workflow Completed!")
        print(f"\nFinal Response: {final_response}")


# ============================================================================
# Main Execution Block
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("OpenTelemetry Manual Instrumentation Example")
    print("Service: my-ai-agent-project")
    print("="*60)

    try:
        # Example 1: Simple LLM call
        print("\n\nExample 1: Simple LLM Call")
        print("-" * 60)
        call_llm_model("What are the best careers in AI?")

        # Example 2: Database tool usage
        print("\n\nExample 2: Database Tool Usage")
        print("-" * 60)
        use_database_tool("SELECT * FROM career_paths WHERE industry = 'tech'")

        # Example 3: Complete agent workflow
        print("\n\nExample 3: Complete Agent Workflow")
        print("-" * 60)
        agent_workflow("I am interested in AI and machine learning careers")

        print("\n\n" + "="*60)
        print("All examples completed successfully!")
        print("\nView traces in Jaeger UI:")
        print("  http://localhost:16686")
        print("\nSelect service: my-ai-agent-project")
        print("="*60 + "\n")

    finally:
        # Ensure all spans are flushed before exit
        print("\nFlushing spans to Jaeger...")
        provider.shutdown()
        print("Shutdown complete.\n")
