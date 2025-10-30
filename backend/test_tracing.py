"""
Enhanced test script to verify OpenTelemetry tracing with real LLM token tracking
"""
import time
import os
from dotenv import load_dotenv
from tracing.setup_tracing import (
    setup_tracing,
    get_tracer,
    shutdown_tracing,
    add_llm_token_attributes,
    extract_and_add_token_attributes_from_response
)
from utils.llm_factory import LLMFactory
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

print("\n" + "="*70)
print("Testing OpenTelemetry Tracing with Real LLM Token Tracking")
print("="*70 + "\n")

# Setup tracing
provider = setup_tracing(
    service_name="test-token-tracking-service",
    otlp_endpoint="http://jaeger:4317",
    insecure=True
)

# Get a tracer
tracer = get_tracer(__name__)

# Test 1: Basic span with manual token attributes
print("\n[Test 1] Creating span with manual token attributes...")
with tracer.start_as_current_span("manual_token_test") as span:
    # Simulate token data
    token_data = {
        'input_tokens': 100,
        'output_tokens': 50,
        'total_tokens': 150,
        'cost_usd': 0.000375,
        'model': 'gpt-4o',
        'pricing_found': True
    }
    add_llm_token_attributes(span, token_data)
    print(f"   ✓ Added token attributes: {token_data['total_tokens']} tokens, ${token_data['cost_usd']:.6f}")
    time.sleep(0.2)

# Test 2: Real LLM call with automatic token extraction
print("\n[Test 2] Making real LLM API call with token tracking...")
try:
    # Create LLM instance
    llm_wrapper = LLMFactory.create_llm(
        model="gpt-4o-mini",  # Using cheaper model for testing
        temperature=0.1
    )

    with tracer.start_as_current_span("real_llm_call") as span:
        span.set_attribute("llm.request.type", "test_query")
        span.set_attribute("llm.model", "gpt-4o-mini")

        # Make actual LLM call
        print("   → Calling LLM API...")
        response = llm_wrapper.invoke([
            HumanMessage(content="What is 2+2? Answer briefly in one word.")
        ])

        # Extract and add token attributes automatically
        token_info = extract_and_add_token_attributes_from_response(
            span, response, "gpt-4o-mini"
        )

        if token_info:
            print(f"   ✓ Real API call completed!")
            print(f"     - Input tokens: {token_info['input_tokens']}")
            print(f"     - Output tokens: {token_info['output_tokens']}")
            print(f"     - Total tokens: {token_info['total_tokens']}")
            print(f"     - Cost: ${token_info['cost_usd']:.6f}")
            print(f"     - Response: {response.content}")
        else:
            print("   ⚠ Token extraction failed, but call succeeded")
            print(f"     - Response: {response.content}")

except Exception as e:
    print(f"   ✗ Test 2 failed: {e}")
    print("     (This is OK if API key is not configured)")

# Test 3: Check LLM wrapper statistics
print("\n[Test 3] Checking LLM wrapper cumulative statistics...")
try:
    stats = llm_wrapper.get_stats()
    print(f"   ✓ LLM Wrapper Stats:")
    print(f"     - Provider: {stats['provider']}")
    print(f"     - Model: {stats['model']}")
    print(f"     - Total API calls: {stats['usage_count']}")
    print(f"     - Total input tokens: {stats['total_input_tokens']}")
    print(f"     - Total output tokens: {stats['total_output_tokens']}")
    print(f"     - Total tokens: {stats['total_tokens']}")
    print(f"     - Total cost: ${stats['total_cost_usd']:.6f}")
except Exception as e:
    print(f"   ⚠ Could not get stats: {e}")

# Test 4: Multiple LLM calls to accumulate tokens
print("\n[Test 4] Making multiple LLM calls to test cumulative tracking...")
try:
    with tracer.start_as_current_span("multiple_llm_calls") as parent_span:
        for i in range(3):
            with tracer.start_as_current_span(f"llm_call_{i+1}") as span:
                print(f"   → Call {i+1}/3...")
                response = llm_wrapper.invoke([
                    HumanMessage(content=f"Count to {i+1}")
                ])
                token_info = extract_and_add_token_attributes_from_response(
                    span, response, "gpt-4o-mini"
                )
                if token_info:
                    print(f"     ✓ Tokens: {token_info['total_tokens']}, Cost: ${token_info['cost_usd']:.6f}")
                time.sleep(0.2)

        # Show cumulative stats
        final_stats = llm_wrapper.get_stats()
        print(f"\n   ✓ Cumulative stats after {final_stats['usage_count']} calls:")
        print(f"     - Total tokens: {final_stats['total_tokens']}")
        print(f"     - Total cost: ${final_stats['total_cost_usd']:.6f}")

except Exception as e:
    print(f"   ✗ Test 4 failed: {e}")

print("\n" + "="*70)
print("Shutting down tracing to flush all spans to Jaeger...")
shutdown_tracing()

print("\n✅ Test complete! View results in Jaeger UI:")
print("   → http://localhost:16686")
print("   → Select service: test-token-tracking-service")
print("\n   Look for these span attributes in Jaeger:")
print("     • llm.input_tokens")
print("     • llm.output_tokens")
print("     • llm.total_tokens")
print("     • llm.cost_usd")
print("     • llm.model")
print("="*70 + "\n")
