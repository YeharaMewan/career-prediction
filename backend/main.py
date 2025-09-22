"""
Main entry point for the career prediction backend

This can be run in different modes:
- API server mode (default): Start FastAPI server
- Interactive mode: Command line interface
- Test mode: Run integration tests
"""
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Career Planning Multi-Agent System")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--test", action="store_true", help="Run integration tests")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    parser.add_argument("query", nargs="?", help="Direct query for career planning")
    
    args = parser.parse_args()
    
    if args.test:
        # Run integration tests
        from test_simple_integration import main as test_main
        return test_main()
    
    elif args.interactive or args.query:
        # Run interactive or direct query mode
        from test_simple_integration import test_main_supervisor_standalone
        if args.query:
            print(f"Processing query: {args.query}")
            # You can extend this to process the query directly
        else:
            print("Interactive mode - running standalone test")
        return test_main_supervisor_standalone()
    
    else:
        # Default: Start API server
        print(f"Starting Career Planning API server on port {args.port}")
        import uvicorn
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0", 
            port=args.port,
            reload=True
        )

if __name__ == "__main__":
    main()