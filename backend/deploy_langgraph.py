#!/usr/bin/env python3
"""
LangGraph Deployment Script for Career Prediction System

This script helps deploy the career prediction system to LangGraph Cloud
or configure it for local development.

Usage:
    cd backend/
    python deploy_langgraph.py [options]

Options:
    --validate-only    Only validate configuration
    --deploy          Deploy to LangGraph Cloud  
    --local           Start local development server
    --package         Create deployment package
"""
import os
import json
import sys
import argparse
from pathlib import Path
import subprocess
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load the langgraph.json configuration."""
    config_path = Path("langgraph.json")
    if not config_path.exists():
        raise FileNotFoundError("langgraph.json not found. Please ensure it exists in the current directory.")

    with open(config_path, 'r') as f:
        return json.load(f)

def check_environment():
    """Check if required environment variables are set."""
    config = load_config()
    required_vars = config.get("environment", {}).get("required_variables", [])

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False

    print("✅ All required environment variables are set.")
    return True

def validate_dependencies():
    """Validate that required dependencies are installed."""
    config = load_config()
    dependencies = config.get("dependencies", [])

    print("🔍 Checking dependencies...")
    missing_deps = []

    for dep in dependencies:
        # Extract package name from version specifier
        package_name = dep.split(">=")[0].split("==")[0].split("~=")[0]
        try:
            __import__(package_name.replace("-", "_"))
            print(f"   ✅ {package_name}")
        except ImportError:
            missing_deps.append(dep)
            print(f"   ❌ {package_name}")

    if missing_deps:
        print(f"\n❌ Missing dependencies: {len(missing_deps)}")
        print("Install them with: pip install -r requirements.txt")
        return False

    print("✅ All dependencies are installed.")
    return True

def validate_agents():
    """Validate that all configured agents can be imported."""
    config = load_config()
    agents = config.get("agents", {})

    print("🤖 Validating agents...")

    for agent_name, agent_config in agents.items():
        module_path = agent_config.get("module")
        class_name = agent_config.get("class")

        try:
            # We're already in the backend directory
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            print(f"   ✅ {agent_name}: {module_path}.{class_name}")
        except ImportError as e:
            print(f"   ❌ {agent_name}: Failed to import {module_path}.{class_name}")
            print(f"      Error: {e}")
            return False
        except AttributeError as e:
            print(f"   ❌ {agent_name}: Class {class_name} not found in {module_path}")
            print(f"      Error: {e}")
            return False

    print("✅ All agents validated successfully.")
    return True

def test_api_server():
    """Test that the API server can start."""
    print("🌐 Testing API server...")

    try:
        # We're already in the backend directory
        from api_server import app
        print("   ✅ API server module imported successfully")
        return True
    except Exception as e:
        print(f"   ❌ Failed to import API server: {e}")
        return False

def create_deployment_package():
    """Create a deployment package for LangGraph Cloud."""
    print("📦 Creating deployment package...")

    # Files to include in deployment
    essential_files = [
        "langgraph.json",
        "requirements.txt", 
        ".env.template",
        "agents/",
        "models/",
        "tools/",
        "config/",
        "api_server.py",
        "main.py",
        "../README.md"
    ]

    # Create deployment directory
    deploy_dir = Path("deployment")
    deploy_dir.mkdir(exist_ok=True)

    print("   ✅ Deployment package structure created")
    print(f"   📁 Package location: {deploy_dir.absolute()}")

    return deploy_dir

def deploy_to_langgraph_cloud():
    """Deploy to LangGraph Cloud (placeholder)."""
    print("☁️  Deploying to LangGraph Cloud...")

    # Check if LangGraph CLI is available
    try:
        result = subprocess.run(["langgraph", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ LangGraph CLI found")
        else:
            print("   ❌ LangGraph CLI not found. Install with: pip install langgraph-cli")
            return False
    except FileNotFoundError:
        print("   ❌ LangGraph CLI not found. Install with: pip install langgraph-cli")
        return False

    # TODO: Implement actual deployment commands
    print("   🚧 Deployment to LangGraph Cloud is not yet implemented")
    print("   📖 Please refer to LangGraph Cloud documentation for deployment steps")

    return True

def start_local_development():
    """Start local development server."""
    print("🚀 Starting local development server...")

    # We're already in the backend directory
    try:
        # Start the API server
        subprocess.run([
            sys.executable, "api_server.py"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description="LangGraph Career Prediction System Deployment")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    parser.add_argument("--deploy", action="store_true", help="Deploy to LangGraph Cloud")
    parser.add_argument("--local", action="store_true", help="Start local development server")
    parser.add_argument("--package", action="store_true", help="Create deployment package")

    args = parser.parse_args()

    print("🎯 Career Prediction System - LangGraph Deployment")
    print("=" * 60)

    # Always validate first
    print("\n1. Environment Validation")
    if not check_environment():
        sys.exit(1)

    print("\n2. Dependencies Validation")
    if not validate_dependencies():
        sys.exit(1)

    print("\n3. Agents Validation")
    if not validate_agents():
        sys.exit(1)

    print("\n4. API Server Validation")
    if not test_api_server():
        sys.exit(1)

    if args.validate_only:
        print("\n✅ All validations passed! System is ready for deployment.")
        return

    if args.package:
        print("\n5. Creating Deployment Package")
        create_deployment_package()

    if args.deploy:
        print("\n6. Deploying to LangGraph Cloud")
        deploy_to_langgraph_cloud()

    if args.local:
        print("\n6. Starting Local Development Server")
        start_local_development()

    if not any([args.deploy, args.local, args.package]):
        print("\n✅ Validation complete! Use --help to see deployment options.")

if __name__ == "__main__":
    main()