"""
FastAPI server for Career Planning Multi-Agent System

This server provides REST API endpoints for the frontend to interact with
the career planning agents.
"""
import os
import logging
import re
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

# Load environment variables from backend folder
from dotenv import load_dotenv
import os
backend_env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(backend_env_path)

# Validate required environment variables
def validate_environment():
    """
    Validate that all required environment variables are present.
    Returns a dict with status and message instead of raising exceptions.
    This allows the app to start in degraded mode if keys are missing.
    """
    required_vars = [
        'OPENAI_API_KEY',
        'LANGSMITH_API_KEY',
        'LANGCHAIN_PROJECT'
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        warning_msg = f"⚠️  Missing environment variables: {', '.join(missing_vars)}"
        print(warning_msg)
        print("   Application will start in degraded mode")
        print("   Please configure your .env file for full functionality")
        return {"status": "degraded", "message": warning_msg, "missing": missing_vars}

    # Test OpenAI API key format
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key and not (openai_key.startswith('sk-') and len(openai_key) > 20):
        warning_msg = "⚠️  OpenAI API key format appears invalid"
        print(warning_msg)
        return {"status": "degraded", "message": warning_msg}

    print(f"✅ Environment validation passed - All API keys loaded")
    return {"status": "healthy", "message": "All environment variables configured"}

# Validate environment on import (non-blocking)
env_status = validate_environment()

# Initialize LangSmith before importing agents
from utils.langsmith_config import setup_langsmith
setup_langsmith()

# OpenTelemetry Manual Instrumentation
# This ensures tracing works even without auto-instrumentation
if os.getenv('ENABLE_TRACING', 'false').lower() == 'true':
    try:
        from tracing.setup_tracing import setup_tracing
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        # Setup OpenTelemetry with Jaeger
        service_name = os.getenv('OTEL_SERVICE_NAME', 'career-planning-system')
        otlp_endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://jaeger:4317')
        setup_tracing(service_name=service_name, otlp_endpoint=otlp_endpoint)
        print(f"✅ Manual OpenTelemetry tracing initialized for {service_name}")
    except Exception as e:
        print(f"⚠️  Failed to initialize OpenTelemetry tracing: {e}")

# Import LLM factory for provider validation
from utils.llm_factory import LLMFactory

# Import your existing agents
try:
    from agents.supervisors.main_supervisor import MainSupervisor
    from agents.workers.user_profiler import UserProfilerAgent
    from models.state_models import AgentState, StudentProfile, CareerBlueprint, TaskResult
except ImportError as e:
    print(f"Warning: Could not import agents: {e}")
    MainSupervisor = None
    UserProfilerAgent = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for agent instances
main_supervisor = None
user_profiler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents on startup and cleanup on shutdown"""
    global main_supervisor, user_profiler

    try:
        # Log tracing status on startup
        tracing_enabled = os.getenv('ENABLE_TRACING', 'false').lower() == 'true'
        if tracing_enabled:
            service_name = os.getenv('OTEL_SERVICE_NAME', 'career-planning-system')
            otlp_endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://jaeger:4317')
            logger.info(f"🔍 Manual tracing ACTIVE: {service_name} → {otlp_endpoint}")
        else:
            logger.info("ℹ️  Manual tracing DISABLED")

        if MainSupervisor:
            main_supervisor = MainSupervisor()
            logger.info("Main Supervisor initialized")

        if UserProfilerAgent:
            user_profiler = UserProfilerAgent()
            logger.info("User Profiler initialized")

        logger.info("Career Planning API server started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")

    yield

    # Cleanup code - shutdown tracing to flush spans
    logger.info("Career Planning API server shutting down")

    # Flush all remaining spans to Jaeger before shutdown
    tracing_enabled = os.getenv('ENABLE_TRACING', 'false').lower() == 'true'
    if tracing_enabled:
        try:
            from tracing.setup_tracing import shutdown_tracing
            logger.info("🔍 Flushing OpenTelemetry spans to Jaeger...")
            shutdown_tracing()
            logger.info("✅ OpenTelemetry tracing shutdown complete")
        except Exception as e:
            logger.warning(f"⚠️  Failed to shutdown tracing: {e}")

# FastAPI app
app = FastAPI(
    title="Career Planning API",
    description="Multi-Agent Career Planning System API",
    version="1.0.0",
    lifespan=lifespan
)

# Instrument FastAPI with OpenTelemetry (if tracing is enabled)
if os.getenv('ENABLE_TRACING', 'false').lower() == 'true':
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        print("✅ FastAPI app instrumented with OpenTelemetry")
    except Exception as e:
        print(f"⚠️  Failed to instrument FastAPI: {e}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatMessage(BaseModel):
    message: str

class UserResponse(BaseModel):
    response: str
    language: Optional[str] = "en"  # User's preferred language (en/si)

class SessionStartRequest(BaseModel):
    initial_message: str
    mode: str = "interactive"  # "interactive" or "batch"
    language: str = "en"  # User's preferred language (en/si)

class SessionInitializeRequest(BaseModel):
    language: str = "en"  # User's preferred language (en/si)

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class CareerPlanningResponse(BaseModel):
    success: bool
    session_id: str
    message: str
    data: Optional[Dict[str, Any]] = None

class InteractiveResponse(BaseModel):
    success: bool
    session_id: str
    question: Optional[str] = None
    awaiting_user_response: bool = False
    conversation_state: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    final_report: Optional[str] = None
    completed: bool = False
    career_predictions: Optional[list] = None
    # Career planning results
    career_title: Optional[str] = None
    academic_plan: Optional[Dict[str, Any]] = None
    skill_plan: Optional[Dict[str, Any]] = None
    academic_message: Optional[str] = None
    skill_message: Optional[str] = None
    message: Optional[str] = None

# Helper function to validate and sanitize response fields
def validate_and_clean_response_fields(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean response fields to prevent undefined/None values in frontend.
    Ensures all text fields are proper strings without 'undefined' concatenations.
    """
    cleaned_data = result_data.copy()

    # Validate and clean question field (most critical)
    if 'question' in cleaned_data:
        question = cleaned_data['question']
        if question is None or (isinstance(question, str) and not question.strip()):
            logger.warning("Empty or None question detected, using fallback")
            cleaned_data['question'] = "Could you tell me more?"
        elif isinstance(question, str):
            # Ensure it's a clean string (no undefined appended)
            question = question.strip()

            # CRITICAL: Remove any "undefined" text
            # Use robust regex to catch all variations (case-insensitive, multiple occurrences, different positions)
            original_question = question
            question = re.sub(r'\s*undefined\s*', ' ', question, flags=re.IGNORECASE)
            question = re.sub(r'[ \t]+', ' ', question).strip()  # Clean up multiple spaces (preserve newlines)

            if original_question != question:
                logger.warning(f"Removed 'undefined' from question. Original length: {len(original_question)}, New length: {len(question)}")

            # Check for cut-off beginnings and fix
            if question and question[0].islower():
                logger.warning(f"Question starts with lowercase, may be cut off: '{question[:50]}...'")
                # Prepend "You " if starts with lowercase
                common_starts = ["mentioned", "said", "told", "indicated", "expressed", "described"]
                for start in common_starts:
                    if question.lower().startswith(start):
                        question = "You " + question
                        logger.info(f"Fixed question beginning in API validation")
                        break

            cleaned_data['question'] = question
        else:
            logger.error(f"Invalid question type: {type(question)}, using fallback")
            cleaned_data['question'] = "Could you tell me more?"

    # Validate other optional text fields
    text_fields = ['message', 'final_report', 'conversation_state', 'career_title',
                   'academic_message', 'skill_message']

    for field in text_fields:
        if field in cleaned_data:
            value = cleaned_data[field]
            if value is None:
                cleaned_data[field] = None  # Keep as None (optional field)
            elif isinstance(value, str):
                cleaned_data[field] = value.strip()
            elif value:  # Non-string, non-None value
                logger.warning(f"Field '{field}' has unexpected type: {type(value)}")
                cleaned_data[field] = str(value).strip()

    # Log message lengths for debugging
    if 'academic_message' in cleaned_data and cleaned_data['academic_message']:
        logger.info(f"📚 API Response - Academic message: {len(cleaned_data['academic_message'])} chars")
    if 'skill_message' in cleaned_data and cleaned_data['skill_message']:
        logger.info(f"🎯 API Response - Skill message: {len(cleaned_data['skill_message'])} chars")

    return cleaned_data

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Career Planning Multi-Agent System API",
        "version": "1.0.0",
        "status": "active",
        "capabilities": [
            "interactive_conversation_sessions",
            "human_in_the_loop_profiling",
            "riasec_psychological_assessment",
            "deterministic_12_question_flow",
            "real_time_qa_flow",
            "session_management",
            "student_profiling",
            "career_exploration",
            "academic_planning",
            "skill_development",
            "future_trends_analysis"
        ],
        "endpoints": {
            "agent_initialize": "/session/initialize",
            "interactive_start": "/sessions/start",
            "interactive_respond": "/sessions/{session_id}/respond",
            "session_status": "/sessions/{session_id}/status",
            "end_session": "/sessions/{session_id}",
            "list_sessions": "/sessions",
            "chat": "/chat",
            "health": "/health",
            "profile": "/profile",
            "career_plans": "/career-plans"
        },
        "modes": {
            "interactive": "Human-in-the-loop conversation with Q&A flow",
            "batch": "Traditional single-request processing"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with environment validation.
    Returns 'healthy' or 'degraded' status.
    This endpoint always returns 200 OK to allow the container to pass health checks.
    """
    try:
        # Validate environment
        env_check = validate_environment()

        # Validate LLM configuration
        try:
            llm_config = LLMFactory.validate_configuration()
            if llm_config["status"] == "error":
                logger.warning(f"LLM configuration issue: {llm_config['message']}")
                status = "degraded"
            else:
                status = env_check.get("status", "healthy")
        except Exception as llm_error:
            logger.warning(f"LLM validation error: {llm_error}")
            status = "degraded"

    except Exception as e:
        logger.warning(f"Health check error: {e}")
        status = "degraded"

    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/tracing/status")
async def tracing_status():
    """Get OpenTelemetry tracing status"""
    tracing_enabled = os.getenv('ENABLE_TRACING', 'false').lower() == 'true'

    if tracing_enabled:
        return {
            "tracing_enabled": True,
            "service_name": os.getenv('OTEL_SERVICE_NAME', 'career-planning-system'),
            "otlp_endpoint": os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://jaeger:4317'),
            "jaeger_ui": "http://localhost:16686",
            "instrumentation": "manual",
            "message": "OpenTelemetry tracing is active"
        }
    else:
        return {
            "tracing_enabled": False,
            "message": "OpenTelemetry tracing is disabled. Set ENABLE_TRACING=true to enable."
        }

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    """Main chat endpoint for streaming responses"""
    if not main_supervisor:
        raise HTTPException(status_code=503, detail="Main supervisor not available")
    
    try:
        # Generate a session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start the career planning workflow
        initial_state = main_supervisor.start_interactive_career_planning(
            student_request=chat_message.message,
            session_id=session_id
        )
        
        logger.info(f"Started workflow for session: {session_id}")
        
        async def generate_streaming_response():
            """Generate streaming response"""
            try:
                yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"
                
                # Process the workflow step by step
                current_state = initial_state
                step_count = 0
                max_steps = 10  # Prevent infinite loops
                
                while current_state is not None and current_state.current_step != "completed" and step_count < max_steps:
                    step_count += 1

                    # Send progress update
                    yield f"data: {json.dumps({'type': 'progress', 'step': current_state.current_step, 'count': step_count})}\n\n"

                    # Process current task
                    result = await main_supervisor.process_task(current_state)
                    
                    # Send intermediate result
                    if result.success:
                        newline = '\n'
                        yield f"data: {json.dumps({'type': 'text', 'content': f'✅ {result.task_type} completed by {result.agent_name}{newline}'})}\n\n"

                        # Update state
                        if result.updated_state is not None:
                            current_state = result.updated_state
                        else:
                            # If no updated state, the workflow is likely complete or waiting
                            yield f"data: {json.dumps({'type': 'warning', 'message': 'No updated state returned, ending workflow'})}\n\n"
                            break

                        # Check if workflow is waiting for user input
                        if result.task_type == "waiting_for_user" or (current_state and current_state.awaiting_user_response):
                            # Send waiting for user response and break the loop
                            waiting_response = {
                                'type': 'waiting_for_user',
                                'session_id': session_id,
                                'question': result.result_data.get('current_question'),
                                'status': 'awaiting_user_response'
                            }
                            yield f"data: {json.dumps(waiting_response)}\n\n"
                            break
                        
                        # If we have profile data, send it
                        if hasattr(current_state, 'student_profile') and current_state.student_profile:
                            profile_data = {
                                'type': 'profile',
                                'data': {
                                    'education_level': current_state.student_profile.current_education_level,
                                    'major_field': current_state.student_profile.major_field,
                                    'technical_skills': current_state.student_profile.technical_skills,
                                    'career_interests': current_state.student_profile.career_interests
                                }
                            }
                            yield f"data: {json.dumps(profile_data)}\n\n"
                        
                        # If we have career blueprints, send them
                        if hasattr(current_state, 'career_blueprints') and current_state.career_blueprints:
                            careers_data = {
                                'type': 'careers',
                                'data': [
                                    {
                                        'title': bp.career_title,
                                        'match_score': bp.match_score,
                                        'description': bp.career_description,
                                        'academic_plan': bp.academic_plan,
                                        'skill_plan': bp.skill_development_plan
                                    }
                                    for bp in current_state.career_blueprints
                                ]
                            }
                            yield f"data: {json.dumps(careers_data)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'❌ Error in {result.task_type}: {result.error_message}'})}\n\n"
                        break
                
                # Send final result
                if current_state.current_step == "completed" or current_state.final_report:
                    final_response = {
                        'type': 'complete',
                        'session_id': session_id,
                        'report': current_state.final_report or "Career planning completed successfully!"
                    }
                    yield f"data: {json.dumps(final_response)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'timeout', 'message': 'Workflow took too long to complete'})}\n\n"
                
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_streaming_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/profile")
async def create_profile(chat_message: ChatMessage):
    """Create student profile from request"""
    if not user_profiler:
        raise HTTPException(status_code=503, detail="User profiler not available")
    
    try:
        profile = user_profiler.create_profile_from_request(chat_message.message)
        
        return {
            "success": True,
            "profile": {
                "education_level": profile.current_education_level,
                "major_field": profile.major_field,
                "technical_skills": profile.technical_skills,
                "soft_skills": profile.soft_skills,
                "career_interests": profile.career_interests,
                "strengths": profile.strengths,
                "goals": profile.goals,
                "summary": profile.to_summary()
            }
        }
    except Exception as e:
        logger.error(f"Error creating profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/career-plans/{session_id}")
async def get_career_plans(session_id: str):
    """Get career plans for a session"""
    # This would typically fetch from a database
    # For now, return a placeholder response
    return {
        "session_id": session_id,
        "status": "completed",
        "career_plans": []
    }

# Interactive Conversation Endpoints

@app.post("/session/initialize", response_model=InteractiveResponse)
async def initialize_agent_session(request: SessionInitializeRequest):
    """Initialize a new agent-initiated career prediction session"""
    if not main_supervisor:
        raise HTTPException(status_code=503, detail="Main supervisor not available")

    try:
        # Generate session ID
        session_id = f"agent_init_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Start agent-initiated session with minimal user request and language
        initial_state = main_supervisor.start_interactive_career_planning(
            student_request="I want career help",  # Minimal trigger
            session_id=session_id,
            language=request.language
        )

        # Store language in session
        if session_id in main_supervisor.active_sessions:
            main_supervisor.active_sessions[session_id]["language"] = request.language

        # Process the initial state to get the agent's first greeting
        result = await main_supervisor.process_task(initial_state)

        if result.success:
            result_data = result.result_data

            # Validate and clean response fields before sending to frontend
            cleaned_data = validate_and_clean_response_fields(result_data)

            return InteractiveResponse(
                success=True,
                session_id=session_id,
                question=cleaned_data.get("question"),
                awaiting_user_response=cleaned_data.get("awaiting_user_response", True),
                conversation_state=cleaned_data.get("conversation_state"),
                progress=cleaned_data.get("progress"),
                completed=False
            )
        else:
            logger.error(f"Failed to initialize agent session: {result.error_message}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize agent session: {result.error_message}")

    except Exception as e:
        logger.error(f"Error initializing agent session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/start", response_model=InteractiveResponse)
async def start_interactive_session(request: SessionStartRequest):
    """Start a new interactive career planning conversation session"""
    if not main_supervisor:
        raise HTTPException(status_code=503, detail="Main supervisor not available")

    try:
        # Generate session ID
        session_id = f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Start interactive session with language
        initial_state = main_supervisor.start_interactive_career_planning(
            student_request=request.initial_message,
            session_id=session_id,
            language=request.language
        )

        # Store language in session
        if session_id in main_supervisor.active_sessions:
            main_supervisor.active_sessions[session_id]["language"] = request.language

        # Process the initial state to get the first question
        result = await main_supervisor.process_task(initial_state)

        if result.success:
            result_data = result.result_data

            # Validate and clean response fields before sending to frontend
            cleaned_data = validate_and_clean_response_fields(result_data)

            return InteractiveResponse(
                success=True,
                session_id=session_id,
                question=cleaned_data.get("question"),
                awaiting_user_response=cleaned_data.get("awaiting_user_response", True),
                conversation_state=cleaned_data.get("conversation_state"),
                progress=cleaned_data.get("progress"),
                completed=False
            )
        else:
            logger.error(f"Failed to start session: {result.error_message}")
            raise HTTPException(status_code=500, detail=f"Failed to start session: {result.error_message}")

    except Exception as e:
        logger.error(f"Error starting interactive session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/respond", response_model=InteractiveResponse)
async def send_user_response(session_id: str, user_response: UserResponse):
    """Send a user response to an active conversation session"""
    if not main_supervisor:
        raise HTTPException(status_code=503, detail="Main supervisor not available")

    try:
        # Get current or provided language
        current_language = user_response.language or "en"

        # Update session language if changed
        if session_id in main_supervisor.active_sessions:
            main_supervisor.active_sessions[session_id]["language"] = current_language

        # Process user response with language
        result = await main_supervisor.process_user_response(
            session_id,
            user_response.response,
            language=current_language
        )

        if result and hasattr(result, 'success') and result.success:
            result_data = result.result_data or {}

            # Validate and clean response fields before sending to frontend
            cleaned_data = validate_and_clean_response_fields(result_data)

            # Check if conversation is complete
            if cleaned_data.get("completed"):
                return InteractiveResponse(
                    success=True,
                    session_id=session_id,
                    question=None,
                    awaiting_user_response=False,
                    conversation_state="completed",
                    final_report=cleaned_data.get("final_report"),
                    completed=True,
                    career_predictions=cleaned_data.get("career_predictions"),
                    # Career planning results
                    career_title=cleaned_data.get("career_title"),
                    academic_message=cleaned_data.get("academic_message"),
                    skill_message=cleaned_data.get("skill_message"),
                    message=cleaned_data.get("message")
                )
            else:
                return InteractiveResponse(
                    success=True,
                    session_id=session_id,
                    question=cleaned_data.get("question"),
                    awaiting_user_response=cleaned_data.get("awaiting_user_response", True),
                    conversation_state=cleaned_data.get("conversation_state"),
                    progress=cleaned_data.get("progress"),
                    completed=False
                )
        else:
            error_msg = result.error_message if result and hasattr(result, 'error_message') else "Unknown error occurred"
            logger.error(f"Failed to process response: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Failed to process response: {error_msg}")

    except Exception as e:
        logger.error(f"Error processing user response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get the current status of a conversation session"""
    if not main_supervisor:
        raise HTTPException(status_code=503, detail="Main supervisor not available")

    try:
        status = main_supervisor.get_session_status(session_id)

        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/confidence")
async def get_session_confidence(session_id: str):
    """Stub endpoint to prevent 404 errors - confidence tracking not implemented"""
    return {
        "session_id": session_id,
        "message": "Confidence tracking not enabled"
    }

@app.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    """End and clean up a conversation session"""
    if not main_supervisor:
        raise HTTPException(status_code=503, detail="Main supervisor not available")

    try:
        success = main_supervisor.end_session(session_id)

        return {
            "success": success,
            "session_id": session_id,
            "message": "Session ended successfully" if success else "Session not found"
        }

    except Exception as e:
        logger.error(f"Error ending session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/sessions/{session_id}/language")
async def update_session_language(session_id: str, request: Dict[str, Any]):
    """Update language preference for active session"""
    if not main_supervisor:
        raise HTTPException(status_code=503, detail="Main supervisor not available")

    try:
        language = request.get("language", "en")

        # Validate language
        if language not in ["en", "si"]:
            raise HTTPException(status_code=400, detail="Invalid language. Must be 'en' or 'si'")

        # Update session
        if session_id in main_supervisor.active_sessions:
            main_supervisor.active_sessions[session_id]["language"] = language

            # Update state if exists
            if hasattr(main_supervisor, 'session_states') and session_id in main_supervisor.session_states:
                main_supervisor.session_states[session_id].preferred_language = language

            return {"success": True, "language": language}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating language: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def list_active_sessions():
    """List all active conversation sessions"""
    if not main_supervisor:
        raise HTTPException(status_code=503, detail="Main supervisor not available")

    try:
        # Get active session IDs (this would need to be implemented in MainSupervisor)
        active_sessions = getattr(main_supervisor, 'active_sessions', {})

        sessions = []
        for session_id, session_info in active_sessions.items():
            sessions.append({
                "session_id": session_id,
                "created_at": session_info.get("created_at"),
                "current_stage": session_info.get("current_stage"),
                "total_interactions": session_info.get("total_interactions", 0)
            })

        return {
            "active_sessions": sessions,
            "total_sessions": len(sessions)
        }

    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# HR-related endpoints for compatibility with frontend
@app.get("/employees")
async def get_employees():
    """Placeholder endpoint for employee data"""
    return []

@app.get("/departments")
async def get_departments():
    """Placeholder endpoint for departments"""
    return []

@app.get("/attendance")
async def get_attendance():
    """Placeholder endpoint for attendance"""
    return []

@app.get("/tasks")
async def get_tasks():
    """Placeholder endpoint for tasks"""
    return []

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )