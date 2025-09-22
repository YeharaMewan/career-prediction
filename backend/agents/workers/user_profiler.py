"""
User Profiler Agent - Human-in-the-Loop Student Assessment

This agent implements the sophisticated conversation flow described in the
User Profiler Agent specification, featuring:
- Finite State Machine conversation management
- RIASEC psychological framework integration
- Chain-of-Thought and Few-Shot prompting
- Confidence-based progression
- Intent recognition and response analysis
- Session management and persistence
- Backward compatibility with batch processing

This is the complete, consolidated implementation that replaces all previous versions.
"""
import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Use absolute imports
from agents.base_agent import WorkerAgent
from agents.conversation_states import (
    ConversationState, ConversationContext, RIASECCategory,
    ConversationStateMachine, create_initial_conversation_context
)
from agents.conversation_manager import ConversationManager, GeneratedQuestion, ResponseAnalysis, QuestionType
from models.state_models import (
    AgentState, TaskResult, StudentProfile, ConversationSession,
    ConversationState as ModelConversationState, RIASECCategory as ModelRIASECCategory
)


class UserProfilerAgent(WorkerAgent):
    """
    Human-in-the-Loop User Profiler Agent implementing advanced career counseling techniques.

    This agent conducts interactive conversations with students to create comprehensive
    profiles using psychological frameworks and sophisticated AI techniques.

    Features:
    - Interactive Q&A conversation flow using FSM
    - RIASEC psychological framework assessment
    - Confidence-based progression through conversation states
    - Chain-of-Thought reasoning for intelligent questions
    - Intent recognition beyond keyword extraction
    - Session persistence and management
    - Backward compatibility with batch processing
    """

    def __init__(self, **kwargs):
        system_prompt = self._create_system_prompt()

        super().__init__(
            name="user_profiler",
            description="Advanced human-in-the-loop student profiler using psychological frameworks and conversation management",
            specialization="interactive_student_profiling_and_assessment",
            system_prompt=system_prompt,
            **kwargs
        )

        # Initialize conversation components
        self.conversation_manager = ConversationManager(
            model=kwargs.get('model', 'gpt-4o-mini'),
            temperature=kwargs.get('temperature', 0.3)
        )
        self.state_machine = ConversationStateMachine()

        # Session storage (in production, use persistent storage)
        self.active_sessions: Dict[str, ConversationContext] = {}

        # Add specific capabilities
        self.capabilities.extend([
            "human_in_the_loop_conversation",
            "psychological_assessment",
            "riasec_profiling",
            "intent_recognition",
            "conversation_state_management",
            "confidence_based_progression",
            "session_management",
            "batch_processing_compatibility"
        ])

        self.logger = logging.getLogger(f"agent.{self.name}")

    def _create_system_prompt(self) -> str:
        """Create the specialized system prompt for interactive career discovery."""
        return """You are an expert AI Career Counselor specializing in human-in-the-loop career discovery.

Your PRIMARY MISSION is to help users discover their most suitable career paths through intelligent conversation.

CORE COMPETENCIES:
1. Career-focused conversation management using Finite State Machines
2. RIASEC psychological framework for career assessment (Realistic, Investigative, Artistic, Social, Enterprising, Conventional)
3. Advanced prompting techniques: Chain-of-Thought reasoning and Few-Shot learning
4. Intent recognition for career-relevant information
5. Confidence-based progression toward career readiness assessment
6. Professional career counseling and guidance methodologies

CONVERSATION FLOW STATES (Career Discovery Focused):
GREETING → ACADEMIC_GATHERING → INTEREST_DISCOVERY → SKILLS_ASSESSMENT → PERSONALITY_PROBE → CAREER_READY → COMPLETED

CAREER ASSESSMENT FRAMEWORK (RIASEC):
- Realistic (R): Practical, hands-on, mechanical work preferences → Engineering, Technical roles
- Investigative (I): Analytical, research-oriented, problem-solving → Data Science, Research roles
- Artistic (A): Creative, imaginative, self-expressive activities → Design, Creative roles
- Social (S): People-oriented, helping, teaching, communication → Management, HR, Education roles
- Enterprising (E): Leadership, business-oriented, persuasive roles → Business, Sales, Leadership roles
- Conventional (C): Organized, systematic, detail-oriented tasks → Finance, Operations, Admin roles

CAREER DISCOVERY PRINCIPLES:
1. Ask targeted questions to gather career-relevant insights
2. Analyze responses for RIASEC indicators and career preferences
3. Build confidence in career assessment accuracy through comprehensive profiling
4. Progress toward identifying suitable career recommendations
5. Ensure suggestions are actionable and well-reasoned
6. Focus conversation on information needed for accurate career matching

ASSESSMENT READINESS CRITERIA:
- High confidence in RIASEC profile (>70%)
- Clear understanding of interests, skills, and work preferences
- Sufficient data to generate distinct, suitable career options
- Strong evidence-based reasoning for each career recommendation

INTERACTION STYLE:
- Enthusiastic about career discovery
- Goal-oriented toward helping find ideal careers
- Professional yet engaging
- Supportive and encouraging throughout the process

Remember: You are an AI career counselor. Your ultimate goal is to help users discover careers that will lead to satisfaction and success, based on comprehensive psychological and preference profiling."""

    def process_task(self, state: AgentState) -> TaskResult:
        """
        Process interactive profiling task with human-in-the-loop conversation.
        """
        start_time = datetime.now()
        session_id = state.session_id or f"profile_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._log_task_start("interactive_profiling", f"session: {session_id}")

        try:
            # Handle different interaction scenarios
            if state.interaction_mode == "interactive":
                return self._handle_interactive_mode(state, session_id)
            else:
                return self._handle_batch_mode(state, session_id)

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_task_completion("interactive_profiling", False, f"Error: {str(e)}")

            return self._create_task_result(
                task_type="interactive_profiling",
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )

    def _handle_interactive_mode(self, state: AgentState, session_id: str) -> TaskResult:
        """Handle interactive conversation mode."""

        # Get or create conversation context
        conversation_context = self._get_or_create_conversation_context(session_id, state)

        # Check if we're waiting for user response
        if state.awaiting_user_response and not state.pending_user_response:
            return self._create_waiting_for_response_result(conversation_context)

        # Process user response if available
        if state.pending_user_response and not state.response_processed:
            return self._process_user_response(state, conversation_context)

        # Generate next question if conversation should continue
        if not self.state_machine.is_conversation_complete(conversation_context):
            return self._generate_next_question(conversation_context)
        else:
            return self._complete_profiling(conversation_context, state)

    def _handle_batch_mode(self, state: AgentState, session_id: str) -> TaskResult:
        """Handle traditional batch processing mode for backward compatibility."""

        student_request = state.student_request
        if not student_request and state.messages:
            for msg in reversed(state.messages):
                if isinstance(msg, HumanMessage):
                    student_request = msg.content
                    break

        if not student_request:
            return self._create_task_result(
                task_type="batch_profiling",
                success=False,
                error_message="No student request found to process"
            )

        # Create a simulated conversation for batch processing
        conversation_context = create_initial_conversation_context()
        conversation_context.session_metadata["session_id"] = session_id
        conversation_context.session_metadata["mode"] = "batch"

        # Process the request as a single comprehensive response
        return self._process_batch_request(student_request, conversation_context, state)

    def _get_or_create_conversation_context(self, session_id: str, state: AgentState) -> ConversationContext:
        """Get existing conversation context or create new one."""

        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        # Create new conversation context
        context = create_initial_conversation_context()
        context.session_metadata["session_id"] = session_id
        context.session_metadata["created_at"] = datetime.now().isoformat()
        context.session_metadata["mode"] = "interactive"

        self.active_sessions[session_id] = context
        return context

    def _process_user_response(self, state: AgentState, context: ConversationContext) -> TaskResult:
        """Process user response and update conversation context."""

        user_response = state.pending_user_response
        if not user_response:
            return self._create_task_result(
                task_type="response_processing",
                success=False,
                error_message="No user response to process"
            )

        # Get the last question that was asked
        last_question = GeneratedQuestion(
            question_text=context.question_history[-1] if context.question_history else "",
            question_type=QuestionType.OPEN_ENDED,
            target_area="general",
            expected_insights=[],
            follow_up_triggers=[],
            riasec_relevance={}
        )

        # Analyze the response
        analysis = self.conversation_manager.analyze_response(user_response, context, last_question)

        # Update conversation context
        context = self.conversation_manager.update_conversation_context(
            context, last_question, user_response, analysis
        )

        # Check if follow-up is needed
        if self.conversation_manager.should_ask_follow_up(context, analysis):
            return self._generate_follow_up_question(context, analysis)

        # Try to transition to next state
        if self.state_machine.can_transition_to_next_state(context):
            self.state_machine.transition_to_next_state(context)

        # Generate next question or complete if ready
        if not self.state_machine.is_conversation_complete(context):
            return self._generate_next_question(context)
        else:
            return self._complete_profiling(context, state)

    def _generate_next_question(self, context: ConversationContext) -> TaskResult:
        """Generate the next question in the conversation."""

        question = self.conversation_manager.generate_question(context)

        # Update context with new question
        context.question_history.append(question.question_text)

        # Store updated context
        session_id = context.session_metadata.get("session_id")
        if session_id:
            self.active_sessions[session_id] = context

        return self._create_task_result(
            task_type="question_generation",
            success=True,
            result_data={
                "question": question.question_text,
                "conversation_state": context.current_state.name,
                "confidence_scores": context.confidence_scores,
                "riasec_scores": {cat.value: score for cat, score in context.riasec_scores.items()},
                "awaiting_response": True,
                "session_id": session_id,
                "progress": {
                    "current_state": context.current_state.name,
                    "questions_asked": len(context.question_history),
                    "overall_confidence": self.state_machine.get_overall_confidence(context),
                    "completion_readiness": self.state_machine.get_completion_readiness(context)
                }
            }
        )

    def _generate_follow_up_question(self, context: ConversationContext, analysis: ResponseAnalysis) -> TaskResult:
        """Generate a follow-up question based on response analysis."""

        follow_up_question = self.conversation_manager.generate_follow_up_question(context, analysis)

        # Update context
        context.question_history.append(follow_up_question.question_text)

        session_id = context.session_metadata.get("session_id")
        if session_id:
            self.active_sessions[session_id] = context

        return self._create_task_result(
            task_type="follow_up_question",
            success=True,
            result_data={
                "question": follow_up_question.question_text,
                "question_type": "follow_up",
                "conversation_state": context.current_state.name,
                "confidence_scores": context.confidence_scores,
                "awaiting_response": True,
                "session_id": session_id,
                "follow_up_reason": "Need more specific information"
            }
        )

    def _create_waiting_for_response_result(self, context: ConversationContext) -> TaskResult:
        """Create result when waiting for user response."""
        return self._create_task_result(
            task_type="waiting_for_response",
            success=True,
            result_data={
                "status": "waiting_for_user_response",
                "current_question": context.question_history[-1] if context.question_history else None,
                "conversation_state": context.current_state.name,
                "session_id": context.session_metadata.get("session_id"),
                "progress": {
                    "questions_asked": len(context.question_history),
                    "responses_received": len(context.response_history),
                    "overall_confidence": self.state_machine.get_overall_confidence(context)
                }
            }
        )

    def _complete_profiling(self, context: ConversationContext, state: AgentState) -> TaskResult:
        """Complete the profiling process and create final student profile."""

        # Create comprehensive student profile
        student_profile = self._create_student_profile_from_context(context)

        # Create session summary
        session_summary = self.conversation_manager.create_session_summary(context)

        # Update state with completed profile
        state.student_profile = student_profile
        state.awaiting_user_response = False
        state.current_question = None
        state.interaction_mode = "completed"

        # Clean up session
        session_id = context.session_metadata.get("session_id")
        if session_id and session_id in self.active_sessions:
            del self.active_sessions[session_id]

        self._log_task_completion("interactive_profiling", True, f"Profile completed for session {session_id}")

        return self._create_task_result(
            task_type="profile_completion",
            success=True,
            result_data={
                "student_profile": student_profile.dict(),
                "session_summary": session_summary,
                "conversation_complete": True,
                "profile_confidence": student_profile.profile_confidence_score,
                "dominant_riasec": student_profile.dominant_riasec_type,
                "total_questions": len(context.question_history),
                "session_id": session_id
            }
        )

    def _create_student_profile_from_context(self, context: ConversationContext) -> StudentProfile:
        """Create a comprehensive StudentProfile from conversation context."""

        collected_data = context.collected_data
        riasec_scores = {cat.value: score for cat, score in context.riasec_scores.items()}

        # Find dominant RIASEC type
        dominant_riasec = None
        if riasec_scores:
            dominant_riasec = max(riasec_scores.items(), key=lambda x: x[1])[0]

        # Calculate overall confidence
        overall_confidence = self.state_machine.get_overall_confidence(context)

        return StudentProfile(
            # Basic information from collected data
            name=collected_data.get("name"),
            age=collected_data.get("age"),

            # Academic background
            current_education_level=collected_data.get("education_level", "Unknown"),
            major_field=collected_data.get("major_field"),
            academic_performance=collected_data.get("academic_performance"),

            # Skills extracted from conversation
            technical_skills=collected_data.get("technical_skills", []),
            soft_skills=collected_data.get("soft_skills", []),

            # Interests and preferences
            career_interests=collected_data.get("career_interests", []),
            work_environment_preferences=collected_data.get("work_environment_preferences", []),
            industry_preferences=collected_data.get("industry_preferences", []),

            # Goals
            short_term_goals=collected_data.get("short_term_goals", []),
            long_term_goals=collected_data.get("long_term_goals", []),

            # Personality and work style
            personality_traits=collected_data.get("personality_traits", []),
            learning_style=collected_data.get("learning_style"),
            work_style=collected_data.get("work_style"),

            # RIASEC results
            riasec_scores=riasec_scores,
            dominant_riasec_type=dominant_riasec,
            riasec_confidence=overall_confidence,

            # Metadata
            profile_confidence_score=overall_confidence,
            conversation_session_id=context.session_metadata.get("session_id"),
            profile_completeness=overall_confidence,
            areas_needing_clarification=context.follow_up_needed
        )

    def _process_batch_request(self, student_request: str, context: ConversationContext, state: AgentState) -> TaskResult:
        """Process a batch request for backward compatibility."""

        # Use the conversation manager to analyze the request
        question = GeneratedQuestion(
            question_text="Please tell me about your background, interests, and career goals",
            question_type=QuestionType.OPEN_ENDED,
            target_area="comprehensive",
            expected_insights=[],
            follow_up_triggers=[],
            riasec_relevance={}
        )

        analysis = self.conversation_manager.analyze_response(student_request, context, question)

        # Update context with analysis
        context = self.conversation_manager.update_conversation_context(
            context, question, student_request, analysis
        )

        # Force completion for batch mode
        context.current_state = ConversationState.COMPLETED

        # Create profile
        student_profile = self._create_student_profile_from_context(context)
        state.student_profile = student_profile

        return self._create_task_result(
            task_type="batch_profiling",
            success=True,
            result_data={
                "student_profile": student_profile.dict(),
                "riasec_scores": {cat.value: score for cat, score in context.riasec_scores.items()},
                "confidence_scores": context.confidence_scores,
                "mode": "batch"
            }
        )

    # Public interface methods for interactive sessions
    def start_interactive_session(self, session_id: str, initial_message: Optional[str] = None) -> TaskResult:
        """Start a new interactive profiling session."""

        context = self._get_or_create_conversation_context(session_id, None)

        # Generate greeting question
        return self._generate_next_question(context)

    def process_user_response_for_session(self, session_id: str, user_response: str) -> TaskResult:
        """Process a user response for a specific session."""

        if session_id not in self.active_sessions:
            return self._create_task_result(
                task_type="session_error",
                success=False,
                error_message=f"Session {session_id} not found"
            )

        context = self.active_sessions[session_id]

        # Create a mock state for processing
        mock_state = AgentState(
            session_id=session_id,
            pending_user_response=user_response,
            response_processed=False,
            interaction_mode="interactive"
        )

        return self._process_user_response(mock_state, context)

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get the current status of a conversation session."""

        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        context = self.active_sessions[session_id]

        return {
            "session_id": session_id,
            "current_state": context.current_state.name,
            "questions_asked": len(context.question_history),
            "responses_received": len(context.response_history),
            "confidence_scores": context.confidence_scores,
            "riasec_scores": {cat.value: score for cat, score in context.riasec_scores.items()},
            "overall_confidence": self.state_machine.get_overall_confidence(context),
            "completion_readiness": self.state_machine.get_completion_readiness(context),
            "is_complete": self.state_machine.is_conversation_complete(context)
        }

    def end_session(self, session_id: str) -> bool:
        """End and clean up a conversation session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False

    def create_profile_from_request(self, student_request: str, session_id: str = None) -> StudentProfile:
        """
        Convenience method for backward compatibility.
        Creates a profile from a single request using batch processing.
        """
        from datetime import datetime

        if not session_id:
            session_id = f"compat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        state = AgentState(
            student_request=student_request,
            current_step="profiling",
            session_id=session_id,
            interaction_mode="batch"  # Force batch mode for compatibility
        )

        result = self.process_task(state)

        if result.success and result.result_data and "student_profile" in result.result_data:
            return StudentProfile(**result.result_data["student_profile"])
        else:
            # Return a minimal profile if analysis fails
            return StudentProfile(
                current_education_level="Unknown",
                career_interests=["General"],
                profile_confidence_score=0.3  # Low confidence for failed analysis
            )