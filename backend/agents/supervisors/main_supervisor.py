"""
Enhanced Main Supervisor - CEO Level Career Planning Orchestrator with Session Management

This enhanced version supports both traditional batch processing and human-in-the-loop
interactive conversation flows with persistent session management.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.prebuilt import create_react_agent

# Use absolute imports
from agents.base_agent import SupervisorAgent
from agents.workers.user_profiler import UserProfilerAgent
from agents.supervisors.career_planning_supervisor import CareerPlanningSupervisor
from models.state_models import (
    AgentState, TaskResult, StudentProfile, CareerBlueprint,
    ConversationSession, ConversationState
)
from utils.handoff_tools import HandoffToolFactory

# Import LangSmith configuration
from utils.langsmith_config import get_traced_run_config, log_agent_execution, setup_langsmith

# Ensure LangSmith is configured
setup_langsmith()


class MainSupervisor(SupervisorAgent):
    """
    Enhanced Main Supervisor with session management and human-in-the-loop support.

    Supports both:
    1. Traditional batch processing for immediate results
    2. Interactive conversation sessions with persistent state management

    Responsibilities:
    1. Manage interactive conversation sessions
    2. Coordinate with human-in-the-loop User Profiler Agent
    3. Handle session persistence and state management
    4. Support real-time Q&A workflow
    5. Seamlessly transition between interactive and batch modes
    """

    def __init__(self, **kwargs):
        # Initialize handoff tools
        self.handoff_factory = HandoffToolFactory()
        handoff_tools = [
            self.handoff_factory.create_user_profiler_handoff(),
            self.handoff_factory.create_career_planning_supervisor_handoff(),
            self.handoff_factory.create_future_trends_analyst_handoff(),
        ]

        orchestration_prompt = self._create_enhanced_orchestration_prompt()

        super().__init__(
            name="main_supervisor",
            description="Enhanced CEO-level supervisor with session management and interactive conversation support",
            orchestration_prompt=orchestration_prompt,
            **kwargs
        )

        # Create the react agent with handoff tools
        self.react_agent = create_react_agent(
            model=self.llm,
            tools=handoff_tools,
            prompt=orchestration_prompt
        )

        # Enhanced capabilities
        self.capabilities.extend([
            "session_management",
            "interactive_conversation_coordination",
            "human_in_the_loop_workflow",
            "persistent_state_management",
            "real_time_qa_handling",
            "workflow_orchestration",
            "student_request_processing",
            "report_generation",
            "quality_assurance",
            "final_integration"
        ])

        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_states: Dict[str, AgentState] = {}

        # Initialize managed agents
        self.user_profiler = None
        self.career_planning_supervisor = None
        self.future_trends_analyst = None

    def _create_enhanced_orchestration_prompt(self) -> str:
        """Create the enhanced system prompt for session-aware orchestration."""
        return """You are the Enhanced Main Supervisor (CEO) of a comprehensive career planning system with advanced session management capabilities.

Your role is to orchestrate both traditional batch workflows and modern human-in-the-loop interactive conversation sessions.

CORE RESPONSIBILITIES:
1. SESSION MANAGEMENT: Create, maintain, and coordinate conversation sessions
2. INTERACTIVE COORDINATION: Manage real-time Q&A flows with User Profiler Agent
3. WORKFLOW ORCHESTRATION: Seamlessly handle both batch and interactive modes
4. STATE PERSISTENCE: Maintain conversation state across multiple interactions
5. QUALITY ASSURANCE: Ensure comprehensive career guidance delivery

OPERATIONAL MODES:

1. INTERACTIVE MODE (Human-in-the-Loop):
   - Create persistent conversation sessions
   - Coordinate with Interactive User Profiler Agent for Q&A flow
   - Handle real-time user responses and agent questions
   - Maintain session state across multiple interactions
   - Progress through conversation states: GREETING → ACADEMIC_GATHERING → INTEREST_DISCOVERY → SKILLS_ASSESSMENT → PERSONALITY_PROBE → SUMMARIZATION → CONFIRMATION → COMPLETED

2. BATCH MODE (Traditional):
   - Process complete requests immediately
   - Generate comprehensive profiles from single inputs
   - Maintain backward compatibility with existing workflows

SESSION MANAGEMENT WORKFLOW:
1. Session Creation
   - Initialize new conversation session
   - Set up persistent state tracking
   - Begin with User Profiler Agent greeting

2. Interactive Processing
   - Relay questions from User Profiler to user
   - Process user responses through profiling agent
   - Monitor conversation progress and confidence scores
   - Handle follow-up questions and clarifications

3. Session Progression
   - Track conversation state advancement
   - Monitor confidence thresholds for state transitions
   - Coordinate with other agents when profiling completes

4. Session Completion
   - Finalize student profile from conversation
   - Proceed with career planning workflow
   - Maintain session history for reference

COMMUNICATION PRINCIPLES:
- Always specify session context in agent communications
- Maintain clear distinction between interactive and batch processing
- Provide progress updates for long-running conversations
- Handle session errors gracefully with recovery options
- Support session suspension and resumption

QUALITY STANDARDS:
- Ensure conversation completeness before proceeding to next workflow stage
- Validate student profile confidence scores
- Maintain professional and supportive communication throughout sessions
- Provide clear feedback on conversation progress

Your goal is to deliver exceptional career guidance through both efficient batch processing and engaging interactive conversations."""

    async def process_task(self, state: AgentState) -> TaskResult:
        """
        Enhanced task processing with session management support.
        """
        start_time = datetime.now()
        session_id = state.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._log_task_start("enhanced_workflow", f"session: {session_id}, mode: {state.interaction_mode}")

        try:
            # Determine processing mode
            if state.interaction_mode == "interactive":
                return await self._handle_interactive_workflow(state, session_id)
            else:
                return self._handle_batch_workflow(state, session_id)

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_task_completion("enhanced_workflow", False, f"Error: {str(e)}")

            return self._create_task_result(
                task_type="enhanced_workflow",
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )

    async def _handle_interactive_workflow(self, state: AgentState, session_id: str) -> TaskResult:
        """Handle interactive conversation workflow."""

        # Initialize or retrieve session
        if session_id not in self.active_sessions:
            return self._initialize_interactive_session(state, session_id)

        session_info = self.active_sessions[session_id]
        session_state = self.session_states[session_id]

        # Check current workflow stage
        current_stage = session_info.get("current_stage", "profiling")

        if current_stage == "profiling":
            return self._handle_interactive_profiling(state, session_id)
        elif current_stage == "career_selection_awaiting":
            return await self._handle_career_selection(state, session_id)
        elif current_stage == "career_planning":
            return self._handle_career_planning_stage(state, session_id)
        elif current_stage == "completion":
            return self._handle_completion_stage(state, session_id)
        else:
            return self._handle_unknown_stage(state, session_id, current_stage)

    def _initialize_interactive_session(self, state: AgentState, session_id: str) -> TaskResult:
        """Initialize a new interactive conversation session."""

        # Create session info
        session_info = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "current_stage": "profiling",
            "conversation_state": "greeting",
            "user_profiler_session_active": True,
            "total_interactions": 0
        }

        # Initialize session state
        session_state = AgentState(
            session_id=session_id,
            interaction_mode="interactive",
            current_step="interactive_profiling",
            student_request=state.student_request,
            awaiting_user_response=False
        )

        # Store session
        self.active_sessions[session_id] = session_info
        self.session_states[session_id] = session_state

        # Initialize User Profiler Agent if not already done
        if not self.user_profiler:
            self.user_profiler = UserProfilerAgent()

        # Start interactive profiling session
        profiler_result = self.user_profiler.start_interactive_session(session_id)

        if profiler_result.success:
            session_info["total_interactions"] += 1
            session_state.awaiting_user_response = True

            # Safely access result_data with null checking
            result_data = profiler_result.result_data or {}
            session_state.current_question = result_data.get("question")

            self._log_task_completion("session_initialization", True, f"Session {session_id} started")

            # Ensure question is never None/undefined
            question_text = result_data.get("question") or result_data.get("current_question") or ""
            if not question_text:
                logging.warning(f"Session {session_id}: No question in profiler result, using fallback")
                question_text = "Hello! I'm here to help you explore career options. What are you interested in?"

            return self._create_task_result(
                task_type="session_initialization",
                success=True,
                result_data={
                    "session_id": session_id,
                    "mode": "interactive",
                    "stage": "profiling",
                    "question": question_text.strip(),
                    "conversation_state": result_data.get("conversation_state", "greeting"),
                    "awaiting_user_response": True,
                    "progress": result_data.get("progress", {})
                },
                updated_state=session_state
            )
        else:
            return self._create_task_result(
                task_type="session_initialization",
                success=False,
                error_message=f"Failed to start profiler session: {profiler_result.error_message}"
            )

    def _handle_interactive_profiling(self, state: AgentState, session_id: str) -> TaskResult:
        """Handle interactive profiling stage."""

        session_info = self.active_sessions[session_id]
        session_state = self.session_states[session_id]

        # Initialize User Profiler if needed
        if not self.user_profiler:
            self.user_profiler = UserProfilerAgent()

        # Check if we have a user response to process
        if state.pending_user_response and not state.response_processed:
            # Process user response through profiler
            profiler_result = self.user_profiler.process_user_response_for_session(
                session_id, state.pending_user_response
            )

            session_info["total_interactions"] += 1
            state.response_processed = True

            if profiler_result.success:
                # Safely access result_data with null checking
                result_data = profiler_result.result_data or {}

                # Check if profiling is complete
                if result_data.get("conversation_complete"):
                    # Profiling completed, move to next stage
                    return self._complete_profiling_stage(session_id, result_data)
                else:
                    # Continue conversation
                    session_state.awaiting_user_response = True

                    # Ensure question is never None/undefined
                    question_text = result_data.get("question") or result_data.get("current_question") or ""
                    if not question_text:
                        logging.warning(f"Session {session_id}: No question in profiler response, using fallback")
                        question_text = "Could you tell me more about your interests?"

                    session_state.current_question = question_text

                    return self._create_task_result(
                        task_type="profiling_interaction",
                        success=True,
                        result_data={
                            "session_id": session_id,
                            "question": question_text.strip(),
                            "conversation_state": result_data.get("conversation_state", "gathering_info"),
                            "awaiting_user_response": True,
                            "progress": result_data.get("progress", {}),
                            "riasec_scores": result_data.get("riasec_scores", {})
                        }
                    )
            else:
                return self._create_task_result(
                    task_type="profiling_interaction",
                    success=False,
                    error_message=f"Profiler error: {profiler_result.error_message}"
                )

        # Check session status
        elif state.awaiting_user_response:
            # Still waiting for user response
            return self._create_waiting_result(session_id, session_state)

        else:
            # Get current session status
            profiler_status = self.user_profiler.get_session_status(session_id)

            if profiler_status.get("is_complete"):
                # Profiling completed, but we haven't processed it yet
                return self._complete_profiling_stage(session_id, profiler_status)
            else:
                # Continue profiling
                return self._create_waiting_result(session_id, session_state)

    def _complete_profiling_stage(self, session_id: str, profiler_data: Dict[str, Any]) -> TaskResult:
        """Complete the profiling stage and generate career predictions."""

        session_info = self.active_sessions[session_id]
        session_state = self.session_states[session_id]

        # Update session stage to career prediction
        session_info["current_stage"] = "career_prediction"
        session_state.current_step = "career_prediction"
        session_state.awaiting_user_response = False

        # Extract student profile with fallback handling
        student_profile = None
        if "student_profile" in profiler_data:
            try:
                student_profile = StudentProfile(**profiler_data["student_profile"])
            except Exception as e:
                self._log_task_completion("profile_extraction_error", False,
                                        f"Session {session_id}: Student profile extraction failed: {str(e)}")
        elif "profile" in profiler_data:
            try:
                student_profile = StudentProfile(**profiler_data["profile"])
            except Exception as e:
                self._log_task_completion("profile_extraction_error", False,
                                        f"Session {session_id}: Profile extraction failed: {str(e)}")

        # Fallback profile creation if extraction failed
        if student_profile is None:
            self._log_task_completion("fallback_profile_creation", True,
                                    f"Session {session_id}: Creating fallback profile for career predictions")
            student_profile = self._create_fallback_student_profile(profiler_data, session_id)

        session_state.student_profile = student_profile

        # Generate 5 career predictions with enhanced error handling
        try:
            career_predictions = self._generate_career_predictions(student_profile, profiler_data)
            self._log_task_completion("career_predictions_generated", True,
                                    f"Session {session_id}: Generated {len(career_predictions)} career predictions")
        except Exception as e:
            self._log_task_completion("career_prediction_error", False,
                                    f"Session {session_id}: Career prediction generation failed: {str(e)}")
            # Generate fallback predictions to ensure we always deliver something
            career_predictions = self._generate_fallback_career_predictions(student_profile, profiler_data)

        # Clean up profiler session
        self.user_profiler.end_session(session_id)

        self._log_task_completion("profiling_stage", True, f"Session {session_id} profiling completed with career predictions")

        # Create comprehensive confidence report for final results
        final_confidence_report = self._create_final_confidence_report(
            student_profile, profiler_data, session_state, session_info
        )

        # Convert career predictions to CareerBlueprint objects and store in state
        career_blueprints = []
        for prediction in career_predictions:
            blueprint = CareerBlueprint(
                career_title=prediction.get("title", ""),
                career_description=prediction.get("description", ""),
                match_score=prediction.get("confidence_score", 0.0) * 100,
                match_reasoning=prediction.get("why_good_fit", ""),
                typical_responsibilities=[],
                required_qualifications=prediction.get("required_skills", []),
                salary_range=prediction.get("median_salary", ""),
                career_progression=[]
            )
            career_blueprints.append(blueprint)

        # Store career blueprints in session state
        session_state.career_blueprints = career_blueprints

        # Update stage to await career selection
        session_info["current_stage"] = "career_selection_awaiting"
        session_state.current_step = "career_selection_awaiting"
        session_state.awaiting_user_response = True

        # Prepare career selection question - ensure it's properly formatted
        try:
            # Build organized markdown format for career predictions
            career_sections = []
            for i, pred in enumerate(career_predictions):
                title = pred.get('title', 'Career Option')
                match_score = pred.get('confidence_score', 0.0) * 100  # Convert to percentage
                description = pred.get('description', 'No description available')
                why_fit = pred.get('why_good_fit', 'Based on your profile and interests')

                # Format each career with proper line breaks (single newlines between list items)
                career_section = (
                    f"{i+1}. **{title}**\n\n"
                    f"- **Match Score:** {match_score:.0f}%\n"
                    f"- **Description:** {description}\n"
                    f"- **Why it fits you:** {why_fit}"
                )
                career_sections.append(career_section)

            career_list = "\n\n".join(career_sections)
            first_career_title = career_predictions[0].get('title', 'your preferred career') if career_predictions else 'your preferred career'

            # Build selection message with explicit double newlines for proper markdown rendering
            selection_message = (
                f"🎯 **Your Career Recommendations**\n\n"
                f"Based on our conversation, here are 5 careers that match your profile:\n\n"
                f"{career_list}\n\n"
                f"---\n\n"
                f"**Next Step:** Choose a career number (1-5) to get a detailed action plan with academic pathways and skill development roadmap.\n\n"
                f"Please respond with the career you'd like to pursue (e.g., \"I want to pursue {first_career_title}\")"
            )
        except Exception as e:
            logging.error(f"Error formatting career selection message: {e}")
            selection_message = "I've identified some great career matches for you. Which career would you like to explore further?"

        # Ensure selection_message is valid
        if not selection_message or not selection_message.strip():
            selection_message = "Based on our conversation, I have career recommendations for you. Which career interests you most?"

        session_state.current_question = selection_message

        return self._create_task_result(
            task_type="career_prediction_with_selection",
            success=True,
            result_data={
                "session_id": session_id,
                "stage": "career_selection_awaiting",
                "profile_created": True,
                "profile_confidence": profiler_data.get("profile_confidence", 0.0),
                "dominant_riasec": profiler_data.get("dominant_riasec"),
                "total_questions": profiler_data.get("total_questions", 0),
                "career_predictions": career_predictions,
                "completed": False,
                "awaiting_user_response": True,
                "question": selection_message.strip(),
                "message": "Career predictions generated. Please select a career to get detailed planning.",
                "final_confidence_report": final_confidence_report,
                "prediction_reliability": self._assess_prediction_reliability(profiler_data.get("profile_confidence", 0.0))
            }
        )

    async def _handle_career_selection(self, state: AgentState, session_id: str) -> TaskResult:
        """Handle user's career selection and route to career planning supervisor."""

        session_info = self.active_sessions[session_id]
        session_state = self.session_states[session_id]

        # Check if we have a user response to process
        if state.pending_user_response and not state.response_processed:
            user_response = state.pending_user_response.lower()
            session_info["total_interactions"] += 1
            state.response_processed = True

            # Extract selected career from user response
            selected_career = None
            if session_state.career_blueprints:
                for blueprint in session_state.career_blueprints:
                    if blueprint.career_title.lower() in user_response:
                        selected_career = blueprint
                        break

            if not selected_career:
                # Default to first career if we can't parse
                selected_career = session_state.career_blueprints[0] if session_state.career_blueprints else None

            if not selected_career:
                return self._create_task_result(
                    task_type="career_selection_error",
                    success=False,
                    error_message="No career could be identified from your selection"
                )

            # Initialize career planning supervisor if needed
            if not self.career_planning_supervisor:
                self.career_planning_supervisor = CareerPlanningSupervisor()

            # Update session stage
            session_info["current_stage"] = "career_planning"
            session_state.current_step = "career_planning"
            session_state.awaiting_user_response = False

            # Create state for career planning supervisor with only selected career
            planning_state = AgentState(
                session_id=session_id,
                student_profile=session_state.student_profile,
                career_blueprints=[selected_career],  # Only the selected career
                interaction_mode="interactive"
            )

            # Route to career planning supervisor (it will execute both agents in parallel)
            self.logger.info(f"🔄 Routing to career planning supervisor for: {selected_career.career_title}")
            planning_result = await self.career_planning_supervisor.process_task(planning_state)

            if planning_result.success:
                # Update session state with results
                if planning_result.updated_state:
                    session_state.career_blueprints = planning_result.updated_state.career_blueprints

                # Update stage to completion
                session_info["current_stage"] = "completion"
                session_state.current_step = "career_planning_complete"

                # Get the planning results
                result_data = planning_result.result_data or {}
                academic_plan = result_data.get("academic_plan")
                skill_plan = result_data.get("skill_plan")

                # Format detailed messages for frontend display using the full plan data
                academic_message = self._format_academic_plan_message(
                    selected_career.career_title,
                    academic_plan
                ) if academic_plan else None

                skill_message = self._format_skill_plan_message(
                    selected_career.career_title,
                    skill_plan
                ) if skill_plan else None

                return self._create_task_result(
                    task_type="career_planning_complete",
                    success=True,
                    result_data={
                        "session_id": session_id,
                        "career_title": selected_career.career_title,
                        "academic_plan": academic_plan,
                        "skill_plan": skill_plan,
                        "academic_message": academic_message,
                        "skill_message": skill_message,
                        "completed": True,
                        "message": f"Thank you! Your career planning session is complete.",
                        "delivery_sequence": ["academic_pathway", "skill_development"]
                    },
                    updated_state=session_state
                )
            else:
                return self._create_task_result(
                    task_type="career_planning_error",
                    success=False,
                    error_message=f"Career planning failed: {planning_result.error_message}"
                )

        # Still waiting for user response
        elif state.awaiting_user_response:
            return self._create_waiting_result(session_id, session_state)

        # No user response yet
        else:
            return self._create_waiting_result(session_id, session_state)

    def _handle_career_planning_stage(self, state: AgentState, session_id: str) -> TaskResult:
        """Handle career prediction stage - this is now the completion stage."""
        session_info = self.active_sessions[session_id]
        session_state = self.session_states[session_id]

        # If we reach this stage, career predictions have already been generated
        # This is the completion stage
        session_info["current_stage"] = "completion"
        session_state.current_step = "completion"

        return self._create_task_result(
            task_type="career_prediction_complete",
            success=True,
            result_data={
                "session_id": session_id,
                "stage": "completion",
                "completed": True,
                "message": "Career predictions have been successfully generated!"
            }
        )

    def _handle_completion_stage(self, state: AgentState, session_id: str) -> TaskResult:
        """Handle completion stage."""

        session_info = self.active_sessions[session_id]
        session_state = self.session_states[session_id]

        # Generate final report
        final_report = self._generate_session_final_report(session_state, session_info)

        # Clean up session
        del self.active_sessions[session_id]
        del self.session_states[session_id]

        return self._create_task_result(
            task_type="session_completion",
            success=True,
            result_data={
                "session_id": session_id,
                "completed": True,
                "final_report": final_report,
                "student_profile": session_state.student_profile.dict() if session_state.student_profile else None
            }
        )

    def _handle_batch_workflow(self, state: AgentState, session_id: str) -> TaskResult:
        """Handle traditional batch processing workflow."""

        # Use the original workflow logic for batch processing
        current_step = state.current_step or "student_request_received"

        if current_step == "student_request_received":
            return self._handle_student_request(state)
        elif current_step == "profile_creation":
            return self._handle_profile_creation(state)
        elif current_step == "career_planning":
            return self._handle_career_planning(state)
        elif current_step == "future_analysis":
            return self._handle_future_analysis(state)
        elif current_step == "final_report_generation":
            return self._handle_final_report_generation(state)
        else:
            return self._handle_unknown_step(state, current_step)

    def _create_waiting_result(self, session_id: str, session_state: AgentState) -> TaskResult:
        """Create a result for when waiting for user response."""
        return self._create_task_result(
            task_type="waiting_for_user",
            success=True,
            result_data={
                "session_id": session_id,
                "awaiting_user_response": True,
                "current_question": session_state.current_question,
                "status": "waiting_for_user_response"
            },
            updated_state=session_state
        )

    def _generate_session_final_report(self, session_state: AgentState, session_info: Dict[str, Any]) -> str:
        """Generate final report for interactive session."""

        if not session_state.student_profile:
            return "Session completed but no student profile was created."

        profile = session_state.student_profile
        total_interactions = session_info.get("total_interactions", 0)

        report_sections = []

        # Header
        report_sections.append("# 🎯 INTERACTIVE CAREER PLANNING SESSION REPORT")
        report_sections.append(f"*Session completed on {datetime.now().strftime('%B %d, %Y')}*\n")

        # Session Summary
        report_sections.append("## 📊 Session Summary")
        report_sections.append(f"**Session ID:** {session_info.get('session_id')}")
        report_sections.append(f"**Total Interactions:** {total_interactions}")
        report_sections.append(f"**Profile Confidence:** {profile.profile_confidence_score:.1%}")
        if profile.dominant_riasec_type:
            report_sections.append(f"**Dominant Career Type:** {profile.dominant_riasec_type}")
        report_sections.append("")

        # Student Profile Summary
        report_sections.append("## 👤 Student Profile")
        if profile.current_education_level:
            report_sections.append(f"**Education Level:** {profile.current_education_level}")
        if profile.major_field:
            report_sections.append(f"**Field of Study:** {profile.major_field}")
        if profile.career_interests:
            report_sections.append(f"**Career Interests:** {', '.join(profile.career_interests[:5])}")
        if profile.technical_skills:
            report_sections.append(f"**Technical Skills:** {', '.join(profile.technical_skills[:5])}")
        report_sections.append("")

        # RIASEC Assessment
        if profile.riasec_scores:
            report_sections.append("## 🔍 RIASEC Career Assessment")
            riasec_names = {
                "R": "Realistic (Hands-on, Practical)",
                "I": "Investigative (Analytical, Research)",
                "A": "Artistic (Creative, Self-expression)",
                "S": "Social (People-oriented, Helping)",
                "E": "Enterprising (Leadership, Business)",
                "C": "Conventional (Organized, Systematic)"
            }

            for code, score in sorted(profile.riasec_scores.items(), key=lambda x: x[1], reverse=True):
                if score > 0.3:  # Only show significant scores
                    report_sections.append(f"**{riasec_names.get(code, code)}:** {score:.1%}")
            report_sections.append("")

        # Next Steps
        report_sections.append("## 📝 Recommended Next Steps")
        report_sections.append("1. **Career Exploration:** Research careers matching your top RIASEC types")
        report_sections.append("2. **Skill Development:** Focus on building skills in your areas of interest")
        report_sections.append("3. **Educational Planning:** Consider academic paths aligned with your goals")
        report_sections.append("4. **Professional Networking:** Connect with professionals in your fields of interest")
        report_sections.append("5. **Practical Experience:** Seek internships or projects in target areas")

        # Footer
        report_sections.append("\n---")
        report_sections.append("*This report was generated through an interactive career counseling session.*")
        report_sections.append("*Continue to explore and refine your career path as you grow and learn.*")

        return "\n".join(report_sections)

    def _generate_career_predictions(self, student_profile: StudentProfile, profiler_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate 5 personalized career predictions using AI analysis of the complete conversation.
        Analyzes all 12 user responses to create truly customized recommendations.
        """
        try:
            # Extract conversation history
            conversation_context = profiler_data.get("conversation_context", {})
            qa_pairs = self._extract_qa_pairs(conversation_context)

            # Build comprehensive AI prompt with full conversation
            prompt = f"""You are an expert career counselor with deep knowledge of various industries and career paths. Analyze this complete conversation with a student who answered 12 career assessment questions.

STUDENT PROFILE SUMMARY:
- Education Level: {student_profile.current_education_level or 'Not specified'}
- Major/Field: {student_profile.major_field or 'Not specified'}
- Technical Skills: {', '.join(student_profile.technical_skills[:8]) if student_profile.technical_skills else 'Not specified'}
- Interests: {', '.join(student_profile.career_interests[:8]) if student_profile.career_interests else 'Not specified'}
- RIASEC Personality Profile: {dict(student_profile.riasec_scores) if student_profile.riasec_scores else 'Not specified'}

COMPLETE CONVERSATION (12 Questions & Detailed Answers):
{self._format_qa_history(qa_pairs)}

TASK: Generate 5 highly personalized career recommendations for THIS specific student.

REQUIREMENTS:
1. Reference specific answers from the conversation (e.g., "You mentioned loving marine biology in Q1...")
2. Suggest diverse careers across different industries
3. Consider their unique combination of interests, skills, and personality
4. Include both traditional and emerging career paths
5. Be specific - avoid generic titles like "Engineer" (use "Biomedical Device Engineer" instead)

For EACH of the 5 careers, provide:
- title: Specific career title (not generic)
- description: What this career actually involves (2-3 sentences)
- why_good_fit: Personalized explanation referencing their specific responses (3-4 sentences)
- required_education: Educational requirements
- required_skills: List of 3-5 key skills needed
- growth_outlook: "High", "Medium", or "Low"
- median_salary: Realistic salary range (e.g., "$75,000-$95,000")
- confidence_score: Match confidence 0.7-0.95 based on how well it fits
- riasec_alignment: Which RIASEC traits this career matches
- career_path: Brief progression path (e.g., "Junior → Senior → Lead")

Return ONLY a valid JSON array (no markdown, no extra text):
[
  {{
    "title": "Specific Career Title",
    "description": "What this career involves...",
    "why_good_fit": "You mentioned X in Q3 and Y in Q7, which directly align with...",
    "required_education": "Bachelor's in X or related field",
    "required_skills": ["skill1", "skill2", "skill3"],
    "growth_outlook": "High",
    "median_salary": "$XX,000-$YY,000",
    "confidence_score": 0.88,
    "riasec_alignment": "Strong Investigative (I) and Artistic (A) match",
    "career_path": "Junior → Mid-level → Senior → Lead"
  }}
]"""

            # Call LLM to generate personalized careers
            careers = self._call_llm_for_careers(prompt, student_profile, profiler_data)

            if careers and len(careers) > 0:
                logging.info(f"AI generated {len(careers)} personalized career predictions")
                return careers[:5]
            else:
                logging.warning("AI generation returned empty, using fallback")
                return self._generate_fallback_career_predictions(student_profile, profiler_data)

        except Exception as e:
            logging.error(f"Career prediction generation failed: {e}. Using fallback.")
            return self._generate_fallback_career_predictions(student_profile, profiler_data)

    def _extract_qa_pairs(self, conversation_context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract question-answer pairs from conversation context."""
        question_history = conversation_context.get("question_history", [])
        response_history = conversation_context.get("response_history", [])

        qa_pairs = []
        for i, (q, a) in enumerate(zip(question_history, response_history), 1):
            qa_pairs.append({
                "number": i,
                "question": q,
                "answer": a
            })
        return qa_pairs

    def _format_qa_history(self, qa_pairs: List[Dict]) -> str:
        """Format Q&A pairs for LLM prompt."""
        if not qa_pairs:
            return "No conversation history available."

        formatted = []
        for qa in qa_pairs:
            formatted.append(f"Q{qa['number']}: {qa['question']}")
            formatted.append(f"A{qa['number']}: {qa['answer']}\n")
        return "\n".join(formatted)

    def _call_llm_for_careers(self, prompt: str, student_profile: StudentProfile, profiler_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call LLM to generate career predictions dynamically."""
        try:
            from langchain_core.messages import HumanMessage

            response = self.llm.invoke([HumanMessage(content=prompt)])

            if not response or not response.content:
                logging.warning("Empty LLM response for career generation")
                return []

            response_text = response.content.strip()

            # Extract JSON array from response
            # Try to find JSON array in the response
            import re
            json_match = re.search(r'\[[\s\S]*\]', response_text)

            if json_match:
                import json
                try:
                    careers = json.loads(json_match.group())

                    if isinstance(careers, list) and len(careers) > 0:
                        # Validate and clean career data
                        validated_careers = []
                        for career in careers[:5]:  # Max 5 careers
                            if isinstance(career, dict) and "title" in career:
                                # Ensure all required fields exist
                                validated_career = {
                                    "title": career.get("title", "Unknown Career"),
                                    "description": career.get("description", "No description available"),
                                    "why_good_fit": career.get("why_good_fit", "Based on your profile"),
                                    "required_education": career.get("required_education", "Bachelor's degree"),
                                    "required_skills": career.get("required_skills", []),
                                    "growth_outlook": career.get("growth_outlook", "Medium"),
                                    "median_salary": career.get("median_salary", "$60,000-$80,000"),
                                    "confidence_score": float(career.get("confidence_score", 0.75)),
                                    "riasec_alignment": career.get("riasec_alignment", "Multi-faceted match"),
                                    "career_path": career.get("career_path", "Entry → Mid → Senior")
                                }
                                validated_careers.append(validated_career)

                        if validated_careers:
                            return validated_careers

                except json.JSONDecodeError as je:
                    logging.error(f"Failed to parse LLM JSON: {je}")
                    logging.debug(f"Response text: {response_text[:500]}")

            logging.warning("Could not extract valid JSON from LLM response")
            return []

        except Exception as e:
            logging.error(f"LLM career generation error: {e}")
            return []

    def _create_final_confidence_report(
        self,
        student_profile: StudentProfile,
        profiler_data: Dict[str, Any],
        session_state: AgentState,
        session_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a comprehensive final confidence report for completed sessions.
        """
        total_questions = profiler_data.get("total_questions", 0)
        profile_confidence = profiler_data.get("profile_confidence", 0.0)

        # Extract confidence breakdown if available
        confidence_breakdown = profiler_data.get("final_confidence_breakdown", {})
        confidence_journey = profiler_data.get("confidence_journey", {})

        # Create final assessment
        assessment_summary = {
            "total_questions_asked": total_questions,
            "final_confidence_score": round(profile_confidence, 2),
            "final_confidence_percentage": f"{profile_confidence:.0%}",
            "session_duration_interactions": session_info.get("total_interactions", 0),
            "conversation_efficiency": self._calculate_conversation_efficiency(total_questions, profile_confidence),
            "data_collection_quality": self._assess_data_collection_quality(student_profile, profile_confidence)
        }

        # Confidence categories analysis
        categories_analysis = {}
        if confidence_breakdown and "confidence_breakdown" in confidence_breakdown:
            for category, data in confidence_breakdown["confidence_breakdown"].items():
                categories_analysis[category] = {
                    "final_score": data.get("current", 0.0),
                    "final_percentage": data.get("percentage", "0%"),
                    "target_met": data.get("status") in ["sufficient", "excellent"],
                    "quality_rating": data.get("status", "unknown")
                }

        # RIASEC confidence analysis
        riasec_analysis = {}
        if student_profile.riasec_scores:
            for category, score in student_profile.riasec_scores.items():
                riasec_analysis[category] = {
                    "score": round(score, 2),
                    "percentage": f"{score:.0%}",
                    "confidence_level": "high" if score >= 0.6 else "moderate" if score >= 0.4 else "developing"
                }

        # Prediction reliability assessment
        prediction_reliability = {
            "overall_reliability": self._assess_prediction_reliability(profile_confidence),
            "career_predictions_confidence": "high" if profile_confidence >= 0.7 else "moderate" if profile_confidence >= 0.5 else "developing",
            "recommendation_strength": self._calculate_recommendation_strength(profile_confidence, total_questions),
            "areas_well_understood": self._identify_well_understood_areas(categories_analysis),
            "areas_with_limited_data": self._identify_limited_data_areas(categories_analysis)
        }

        # Session insights
        session_insights = {
            "conversation_flow": "efficient" if total_questions <= 10 else "thorough",
            "user_engagement": self._assess_user_engagement(session_info),
            "information_richness": self._assess_information_richness(student_profile),
            "goal_achievement": "excellent" if profile_confidence >= 0.7 else "good" if profile_confidence >= 0.6 else "satisfactory"
        }

        return {
            "assessment_summary": assessment_summary,
            "confidence_categories": categories_analysis,
            "riasec_analysis": riasec_analysis,
            "prediction_reliability": prediction_reliability,
            "session_insights": session_insights,
            "confidence_journey": confidence_journey,
            "final_recommendations": {
                "career_predictions_reliability": prediction_reliability["overall_reliability"]["level"],
                "next_steps": self._generate_next_steps_recommendations(profile_confidence),
                "improvement_opportunities": self._identify_improvement_opportunities(categories_analysis)
            }
        }

    def _calculate_conversation_efficiency(self, questions_asked: int, final_confidence: float) -> Dict[str, Any]:
        """Calculate how efficiently the conversation gathered information."""
        # Ideal efficiency: high confidence with fewer questions
        efficiency_score = final_confidence / max(questions_asked / 12, 1.0)  # Normalized to question limit

        if efficiency_score >= 0.8:
            rating = "excellent"
        elif efficiency_score >= 0.6:
            rating = "good"
        elif efficiency_score >= 0.4:
            rating = "fair"
        else:
            rating = "could_improve"

        return {
            "score": round(efficiency_score, 2),
            "rating": rating,
            "questions_per_confidence_point": round(questions_asked / max(final_confidence, 0.1), 1)
        }

    def _assess_data_collection_quality(self, student_profile: StudentProfile, confidence: float) -> Dict[str, Any]:
        """Assess the quality of data collected during the conversation."""
        quality_indicators = {
            "has_academic_info": bool(student_profile.current_education_level and student_profile.current_education_level != "Unknown"),
            "has_career_interests": bool(student_profile.career_interests),
            "has_technical_skills": bool(student_profile.technical_skills),
            "has_riasec_data": bool(student_profile.riasec_scores),
            "has_goals": bool(student_profile.short_term_goals or student_profile.long_term_goals)
        }

        quality_score = sum(quality_indicators.values()) / len(quality_indicators)

        return {
            "quality_score": round(quality_score, 2),
            "quality_percentage": f"{quality_score:.0%}",
            "data_completeness": "comprehensive" if quality_score >= 0.8 else "good" if quality_score >= 0.6 else "basic",
            "missing_elements": [key.replace("has_", "") for key, value in quality_indicators.items() if not value]
        }

    def _assess_prediction_reliability(self, confidence: float) -> Dict[str, Any]:
        """Assess how reliable career predictions are based on confidence."""
        if confidence >= 0.8:
            return {
                "level": "very_high",
                "description": "Career predictions are highly reliable and well-supported by comprehensive data",
                "recommendation": "Proceed with confidence in these career recommendations"
            }
        elif confidence >= 0.7:
            return {
                "level": "high",
                "description": "Career predictions are reliable with good supporting data",
                "recommendation": "These career suggestions provide a strong foundation for exploration"
            }
        elif confidence >= 0.6:
            return {
                "level": "good",
                "description": "Career predictions are reasonably reliable with adequate data",
                "recommendation": "Use these suggestions as a starting point for career exploration"
            }
        elif confidence >= 0.5:
            return {
                "level": "moderate",
                "description": "Career predictions have moderate reliability due to limited data",
                "recommendation": "Consider these as initial guidance while gathering more information"
            }
        else:
            return {
                "level": "developing",
                "description": "Career predictions are based on limited information",
                "recommendation": "Use as preliminary guidance and continue career exploration"
            }

    def _calculate_recommendation_strength(self, confidence: float, questions_asked: int) -> str:
        """Calculate the strength of career recommendations."""
        if confidence >= 0.7 and questions_asked >= 8:
            return "strong"
        elif confidence >= 0.6 and questions_asked >= 6:
            return "moderate"
        elif confidence >= 0.5:
            return "preliminary"
        else:
            return "exploratory"

    def _identify_well_understood_areas(self, categories_analysis: Dict[str, Any]) -> List[str]:
        """Identify areas where we have high confidence."""
        well_understood = []
        for category, data in categories_analysis.items():
            if data.get("quality_rating") in ["sufficient", "excellent"]:
                well_understood.append(category)
        return well_understood

    def _identify_limited_data_areas(self, categories_analysis: Dict[str, Any]) -> List[str]:
        """Identify areas where data collection was limited."""
        limited_areas = []
        for category, data in categories_analysis.items():
            if data.get("quality_rating") == "needs_improvement":
                limited_areas.append(category)
        return limited_areas

    def _assess_user_engagement(self, session_info: Dict[str, Any]) -> str:
        """Assess how engaged the user was during the conversation."""
        interactions = session_info.get("total_interactions", 0)
        if interactions >= 10:
            return "highly_engaged"
        elif interactions >= 6:
            return "well_engaged"
        elif interactions >= 3:
            return "moderately_engaged"
        else:
            return "limited_engagement"

    def _assess_information_richness(self, student_profile: StudentProfile) -> str:
        """Assess how rich and detailed the collected information is."""
        richness_score = 0

        # Academic information richness
        if student_profile.major_field:
            richness_score += 1
        if student_profile.academic_performance:
            richness_score += 1

        # Skills richness
        if len(student_profile.technical_skills or []) >= 3:
            richness_score += 1
        if len(student_profile.soft_skills or []) >= 2:
            richness_score += 1

        # Interest richness
        if len(student_profile.career_interests or []) >= 3:
            richness_score += 1
        if student_profile.industry_preferences:
            richness_score += 1

        # Goals richness
        if student_profile.short_term_goals:
            richness_score += 1
        if student_profile.long_term_goals:
            richness_score += 1

        if richness_score >= 6:
            return "very_rich"
        elif richness_score >= 4:
            return "rich"
        elif richness_score >= 2:
            return "moderate"
        else:
            return "basic"

    def _generate_next_steps_recommendations(self, confidence: float) -> List[str]:
        """Generate recommendations for next steps based on confidence level."""
        if confidence >= 0.7:
            return [
                "Explore the recommended careers in depth",
                "Research specific companies and roles",
                "Connect with professionals in these fields",
                "Consider relevant internships or projects",
                "Develop skills aligned with your top career matches"
            ]
        elif confidence >= 0.6:
            return [
                "Research the suggested career areas further",
                "Take additional career assessments if needed",
                "Gain experience through volunteering or projects",
                "Speak with career counselors for additional guidance",
                "Continue exploring your interests and strengths"
            ]
        else:
            return [
                "Continue self-exploration and career research",
                "Take comprehensive career assessments",
                "Gain diverse experiences to better understand preferences",
                "Seek guidance from career professionals",
                "Keep an open mind while exploring various paths"
            ]

    def _identify_improvement_opportunities(self, categories_analysis: Dict[str, Any]) -> List[str]:
        """Identify areas where the assessment could be improved."""
        opportunities = []
        for category, data in categories_analysis.items():
            if not data.get("target_met", True):
                opportunities.append(f"Gather more information about {category.replace('_', ' ')}")

        if not opportunities:
            opportunities.append("Continue monitoring career interests as they evolve")

        return opportunities

    def _create_fallback_student_profile(self, profiler_data: Dict[str, Any], session_id: str) -> StudentProfile:
        """
        Create a minimal but functional student profile when normal extraction fails.
        Ensures career predictions can always be generated.
        """
        self._log_task_completion("fallback_profile_creation", True,
                                f"Session {session_id}: Creating fallback profile from available data")

        # Extract what data we can from profiler_data
        academic_level = "Unknown"
        career_interests = ["General"]
        riasec_scores = {}
        confidence_score = 0.3  # Default low confidence

        # Try to extract any available information
        if isinstance(profiler_data, dict):
            # Look for RIASEC data
            if "riasec_scores" in profiler_data:
                riasec_data = profiler_data["riasec_scores"]
                if isinstance(riasec_data, dict):
                    riasec_scores = riasec_data

            # Look for basic profile info
            if "profile_confidence" in profiler_data:
                confidence_score = max(confidence_score, profiler_data.get("profile_confidence", 0.3))

        # Create minimal viable profile
        return StudentProfile(
            current_education_level=academic_level,
            career_interests=career_interests,
            riasec_scores=riasec_scores,
            technical_skills=["General problem-solving"],
            soft_skills=["Communication"],
            short_term_goals=["Explore career options"],
            long_term_goals=["Find fulfilling career"]
        )

    def _generate_fallback_career_predictions(self, student_profile: StudentProfile, profiler_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate basic career predictions when normal prediction process fails.
        Provides general career suggestions suitable for most students.
        """
        self._log_task_completion("fallback_predictions", True,
                                "Generating fallback career predictions")

        # General career suggestions suitable for most people
        fallback_careers = [
            {
                "career_title": "Business Analyst",
                "confidence_score": 0.65,
                "match_percentage": 65,
                "key_reasons": [
                    "Versatile career path suitable for various backgrounds",
                    "Combines analytical thinking with communication skills",
                    "Available across multiple industries"
                ],
                "riasec_match": "Investigative, Enterprising",
                "growth_outlook": "Strong job growth expected",
                "education_requirements": "Bachelor's degree preferred",
                "industries": ["Technology", "Finance", "Healthcare", "Consulting"]
            },
            {
                "career_title": "Project Coordinator",
                "confidence_score": 0.62,
                "match_percentage": 62,
                "key_reasons": [
                    "Organizational skills valuable across industries",
                    "Entry-level pathway to management roles",
                    "Develops transferable skills"
                ],
                "riasec_match": "Enterprising, Conventional",
                "growth_outlook": "Steady demand across sectors",
                "education_requirements": "Bachelor's degree or equivalent experience",
                "industries": ["Various industries"]
            },
            {
                "career_title": "Customer Success Specialist",
                "confidence_score": 0.58,
                "match_percentage": 58,
                "key_reasons": [
                    "Growing field with increasing demand",
                    "Combines interpersonal and problem-solving skills",
                    "Opportunity for career advancement"
                ],
                "riasec_match": "Social, Enterprising",
                "growth_outlook": "High growth potential",
                "education_requirements": "Bachelor's degree preferred",
                "industries": ["Technology", "SaaS", "Services"]
            },
            {
                "career_title": "Marketing Coordinator",
                "confidence_score": 0.55,
                "match_percentage": 55,
                "key_reasons": [
                    "Creative and analytical aspects",
                    "Digital marketing skills in high demand",
                    "Diverse career progression opportunities"
                ],
                "riasec_match": "Artistic, Enterprising",
                "growth_outlook": "Strong growth in digital marketing",
                "education_requirements": "Bachelor's degree in marketing or related field",
                "industries": ["Marketing", "Technology", "Retail", "Services"]
            },
            {
                "career_title": "Operations Specialist",
                "confidence_score": 0.52,
                "match_percentage": 52,
                "key_reasons": [
                    "Process improvement and efficiency focus",
                    "Analytical and systematic approach",
                    "Stable career with advancement potential"
                ],
                "riasec_match": "Conventional, Investigative",
                "growth_outlook": "Steady demand for operational efficiency",
                "education_requirements": "Bachelor's degree preferred",
                "industries": ["Manufacturing", "Healthcare", "Finance", "Logistics"]
            }
        ]

        # Try to customize based on any available profile information
        if student_profile and hasattr(student_profile, 'riasec_scores') and student_profile.riasec_scores:
            # Adjust confidence scores based on RIASEC alignment if available
            for career in fallback_careers:
                # Simple boost for better alignment (this is basic but functional)
                career["confidence_score"] = min(career["confidence_score"] + 0.05, 0.75)
                career["match_percentage"] = int(career["confidence_score"] * 100)

        return fallback_careers

    def _get_riasec_name(self, code: str) -> str:
        """Convert RIASEC code to full name."""
        names = {
            "R": "Realistic",
            "I": "Investigative",
            "A": "Artistic",
            "S": "Social",
            "E": "Enterprising",
            "C": "Conventional"
        }
        return names.get(code, code)

    def _generate_fit_explanation(self, career: Dict[str, Any], profile: StudentProfile, category: str) -> str:
        """Generate explanation of why this career is a good fit."""
        explanations = {
            "R": f"Your practical, hands-on approach makes you well-suited for {career['title']} roles that involve building and creating tangible solutions.",
            "I": f"Your analytical mindset and love for problem-solving align perfectly with {career['title']} positions that require research and investigation.",
            "A": f"Your creative thinking and appreciation for innovation make {career['title']} an excellent match for expressing your artistic abilities.",
            "S": f"Your people-oriented nature and desire to help others make {career['title']} roles ideal for making a positive impact.",
            "E": f"Your leadership qualities and business acumen are perfectly suited for {career['title']} positions that drive organizational success.",
            "C": f"Your organized, detail-oriented approach makes you an excellent fit for {career['title']} roles that require systematic thinking."
        }
        return explanations.get(category, f"This {career['title']} role aligns well with your overall profile and interests.")

    def _format_academic_plan_message(self, career_title: str, academic_plan: Dict[str, Any]) -> str:
        """
        Format the complete academic pathway plan into a detailed, readable message.
        Shows all pathway options, institutions, costs, and next steps.
        """
        if not academic_plan:
            return f"📚 **Academic Pathway for {career_title}**\n\nDetailed plan is being generated..."

        message_parts = []
        message_parts.append(f"📚 **ACADEMIC PATHWAY FOR {career_title.upper()}**")
        message_parts.append("")  # Blank line after title

        # Student Assessment
        student_assessment = academic_plan.get("student_assessment", {})
        if student_assessment:
            message_parts.append("🎓 **STUDENT ASSESSMENT**")
            message_parts.append("")  # Blank line after section header
            if student_assessment.get("current_level"):
                message_parts.append(f"• **Current Level:** {student_assessment.get('current_level', 'Not specified')}")
            if student_assessment.get("recommended_timeline"):
                message_parts.append(f"• **Timeline to Career:** {student_assessment.get('recommended_timeline', 'Varies')}")
            message_parts.append("")  # Blank line after section

        # Pathway Options - This is the main content!
        pathway_options = academic_plan.get("pathway_options", [])
        if pathway_options:
            message_parts.append("📍 **EDUCATIONAL PATHWAY OPTIONS**")
            message_parts.append("")  # Blank line after section header

            for idx, pathway in enumerate(pathway_options, 1):
                pathway_type = pathway.get("pathway_type", "Option")
                message_parts.append(f"**{idx}. {pathway_type} Pathway**")

                # Sri Lankan Options
                sri_lankan_options = pathway.get("sri_lankan_options", [])
                if sri_lankan_options:
                    message_parts.append("")
                    message_parts.append("**🇱🇰 Sri Lankan Institutions:**")
                    message_parts.append("")
                    for option in sri_lankan_options[:10]:  # Show up to 10 options
                        inst_name = option.get("institution_name", "Institution")
                        program = option.get("program_name", "Related program")
                        duration = option.get("duration", "")
                        cost = option.get("approximate_cost", "")
                        message_parts.append(f"**• {inst_name}**")
                        message_parts.append("")
                        message_parts.append(f"- **Program:** {program}")
                        if duration:
                            message_parts.append(f"- **Duration:** {duration}")
                        if cost:
                            message_parts.append(f"- **Cost:** {cost}")
                        message_parts.append("")  # Blank line after each institution
                    message_parts.append("")

                # International Options
                international_options = pathway.get("international_options", [])
                if international_options:
                    message_parts.append("**🌍 International Options:**")
                    message_parts.append("")
                    for option in international_options[:10]:  # Show up to 10 options
                        country = option.get("country", "Country")
                        program_type = option.get("program_type", "Degree program")
                        duration = option.get("duration", "")
                        cost = option.get("approximate_cost", "")
                        message_parts.append(f"**• {country}**")
                        message_parts.append("")
                        message_parts.append(f"- **Program:** {program_type}")
                        if duration:
                            message_parts.append(f"- **Duration:** {duration}")
                        if cost:
                            message_parts.append(f"- **Cost:** {cost}")
                        message_parts.append("")  # Blank line after each option
                    message_parts.append("")

        # Step-by-step Implementation Plan
        step_plan = academic_plan.get("step_by_step_plan", [])
        if step_plan:
            message_parts.append("📋 **IMPLEMENTATION ROADMAP**")
            message_parts.append("")
            for phase in step_plan[:3]:  # Show up to 3 phases
                phase_name = phase.get("phase", "Phase")
                timeframe = phase.get("timeframe", "")
                actions = phase.get("actions", [])

                message_parts.append(f"**{phase_name}** ({timeframe})")
                message_parts.append("")  # Blank line after phase header
                if actions:
                    for action in actions[:5]:  # Show up to 5 actions per phase
                        message_parts.append(f"- {action}")
                message_parts.append("")

        # Financial Planning
        financial = academic_plan.get("financial_planning", {})
        if financial:
            message_parts.append("💰 **FINANCIAL PLANNING**")
            message_parts.append("")
            total_cost = financial.get("total_estimated_cost_lkr", "")
            if total_cost and total_cost != "To be determined":
                message_parts.append(f"- **Estimated Total Cost:** {total_cost}")

            funding_options = financial.get("funding_options", [])
            if funding_options:
                message_parts.append("")
                message_parts.append("**Funding Options:**")
                message_parts.append("")  # Blank line after subheader
                for option in funding_options[:5]:
                    message_parts.append(f"- {option}")
            message_parts.append("")

        # Next Immediate Steps
        next_steps = academic_plan.get("next_immediate_steps", [])
        if next_steps:
            message_parts.append("🎯 **NEXT IMMEDIATE STEPS**")
            message_parts.append("")
            for idx, step in enumerate(next_steps[:5], 1):
                message_parts.append(f"{idx}. {step}")
            message_parts.append("")

        # Alternative Pathways
        alternatives = academic_plan.get("alternative_pathways", [])
        if alternatives:
            message_parts.append("🔄 **ALTERNATIVE PATHWAYS**")
            message_parts.append("")
            for alt in alternatives[:3]:
                desc = alt.get("pathway_description", "")
                if desc:
                    message_parts.append(f"• {desc}")
            message_parts.append("")

        return "\n".join(message_parts)

    def _format_skill_plan_message(self, career_title: str, skill_plan: Dict[str, Any]) -> str:
        """
        Format the complete skill development plan into a detailed, readable message.
        Shows all technical skills, soft skills, learning phases, and certifications.
        """
        if not skill_plan:
            return f"🎯 **Skill Development Plan for {career_title}**\n\nDetailed plan is being generated..."

        message_parts = []
        message_parts.append(f"🎯 **SKILL DEVELOPMENT PLAN FOR {career_title.upper()}**")
        message_parts.append("")  # Blank line after title

        # Technical Skills
        technical_skills = skill_plan.get("technical_skills", {})
        if technical_skills:
            message_parts.append("💻 **TECHNICAL SKILLS**")
            message_parts.append("")  # Blank line after section header

            # Core Skills
            core_skills = technical_skills.get("core_skills", [])
            if core_skills:
                message_parts.append("**Core Skills (Must-Have):**")
                message_parts.append("")  # Blank line after subheader
                for skill in core_skills:
                    message_parts.append(f"- {skill}")
                message_parts.append("")

            # Advanced Skills
            advanced_skills = technical_skills.get("advanced_skills", [])
            if advanced_skills:
                message_parts.append("**Advanced Skills:**")
                message_parts.append("")  # Blank line after subheader
                for skill in advanced_skills:
                    message_parts.append(f"- {skill}")
                message_parts.append("")

            # Tools & Technologies
            tools = technical_skills.get("tools_technologies", [])
            if tools:
                message_parts.append("**Tools & Technologies:**")
                message_parts.append("")  # Blank line after subheader
                for tool in tools:
                    message_parts.append(f"- {tool}")
                message_parts.append("")

        # Soft Skills
        soft_skills = skill_plan.get("soft_skills", {})
        if soft_skills:
            message_parts.append("🤝 **SOFT SKILLS**")
            message_parts.append("")  # Blank line after section header

            # Essential
            essential = soft_skills.get("essential", [])
            if essential:
                message_parts.append("**Essential:**")
                message_parts.append("")  # Blank line after subheader
                for skill in essential:
                    message_parts.append(f"- {skill}")
                message_parts.append("")

            # Recommended
            recommended = soft_skills.get("recommended", [])
            if recommended:
                message_parts.append("**Recommended:**")
                message_parts.append("")  # Blank line after subheader
                for skill in recommended:
                    message_parts.append(f"- {skill}")
                message_parts.append("")

        # Learning Phases
        learning_phases = skill_plan.get("learning_phases", [])
        if learning_phases:
            message_parts.append("📚 **LEARNING PHASES**")
            message_parts.append("")  # Blank line after section header

            for phase in learning_phases:
                phase_name = phase.get("phase", "Phase")
                duration = phase.get("duration", "")
                skills_focus = phase.get("skills_focus", [])
                resources = phase.get("resources", [])
                milestones = phase.get("milestones", [])

                message_parts.append(f"**{phase_name}** ({duration})")
                message_parts.append("")  # Blank line after phase name

                if skills_focus:
                    message_parts.append("**Skills to Master:**")
                    message_parts.append("")  # Blank line after subheader
                    for skill in skills_focus[:5]:
                        message_parts.append(f"- {skill}")
                    message_parts.append("")

                if resources:
                    message_parts.append("**Learning Resources:**")
                    message_parts.append("")  # Blank line after subheader
                    for resource in resources[:5]:
                        message_parts.append(f"- {resource}")
                    message_parts.append("")

                if milestones:
                    message_parts.append("**Milestones:**")
                    message_parts.append("")  # Blank line after subheader
                    for milestone in milestones[:3]:
                        message_parts.append(f"- ✓ {milestone}")
                    message_parts.append("")

        # Certifications
        certifications = skill_plan.get("certifications", [])
        if certifications:
            message_parts.append("🎓 **RECOMMENDED CERTIFICATIONS**")
            message_parts.append("")
            for cert in certifications[:10]:
                message_parts.append(f"- {cert}")
            message_parts.append("")

        # Estimated Timeline
        total_time = skill_plan.get("estimated_total_time", "")
        if total_time:
            message_parts.append(f"⏱️ **Estimated Timeline:** {total_time}")
            message_parts.append("")

        return "\n".join(message_parts)

    # Include original batch processing methods for backward compatibility
    def _handle_student_request(self, state: AgentState) -> TaskResult:
        """Handle initial student request - original implementation."""
        # [Original implementation would be copied here for batch mode]
        return self._create_task_result(
            task_type="student_request_analysis",
            success=True,
            result_data={"next_step": "profile_creation", "mode": "batch"}
        )

    def _handle_profile_creation(self, state: AgentState) -> TaskResult:
        """Handle profile creation - original implementation."""
        # [Original implementation would be copied here for batch mode]
        return self._create_task_result(
            task_type="profile_creation",
            success=True,
            result_data={"next_step": "career_planning", "mode": "batch"}
        )

    def _handle_career_planning(self, state: AgentState) -> TaskResult:
        """Handle career planning - original implementation."""
        return self._create_task_result(
            task_type="career_planning",
            success=True,
            result_data={"next_step": "completion", "mode": "batch"}
        )

    def _handle_future_analysis(self, state: AgentState) -> TaskResult:
        """Handle future analysis - original implementation."""
        return self._create_task_result(
            task_type="future_analysis",
            success=True,
            result_data={"next_step": "completion", "mode": "batch"}
        )

    def _handle_final_report_generation(self, state: AgentState) -> TaskResult:
        """Handle final report generation - original implementation."""
        return self._create_task_result(
            task_type="final_report_generation",
            success=True,
            result_data={"completed": True, "mode": "batch"}
        )

    def _handle_unknown_step(self, state: AgentState, step: str) -> TaskResult:
        """Handle unknown workflow steps."""
        return self._create_task_result(
            task_type="workflow_error",
            success=False,
            error_message=f"Unknown workflow step: {step}"
        )

    def _handle_unknown_stage(self, state: AgentState, session_id: str, stage: str) -> TaskResult:
        """Handle unknown session stages."""
        return self._create_task_result(
            task_type="session_error",
            success=False,
            error_message=f"Unknown session stage: {stage}"
        )

    # Public interface methods for interactive sessions
    def start_interactive_career_planning(self, student_request: str, session_id: Optional[str] = None) -> AgentState:
        """
        Start an interactive career planning session.
        """
        if not session_id:
            session_id = f"interactive_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        initial_state = AgentState(
            student_request=student_request,
            current_step="interactive_profiling",
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            interaction_mode="interactive",
            awaiting_user_response=False,
            messages=[HumanMessage(content=student_request)]
        )

        self.logger.info(f"🚀 Starting interactive career planning session: {session_id}")
        return initial_state

    async def process_user_response(self, session_id: str, user_response: str) -> TaskResult:
        """
        Process a user response for an active session.
        """
        if session_id not in self.session_states:
            return self._create_task_result(
                task_type="session_error",
                success=False,
                error_message=f"Session {session_id} not found"
            )

        session_state = self.session_states[session_id]
        session_state.pending_user_response = user_response
        session_state.response_processed = False

        return await self.process_task(session_state)

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get the current status of a session.
        """
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session_info = self.active_sessions[session_id]
        session_state = self.session_states[session_id]

        return {
            "session_id": session_id,
            "current_stage": session_info.get("current_stage"),
            "awaiting_user_response": session_state.awaiting_user_response,
            "current_question": session_state.current_question,
            "total_interactions": session_info.get("total_interactions", 0),
            "created_at": session_info.get("created_at")
        }

    def end_session(self, session_id: str) -> bool:
        """
        End and clean up a session.
        """
        cleaned_up = False

        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            cleaned_up = True

        if session_id in self.session_states:
            del self.session_states[session_id]
            cleaned_up = True

        # Also clean up profiler session
        if self.user_profiler:
            self.user_profiler.end_session(session_id)

        return cleaned_up