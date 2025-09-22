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

    def process_task(self, state: AgentState) -> TaskResult:
        """
        Enhanced task processing with session management support.
        """
        start_time = datetime.now()
        session_id = state.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._log_task_start("enhanced_workflow", f"session: {session_id}, mode: {state.interaction_mode}")

        try:
            # Determine processing mode
            if state.interaction_mode == "interactive":
                return self._handle_interactive_workflow(state, session_id)
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

    def _handle_interactive_workflow(self, state: AgentState, session_id: str) -> TaskResult:
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

            return self._create_task_result(
                task_type="session_initialization",
                success=True,
                result_data={
                    "session_id": session_id,
                    "mode": "interactive",
                    "stage": "profiling",
                    "question": result_data.get("question"),
                    "conversation_state": result_data.get("conversation_state"),
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
                    session_state.current_question = result_data.get("question")

                    return self._create_task_result(
                        task_type="profiling_interaction",
                        success=True,
                        result_data={
                            "session_id": session_id,
                            "question": result_data.get("question"),
                            "conversation_state": result_data.get("conversation_state"),
                            "awaiting_user_response": True,
                            "progress": result_data.get("progress", {}),
                            "confidence_scores": result_data.get("confidence_scores", {}),
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

        # Extract student profile
        if "student_profile" in profiler_data:
            session_state.student_profile = StudentProfile(**profiler_data["student_profile"])
        elif "profile" in profiler_data:
            session_state.student_profile = StudentProfile(**profiler_data["profile"])

        # Generate 5 career predictions
        career_predictions = self._generate_career_predictions(session_state.student_profile, profiler_data)

        # Clean up profiler session
        self.user_profiler.end_session(session_id)

        self._log_task_completion("profiling_stage", True, f"Session {session_id} profiling completed with career predictions")

        return self._create_task_result(
            task_type="career_prediction_completion",
            success=True,
            result_data={
                "session_id": session_id,
                "stage": "career_prediction_complete",
                "profile_created": True,
                "profile_confidence": profiler_data.get("profile_confidence", 0.0),
                "dominant_riasec": profiler_data.get("dominant_riasec"),
                "total_questions": profiler_data.get("total_questions", 0),
                "career_predictions": career_predictions,
                "completed": True,
                "message": f"Perfect! Based on our conversation, I've identified some great career matches for you. Here are my recommendations:"
            }
        )

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
        Generate 5 career predictions based on student profile and RIASEC scores.
        """
        # Career database mapped to RIASEC categories
        career_database = {
            "R": [  # Realistic
                {"title": "Software Engineer", "description": "Design and develop software applications", "growth_outlook": "High", "median_salary": "$95,000"},
                {"title": "Mechanical Engineer", "description": "Design and build mechanical systems", "growth_outlook": "Medium", "median_salary": "$88,000"},
                {"title": "Civil Engineer", "description": "Design infrastructure and construction projects", "growth_outlook": "Medium", "median_salary": "$85,000"},
                {"title": "Electrical Engineer", "description": "Design electrical systems and components", "growth_outlook": "Medium", "median_salary": "$92,000"},
                {"title": "Industrial Designer", "description": "Create functional and aesthetic product designs", "growth_outlook": "Medium", "median_salary": "$68,000"}
            ],
            "I": [  # Investigative
                {"title": "Data Scientist", "description": "Analyze complex data to derive insights", "growth_outlook": "Very High", "median_salary": "$110,000"},
                {"title": "Research Scientist", "description": "Conduct scientific research and experiments", "growth_outlook": "Medium", "median_salary": "$85,000"},
                {"title": "Cybersecurity Analyst", "description": "Protect systems from security threats", "growth_outlook": "Very High", "median_salary": "$98,000"},
                {"title": "Biomedical Engineer", "description": "Apply engineering to medical and biological problems", "growth_outlook": "High", "median_salary": "$95,000"},
                {"title": "Market Research Analyst", "description": "Study market conditions and consumer behavior", "growth_outlook": "High", "median_salary": "$65,000"}
            ],
            "A": [  # Artistic
                {"title": "UX/UI Designer", "description": "Design user interfaces and experiences", "growth_outlook": "High", "median_salary": "$75,000"},
                {"title": "Creative Director", "description": "Lead creative teams and campaigns", "growth_outlook": "Medium", "median_salary": "$95,000"},
                {"title": "Game Developer", "description": "Create video games and interactive media", "growth_outlook": "High", "median_salary": "$72,000"},
                {"title": "Digital Marketing Specialist", "description": "Create and manage digital marketing campaigns", "growth_outlook": "High", "median_salary": "$58,000"},
                {"title": "Graphic Designer", "description": "Create visual concepts and designs", "growth_outlook": "Medium", "median_salary": "$52,000"}
            ],
            "S": [  # Social
                {"title": "Product Manager", "description": "Lead product development and strategy", "growth_outlook": "High", "median_salary": "$115,000"},
                {"title": "Human Resources Manager", "description": "Manage employee relations and policies", "growth_outlook": "Medium", "median_salary": "$78,000"},
                {"title": "Training and Development Specialist", "description": "Design and implement training programs", "growth_outlook": "High", "median_salary": "$62,000"},
                {"title": "Social Media Manager", "description": "Manage social media presence and engagement", "growth_outlook": "High", "median_salary": "$55,000"},
                {"title": "Customer Success Manager", "description": "Ensure customer satisfaction and retention", "growth_outlook": "High", "median_salary": "$72,000"}
            ],
            "E": [  # Enterprising
                {"title": "Business Analyst", "description": "Analyze business processes and requirements", "growth_outlook": "High", "median_salary": "$82,000"},
                {"title": "Sales Manager", "description": "Lead sales teams and strategies", "growth_outlook": "Medium", "median_salary": "$85,000"},
                {"title": "Project Manager", "description": "Plan and execute project deliverables", "growth_outlook": "High", "median_salary": "$88,000"},
                {"title": "Startup Founder", "description": "Create and lead new business ventures", "growth_outlook": "Variable", "median_salary": "Variable"},
                {"title": "Business Development Manager", "description": "Identify and develop new business opportunities", "growth_outlook": "Medium", "median_salary": "$78,000"}
            ],
            "C": [  # Conventional
                {"title": "Financial Analyst", "description": "Analyze financial data and investment opportunities", "growth_outlook": "Medium", "median_salary": "$78,000"},
                {"title": "Operations Manager", "description": "Oversee daily business operations", "growth_outlook": "Medium", "median_salary": "$82,000"},
                {"title": "Quality Assurance Specialist", "description": "Ensure products meet quality standards", "growth_outlook": "Medium", "median_salary": "$65,000"},
                {"title": "Database Administrator", "description": "Manage and maintain database systems", "growth_outlook": "Medium", "median_salary": "$92,000"},
                {"title": "Systems Analyst", "description": "Analyze and improve IT systems", "growth_outlook": "Medium", "median_salary": "$88,000"}
            ]
        }

        # Get RIASEC scores from profile
        riasec_scores = student_profile.riasec_scores or {}

        # Convert string keys to proper format if needed
        normalized_scores = {}
        for key, value in riasec_scores.items():
            if isinstance(key, str) and len(key) == 1:
                normalized_scores[key.upper()] = value
            elif isinstance(key, str) and len(key) > 1:
                # Handle full names like "Realistic" -> "R"
                first_letter = key[0].upper()
                if first_letter in ["R", "I", "A", "S", "E", "C"]:
                    normalized_scores[first_letter] = value

        # Sort RIASEC categories by score (highest first)
        sorted_categories = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

        # Generate career predictions
        predictions = []
        used_careers = set()

        # Get careers from top RIASEC categories
        for category, score in sorted_categories:
            if len(predictions) >= 5:
                break

            category_careers = career_database.get(category, [])

            for career in category_careers:
                if len(predictions) >= 5:
                    break

                if career["title"] not in used_careers:
                    # Calculate match confidence based on RIASEC score and other factors
                    base_confidence = score * 0.7  # RIASEC score contributes 70%

                    # Add bonus for relevant skills or interests
                    skill_bonus = 0.0
                    if student_profile.technical_skills:
                        if any(skill.lower() in career["description"].lower() for skill in student_profile.technical_skills):
                            skill_bonus = 0.1

                    interest_bonus = 0.0
                    if student_profile.career_interests:
                        if any(interest.lower() in career["title"].lower() or interest.lower() in career["description"].lower()
                               for interest in student_profile.career_interests):
                            interest_bonus = 0.1

                    confidence = min(base_confidence + skill_bonus + interest_bonus, 0.95)

                    # Create prediction with reasoning
                    reasoning_parts = []
                    reasoning_parts.append(f"Strong {self._get_riasec_name(category)} characteristics (score: {score:.1%})")

                    if skill_bonus > 0:
                        reasoning_parts.append("Relevant technical skills identified")
                    if interest_bonus > 0:
                        reasoning_parts.append("Aligns with expressed career interests")

                    if student_profile.dominant_riasec_type and category in student_profile.dominant_riasec_type:
                        reasoning_parts.append("Matches your dominant personality type")

                    prediction = {
                        "title": career["title"],
                        "description": career["description"],
                        "confidence_score": round(confidence, 2),
                        "reasoning": "; ".join(reasoning_parts),
                        "riasec_category": self._get_riasec_name(category),
                        "growth_outlook": career["growth_outlook"],
                        "median_salary": career["median_salary"],
                        "why_good_fit": self._generate_fit_explanation(career, student_profile, category)
                    }

                    predictions.append(prediction)
                    used_careers.add(career["title"])

        # If we don't have 5 predictions, add some general technology careers
        if len(predictions) < 5:
            fallback_careers = [
                {"title": "Technology Consultant", "description": "Advise organizations on technology solutions", "growth_outlook": "High", "median_salary": "$85,000"},
                {"title": "Business Intelligence Analyst", "description": "Transform data into business insights", "growth_outlook": "High", "median_salary": "$78,000"},
                {"title": "Digital Transformation Specialist", "description": "Lead digital innovation initiatives", "growth_outlook": "Very High", "median_salary": "$92,000"}
            ]

            for career in fallback_careers:
                if len(predictions) >= 5:
                    break
                if career["title"] not in used_careers:
                    prediction = {
                        "title": career["title"],
                        "description": career["description"],
                        "confidence_score": 0.65,
                        "reasoning": "Good general fit based on overall profile",
                        "riasec_category": "Multi-category fit",
                        "growth_outlook": career["growth_outlook"],
                        "median_salary": career["median_salary"],
                        "why_good_fit": "This role combines multiple aspects of your interests and skills"
                    }
                    predictions.append(prediction)

        return predictions[:5]  # Ensure we return exactly 5 predictions

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

    def process_user_response(self, session_id: str, user_response: str) -> TaskResult:
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

        return self.process_task(session_state)

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