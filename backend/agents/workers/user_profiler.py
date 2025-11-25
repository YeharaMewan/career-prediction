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
            model=kwargs.get('model', 'gpt-4o'),
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
        return """You are a natural, adaptive Career Counselor having a genuine conversation with a HIGH SCHOOL STUDENT exploring career options.

YOUR ROLE: You're a supportive counselor who adapts to each student - you help them understand themselves through real conversation, not robotic questions. You sound like a real person who cares.

ADAPTIVE & NATURAL APPROACH (Core Behaviors):

1. MATCH THEIR ENERGY & TONE (Be Adaptive)
   • If they're enthusiastic and excited → Mirror that energy and be encouraging
   • If they're thoughtful and reserved → Be calm and reflective with them
   • If they're uncertain → Be patient and reassuring
   • If they're confident → Match their confidence and dig deeper
   • Always adapt your personality to create rapport with THEIR communication style

2. VARY YOUR RESPONSES (Avoid Repetition)
   • Never use the same validation phrase twice in one conversation
   • Draw from diverse natural reactions: affirmations, encouragement, curiosity, empathy, humor
   • Examples of VARIED validations you might use (don't repeat these exact phrases):
     - "I can see why that appeals to you"
     - "That really shows through in what you're saying"
     - "You're picking up on something important there"
     - "I hear what you're saying"
     - "That makes a lot of sense given what you've shared"
     - "There's definitely something to that"
     - "You've got a good read on yourself"
     - "I'm following you"
     - "Interesting perspective"
   • Create NEW natural responses, don't recycle the same phrases

3. ANSWER QUESTIONS DIRECTLY (When They Ask)
   • If they say "...right?" or ask a question, ANSWER IT naturally first
   • Then smoothly continue the conversation
   • Be genuine in your answers, not scripted

4. CONNECT SMOOTHLY (Natural Transitions)
   • Reference specific details they mentioned (use their actual words)
   • Build bridges between topics so it flows naturally
   • Show you're actively listening and tracking the conversation
   • Don't jump topics abruptly - make connections explicit

5. USE PERSONALITY VARIATIONS (Sound Human)
   • Empathetic: Show you understand how they feel
   • Encouraging: Build their confidence when appropriate
   • Curious: Express genuine interest in learning more
   • Playful: Use light humor when the mood fits (age-appropriate)
   • Reflective: Help them see patterns in what they're sharing

6. MID-CONVERSATION CHECK-IN (After about 6-7 questions)
   • Pause and summarize what you've learned
   • Give them a chance to correct or add
   • Make it conversational, not formal

CONVERSATION EXAMPLES (Notice the variety):

Example 1 - Enthusiastic Student:
User: "I love organizing events. It felt really impactful, you know?"
You: "Absolutely! There's something really satisfying about seeing all the pieces come together like that. It sounds like you get a real kick out of making things happen. When you picture your ideal career, what kind of impact are you hoping to have? Quick wins like events, or are you also drawn to longer-term projects?"

Example 2 - Uncertain Student:
User: "I'm good at math, I guess. Is that useful for careers?"
You: "Definitely - math skills open up a lot of doors across different fields. And if it comes naturally to you, that's worth paying attention to. What is it about math that works for you? Do you like solving problems independently, or do you prefer working through them with others?"

Example 3 - Thoughtful Student:
User: "I just like helping people feel better, right?"
You: "Yeah, exactly - helping others and seeing that positive impact is at the core of a lot of meaningful work. That's a real strength. What aspect of that resonates most with you? Is it the problem-solving side, the connection with people, or something else?"

Notice: Each response is different, matches the student's energy, and flows naturally.

Avoid These Patterns:
✗ Repeating the same validation phrase ("That's awesome", "Got it") multiple times
✗ Sounding scripted or robotic
✗ Using overly complex vocabulary or formal language
✗ Ignoring their questions or emotional tone
✗ Jumping topics with no connection
✗ Generic responses like "You mentioned [X]. [Question]?"

ASSESSMENT FRAMEWORK (Use naturally, don't mention explicitly):
You're gathering insights about their career personality profile across these dimensions:
- Realistic: Hands-on, practical, building/fixing things
- Investigative: Analytical, research-oriented, problem-solving
- Artistic: Creative, expressive, design-oriented
- Social: People-oriented, helping, teaching, collaborating
- Enterprising: Leadership, persuasion, business-minded
- Conventional: Organized, detail-oriented, systematic

CONVERSATION STRUCTURE (12 Total Questions):

Phase 1: Questions 1-6 (Initial Exploration)
• Academic interests, hobbies, what they're good at
• Adapt to their communication style
• Vary your language naturally

Phase 2: Question 6-7 (MID-CONVERSATION SUMMARY)
• Pause and naturally summarize what you've learned
• Use conversational language, not formal summaries
• Let them correct or add details
• This builds rapport and shows you're listening

Phase 3: Questions 7-12 (Deeper Exploration)
• Work style, values, personality exploration
• Continue adapting and varying your responses
• Build naturally toward career recommendations

CRITICAL GUIDELINES (Natural & Adaptive):

LANGUAGE APPROACH:
• Use clear, everyday language (high school appropriate)
• Keep sentences conversational and flowing
• Adapt complexity to match their responses
• Avoid overly formal or academic vocabulary
• Examples to avoid: "elaborate", "articulate", "astute", "facilitate", "comprehend"

VALIDATION APPROACH:
• Answer their questions directly and genuinely
• Vary how you acknowledge them - never repeat the same phrase
• Match their emotional tone
• Show authentic understanding

CONNECTION APPROACH:
• Reference specific things they said (use their words)
• Build natural bridges between topics
• Make transitions smooth and logical
• Show you're tracking the full conversation, not just the last response

PERSONALITY VARIATIONS TO USE:
• Empathetic responses when they share something personal
• Encouraging responses when they show uncertainty
• Curious responses when they mention interesting details
• Reflective responses to help them see patterns
• Light humor when appropriate to their communication style

Remember: You're an adaptive counselor who helps students understand themselves through natural conversation. Each response should feel fresh and tailored to what they just shared. They should feel genuinely heard, understood, and excited about discovering their potential."""

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

        # Check completion: exactly 12 questions asked
        if len(conversation_context.question_history) >= 12:
            return self._complete_profiling(conversation_context, state)

        # Generate next question if conversation should continue
        return self._generate_next_question(conversation_context)

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

        # Get language with null safety
        language = "en"
        if state and hasattr(state, 'preferred_language'):
            language = state.preferred_language or "en"

        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            # Update language if changed
            context.session_metadata["language"] = language
            return context

        # Create new conversation context
        context = create_initial_conversation_context()
        context.session_metadata["session_id"] = session_id
        context.session_metadata["created_at"] = datetime.now().isoformat()
        context.session_metadata["mode"] = "interactive"
        context.session_metadata["language"] = language  # Store language

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

        # Check if follow-up is needed (only if under question limit)
        if self.conversation_manager.should_ask_follow_up(context, analysis):
            return self._generate_follow_up_question(context, analysis)

        # Try to transition to next state based on question count
        if self.state_machine.can_transition_to_next_state(context):
            self.state_machine.transition_to_next_state(context)

        # Check for completion: exactly 12 questions
        session_id = context.session_metadata.get("session_id", "unknown")
        questions_asked = len(context.question_history)

        # Log current conversation state for debugging
        logging.info(f"Session {session_id}: Processing response #{questions_asked}, "
                    f"state: {context.current_state.name}")

        if questions_asked >= 12:
            logging.info(f"Session {session_id}: Reached 12 questions - generating career predictions")
            return self._complete_profiling(context, state)
        else:
            logging.info(f"Session {session_id}: Generating next question - "
                        f"{12 - questions_asked} questions remaining")
            return self._generate_next_question(context)

    def _generate_next_question(self, context: ConversationContext) -> TaskResult:
        """Generate the next question in the conversation."""

        try:
            question = self.conversation_manager.generate_question(context)

            # Store updated context (question will be added to history when response is processed)
            session_id = context.session_metadata.get("session_id")
            if session_id:
                self.active_sessions[session_id] = context

            # Get simple progress information
            progress_summary = self._create_progress_summary(context)

            return self._create_task_result(
                task_type="question_generation",
                success=True,
                result_data={
                    "question": question.question_text,
                    "conversation_state": context.current_state.name,
                    "riasec_scores": {cat.value: score for cat, score in context.riasec_scores.items()},
                    "awaiting_response": True,
                    "session_id": session_id,
                    "progress": progress_summary
                }
            )

        except ValueError as e:
            # Question limit reached - force completion
            if "Maximum question limit" in str(e):
                # Create a mock state to trigger completion
                mock_state = AgentState(
                    session_id=context.session_metadata.get("session_id"),
                    interaction_mode="interactive"
                )
                return self._complete_profiling(context, mock_state)
            else:
                # Re-raise other ValueError instances
                raise e

    def _generate_follow_up_question(self, context: ConversationContext, analysis: ResponseAnalysis) -> TaskResult:
        """Generate a follow-up question based on response analysis."""

        follow_up_question = self.conversation_manager.generate_follow_up_question(context, analysis)

        # Update context
        context.question_history.append(follow_up_question.question_text)

        session_id = context.session_metadata.get("session_id")
        if session_id:
            self.active_sessions[session_id] = context

        # Get simple progress information
        progress_summary = self._create_progress_summary(context)

        return self._create_task_result(
            task_type="follow_up_question",
            success=True,
            result_data={
                "question": follow_up_question.question_text,
                "question_type": "follow_up",
                "conversation_state": context.current_state.name,
                "awaiting_response": True,
                "session_id": session_id,
                "follow_up_reason": "Need more specific information",
                "progress": progress_summary
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

        # Get final progress summary
        final_progress = self._create_progress_summary(context)

        return self._create_task_result(
            task_type="profile_completion",
            success=True,
            result_data={
                "student_profile": student_profile.dict(),
                "session_summary": session_summary,
                "conversation_complete": True,
                "dominant_riasec": student_profile.dominant_riasec_type,
                "total_questions": len(context.question_history),
                "session_id": session_id,
                "final_progress": final_progress,
                "conversation_context": {
                    "question_history": context.question_history,
                    "response_history": context.response_history,
                    "collected_data": context.collected_data
                }
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

        # Calculate RIASEC assessment quality based on question distribution
        riasec_quality = sum(1 for count in context.riasec_question_count.values() if count >= 2) / 6.0

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
            riasec_confidence=riasec_quality,

            # Metadata
            conversation_session_id=context.session_metadata.get("session_id"),
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
                "mode": "batch"
            }
        )

    # Public interface methods for interactive sessions
    def start_interactive_session(self, session_id: str, state: Optional[AgentState] = None) -> TaskResult:
        """Start a new interactive profiling session with language context."""

        # Pass state to ensure language preference is preserved
        context = self._get_or_create_conversation_context(session_id, state)

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

        progress_summary = self._create_progress_summary(context)

        return {
            "session_id": session_id,
            "current_state": context.current_state.name,
            "questions_asked": len(context.question_history),
            "questions_remaining": 12 - len(context.question_history),
            "responses_received": len(context.response_history),
            "riasec_scores": {cat.value: score for cat, score in context.riasec_scores.items()},
            "is_complete": self.state_machine.is_conversation_complete(context),
            "progress": progress_summary
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
                career_interests=["General"]
            )

    def _should_complete_conversation(self, context: ConversationContext) -> bool:
        """
        Determine if conversation should be completed.
        Complete after exactly 12 questions.
        """
        questions_asked = len(context.question_history)

        # DETERMINISTIC RULE: Complete after exactly 12 questions
        if questions_asked >= 12:
            session_id = context.session_metadata.get("session_id", "unknown")
            logging.info(f"Session {session_id}: Reached 12 questions - generating career predictions")
            return True

        return False

    def _create_progress_summary(self, context: ConversationContext) -> Dict[str, Any]:
        """
        Create simple progress summary for API responses.
        """
        questions_asked = len(context.question_history)
        questions_remaining = 12 - questions_asked

        # Determine progress stage based on question count
        if questions_asked <= 3:
            stage = "Academic Background"
        elif questions_asked <= 6:
            stage = "Interest Discovery"
        elif questions_asked <= 9:
            stage = "Skills Assessment"
        else:
            stage = "Personality & Final Questions"

        return {
            "questions_asked": questions_asked,
            "questions_remaining": questions_remaining,
            "total_questions": 12,
            "progress_percentage": round((questions_asked / 12) * 100, 1),
            "current_stage": stage,
            "riasec_distribution": {cat.value: count for cat, count in context.riasec_question_count.items()}
        }