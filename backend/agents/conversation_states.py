"""
Conversation State Management for Human-in-the-Loop User Profiler Agent

This module implements the Finite State Machine (FSM) for managing
the conversation flow during student profiling sessions.
"""
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


class ConversationState(Enum):
    """
    Finite State Machine states for the User Profiler conversation flow.
    Based on the psychological career counseling framework.
    """
    GREETING = auto()
    ACADEMIC_GATHERING = auto()
    INTEREST_DISCOVERY = auto()
    SKILLS_ASSESSMENT = auto()
    PERSONALITY_PROBE = auto()
    SUMMARIZATION = auto()
    CONFIRMATION = auto()
    COMPLETED = auto()
    ERROR = auto()


class RIASECCategory(Enum):
    """RIASEC model categories for career assessment."""
    REALISTIC = "R"       # Practical, hands-on
    INVESTIGATIVE = "I"   # Analytical, research-oriented
    ARTISTIC = "A"        # Creative, self-expression
    SOCIAL = "S"          # People-oriented, helping
    ENTERPRISING = "E"    # Leadership, business-oriented
    CONVENTIONAL = "C"    # Organized, detail-oriented


@dataclass
class ConversationContext:
    """
    Context information for the current conversation state.
    Simplified to track only question count and RIASEC assessment.
    """
    current_state: ConversationState
    riasec_scores: Dict[RIASECCategory, float]  # RIASEC category scores
    riasec_question_count: Dict[RIASECCategory, int]  # Questions asked per RIASEC category
    collected_data: Dict[str, Any]  # Information gathered so far
    question_history: List[str]  # Questions already asked
    response_history: List[str]  # User responses
    follow_up_needed: List[str]  # Areas needing more clarification
    session_metadata: Dict[str, Any]  # Session tracking info

    def __post_init__(self):
        """Initialize default values."""
        if not self.riasec_scores:
            self.riasec_scores = {category: 0.0 for category in RIASECCategory}

        # Initialize RIASEC question tracking
        if not hasattr(self, 'riasec_question_count') or not self.riasec_question_count:
            self.riasec_question_count = {category: 0 for category in RIASECCategory}

        if not self.collected_data:
            self.collected_data = {}

        if not self.question_history:
            self.question_history = []

        if not self.response_history:
            self.response_history = []

        if not self.follow_up_needed:
            self.follow_up_needed = []

        if not self.session_metadata:
            self.session_metadata = {}


class StateTransitionRule(ABC):
    """
    Abstract base class for state transition rules.
    """

    @abstractmethod
    def can_transition(self, context: ConversationContext) -> bool:
        """Check if transition is allowed based on current context."""
        pass

    @abstractmethod
    def get_next_state(self, context: ConversationContext) -> ConversationState:
        """Get the next state to transition to."""
        pass


class QuestionCountTransition(StateTransitionRule):
    """
    Transition rule based on minimum number of questions in current state.
    """

    def __init__(self, min_questions: int, target_state: ConversationState):
        self.min_questions = min_questions
        self.target_state = target_state

    def can_transition(self, context: ConversationContext) -> bool:
        """Check if minimum questions have been asked."""
        return len(context.question_history) >= self.min_questions

    def get_next_state(self, context: ConversationContext) -> ConversationState:
        """Return the target state."""
        return self.target_state


class QuestionLimitTransition(StateTransitionRule):
    """
    Transition rule that forces completion when approaching question limit.
    """

    def __init__(self, question_limit: int, target_state: ConversationState):
        self.question_limit = question_limit
        self.target_state = target_state

    def can_transition(self, context: ConversationContext) -> bool:
        """Check if we're approaching the question limit and should force completion."""
        # Force transition when we have 2 or fewer questions remaining
        return len(context.question_history) >= (self.question_limit - 2)

    def get_next_state(self, context: ConversationContext) -> ConversationState:
        """Return the target state."""
        return self.target_state


class ConversationStateMachine:
    """
    Finite State Machine for managing the User Profiler conversation flow.
    """

    def __init__(self):
        self.transitions = self._setup_transitions()
        self.state_objectives = self._setup_state_objectives()

    def _setup_transitions(self) -> Dict[ConversationState, List[StateTransitionRule]]:
        """Setup state transition rules - SIMPLIFIED: Based purely on question count, no confidence checks."""
        question_limit = 12  # Exactly 12 questions: 2 per RIASEC category (R, I, A, S, E, C)

        return {
            ConversationState.GREETING: [
                QuestionCountTransition(1, ConversationState.ACADEMIC_GATHERING),
                QuestionLimitTransition(question_limit, ConversationState.COMPLETED)
            ],

            ConversationState.ACADEMIC_GATHERING: [
                QuestionCountTransition(3, ConversationState.INTEREST_DISCOVERY),  # After 3 questions, move on
                QuestionLimitTransition(question_limit, ConversationState.COMPLETED)
            ],

            ConversationState.INTEREST_DISCOVERY: [
                QuestionCountTransition(6, ConversationState.SKILLS_ASSESSMENT),  # After 6 questions, move on
                QuestionLimitTransition(question_limit, ConversationState.COMPLETED)
            ],

            ConversationState.SKILLS_ASSESSMENT: [
                QuestionCountTransition(9, ConversationState.PERSONALITY_PROBE),  # After 9 questions, move on
                QuestionLimitTransition(question_limit, ConversationState.COMPLETED)
            ],

            ConversationState.PERSONALITY_PROBE: [
                QuestionCountTransition(12, ConversationState.COMPLETED),  # After 12 questions, complete
                QuestionLimitTransition(question_limit, ConversationState.COMPLETED)
            ],

            ConversationState.SUMMARIZATION: [
                QuestionCountTransition(1, ConversationState.COMPLETED),  # Should never reach here
                QuestionLimitTransition(question_limit, ConversationState.COMPLETED)
            ],

            ConversationState.CONFIRMATION: [
                QuestionCountTransition(1, ConversationState.COMPLETED),
                QuestionLimitTransition(question_limit, ConversationState.COMPLETED)
            ]
        }

    def _setup_state_objectives(self) -> Dict[ConversationState, str]:
        """Setup objectives for each conversation state."""
        return {
            ConversationState.GREETING: "Welcome the student and establish rapport",
            ConversationState.ACADEMIC_GATHERING: "Collect comprehensive academic background information",
            ConversationState.INTEREST_DISCOVERY: "Identify career interests and industry preferences",
            ConversationState.SKILLS_ASSESSMENT: "Assess technical and soft skills",
            ConversationState.PERSONALITY_PROBE: "Understand personality traits and work style preferences",
            ConversationState.SUMMARIZATION: "Create comprehensive profile summary",
            ConversationState.CONFIRMATION: "Confirm accuracy and completeness of profile",
            ConversationState.COMPLETED: "Profile creation completed successfully"
        }

    def can_transition_to_next_state(self, context: ConversationContext) -> bool:
        """
        Check if the conversation can transition to the next state.
        """
        current_state = context.current_state
        transition_rules = self.transitions.get(current_state, [])

        for rule in transition_rules:
            if rule.can_transition(context):
                return True

        return False

    def get_next_state(self, context: ConversationContext) -> Optional[ConversationState]:
        """
        Get the next state if transition is possible.
        """
        current_state = context.current_state
        transition_rules = self.transitions.get(current_state, [])

        for rule in transition_rules:
            if rule.can_transition(context):
                return rule.get_next_state(context)

        return None

    def transition_to_next_state(self, context: ConversationContext) -> bool:
        """
        Attempt to transition to the next state - SIMPLIFIED: Based purely on question count.
        Returns True if transition was successful.
        """
        next_state = self.get_next_state(context)
        if next_state:
            context.current_state = next_state
            return True

        # Simple fallback: Force completion at 12 questions
        questions_asked = len(context.question_history)

        if questions_asked >= 12:
            context.current_state = ConversationState.COMPLETED
            return True

        return False

    def get_state_objective(self, state: ConversationState) -> str:
        """Get the objective for a given state."""
        return self.state_objectives.get(state, "Continue conversation")

    def is_conversation_complete(self, context: ConversationContext) -> bool:
        """
        Check if the conversation has reached completion - SIMPLIFIED: Based purely on question count.
        """
        # Standard completion check
        if context.current_state == ConversationState.COMPLETED:
            return True

        # Simple rule: Complete after exactly 12 questions
        questions_asked = len(context.question_history)

        if questions_asked >= 12:
            # Force transition to completed state
            context.current_state = ConversationState.COMPLETED
            return True

        return False

def create_initial_conversation_context() -> ConversationContext:
    """
    Create an initial conversation context for a new session.
    """
    return ConversationContext(
        current_state=ConversationState.GREETING,
        riasec_scores={},
        riasec_question_count={},
        collected_data={},
        question_history=[],
        response_history=[],
        follow_up_needed=[],
        session_metadata={}
    )