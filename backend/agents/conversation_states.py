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
    """
    current_state: ConversationState
    confidence_scores: Dict[str, float]  # Confidence in collected information
    riasec_scores: Dict[RIASECCategory, float]  # RIASEC category scores
    collected_data: Dict[str, Any]  # Information gathered so far
    question_history: List[str]  # Questions already asked
    response_history: List[str]  # User responses
    follow_up_needed: List[str]  # Areas needing more clarification
    session_metadata: Dict[str, Any]  # Session tracking info

    def __post_init__(self):
        """Initialize default values."""
        if not self.confidence_scores:
            self.confidence_scores = {
                "academic_background": 0.0,
                "career_interests": 0.0,
                "skills": 0.0,
                "personality": 0.0,
                "goals": 0.0
            }

        if not self.riasec_scores:
            self.riasec_scores = {category: 0.0 for category in RIASECCategory}

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


class ConfidenceBasedTransition(StateTransitionRule):
    """
    Transition rule based on confidence thresholds.
    """

    def __init__(
        self,
        required_confidence: Dict[str, float],
        target_state: ConversationState,
        min_overall_confidence: float = 0.7
    ):
        self.required_confidence = required_confidence
        self.target_state = target_state
        self.min_overall_confidence = min_overall_confidence

    def can_transition(self, context: ConversationContext) -> bool:
        """Check if confidence thresholds are met."""
        for area, min_confidence in self.required_confidence.items():
            if context.confidence_scores.get(area, 0.0) < min_confidence:
                return False

        # Check overall confidence
        overall = sum(context.confidence_scores.values()) / len(context.confidence_scores)
        return overall >= self.min_overall_confidence

    def get_next_state(self, context: ConversationContext) -> ConversationState:
        """Return the target state."""
        return self.target_state


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


class ConversationStateMachine:
    """
    Finite State Machine for managing the User Profiler conversation flow.
    """

    def __init__(self):
        self.transitions = self._setup_transitions()
        self.confidence_thresholds = self._setup_confidence_thresholds()
        self.state_objectives = self._setup_state_objectives()

    def _setup_transitions(self) -> Dict[ConversationState, List[StateTransitionRule]]:
        """Setup state transition rules."""
        return {
            ConversationState.GREETING: [
                QuestionCountTransition(1, ConversationState.ACADEMIC_GATHERING)
            ],

            ConversationState.ACADEMIC_GATHERING: [
                ConfidenceBasedTransition(
                    {"academic_background": 0.8},
                    ConversationState.INTEREST_DISCOVERY,
                    min_overall_confidence=0.3
                )
            ],

            ConversationState.INTEREST_DISCOVERY: [
                ConfidenceBasedTransition(
                    {"career_interests": 0.7},
                    ConversationState.SKILLS_ASSESSMENT,
                    min_overall_confidence=0.4
                )
            ],

            ConversationState.SKILLS_ASSESSMENT: [
                ConfidenceBasedTransition(
                    {"skills": 0.7},
                    ConversationState.PERSONALITY_PROBE,
                    min_overall_confidence=0.5
                )
            ],

            ConversationState.PERSONALITY_PROBE: [
                ConfidenceBasedTransition(
                    {"personality": 0.7},
                    ConversationState.SUMMARIZATION,
                    min_overall_confidence=0.6
                )
            ],

            ConversationState.SUMMARIZATION: [
                ConfidenceBasedTransition(
                    {"academic_background": 0.7, "career_interests": 0.7, "skills": 0.6, "personality": 0.6},
                    ConversationState.CONFIRMATION,
                    min_overall_confidence=0.7
                )
            ],

            ConversationState.CONFIRMATION: [
                QuestionCountTransition(1, ConversationState.COMPLETED)
            ]
        }

    def _setup_confidence_thresholds(self) -> Dict[ConversationState, Dict[str, float]]:
        """Setup minimum confidence thresholds for each state."""
        return {
            ConversationState.ACADEMIC_GATHERING: {"academic_background": 0.8},
            ConversationState.INTEREST_DISCOVERY: {"career_interests": 0.7},
            ConversationState.SKILLS_ASSESSMENT: {"skills": 0.7},
            ConversationState.PERSONALITY_PROBE: {"personality": 0.7},
            ConversationState.SUMMARIZATION: {
                "academic_background": 0.7,
                "career_interests": 0.7,
                "skills": 0.6,
                "personality": 0.6
            }
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
        Attempt to transition to the next state.
        Returns True if transition was successful.
        """
        next_state = self.get_next_state(context)
        if next_state:
            context.current_state = next_state
            return True
        return False

    def get_state_objective(self, state: ConversationState) -> str:
        """Get the objective for a given state."""
        return self.state_objectives.get(state, "Continue conversation")

    def get_required_confidence(self, state: ConversationState) -> Dict[str, float]:
        """Get required confidence thresholds for a state."""
        return self.confidence_thresholds.get(state, {})

    def is_conversation_complete(self, context: ConversationContext) -> bool:
        """Check if the conversation has reached completion."""
        return context.current_state == ConversationState.COMPLETED

    def get_overall_confidence(self, context: ConversationContext) -> float:
        """Calculate overall confidence score."""
        scores = list(context.confidence_scores.values())
        return sum(scores) / len(scores) if scores else 0.0

    def get_completion_readiness(self, context: ConversationContext) -> Dict[str, Any]:
        """
        Assess how ready the conversation is for completion.
        """
        overall_confidence = self.get_overall_confidence(context)
        required_confidence = self.get_required_confidence(ConversationState.SUMMARIZATION)

        missing_areas = []
        for area, min_confidence in required_confidence.items():
            if context.confidence_scores.get(area, 0.0) < min_confidence:
                missing_areas.append(area)

        return {
            "overall_confidence": overall_confidence,
            "completion_ready": len(missing_areas) == 0 and overall_confidence >= 0.7,
            "missing_areas": missing_areas,
            "next_recommended_state": self.get_next_state(context) or context.current_state
        }


def create_initial_conversation_context() -> ConversationContext:
    """
    Create an initial conversation context for a new session.
    """
    return ConversationContext(
        current_state=ConversationState.GREETING,
        confidence_scores={},
        riasec_scores={},
        collected_data={},
        question_history=[],
        response_history=[],
        follow_up_needed=[],
        session_metadata={}
    )