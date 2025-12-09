"""
Enhanced Conversation Manager for Human-in-the-Loop User Profiler Agent

This module handles intelligent question generation, response analysis,
and conversation flow control using advanced AI techniques.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage

from agents.conversation_states import (
    ConversationState,
    ConversationContext,
    RIASECCategory,
    ConversationStateMachine,
    create_initial_conversation_context,
)
from agents.personality_analyzer import PersonalityAnalyzer

# Import LangSmith configuration
from utils.langsmith_config import (
    get_traced_run_config,
    log_agent_execution,
    setup_langsmith,
)

# Import LLM factory for fallback support
from utils.llm_factory import LLMFactory

# Import MessagePool for dynamic greeting variation
from agents.message_pools import MessagePool

# Ensure LangSmith is configured
setup_langsmith()


class QuestionType(Enum):
    """Types of questions that can be asked."""

    OPEN_ENDED = "open_ended"
    MULTIPLE_CHOICE = "multiple_choice"
    SCALE_RATING = "scale_rating"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"


@dataclass
class GeneratedQuestion:
    """
    A generated question with metadata.
    """

    question_text: str
    question_type: QuestionType
    target_area: str  # What area this question is meant to assess
    expected_insights: List[str]  # What insights we expect to gain
    follow_up_triggers: List[str]  # Conditions that would trigger follow-ups
    riasec_relevance: Dict[RIASECCategory, float]  # RIASEC category relevance scores


@dataclass
class ResponseAnalysis:
    """
    Analysis results from a user response.
    Simplified to focus on RIASEC assessment without confidence tracking.
    Enhanced with dynamic follow-up and ambiguity detection fields.
    """

    riasec_updates: Dict[RIASECCategory, float]  # Updates to RIASEC scores
    extracted_data: Dict[str, Any]  # Extracted structured data
    follow_up_needed: List[str]  # Areas needing follow-up
    intent_classification: str  # Classified intent
    sentiment: str  # Response sentiment
    completeness_score: float  # How complete the response is (0-1)

    # NEW: Dynamic follow-up fields
    depth_score: float = 0.5  # How detailed is the response? (0-1)
    specificity_score: float = 0.5  # How specific vs generic? (0-1)
    follow_up_type: Optional[str] = None  # "clarification", "expansion", "exploration", or None
    follow_up_focus: Optional[str] = None  # Specific area to explore (e.g., "IT interest")
    key_concepts: List[str] = field(default_factory=list)  # Broader concepts (NOT exact user phrases) - max 3

    # NEW: Ambiguity detection
    ambiguity_score: float = 0.0  # How vague/ambiguous is the response? (0-1)


class ConversationManager:
    """
    Advanced conversation manager for intelligent Q&A flow.
    
    Uses GPT-4o-mini for cost-effective question generation and response analysis.

    Implements Chain-of-Thought reasoning, Few-Shot prompting,
    and sophisticated intent recognition with question limiting.
    """

    # Maximum number of questions to ask - Exactly 12 questions for deterministic flow
    MAX_QUESTIONS = 12  # Fixed: 2 questions per RIASEC category (R, I, A, S, E, C)

    # Minimum questions to ensure adequate assessment
    MIN_QUESTIONS = 6

    # Maximum questions per RIASEC category for balanced assessment
    MAX_QUESTIONS_PER_RIASEC = 2  # Exactly 2 questions per category

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):  # Use gpt-4o-mini for cost savings
        # Use LLM factory with fallback support instead of direct ChatOpenAI
        self.llm_wrapper = LLMFactory.create_llm(
            model=model, temperature=temperature, enable_fallback=True
        )
        self.llm = self.llm_wrapper.llm
        logging.info(
            f"✅ ConversationManager LLM initialized using {self.llm_wrapper.strategy.provider_name} with model: {model}"
        )

        self.state_machine = ConversationStateMachine()
        self.question_bank = self._initialize_question_bank()
        self.riasec_keywords = self._initialize_riasec_keywords()
        self.few_shot_examples = self._initialize_few_shot_examples()

        # Initialize message pool for dynamic greeting variation (Phase 2A enhancement)
        self.message_pool = MessagePool()
        logging.info("✅ MessagePool initialized for dynamic greetings")

        # Initialize rephrased question cache (Phase 2B enhancement)
        # Cache structure: {(question_text, tone_profile): rephrased_text}
        # tone_profile = f"{energy_level}_{formality}" (e.g., "enthusiastic_casual")
        self.rephrased_question_cache = {}
        logging.info("✅ Rephrased question cache initialized for adaptive tone matching")

    def _initialize_question_bank(
        self,
    ) -> Dict[ConversationState, List[Dict[str, Any]]]:
        """Initialize curated question bank for each conversation state with simplified, accessible questions."""
        return {
            ConversationState.GREETING: [
                # No predefined greeting - will be generated dynamically
            ],
            ConversationState.INTEREST_DISCOVERY: [
                {
                    "question": "Outside of school, what kinds of activities or hobbies really draw you in? What makes them appealing?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "work_preferences",
                    "follow_up_triggers": [
                        "general_response",
                        "multiple_activities",
                        "unclear",
                    ],
                    "priority": 1,
                },
                {
                    "question": "What do you find yourself doing when you have free time and no obligations? What pulls you in?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "work_preferences",
                    "follow_up_triggers": [
                        "general_response",
                        "multiple_activities",
                        "unclear",
                    ],
                    "priority": 1,
                },
                {
                    "question": "If you had a completely free weekend, what would you choose to spend your time on?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "work_preferences",
                    "follow_up_triggers": [
                        "general_response",
                        "multiple_activities",
                        "unclear",
                    ],
                    "priority": 1,
                },
                {
                    "question": "Think about times when you've felt really engaged and energized. Was it working independently on something, or collaborating with others? What made that experience work for you?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "work_style",
                    "follow_up_triggers": [
                        "combination",
                        "uncertain",
                        "strong_preference",
                    ],
                    "priority": 1,
                },
                {
                    "question": "Do you prefer tackling projects on your own, or do you thrive when working as part of a team? What's your ideal setup?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "work_style",
                    "follow_up_triggers": [
                        "combination",
                        "uncertain",
                        "strong_preference",
                    ],
                    "priority": 1,
                },
                {
                    "question": "When you picture yourself in a work environment, what kind of setting appeals to you? What atmosphere helps you do your best work?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "industry_interests",
                    "follow_up_triggers": [
                        "multiple_industries",
                        "superficial_knowledge",
                        "specific_companies",
                    ],
                    "priority": 1,
                },
                {
                    "question": "Everyone values different things. What matters most to you in your future work - is it making an impact, creative freedom, financial stability, helping others, or something else?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "career_values",
                    "follow_up_triggers": [
                        "multiple_motivations",
                        "unclear_values",
                        "specific_impact",
                    ],
                    "priority": 1,
                },
            ],
            ConversationState.ACADEMIC_GATHERING: [
                {
                    "question": "Tell me about your experience with school so far. What subjects or topics have really clicked for you?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "academic_background",
                    "follow_up_triggers": [
                        "incomplete_info",
                        "career_change",
                        "multiple_fields",
                    ],
                    "priority": 1,
                },
                {
                    "question": "What subjects in school have you found most interesting or engaging? What makes them stand out?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "academic_background",
                    "follow_up_triggers": [
                        "incomplete_info",
                        "career_change",
                        "multiple_fields",
                    ],
                    "priority": 1,
                },
                {
                    "question": "Looking at your classes and subjects, which ones have felt the most natural or enjoyable to you?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "academic_background",
                    "follow_up_triggers": [
                        "incomplete_info",
                        "career_change",
                        "multiple_fields",
                    ],
                    "priority": 1,
                },
                {
                    "question": "When you're learning something new, what helps it stick? Do you prefer hands-on practice, working through problems, creative projects, or discussing ideas with others?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "learning_style",
                    "follow_up_triggers": [
                        "specific_methods",
                        "challenges",
                        "preferences",
                    ],
                    "priority": 1,
                },
                {
                    "question": "What's your go-to approach when picking up a new skill? Do you dive in and experiment, or do you prefer structured guidance?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "learning_style",
                    "follow_up_triggers": [
                        "specific_methods",
                        "challenges",
                        "preferences",
                    ],
                    "priority": 1,
                },
            ],
            ConversationState.SKILLS_ASSESSMENT: [
                {
                    "question": "Let's talk about your strengths. What comes naturally to you that others might find challenging? What do people tend to come to you for?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "technical_skills",
                    "follow_up_triggers": [
                        "limited_skills",
                        "advanced_skills",
                        "self_taught",
                    ],
                    "priority": 1,
                },
                {
                    "question": "What skills or abilities do you have that seem to come easily to you, even if others struggle with them?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "technical_skills",
                    "follow_up_triggers": [
                        "limited_skills",
                        "advanced_skills",
                        "self_taught",
                    ],
                    "priority": 1,
                },
                {
                    "question": "Think about what your friends or classmates ask for your help with. What are you known for being good at?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "technical_skills",
                    "follow_up_triggers": [
                        "limited_skills",
                        "advanced_skills",
                        "self_taught",
                    ],
                    "priority": 1,
                },
                {
                    "question": "When you're faced with a challenge or problem, how do you typically approach it? Walk me through your process.",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "problem_solving",
                    "follow_up_triggers": ["analytical", "creative", "collaborative"],
                    "priority": 1,
                },
                {
                    "question": "When something goes wrong or doesn't work out as planned, what's your first instinct? How do you handle it?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "problem_solving",
                    "follow_up_triggers": ["analytical", "creative", "collaborative"],
                    "priority": 1,
                },
            ],
            ConversationState.PERSONALITY_PROBE: [
                {
                    "question": "Some people recharge by being around others, while others need solo time to refuel. What's your pattern? When do you feel most like yourself?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "introversion_extraversion",
                    "follow_up_triggers": [
                        "extreme_preference",
                        "situational",
                        "unclear",
                    ],
                    "priority": 1,
                },
                {
                    "question": "Think about a time recently when you felt really satisfied or accomplished. What were you doing, and what about that moment made it meaningful for you?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "core_motivation",
                    "follow_up_triggers": [
                        "multiple_motivations",
                        "unclear",
                        "specific_examples",
                    ],
                    "priority": 1,
                },
            ],
        }

    def _initialize_riasec_keywords(self) -> Dict[RIASECCategory, List[str]]:
        """Initialize RIASEC category keywords for scoring."""
        return {
            RIASECCategory.REALISTIC: [
                "hands-on",
                "practical",
                "mechanical",
                "building",
                "tools",
                "outdoors",
                "physical",
                "construction",
                "repair",
                "engineering",
                "technical",
                "tangible",
            ],
            RIASECCategory.INVESTIGATIVE: [
                "research",
                "analyze",
                "investigate",
                "scientific",
                "data",
                "theory",
                "logical",
                "experimental",
                "academic",
                "intellectual",
                "problem-solving",
                "study",
            ],
            RIASECCategory.ARTISTIC: [
                "creative",
                "design",
                "artistic",
                "imaginative",
                "aesthetic",
                "visual",
                "music",
                "writing",
                "innovative",
                "expressive",
                "original",
                "beauty",
            ],
            RIASECCategory.SOCIAL: [
                "helping",
                "teaching",
                "caring",
                "people",
                "service",
                "community",
                "counseling",
                "teamwork",
                "communication",
                "mentoring",
                "support",
                "relationships",
            ],
            RIASECCategory.ENTERPRISING: [
                "leadership",
                "management",
                "business",
                "sales",
                "persuading",
                "competitive",
                "entrepreneurial",
                "ambitious",
                "influence",
                "negotiate",
                "risk-taking",
                "profit",
            ],
            RIASECCategory.CONVENTIONAL: [
                "organized",
                "systematic",
                "detail-oriented",
                "procedures",
                "administrative",
                "structured",
                "accuracy",
                "routine",
                "data-entry",
                "planning",
                "methodical",
                "efficient",
            ],
        }
    def _initialize_few_shot_examples(self) -> List[Dict[str, str]]:
        """Initialize few-shot prompting examples for better LLM performance."""
        return [
            {
                "user_response": "I really enjoy working with computers and solving technical problems. I'm studying computer science and I like programming.",
                "analysis": "High confidence in technical interests (0.9), strong RIASEC Investigative score (0.8), some Realistic elements (0.6). Academic background clearly defined. Follow-up needed on specific programming areas and career goals."
            },
            {
                "user_response": "I want to help people and make a difference. I'm not sure exactly how, but something in healthcare or education maybe.",
                "analysis": "High Social RIASEC score (0.9), moderate confidence in career interests (0.6) due to uncertainty. Need follow-up on specific helping preferences and educational background."
            },
            {
                "user_response": "I like being creative and working on design projects. I also enjoy the business side of things.",
                "analysis": "High Artistic RIASEC score (0.8), moderate Enterprising score (0.6). Good confidence in interests (0.7). Follow-up needed on specific design areas and business interests."
            }
        ]

    def _translate_if_needed(self, text: str, context: ConversationContext) -> str:
        """Translate text to target language if needed."""
        language = context.session_metadata.get("language", "en")
        
        # If language is English or text is empty, return original
        if language == "en" or not text:
            return text
            
        # For Sinhala, translate using LLM
        if language == "si":
            # Log translation fallback (indicates LLM didn't follow instructions)
            session_id = context.session_metadata.get("session_id", "unknown")
            logging.warning(f"Session {session_id}: Translation fallback triggered - LLM may not have followed language instructions")

            prompt = f"""Translate the following career counseling question/statement into natural, conversational Sinhala (Sinhala script).
            
            ORIGINAL: "{text}"
            
            REQUIREMENTS:
            - Use natural, spoken-style Sinhala suitable for a high school student
            - Maintain the warm, supportive tone
            - Write in Sinhala script (සිංහල අකුරු)
            - Return ONLY the translated text, no other words
            
            TRANSLATION:"""
            
            try:
                response = self.llm_wrapper.invoke([HumanMessage(content=prompt)])
                if response and response.content:
                    return response.content.strip()
            except Exception as e:
                logging.error(f"Translation failed: {e}")
                return text
                
        return text

    def generate_question(self, context: ConversationContext) -> GeneratedQuestion:
        """
        Generate an intelligent question based on current conversation context.
        Uses Chain-of-Thought reasoning and context analysis.
        """
        start_time = datetime.now()
        
        # Get tracing configuration
        state_value = context.current_state.value if context.current_state else "unknown"
        run_config = get_traced_run_config(
            session_type="conversation_flow",
            agent_name="conversation_manager",
            session_id=getattr(context, 'session_id', None),
            additional_tags=["question_generation", f"state:{state_value}"],
            additional_metadata={
                "current_state": state_value,
                "conversation_turn": len(context.question_history),
                "questions_remaining": 12 - len(context.question_history)
            }
        )
        
        current_state = context.current_state

        # Check if we've reached the question limit
        if self._has_reached_question_limit(context):
            # Force conversation completion by marking profile as ready
            # This is handled by the caller, but we shouldn't generate more questions
            raise ValueError(f"Maximum question limit of {self.MAX_QUESTIONS} reached")

        # MID-CONVERSATION SUMMARY: Generate after 6-7 questions
        questions_asked = len(context.question_history)
        if questions_asked == 6 or questions_asked == 7:
            # Check if we already did a summary (avoid duplicate)
            if not hasattr(context, '_summary_done') or not context._summary_done:
                logging.info(f"Generating mid-conversation summary at question {questions_asked}")
                return self._generate_mid_conversation_summary(context)

        # Handle GREETING state specifically
        if current_state == ConversationState.GREETING:
            return self._generate_dynamic_greeting(context)

        # Get available questions with priority consideration
        available_questions = self._get_available_questions_by_priority(context)

        if available_questions:
            # Select best question based on priority and context
            selected_question = self._select_best_question(available_questions, context)

            # Translate if needed
            if context.session_metadata.get("language") == "si":
                selected_question["question"] = self._translate_if_needed(selected_question["question"], context)

            result = self._convert_to_generated_question(selected_question, context)
        else:
            # Only generate dynamic questions if we haven't reached limit and no predefined questions available
            result = self._generate_dynamic_question(context)

        # ADAPTIVE REPHRASING (Phase 2B enhancement)
        # Apply tone-matching rephrasing after question 2 (once we have baseline tone data)
        questions_asked = len(context.question_history)
        if questions_asked >= 2 and current_state != ConversationState.GREETING:
            # Skip rephrasing for mid-conversation summaries (they're already personalized)
            if result.target_area != "mid_summary":
                original_question = result.question_text
                rephrased_question = self._rephrase_question_for_student(original_question, context)

                # Update the result with rephrased question if different
                if rephrased_question != original_question:
                    result.question_text = rephrased_question
                    logging.debug(f"Applied adaptive rephrasing (Q{questions_asked + 1})")

        # Log execution
        duration = (datetime.now() - start_time).total_seconds()
        state_value = context.current_state.value if context.current_state else "unknown"
        question_type_value = result.question_type.value if result.question_type else "unknown"
        log_agent_execution(
            "conversation_manager",
            f"generate_question_{state_value}",
            f"Generated {question_type_value} question",
            duration
        )
        
        return result

    def _generate_mid_conversation_summary(self, context: ConversationContext) -> GeneratedQuestion:
        """Generate a summary of what we've learned so far."""
        # Mark summary as done
        context._summary_done = True
        
        summary_prompt = f"""Summarize the key career interests and strengths identified so far for this student.
        
        CONTEXT:
        - RIASEC Scores: {context.riasec_scores}
        - Collected Data: {context.collected_data}
        
        TASK:
        Create a brief, encouraging summary (2-3 sentences) that reflects back what you've heard.
        Then ask a transition question to explore deeper.
        """
        
        try:
            response = self.llm_wrapper.invoke([HumanMessage(content=summary_prompt)])
            summary_text = response.content.strip()
            
            # Translate if needed
            summary_text = self._translate_if_needed(summary_text, context)
            
            return GeneratedQuestion(
                question_text=summary_text,
                question_type=QuestionType.OPEN_ENDED,
                target_area="mid_summary",
                expected_insights=["confirmation", "correction"],
                follow_up_triggers=[],
                riasec_relevance={}
            )
        except Exception as e:
            logging.error(f"Mid-conversation summary generation failed: {e}")
            return self._generate_dynamic_question(context)

    def _generate_dynamic_greeting(self, context: ConversationContext) -> GeneratedQuestion:
        """
        Generate a dynamic, friendly greeting with random variation.

        Strategy (Phase 2A Enhancement):
        - 70% use pre-generated pool (5 greetings per language) - fast, cost-effective
        - 30% generate fresh greeting with LLM - variety, prevents staleness

        Args:
            context: Conversation context containing session metadata

        Returns:
            GeneratedQuestion with greeting text
        """
        import random

        language = context.session_metadata.get("language", "en")

        # Use random selection 70% of time, LLM generation 30%
        use_pool = random.random() < 0.7

        if use_pool:
            # Pick random greeting from pool (70% of cases)
            greeting = self.message_pool.get_random_greeting(language)
            logging.debug(f"Using pooled greeting for language: {language}")
        else:
            # Generate fresh greeting with LLM (30% of cases)
            greeting = self._generate_fresh_greeting_with_llm(language)
            logging.debug(f"Generated fresh LLM greeting for language: {language}")

            # Optionally add this fresh greeting back to the pool
            self.message_pool.add_greeting(language, greeting)

        return GeneratedQuestion(
            question_text=greeting,
            question_type=QuestionType.OPEN_ENDED,
            target_area="greeting",
            expected_insights=["initial_interest"],
            follow_up_triggers=[],
            riasec_relevance={}
        )

    def _generate_fresh_greeting_with_llm(self, language: str) -> str:
        """
        Generate a unique greeting using LLM (30% of cases).

        Creates a fresh, varied greeting that maintains the required structure
        but uses natural, conversational language.

        Args:
            language: Language code ("en" or "si")

        Returns:
            Generated greeting text
        """
        lang_instruction = "සිංහල භාෂාවෙන්" if language == "si" else "in English"

        prompt = f"""Generate a warm, friendly greeting for an AI conversation agent {lang_instruction}.

STRUCTURE (must include):
1. Greeting (1 sentence)
2. Brief introduction (1 sentence)
3. Opening question about interests/hobbies (1 sentence)

CRITICAL REQUIREMENTS:
- DO NOT mention: "career", "job", "future", "calling", "profession", "work"
- DO ask about: hobbies, interests, free time, what they enjoy, passions
- Tone: Friendly, casual, genuinely curious
- Style: Natural conversation starter (not career counseling)
- Length: 2-3 sentences total
- Be warm and approachable

⚠️ AVOID ASKING ABOUT "OUTSIDE OF SCHOOL" OR "FREE TIME" - These topics will be covered later
Instead ask about: general interests, what they're into, what excites them, recent activities

EXAMPLES OF GOOD GREETINGS (for inspiration, don't copy):
- "Hey! I'm here to chat and learn about what makes you tick. What are some of your favorite hobbies or interests?"
- "Welcome! Let's get to know each other. What's something you're really passionate about these days?"
- "Hi there! I'd love to hear about you. What kinds of things do you enjoy doing?"

Generate ONLY the greeting text, no labels or formatting."""

        try:
            response = self.llm_wrapper.llm.invoke([HumanMessage(content=prompt)])
            greeting = response.content.strip()

            # Clean up any unwanted formatting
            greeting = self._clean_generated_text(greeting)

            logging.info(f"✅ Generated fresh LLM greeting ({len(greeting)} chars)")
            return greeting

        except Exception as e:
            logging.error(f"❌ Failed to generate LLM greeting: {e}")

            # Fallback to pool if LLM generation fails
            logging.warning("Using pool fallback for greeting")
            return self.message_pool.get_random_greeting(language)

    def _clean_generated_text(self, text: str) -> str:
        """
        Clean generated text by removing unwanted formatting.

        Removes:
        - Quotes at start/end
        - Extra whitespace
        - Structural markers like [GREETING], [QUESTION], etc.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove quotes
        text = text.strip('"\'')

        # Remove structural markers
        markers = ["[GREETING]", "[ROLE]", "[QUESTION]", "[END]"]
        for marker in markers:
            text = text.replace(marker, "")

        # Clean extra whitespace
        text = " ".join(text.split())

        return text.strip()

    def _rephrase_question_for_student(
        self, question: str, context: ConversationContext
    ) -> str:
        """
        Rephrase a question to match the student's communication tone and style.

        Uses the student's detected tone (energy, formality, confidence) to adapt
        question phrasing for better engagement. Results are cached to minimize LLM costs.

        Args:
            question: Original question text
            context: Conversation context with tone analysis data

        Returns:
            Rephrased question matching student's tone, or original if rephrasing fails
        """
        # Extract tone insights from context (stored in session_metadata by _analyze_response_for_adaptation)
        energy_level = context.session_metadata.get("energy_level", "balanced")
        formality = context.session_metadata.get("formality", "neutral")
        confidence_level = context.session_metadata.get("confidence_level", "moderate")
        suggested_personality = context.session_metadata.get("suggested_personality", "balanced_and_curious")

        # If no tone data available yet (first 1-2 questions), return original
        if energy_level == "balanced" and formality == "neutral" and confidence_level == "moderate":
            logging.debug("No tone insights available yet, using original question")
            return question

        tone_profile = f"{energy_level}_{formality}_{confidence_level}"
        cache_key = (question, tone_profile)

        # Check cache first
        if cache_key in self.rephrased_question_cache:
            logging.debug(f"Using cached rephrased question for tone: {tone_profile}")
            return self.rephrased_question_cache[cache_key]

        # Get language
        language = context.session_metadata.get("language", "en")
        lang_instruction = "සිංහල භාෂාවෙන්" if language == "si" else "in English"

        # Build adaptive rephrasing prompt
        prompt = f"""Rephrase this career counseling question to match the student's communication style {lang_instruction}.

ORIGINAL QUESTION:
"{question}"

STUDENT'S COMMUNICATION STYLE:
- Energy Level: {energy_level}
- Formality: {formality}
- Confidence: {confidence_level}
- Suggested Personality: {suggested_personality}

REPHRASING GUIDELINES:

1. If student is ENTHUSIASTIC:
   - Use more energetic language ("That's awesome!", "I'm curious to know...")
   - Add exclamation points sparingly
   - Mirror their excitement

2. If student is RESERVED:
   - Use calmer, gentler phrasing
   - Avoid excessive enthusiasm
   - Be more thoughtful and patient

3. If student is CASUAL:
   - Use conversational language ("How do you feel about...", "What's your take on...")
   - Avoid formal words like "furthermore", "indeed"
   - Keep it natural and friendly

4. If student is FORMAL:
   - Use more structured language
   - Maintain professional tone
   - Avoid slang

5. If student is UNCERTAIN/LOW CONFIDENCE:
   - Use reassuring language
   - Soften the question ("I'm wondering...", "If you're comfortable sharing...")
   - Make it feel safe to explore

6. If student is CONFIDENT:
   - Be more direct
   - Ask deeper, more challenging questions

CRITICAL REQUIREMENTS:
- Keep the SAME core meaning and intent
- Maintain {lang_instruction} language ONLY
- Output ONLY the rephrased question, no explanations
- Keep it natural and conversational (not robotic)
- Length: 1-2 sentences maximum

Generate the rephrased question now:"""

        try:
            response = self.llm_wrapper.llm.invoke([HumanMessage(content=prompt)])
            rephrased = response.content.strip()
            rephrased = self._clean_generated_text(rephrased)

            # Validate it's not empty
            if len(rephrased) < 5:
                logging.warning("Rephrased question too short, using original")
                return question

            # Cache for future use
            self.rephrased_question_cache[cache_key] = rephrased
            logging.info(f"✅ Rephrased question for tone: {tone_profile} ({len(rephrased)} chars)")

            return rephrased

        except Exception as e:
            logging.error(f"❌ Question rephrasing failed: {e}. Using original question.")
            return question

    def _generate_dynamic_question(self, context: ConversationContext) -> GeneratedQuestion:
        """Generate a dynamic question when no predefined ones are suitable."""
        language = context.session_metadata.get("language", "en")

        # CRITICAL: Strong language enforcement at prompt START
        if language == "si":
            language_instruction = """=== CRITICAL LANGUAGE REQUIREMENT ===
🚨 YOUR QUESTION MUST BE IN SINHALA (සිංහල) 🚨

EVEN IF THE USER TYPED IN ENGLISH, YOUR QUESTION IS IN SINHALA.
DO NOT MIRROR THEIR LANGUAGE. SINHALA ONLY.
==========================================

"""
        else:
            language_instruction = "Language: English\n\n"

        prompt = f"""{language_instruction}Generate a career counseling question for the current state: {context.current_state.name}.

        CONTEXT:
        - Previous Questions: {context.question_history[-3:]}
        - RIASEC Scores: {context.riasec_scores}

        🚨 CRITICAL RESTRICTION:
        - NEVER ask directly: "What career?", "What job?", "What do you want to be?"
        - INSTEAD ask about: interests, strengths, how they work, what they enjoy, their values
        - Focus on understanding their PERSONALITY and INTERESTS - let AI infer career matches

        Generate a single, engaging open-ended question."""
        
        try:
            response = self.llm_wrapper.invoke([HumanMessage(content=prompt)])
            question_text = response.content.strip()
            
            return GeneratedQuestion(
                question_text=question_text,
                question_type=QuestionType.OPEN_ENDED,
                target_area="dynamic_exploration",
                expected_insights=["general_insight"],
                follow_up_triggers=[],
                riasec_relevance={}
            )
        except Exception as e:
            logging.error(f"Dynamic question generation failed: {e}")
            fallback = "Tell me more about your interests."
            if language == "si":
                fallback = "ඔබේ රුචිකත්වයන් ගැන මට තව කියන්න."
            return GeneratedQuestion(
                question_text=fallback,
                question_type=QuestionType.OPEN_ENDED,
                target_area="fallback",
                expected_insights=[],
                follow_up_triggers=[],
                riasec_relevance={}
            )

    def _select_best_question(self, available_questions: List[Dict[str, Any]], context: ConversationContext) -> Dict[str, Any]:
        """
        Select the best question from the available list.
        
        Enhanced with random selection among same-priority questions to add variety
        and prevent users from seeing the exact same questions every time.
        """
        import random
        
        if not available_questions:
            raise ValueError("No available questions to select from")
        
        # Group questions by priority
        priority_groups = {}
        for question in available_questions:
            priority = question.get("priority", 1)
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(question)
        
        # Get highest priority group
        highest_priority = min(priority_groups.keys())
        top_questions = priority_groups[highest_priority]
        
        # Randomly select from top priority questions for variety
        selected = random.choice(top_questions)
        
        logging.debug(f"Selected question from {len(top_questions)} priority-{highest_priority} options")
        return selected

    def _convert_to_generated_question(self, question_data: Dict[str, Any], context: ConversationContext) -> GeneratedQuestion:
        """Convert dictionary question data to GeneratedQuestion object."""
        return GeneratedQuestion(
            question_text=question_data["question"],
            question_type=question_data["type"],
            target_area=question_data["target_area"],
            expected_insights=[],
            follow_up_triggers=question_data.get("follow_up_triggers", []),
            riasec_relevance=self._calculate_question_riasec_relevance(question_data["question"])
        )

    def analyze_response(self, user_response: str, context: ConversationContext, question: GeneratedQuestion) -> ResponseAnalysis:
        """
        Analyze user response using advanced AI techniques.
        Implements intent recognition, confidence scoring, and RIASEC assessment.
        """
        start_time = datetime.now()
        
        # Get tracing configuration
        state_value = context.current_state.value if context.current_state else "unknown"
        run_config = get_traced_run_config(
            session_type="response_analysis",
            agent_name="conversation_manager",
            session_id=getattr(context, 'session_id', None),
            additional_tags=["response_analysis", f"state:{state_value}", question.target_area],
            additional_metadata={
                "question_target_area": question.target_area,
                "response_length": len(user_response),
                "current_riasec_scores": context.riasec_scores
            }
        )

        # Create analysis prompt - concise format without verbose reasoning
        analysis_prompt = f"""You are an expert career counselor analyzing a student's response.

QUESTION ASKED: {question.question_text}
TARGET AREA: {question.target_area}
USER RESPONSE: "{user_response}"

CONVERSATION CONTEXT:
- Current State: {context.current_state.name}
- Current RIASEC Scores: {context.riasec_scores}

FEW-SHOT EXAMPLES:
{self._format_few_shot_examples()}

ANALYSIS TASK:
Analyze the response and provide structured assessment covering: completeness, extracted data, RIASEC scores, follow-up areas, intent, sentiment, PLUS new fields for ambiguity detection and dynamic follow-ups.

AMBIGUITY ASSESSMENT:
- Evaluate if the response is too vague to be actionable (generic terms, very short, no specifics)
- Score ambiguity from 0.0 (very specific) to 1.0 (extremely vague)

DEPTH & SPECIFICITY:
- depth_score (0-1): How detailed is the response? (0=very brief, 1=very detailed)
- specificity_score (0-1): How specific vs generic? (0=generic "I like tech", 1=specific "I enjoy Python web development")

FOLLOW-UP OPPORTUNITY:
- follow_up_type: "clarification" (vague/unclear), "expansion" (interesting but brief), "exploration" (passion detected), or null
- follow_up_focus: Specific topic to explore (e.g., "AI interest", "helping people motivation")
- key_concepts: Extract 1-3 CONCEPTS they discussed (NOT their exact words) - paraphrase to broader themes
  Example: If they say "React and Flask", extract ["full-stack development", "web technologies"]
  Example: If they say "organizing events", extract ["project coordination", "leadership"]

Provide analysis in this JSON format (JSON only, no explanations):
{{
    "completeness_score": 0.0-1.0,
    "extracted_data": {{"key": "value"}},
    "riasec_updates": {{"R": 0.0-1.0, "I": 0.0-1.0, "A": 0.0-1.0, "S": 0.0-1.0, "E": 0.0-1.0, "C": 0.0-1.0}},
    "follow_up_needed": ["area1", "area2"],
    "intent_classification": "description",
    "sentiment": "positive/neutral/negative",
    "ambiguity_score": 0.0-1.0,
    "depth_score": 0.0-1.0,
    "specificity_score": 0.0-1.0,
    "follow_up_type": "clarification|expansion|exploration|null",
    "follow_up_focus": "topic area or null",
    "key_concepts": ["concept1", "concept2", "concept3"]
}}"""

        try:
            response = self.llm_wrapper.invoke([HumanMessage(content=analysis_prompt)])

            if not response or not response.content:
                raise ValueError("Empty response from LLM")

            analysis_text = response.content.strip()

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                try:
                    analysis_data = json.loads(json_match.group())
                except json.JSONDecodeError as je:
                    logging.warning(f"Failed to parse LLM JSON response: {je}. Using fallback analysis.")
                    return self._fallback_analysis(user_response, question)
            else:
                logging.warning("No JSON found in LLM response. Using fallback analysis.")
                return self._fallback_analysis(user_response, question)

            # Convert to ResponseAnalysis object
            result = ResponseAnalysis(
                riasec_updates={
                    RIASECCategory(k): v for k, v in analysis_data.get("riasec_updates", {}).items()
                    if k in ["R", "I", "A", "S", "E", "C"]
                },
                extracted_data=analysis_data.get("extracted_data", {}),
                follow_up_needed=analysis_data.get("follow_up_needed", []),
                intent_classification=analysis_data.get("intent_classification", "general_response"),
                sentiment=analysis_data.get("sentiment", "neutral"),
                completeness_score=analysis_data.get("completeness_score", 0.5),
                # NEW: Dynamic follow-up fields
                depth_score=analysis_data.get("depth_score", 0.5),
                specificity_score=analysis_data.get("specificity_score", 0.5),
                follow_up_type=analysis_data.get("follow_up_type"),
                follow_up_focus=analysis_data.get("follow_up_focus"),
                key_concepts=analysis_data.get("key_concepts", []),
                # NEW: Ambiguity detection
                ambiguity_score=analysis_data.get("ambiguity_score", 0.0)
            )
            
            # Log execution
            duration = (datetime.now() - start_time).total_seconds()
            log_agent_execution(
                "conversation_manager", 
                f"analyze_response_{question.target_area}",
                f"Analysis completed - Intent: {result.intent_classification}, Completeness: {result.completeness_score:.2f}",
                duration
            )
            
            return result

        except Exception as e:
            # Log the specific error for debugging
            logging.error(f"LLM analysis failed: {type(e).__name__}: {str(e)}. Using fallback analysis.")

            # Log execution with error
            duration = (datetime.now() - start_time).total_seconds()
            log_agent_execution(
                "conversation_manager",
                f"analyze_response_{question.target_area}",
                f"Analysis failed, using fallback - Error: {type(e).__name__}",
                duration
            )

            # Fallback to rule-based analysis
            return self._fallback_analysis(user_response, question)

    def detect_ambiguity(
        self,
        user_response: str,
        question_context: str = ""
    ) -> tuple[bool, float, List[str]]:
        """
        Detect if a response is too vague/ambiguous to be useful.

        Args:
            user_response: The user's response text
            question_context: The question that was asked (for context)

        Returns:
            Tuple of (is_ambiguous, ambiguity_score, ambiguous_phrases)
            - is_ambiguous: True if ambiguity_score >= 0.5
            - ambiguity_score: 0.0 (clear) to 1.0 (extremely vague)
            - ambiguous_phrases: List of specific vague phrases detected
        """
        response_lower = user_response.lower().strip()
        word_count = len(user_response.split())

        ambiguity_score = 0.0
        ambiguous_phrases = []

        # 1. Length check - very short responses are often vague
        if word_count < 10:
            ambiguity_score += 0.3
            if word_count < 5:
                ambiguity_score += 0.1  # Extra penalty for extremely short

        # 2. Generic career phrases
        generic_phrases = [
            "good job", "make money", "be rich", "successful", "nice career",
            "stable job", "high salary", "comfortable life", "good salary",
            "make a living", "earn well", "decent job"
        ]
        for phrase in generic_phrases:
            if phrase in response_lower:
                ambiguity_score += 0.25
                ambiguous_phrases.append(phrase)

        # 3. Vague interest statements
        vague_interests = [
            "i like technology", "i want to help people", "business is interesting",
            "i'm interested in everything", "i don't know yet", "anything related to",
            "something with", "something in", "work with computers",
            "be my own boss"
        ]
        for statement in vague_interests:
            if statement in response_lower:
                ambiguity_score += 0.3
                ambiguous_phrases.append(statement)

        # 4. Non-committal language
        uncertainty_markers = [
            "maybe", "i guess", "i don't know", "not sure", "kind of",
            "sort of", "possibly", "might", "whatever", "anything"
        ]
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in response_lower)
        if uncertainty_count >= 2:
            ambiguity_score += 0.2
            ambiguous_phrases.append(f"{uncertainty_count} uncertainty markers")

        # 5. Single-word or very minimal responses
        if word_count <= 2 and not any(marker in response_lower for marker in ["yes", "no"]):
            ambiguity_score += 0.4
            ambiguous_phrases.append("minimal response")

        # Cap at 1.0
        ambiguity_score = min(ambiguity_score, 1.0)

        # Threshold for ambiguity
        is_ambiguous = ambiguity_score >= 0.5

        logging.debug(f"Ambiguity detection: score={ambiguity_score:.2f}, "
                     f"is_ambiguous={is_ambiguous}, phrases={ambiguous_phrases}")

        return is_ambiguous, ambiguity_score, ambiguous_phrases

    def _fallback_analysis(self, user_response: str, question: GeneratedQuestion) -> ResponseAnalysis:
        """Fallback rule-based analysis when LLM analysis fails."""

        response_lower = user_response.lower()

        # Simple keyword-based RIASEC scoring
        riasec_updates = {}
        for category, keywords in self.riasec_keywords.items():
            score = sum(1 for keyword in keywords if keyword in response_lower)
            riasec_updates[category] = min(score / len(keywords), 1.0)

        # Simple completeness scoring based on response length
        word_count = len(user_response.split())
        completeness_score = min(word_count / 20, 1.0)  # Normalize to 0-1

        # Calculate ambiguity using the detect_ambiguity method
        is_ambiguous, ambiguity_score, ambiguous_phrases = self.detect_ambiguity(user_response)

        return ResponseAnalysis(
            riasec_updates=riasec_updates,
            extracted_data={"response": user_response},
            follow_up_needed=[] if completeness_score > 0.7 else [question.target_area],
            intent_classification="general_response",
            sentiment="neutral",
            completeness_score=completeness_score,
            # NEW: Dynamic follow-up fields (defaults for fallback)
            depth_score=completeness_score,  # Use completeness as proxy for depth
            specificity_score=0.5,  # Neutral default
            follow_up_type="clarification" if is_ambiguous else None,
            follow_up_focus=question.target_area if is_ambiguous else None,
            key_concepts=[],
            # NEW: Ambiguity detection
            ambiguity_score=ambiguity_score
        )

    def _calculate_question_riasec_relevance(self, question_text: str) -> Dict[RIASECCategory, float]:
        """Calculate how relevant a question is to each RIASEC category."""
        question_lower = question_text.lower()
        relevance = {}

        for category, keywords in self.riasec_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            relevance[category] = min(matches / len(keywords), 1.0)

        return relevance

    def _format_few_shot_examples(self) -> str:
        """Format few-shot examples for prompting."""
        formatted = []
        for example in self.few_shot_examples:
            formatted.append(f"User: {example['user_response']}")
            formatted.append(f"Analysis: {example['analysis']}")
            formatted.append("")
        return "\n".join(formatted)

    def _has_reached_question_limit(self, context: ConversationContext) -> bool:
        """Check if we've reached the maximum question limit."""
        return len(context.question_history) >= self.MAX_QUESTIONS

    def _get_riasec_priority_categories(self, context: ConversationContext) -> List[RIASECCategory]:
        """Get RIASEC categories that need more questions, prioritized by need."""
        priority_categories = []

        for category in RIASECCategory:
            questions_count = context.riasec_question_count.get(category, 0)

            # Only include categories that haven't reached the limit (2 questions each)
            if questions_count < self.MAX_QUESTIONS_PER_RIASEC:
                priority_categories.append(category)

        # Sort by lowest question count first
        priority_categories.sort(key=lambda cat: context.riasec_question_count.get(cat, 0))

        return priority_categories

    def _strip_structural_markers(self, text: str) -> str:
        """
        Remove structural markers, prefixes, and verbose agent output from LLM responses.
        These are used in prompts for instruction but should never appear in user-facing output.
        """
        if not text:
            return text

        # Remove all structural markers in brackets (case-insensitive)
        markers = [
            r'\[VALIDATION\]', r'\[BRIDGE\]', r'\[QUESTION\]',
            r'\[VALIDATE\]', r'\[EMPATHIZE\]', r'\[FOLLOW-UP QUESTION\]',
            r'\[SUMMARY\]', r'\[TRANSITION\]', r'\[ACKNOWLEDGE\]', r'\[CONNECT\]', r'\[ASK\]'
        ]

        cleaned_text = text
        for marker in markers:
            cleaned_text = re.sub(marker, '', cleaned_text, flags=re.IGNORECASE)

        # Remove common prefixes that indicate agent reasoning/structure
        prefixes = [
            r'^Acknowledge:\s*', r'^Connect:\s*', r'^Ask:\s*',
            r'^Reasoning:\s*', r'^Analysis:\s*', r'^Response:\s*',
            r'^Question:\s*', r'^Explanation:\s*', r'^Validation:\s*'
        ]
        for prefix in prefixes:
            cleaned_text = re.sub(prefix, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove content in parentheses that looks like meta-commentary
        # e.g., "(explanation)", "(reasoning)", "(acknowledging their response)"
        cleaned_text = re.sub(r'\s*\([^)]*(?:explanation|reasoning|note|meta|thinking|analysis)[^)]*\)\s*', ' ', cleaned_text, flags=re.IGNORECASE)

        # Remove verbose introductions like "Let me ask you..."
        cleaned_text = re.sub(r'^Let me ask you[,:]?\s*', '', cleaned_text, flags=re.IGNORECASE)

        # Clean up any extra whitespace that might result from removals
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Log if significant cleaning occurred
        if len(text) - len(cleaned_text) > 20:
            logging.info(f"Stripped verbose output from response. Original: {len(text)} chars, Cleaned: {len(cleaned_text)} chars")

        return cleaned_text

    def _fix_question_beginning(self, question_text: str) -> str:
        """Fix questions that start with lowercase or missing subject (e.g., 'mentioned...' -> 'You mentioned...')."""
        if not question_text or not question_text.strip():
            return question_text

        question_text = question_text.strip()

        # Check if question starts with lowercase letter (indicates missing subject)
        if question_text and question_text[0].islower():
            logging.warning(f"Question starts with lowercase: '{question_text[:50]}...'")

            # Common fragments that need "You" prepended
            lowercase_starts = [
                "mentioned", "said", "told", "indicated", "expressed", "described",
                "shared", "explained", "talked about", "spoke about"
            ]

            for start in lowercase_starts:
                if question_text.lower().startswith(start):
                    question_text = "You " + question_text
                    logging.info(f"Fixed question beginning: '{question_text[:50]}...'")
                    break

        # Check for other common cut-off patterns
        incomplete_starts = [
            "are some ", "is your ", "do you ", "did you ", "have you ",
            "can you ", "could you ", "would you ", "will you "
        ]

        for incomplete in incomplete_starts:
            if question_text.lower().startswith(incomplete):
                # These likely need "What " prepended
                question_text = "What " + question_text
                logging.info(f"Fixed incomplete question: '{question_text[:50]}...'")
                break

        return question_text

    def _ensure_single_question(self, question_text: str) -> str:
        """Ensure only ONE question is asked per agent message - FIXED to prevent text cutoff."""
        if not question_text or not question_text.strip():
            return question_text

        # Remove any newlines or extra spacing
        question_text = ' '.join(question_text.split())

        # Count question marks
        question_count = question_text.count('?')

        if question_count == 0:
            # No question mark - check if it looks like a question
            sentences = question_text.split('.')
            if sentences and sentences[0].strip():
                first_sentence = sentences[0].strip()
                # Add question mark if it seems like a question
                if any(word in first_sentence.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which', 'do you', 'are you', 'can you', 'could you', 'would you']):
                    return first_sentence + '?'
                return first_sentence + '.'
            return question_text

        if question_count == 1:
            # Perfect - exactly one question, return as-is
            return question_text.strip()

        # Multiple questions - take ONLY the first complete question
        # FIXED: Preserve full question text, don't cut off the beginning
        logging.warning(f"Multiple questions detected ({question_count} question marks). Taking first question only.")

        # Split by question marks
        parts = question_text.split('?')

        if len(parts) >= 2 and parts[0].strip():
            # Take the first question (everything before the first ?)
            first_question = parts[0].strip() + '?'
            logging.info(f"Extracted first question: {first_question[:100]}...")
            return first_question

        # Fallback
        return question_text.split('?')[0].strip() + '?' if '?' in question_text else question_text

    def _ensure_greeting_has_single_question(self, greeting_text: str) -> str:
        """Ensure greeting ends with exactly ONE question - FIXED to prevent text cutoff."""
        if not greeting_text or not greeting_text.strip():
            logging.warning("Empty greeting, using fallback")
            return "Hi! I'm your AI career helper. What interests you most about finding a career?"

        greeting_text = greeting_text.strip()

        # Count question marks
        question_count = greeting_text.count('?')

        if question_count == 0:
            # No question - this is invalid, return fallback
            logging.warning("Greeting has no question, using fallback")
            return "Hi! I'm your AI career helper. What interests you most about finding a career?"

        if question_count == 1:
            # Perfect - exactly one question, return as-is
            return greeting_text

        # Multiple questions - keep ALL text but remove extra question marks except the last one
        # FIXED: Previously this was cutting off text, now we preserve everything
        logging.warning(f"Multiple questions detected in greeting ({question_count} question marks)")

        # Find the position of the last question mark
        last_q_index = greeting_text.rfind('?')

        # Strategy: Keep everything up to and including the last question mark
        # Remove any text after the last question mark (cleanup)
        result = greeting_text[:last_q_index + 1].strip()

        # Now remove intermediate question marks by replacing ? with . in the intro part
        # Find where the last sentence before final ? starts
        # Look backwards from the last ? to find sentence boundaries
        text_before_final_q = result[:last_q_index]

        # Replace all ? in the intro with . to maintain text but remove extra questions
        if '?' in text_before_final_q:
            # Find the start of the last sentence
            sentence_starters = []
            for i, char in enumerate(text_before_final_q):
                if char in '.!?' and i < len(text_before_final_q) - 1:
                    sentence_starters.append(i + 1)

            if sentence_starters:
                # Get the start of the last sentence (last question)
                last_sentence_start = sentence_starters[-1] if sentence_starters else 0
                intro = text_before_final_q[:last_sentence_start].replace('?', '.')
                last_question = text_before_final_q[last_sentence_start:]
                result = intro + last_question + '?'
            else:
                # No sentence boundaries found, just replace ? with . in intro
                result = text_before_final_q.replace('?', '.') + '?'

        logging.info(f"Cleaned greeting to single question: {result[:100]}...")
        return result

    def _update_riasec_question_tracking(self, context: ConversationContext, question_text: str, riasec_relevance: Dict[RIASECCategory, float]):
        """Update RIASEC question counts - STRICT: Each question counts for exactly ONE category."""
        # Find the primary RIASEC category for this question (highest relevance)
        if riasec_relevance:
            primary_category = max(riasec_relevance.items(), key=lambda x: x[1])[0]

            # STRICT: Only increment the primary category by exactly 1
            # This ensures we ask exactly 2 questions per RIASEC category
            if riasec_relevance[primary_category] > 0.3:
                context.riasec_question_count[primary_category] = context.riasec_question_count.get(primary_category, 0) + 1
                logging.info(f"RIASEC tracking: {primary_category.value} now has {context.riasec_question_count[primary_category]} questions")

    def _get_available_questions_by_priority(self, context: ConversationContext) -> List[Dict[str, Any]]:
        """Get available questions - STRICT RIASEC BALANCE: Prioritize categories with fewer than 2 questions."""
        current_state = context.current_state
        state_questions = self.question_bank.get(current_state, [])

        # Enhanced filtering to prevent question repetition
        asked_questions = set(context.question_history)
        available_questions = []

        for q in state_questions:
            question_text = q["question"]

            # Exact match filtering
            if question_text in asked_questions:
                continue

            # Similarity filtering - prevent questions that are too similar to already asked ones
            if self._is_too_similar_to_asked_questions(question_text, context.question_history):
                continue

            # STRICT RIASEC FILTERING: Check if this question's primary RIASEC category already has 2 questions
            riasec_relevance = self._calculate_question_riasec_relevance(question_text)
            if riasec_relevance:
                primary_category = max(riasec_relevance.items(), key=lambda x: x[1])[0]
                category_count = context.riasec_question_count.get(primary_category, 0)

                # Skip if this category already has 2 questions
                if category_count >= self.MAX_QUESTIONS_PER_RIASEC:
                    logging.info(f"Skipping question - {primary_category.value} already has {category_count} questions")
                    continue

            available_questions.append(q)

        # PRIORITIZE categories with 0 or 1 questions to ensure balance
        # Sort by: 1) RIASEC category need, 2) priority
        def sort_key(q):
            riasec_relevance = self._calculate_question_riasec_relevance(q["question"])
            if riasec_relevance:
                primary_category = max(riasec_relevance.items(), key=lambda x: x[1])[0]
                category_count = context.riasec_question_count.get(primary_category, 0)
                # Categories with fewer questions get higher priority (lower number = higher priority)
                return (category_count, q.get("priority", 3))
            return (999, q.get("priority", 3))  # No RIASEC relevance = lowest priority

        available_questions.sort(key=sort_key)

        # Log question selection for debugging
        session_id = context.session_metadata.get("session_id", "unknown")
        riasec_status = {cat.value: context.riasec_question_count.get(cat, 0) for cat in RIASECCategory}
        logging.info(f"Session {session_id}: {len(available_questions)} questions available, RIASEC: {riasec_status}")

        return available_questions

    def _extract_question_keywords(self, question: str) -> List[str]:
        """
        Extract meaningful keywords from a question for similarity comparison.

        Removes:
        - Common stop words (what, how, do, you, are, etc.)
        - Question words (who, what, where, when, why, how)
        - Very short words (< 3 characters)

        Keeps:
        - Nouns, verbs, adjectives (content words)
        - Domain-specific terms (career, skills, interests, etc.)

        Args:
            question: Question text to extract keywords from

        Returns:
            List of meaningful keywords in lowercase
        """
        # Stop words to remove (English)
        stop_words = {
            'what', 'how', 'do', 'does', 'did', 'you', 'your', 'are', 'is', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'the', 'a', 'an',
            'to', 'of', 'in', 'on', 'at', 'for', 'with', 'about', 'as', 'by',
            'that', 'this', 'these', 'those', 'it', 'they', 'them', 'their',
            'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might',
            'tell', 'me', 'us', 'some', 'any', 'or', 'and', 'but', 'so', 'if',
            'when', 'where', 'who', 'which', 'why', 'there', 'here'
        }

        # Sinhala stop words (basic set)
        sinhala_stop_words = {
            'මම', 'ඔබ', 'ඔබේ', 'මගේ', 'මේ', 'එම', 'කරන', 'කරනවා',
            'කළ', 'කරන්න', 'ඇති', 'තියෙන', 'තියෙනවා'
        }

        # Combine stop words
        all_stop_words = stop_words | sinhala_stop_words

        # Tokenize and clean
        import re

        # Remove punctuation and split
        cleaned = re.sub(r'[^\w\s]', ' ', question.lower())
        words = cleaned.split()

        # Filter keywords
        keywords = [
            word for word in words
            if len(word) >= 3  # At least 3 characters
            and word not in all_stop_words  # Not a stop word
        ]

        return keywords

    def _is_too_similar_to_asked_questions(self, new_question: str, question_history: List[str]) -> bool:
        """
        Check if a new question is too similar to previously asked questions.
        Enhanced similarity detection to prevent repetitive questions.
        """
        if not question_history:
            return False

        new_question_lower = new_question.lower()
        new_keywords = set(self._extract_question_keywords(new_question))

        # Check against recent questions (last 5) for stronger similarity filtering
        recent_questions = question_history[-5:] if len(question_history) >= 5 else question_history

        for asked_question in recent_questions:
            asked_question_lower = asked_question.lower()
            asked_keywords = set(self._extract_question_keywords(asked_question))

            # 1. Exact substring matching
            if new_question_lower in asked_question_lower or asked_question_lower in new_question_lower:
                return True

            # 2. High keyword overlap (indicates very similar topics)
            keyword_overlap = len(new_keywords & asked_keywords)
            total_keywords = len(new_keywords | asked_keywords)

            if total_keywords > 0:
                similarity_ratio = keyword_overlap / total_keywords
                if similarity_ratio >= 0.6:  # 60% keyword overlap threshold
                    return True

            # 3. Check for similar question patterns
            if self._have_similar_patterns(new_question_lower, asked_question_lower):
                return True

            # 4. Check for semantic similarity in key phrases
            if self._have_similar_key_phrases(new_question, asked_question):
                return True

        return False

    def _have_similar_patterns(self, question1: str, question2: str) -> bool:
        """Check if two questions follow similar patterns."""
        # Common question patterns that might indicate repetition
        patterns = [
            ("what are you", "what do you"),
            ("tell me about", "describe your"),
            ("how do you", "what is your"),
            ("what kind of", "what type of"),
            ("what makes you", "what drives you"),
            ("where do you", "how do you")
        ]

        for pattern1, pattern2 in patterns:
            if (pattern1 in question1 and pattern2 in question2) or \
               (pattern2 in question1 and pattern1 in question2):
                return True

        return False

    def _have_similar_key_phrases(self, question1: str, question2: str) -> bool:
        """Check if questions contain similar key phrases indicating the same topic."""
        # Key phrases that indicate similar questions
        key_phrase_groups = [
            ["academic", "education", "study", "school", "learning"],
            ["skills", "abilities", "talents", "strengths", "capabilities"],
            ["interests", "passion", "enjoy", "like", "prefer"],
            ["work", "job", "career", "profession", "occupation"],
            ["personality", "character", "nature", "style", "approach"],
            ["goals", "ambitions", "objectives", "aims", "aspirations"],
            ["experience", "background", "history", "past"]
        ]

        question1_lower = question1.lower()
        question2_lower = question2.lower()

        for phrase_group in key_phrase_groups:
            q1_has_phrases = sum(1 for phrase in phrase_group if phrase in question1_lower)
            q2_has_phrases = sum(1 for phrase in phrase_group if phrase in question2_lower)

            # If both questions have multiple phrases from the same group, they're likely similar
            if q1_has_phrases >= 2 and q2_has_phrases >= 2:
                return True

        return False

    def update_conversation_context(self, context: ConversationContext, question: GeneratedQuestion, user_response: str, analysis: ResponseAnalysis) -> ConversationContext:
        """
        Update the conversation context with new question, response, and analysis.
        Simplified to track only RIASEC scores and question count.
        """
        # Add to history
        context.question_history.append(question.question_text)
        context.response_history.append(user_response)

        # Enhanced response analysis for question adaptation
        response_insights = self._analyze_response_for_adaptation(user_response, context)
        context.session_metadata.update(response_insights)

        # Update RIASEC scores
        for category, update in analysis.riasec_updates.items():
            current = context.riasec_scores.get(category, 0.0)
            new_score = (current * 0.8) + (update * 0.2)
            change = new_score - current

            # Store RIASEC changes
            context.session_metadata[f"riasec_change_{category.value}"] = change
            context.riasec_scores[category] = new_score

        # Track RIASEC question coverage for this question
        riasec_relevance = self._calculate_question_riasec_relevance(question.question_text)
        self._update_riasec_question_tracking(context, question.question_text, riasec_relevance)

        # Update collected data with enhanced extraction
        extracted_data = analysis.extracted_data.copy()
        extracted_data.update(self._extract_additional_data_from_response(user_response, context))
        context.collected_data.update(extracted_data)

        # Update follow-up needs based on response analysis
        follow_up_needs = analysis.follow_up_needed.copy()
        follow_up_needs.extend(self._identify_follow_up_opportunities(user_response, context))
        context.follow_up_needed.extend(follow_up_needs)
        # Remove duplicates
        context.follow_up_needed = list(set(context.follow_up_needed))

        # NEW: Personality analysis (per-response)
        if not hasattr(context, 'personality_analyzer'):
            # Initialize personality analyzer on first use
            context.personality_analyzer = PersonalityAnalyzer()
            logging.debug("Initialized PersonalityAnalyzer for this session")

        # Analyze current response for personality traits
        personality_insights = context.personality_analyzer.analyze_response(user_response)

        # Update cumulative scores
        context.personality_analyzer.update_cumulative_scores(personality_insights)

        # Get dominant traits and confidence
        dominant_traits = context.personality_analyzer.get_dominant_traits(threshold=0.25)
        personality_confidence = context.personality_analyzer.get_confidence_score()

        # Store in session metadata
        context.session_metadata["personality_traits_detected"] = dominant_traits
        context.session_metadata["personality_confidence"] = personality_confidence
        context.session_metadata["personality_summary"] = context.personality_analyzer.get_trait_summary()

        logging.debug(f"Personality analysis: {len(dominant_traits)} traits detected "
                     f"(confidence: {personality_confidence:.2f})")

        # Update session metadata with progress tracking
        question_type_value = question.question_type.value if question.question_type else "unknown"
        context.session_metadata.update({
            "last_question_type": question_type_value,
            "last_response_sentiment": analysis.sentiment,
            "last_response_completeness": analysis.completeness_score,
            "total_questions_asked": len(context.question_history),
            "questions_remaining": self.MAX_QUESTIONS - len(context.question_history),
            "last_update_timestamp": datetime.now().isoformat()
        })

        return context

    def _analyze_response_for_adaptation(self, user_response: str, context: ConversationContext) -> Dict[str, Any]:
        """Analyze user response to guide future question adaptation."""
        response_lower = user_response.lower()
        insights = {}

        # Detect response depth and detail level
        word_count = len(user_response.split())
        insights["response_word_count"] = word_count
        insights["response_detail_level"] = "detailed" if word_count > 30 else "moderate" if word_count > 10 else "brief"

        # Identify specific interests mentioned
        interests_mentioned = []
        interest_keywords = {
            "technology": ["tech", "computer", "programming", "software", "coding", "digital", "data"],
            "healthcare": ["health", "medical", "doctor", "nurse", "care", "patient", "medicine"],
            "education": ["teach", "education", "school", "student", "learn", "training"],
            "business": ["business", "management", "sales", "marketing", "finance", "entrepreneurship"],
            "creative": ["art", "design", "creative", "writing", "music", "innovation"],
            "science": ["science", "research", "biology", "chemistry", "physics", "lab"],
            "engineering": ["engineering", "build", "design", "technical", "systems"]
        }

        for interest, keywords in interest_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                interests_mentioned.append(interest)

        insights["interests_mentioned"] = interests_mentioned

        # Detect skill areas referenced
        skills_mentioned = []
        skill_indicators = {
            "analytical": ["analysis", "research", "problem", "solve", "think", "logic"],
            "creative": ["creative", "design", "art", "innovative", "original"],
            "social": ["people", "team", "communicate", "help", "social"],
            "technical": ["technical", "computer", "programming", "tools", "software"],
            "leadership": ["lead", "manage", "organize", "coordinate", "direct"]
        }

        for skill, indicators in skill_indicators.items():
            if any(indicator in response_lower for indicator in indicators):
                skills_mentioned.append(skill)

        insights["skills_mentioned"] = skills_mentioned

        # Detect enthusiasm level
        enthusiasm_indicators = ["love", "enjoy", "excited", "passionate", "interested", "fascinated"]
        enthusiasm_count = sum(1 for indicator in enthusiasm_indicators if indicator in response_lower)
        insights["enthusiasm_level"] = "high" if enthusiasm_count >= 2 else "moderate" if enthusiasm_count >= 1 else "low"

        # NEW: Analyze student's tone and energy for adaptive responses
        tone_analysis = self._analyze_student_tone(user_response, context)
        insights.update(tone_analysis)

        return insights

    def _analyze_student_tone(self, user_response: str, context: ConversationContext) -> Dict[str, Any]:
        """
        Analyze the student's communication style and tone to enable adaptive responses.
        This helps the agent mirror their energy and match their personality.
        """
        response_lower = user_response.lower()
        tone_insights = {}

        # 1. Energy Level Detection
        high_energy_indicators = ["!", "love", "awesome", "amazing", "super", "really", "so", "very", "excited"]
        low_energy_indicators = ["i guess", "maybe", "i don't know", "not sure", "kind of", "sort of"]

        high_energy_count = sum(1 for indicator in high_energy_indicators if indicator in user_response or indicator in response_lower)
        low_energy_count = sum(1 for indicator in low_energy_indicators if indicator in response_lower)

        if high_energy_count > low_energy_count and high_energy_count >= 2:
            tone_insights["energy_level"] = "enthusiastic"
        elif low_energy_count > high_energy_count:
            tone_insights["energy_level"] = "reserved"
        else:
            tone_insights["energy_level"] = "balanced"

        # 2. Formality Level
        casual_indicators = ["yeah", "gonna", "wanna", "kinda", "sorta", "cool", "stuff"]
        formal_indicators = ["however", "therefore", "furthermore", "indeed", "particularly"]

        casual_count = sum(1 for indicator in casual_indicators if indicator in response_lower)
        formal_count = sum(1 for indicator in formal_indicators if indicator in response_lower)

        if casual_count > formal_count:
            tone_insights["formality"] = "casual"
        elif formal_count > casual_count:
            tone_insights["formality"] = "formal"
        else:
            tone_insights["formality"] = "neutral"

        # 3. Confidence Level
        confident_indicators = ["i know", "i'm good at", "definitely", "absolutely", "for sure"]
        uncertain_indicators = ["i guess", "maybe", "not sure", "i think", "possibly", "i'm not great"]

        confident_count = sum(1 for indicator in confident_indicators if indicator in response_lower)
        uncertain_count = sum(1 for indicator in uncertain_indicators if indicator in response_lower)

        if confident_count > uncertain_count:
            tone_insights["confidence_level"] = "confident"
        elif uncertain_count > confident_count:
            tone_insights["confidence_level"] = "uncertain"
        else:
            tone_insights["confidence_level"] = "moderate"

        # 4. Emotional State
        positive_indicators = ["happy", "enjoy", "love", "excited", "fun", "interesting"]
        negative_indicators = ["worried", "stressed", "concerned", "difficult", "struggle", "hard"]

        positive_count = sum(1 for indicator in positive_indicators if indicator in response_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in response_lower)

        if positive_count > negative_count and positive_count >= 1:
            tone_insights["emotional_state"] = "positive"
        elif negative_count > positive_count:
            tone_insights["emotional_state"] = "concerned"
        else:
            tone_insights["emotional_state"] = "neutral"

        # 5. Response Length Pattern (Brief vs Elaborate)
        word_count = len(user_response.split())
        if word_count < 10:
            tone_insights["response_style"] = "brief"
        elif word_count > 40:
            tone_insights["response_style"] = "elaborate"
        else:
            tone_insights["response_style"] = "moderate"

        # 6. Question Asking Behavior (Are they asking questions?)
        has_question = "?" in user_response or any(q in response_lower for q in ["right?", "correct?", "is that", "do you think"])
        tone_insights["asks_questions"] = has_question

        # 7. Create Adaptive Response Guidance
        # This provides guidance on how the agent should respond
        if tone_insights["energy_level"] == "enthusiastic":
            tone_insights["suggested_personality"] = "encouraging_and_energetic"
        elif tone_insights["confidence_level"] == "uncertain":
            tone_insights["suggested_personality"] = "reassuring_and_patient"
        elif tone_insights["energy_level"] == "reserved":
            tone_insights["suggested_personality"] = "calm_and_thoughtful"
        elif tone_insights["emotional_state"] == "concerned":
            tone_insights["suggested_personality"] = "empathetic_and_supportive"
        else:
            tone_insights["suggested_personality"] = "balanced_and_curious"

        return tone_insights

    def _extract_additional_data_from_response(self, user_response: str, context: ConversationContext) -> Dict[str, Any]:
        """Extract additional structured data from user responses."""
        additional_data = {}
        response_lower = user_response.lower()

        # Extract specific programs, tools, or technologies mentioned
        tech_tools = ["python", "java", "excel", "photoshop", "autocad", "sql", "javascript"]
        tools_mentioned = [tool for tool in tech_tools if tool in response_lower]
        if tools_mentioned:
            additional_data["tools_mentioned"] = tools_mentioned

        # Extract specific industries or companies mentioned
        industries = ["healthcare", "finance", "technology", "education", "nonprofit", "government", "startup"]
        industries_mentioned = [industry for industry in industries if industry in response_lower]
        if industries_mentioned:
            additional_data["industries_mentioned"] = industries_mentioned

        # Extract specific career titles mentioned
        career_titles = ["engineer", "teacher", "manager", "designer", "analyst", "developer", "consultant"]
        careers_mentioned = [career for career in career_titles if career in response_lower]
        if careers_mentioned:
            additional_data["careers_mentioned"] = careers_mentioned

        return additional_data

    def _identify_follow_up_opportunities(self, user_response: str, context: ConversationContext) -> List[str]:
        """Identify opportunities for valuable follow-up questions based on response content."""
        follow_ups = []
        response_lower = user_response.lower()

        # Look for vague or general statements that could benefit from specificity
        vague_indicators = ["kind of", "sort of", "maybe", "i guess", "probably", "sometimes"]
        if any(indicator in response_lower for indicator in vague_indicators):
            follow_ups.append("needs_clarification")

        # Look for mentions of multiple options that could be prioritized
        if "and" in response_lower and len(response_lower.split("and")) > 3:
            follow_ups.append("needs_prioritization")

        # Look for passionate language that suggests deeper exploration
        passion_indicators = ["love", "passionate", "excited", "fascinated"]
        if any(indicator in response_lower for indicator in passion_indicators):
            follow_ups.append("explore_passion")

        # Look for experience mentions that could be detailed
        experience_indicators = ["worked", "experience", "did", "used", "learned", "studied"]
        if any(indicator in response_lower for indicator in experience_indicators):
            follow_ups.append("detail_experience")

        return follow_ups

    def should_ask_follow_up(self, context: ConversationContext, analysis: ResponseAnalysis) -> bool:
        """
        Determine if a follow-up question should be asked before proceeding.
        Simplified to avoid exceeding question limit.
        """
        # Don't ask follow-ups if close to question limit
        questions_asked = len(context.question_history)
        if questions_asked >= (self.MAX_QUESTIONS - 2):  # Save room for remaining questions
            return False

        # Ask follow-up if:
        # 1. Response completeness is very low (< 0.3)
        # 2. There are critical follow-up needs identified

        if analysis.completeness_score < 0.3:
            return True

        if analysis.follow_up_needed and questions_asked < 8:  # Only early in conversation
            return True

        return False

    def should_ask_dynamic_follow_up(
        self,
        context: ConversationContext,
        analysis: ResponseAnalysis
    ) -> bool:
        """
        Determine if a dynamic follow-up is warranted based on response depth and specificity.
        This is separate from the basic should_ask_follow_up() method.

        Dynamic follow-up triggers:
        - High interest + low specificity (depth_score > 0.7 but specificity < 0.5)
        - Passion indicators detected
        - Multiple interests without prioritization
        - Follow-up type is "expansion" or "exploration"

        Budget constraints:
        - Only if questions_asked < 10 (save 2 questions for RIASEC balance)
        - Limit to 2-3 dynamic follow-ups per session
        """
        questions_asked = len(context.question_history)
        dynamic_followups_used = context.session_metadata.get("dynamic_followups_count", 0)

        # Budget constraints
        if questions_asked >= 10:  # Save room for final RIASEC questions
            return False

        if dynamic_followups_used >= 3:  # Max 3 dynamic follow-ups per session
            return False

        # High-value follow-up triggers
        if hasattr(analysis, 'follow_up_type') and analysis.follow_up_type in ["expansion", "exploration"]:
            # Check if depth/specificity warrant follow-up
            depth_score = getattr(analysis, 'depth_score', 0.5)
            specificity_score = getattr(analysis, 'specificity_score', 0.5)

            # High depth but low specificity → expand on interesting topic
            if depth_score > 0.7 and specificity_score < 0.5:
                logging.debug(f"Dynamic follow-up triggered: high depth ({depth_score:.2f}), "
                             f"low specificity ({specificity_score:.2f})")
                return True

            # Exploration type with reasonable depth
            if analysis.follow_up_type == "exploration" and depth_score > 0.5:
                logging.debug(f"Dynamic follow-up triggered: exploration type with depth {depth_score:.2f}")
                return True

        return False

    def _get_follow_up_type_guidance(self, follow_up_type: Optional[str]) -> str:
        """Get specific guidance based on follow-up type."""
        if follow_up_type == "clarification":
            return " (Ask them to clarify or give specific examples)"
        elif follow_up_type == "expansion":
            return " (Dig deeper - ask 'what specifically' or 'why that appeals')"
        elif follow_up_type == "exploration":
            return " (Explore their passion - ask about experiences, what drew them in, favorite aspects)"
        else:
            return ""

    def generate_follow_up_question(self, context: ConversationContext, analysis: ResponseAnalysis) -> GeneratedQuestion:
        """
        Generate a targeted follow-up question based on the analysis.
        Enhanced to use dynamic follow-up fields (key_concepts, follow_up_focus, follow_up_type).
        """
        # NEW: Check for dynamic follow-up fields
        follow_up_type = getattr(analysis, 'follow_up_type', None)
        follow_up_focus = getattr(analysis, 'follow_up_focus', None)
        key_concepts = getattr(analysis, 'key_concepts', [])

        # Identify the most critical area needing follow-up
        # Prioritize follow_up_focus if available
        follow_up_area = follow_up_focus if follow_up_focus else None
        if not follow_up_area and analysis.follow_up_needed:
            follow_up_area = analysis.follow_up_needed[0]
        if not follow_up_area:
            follow_up_area = "general"

        # Extract key insights from their response
        # NEW: Use key_concepts if available (broader themes, NOT exact phrases)
        insights_context = ""
        if key_concepts:
            # Include broader concepts to provide context for paraphrasing
            insights_context = f"\n\nKEY THEMES IDENTIFIED: {', '.join(key_concepts[:3])}\n(Note: These are broader concepts - use your own words, don't copy user's exact phrases)"
        elif hasattr(analysis, 'key_insights') and analysis.key_insights:
            insights_context = f"\nKey insights from their response: {', '.join(analysis.key_insights[:3])}"

        # Get student's tone for adaptive response
        tone_context = ""
        latest_tone = context.session_metadata.get("energy_level", "balanced")
        confidence = context.session_metadata.get("confidence_level", "moderate")
        suggested_personality = context.session_metadata.get("suggested_personality", "balanced_and_curious")

        tone_context = f"\n\nSTUDENT'S COMMUNICATION STYLE:\n- Energy: {latest_tone}\n- Confidence: {confidence}\n- Match this tone: {suggested_personality}"

        # Get language and build STRONG enforcement
        language = context.session_metadata.get("language", "en")

        if language == "si":
            language_instruction = """=== CRITICAL LANGUAGE REQUIREMENT ===
🚨 YOUR FOLLOW-UP MUST BE IN SINHALA (සිංහල) 🚨

THE USER JUST RESPONDED (MAYBE IN ENGLISH) - YOUR FOLLOW-UP IS STILL IN SINHALA.
IGNORE THEIR INPUT LANGUAGE. RESPOND IN SINHALA ALWAYS.
==========================================

"""
        else:
            language_instruction = ""

        # Get random acknowledgment from message pool for variety
        random_acknowledgment = self.message_pool.get_random_acknowledgment(language)

        prompt = f"""{language_instruction}You are a natural, adaptive career counselor having a real conversation with a HIGH SCHOOL STUDENT. They just shared something that needs more exploration. Dig deeper in a genuine, conversational way.{tone_context}

THEIR RESPONSE: "{context.response_history[-1] if context.response_history else 'None'}"

WHAT NEEDS MORE INFO: {follow_up_area}{insights_context}

🚨 CRITICAL RESTRICTION:
- NEVER ask directly about "career", "job", "profession", or "what you want to be"
- INSTEAD ask about: interests, motivations, what they enjoy, how they work, their strengths
- Let the AI system INFER career matches - your job is to understand their PERSONALITY and INTERESTS

YOUR APPROACH - Respond naturally using this structure:

1. ACKNOWLEDGE (1 sentence)
   • START WITH THIS EXACT PHRASE: "{random_acknowledgment}"
   • Then naturally connect it to what they said
   • Match their tone and energy level

2. CONNECT (1 sentence)
   • Reference what they shared, but PARAPHRASE naturally
   • Show you're tracking the conversation by connecting to the MEANING, not copying words
   • Transform their phrases into your own natural language
   • Example: "React and Flask" → "full-stack work" or "backend and frontend"

3. DIG DEEPER (1-2 sentences)
   • Ask a natural follow-up about {follow_up_area}{self._get_follow_up_type_guidance(follow_up_type)}
   • Show authentic curiosity
   • Make it conversational, not interrogative
   • PARAPHRASE their key concepts - don't copy their exact technical terms{f'. Themes: {", ".join(key_concepts[:2])}' if key_concepts else ''}

IMPORTANT:
- Do NOT include section labels
- Flow naturally from thought to thought
- Start with the acknowledgment phrase provided above

EXAMPLE (using different acknowledgment):

Their response: "I guess I like art. Is that useful for jobs?"
Acknowledgment provided: "I can see why that resonates with you."

Your response: "I can see why that resonates with you - creative skills are in demand across many fields.

It sounds like art's something you're drawn to.

What kind of creative work do you find yourself doing? Are you more into traditional stuff like drawing, or do you mess around with digital design?"

Generate your natural follow-up response:"""

        try:
            response = self.llm_wrapper.invoke([HumanMessage(content=prompt)])

            if not response or not response.content:
                raise ValueError("Empty response from LLM")

            question_text = response.content.strip()

            # Strip structural markers that should not appear in user-facing output
            question_text = self._strip_structural_markers(question_text)

            # CRITICAL: Remove any "undefined" text that might be appended
            # Use robust regex to catch all variations (case-insensitive, multiple occurrences, different positions)
            original_text = question_text
            question_text = re.sub(r'\s*undefined\s*', ' ', question_text, flags=re.IGNORECASE)
            question_text = re.sub(r'\s+', ' ', question_text).strip()  # Clean up multiple spaces

            if original_text != question_text:
                logging.warning(f"Removed 'undefined' from question. Original length: {len(original_text)}, New length: {len(question_text)}")

            # Enforce single question
            question_text = self._ensure_single_question(question_text)

            # CRITICAL: Fix cut-off beginnings - ensure proper sentence start
            question_text = self._fix_question_beginning(question_text)

            # Ensure the question is valid
            if not question_text or len(question_text) < 10:
                raise ValueError("Follow-up question too short or empty")

        except Exception as e:
            logging.error(f"Follow-up question generation failed: {type(e).__name__}: {str(e)}. Using fallback question.")

            # Professional fallback follow-up questions
            if follow_up_area:
                area_prompts = {
                    "academic_background": "Could you elaborate on your educational experience and what specific aspects have been most valuable or interesting to you?",
                    "career_interests": "I'd like to understand more about what draws you to this area. What specific aspects or activities in this field appeal to you most?",
                    "skills": "That's interesting - could you tell me more about how you developed this ability and in what situations you've found it most useful?",
                    "technical_skills": "Could you provide more detail about your experience with this? What specific tools, methods, or applications have you worked with?",
                    "work_preferences": "Help me understand what makes this appealing to you. What would an ideal day doing this type of work look like?",
                    "personality": "Could you give me more insight into your working style and what environments or situations bring out your best performance?"
                }
                question_text = area_prompts.get(follow_up_area, "Could you provide more specific details about what you mentioned? I'd like to better understand this aspect of your background for career planning purposes.")
            else:
                question_text = "I'd like to explore this topic further - could you share more specific details or examples that would help me better understand your interests and capabilities?"

        return GeneratedQuestion(
            question_text=question_text,
            question_type=QuestionType.FOLLOW_UP,
            target_area=follow_up_area or "general",
            expected_insights=[],
            follow_up_triggers=[],
            riasec_relevance={}
        )

    def create_session_summary(self, context: ConversationContext) -> Dict[str, Any]:
        """
        Create a comprehensive summary of the conversation session.
        """
        return {
            "session_metadata": context.session_metadata,
            "final_state": context.current_state.name,
            "total_questions": len(context.question_history),
            "riasec_scores": {cat.value: score for cat, score in context.riasec_scores.items()},
            "dominant_riasec": max(context.riasec_scores.items(), key=lambda x: x[1]) if context.riasec_scores else None,
            "collected_data": context.collected_data,
            "riasec_question_distribution": {cat.value: count for cat, count in context.riasec_question_count.items()}
        }
