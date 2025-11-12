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
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage

from agents.conversation_states import (
    ConversationState, ConversationContext, RIASECCategory,
    ConversationStateMachine, create_initial_conversation_context
)

# Import LangSmith configuration
from utils.langsmith_config import get_traced_run_config, log_agent_execution, setup_langsmith

# Import LLM factory for fallback support
from utils.llm_factory import LLMFactory

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
    """
    riasec_updates: Dict[RIASECCategory, float]  # Updates to RIASEC scores
    extracted_data: Dict[str, Any]  # Extracted structured data
    follow_up_needed: List[str]  # Areas needing follow-up
    intent_classification: str  # Classified intent
    sentiment: str  # Response sentiment
    completeness_score: float  # How complete the response is (0-1)


class ConversationManager:
    """
    Advanced conversation manager for intelligent Q&A flow.

    Implements Chain-of-Thought reasoning, Few-Shot prompting,
    and sophisticated intent recognition with question limiting.
    """

    # Maximum number of questions to ask - Exactly 12 questions for deterministic flow
    MAX_QUESTIONS = 12  # Fixed: 2 questions per RIASEC category (R, I, A, S, E, C)

    # Minimum questions to ensure adequate assessment
    MIN_QUESTIONS = 6

    # Maximum questions per RIASEC category for balanced assessment
    MAX_QUESTIONS_PER_RIASEC = 2  # Exactly 2 questions per category

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3):
        # Use LLM factory with fallback support instead of direct ChatOpenAI
        self.llm_wrapper = LLMFactory.create_llm(
            model=model,
            temperature=temperature,
            enable_fallback=True
        )
        self.llm = self.llm_wrapper.llm
        logging.info(f"✅ ConversationManager LLM initialized using {self.llm_wrapper.strategy.provider_name}")

        self.state_machine = ConversationStateMachine()
        self.question_bank = self._initialize_question_bank()
        self.riasec_keywords = self._initialize_riasec_keywords()
        self.few_shot_examples = self._initialize_few_shot_examples()

    def _initialize_question_bank(self) -> Dict[ConversationState, List[Dict[str, Any]]]:
        """Initialize curated question bank for each conversation state with simplified, accessible questions."""
        return {
            ConversationState.GREETING: [
                # No predefined greeting - will be generated dynamically
            ],

            ConversationState.ACADEMIC_GATHERING: [
                {
                    "question": "Tell me about your experience with school so far. What subjects or topics have really clicked for you?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "academic_background",
                    "follow_up_triggers": ["incomplete_info", "career_change", "multiple_fields"],
                    "priority": 1
                },
                {
                    "question": "When you're learning something new, what helps it stick? Do you prefer hands-on practice, working through problems, creative projects, or discussing ideas with others?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "learning_style",
                    "follow_up_triggers": ["specific_methods", "challenges", "preferences"],
                    "priority": 1
                }
            ],

            ConversationState.INTEREST_DISCOVERY: [
                {
                    "question": "Outside of school, what kinds of activities or hobbies really draw you in? What makes them appealing?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "work_preferences",
                    "follow_up_triggers": ["general_response", "multiple_activities", "unclear"],
                    "priority": 1
                },
                {
                    "question": "Think about times when you've felt really engaged and energized. Was it working independently on something, or collaborating with others? What made that experience work for you?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "work_style",
                    "follow_up_triggers": ["combination", "uncertain", "strong_preference"],
                    "priority": 1
                },
                {
                    "question": "When you picture yourself in a work environment, what kind of setting appeals to you? What atmosphere helps you do your best work?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "industry_interests",
                    "follow_up_triggers": ["multiple_industries", "superficial_knowledge", "specific_companies"],
                    "priority": 1
                },
                {
                    "question": "Everyone values different things in their work. What matters most to you when you think about your future career - is it making an impact, creative freedom, financial stability, something else?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "career_values",
                    "follow_up_triggers": ["multiple_motivations", "unclear_values", "specific_impact"],
                    "priority": 1
                }
            ],

            ConversationState.SKILLS_ASSESSMENT: [
                {
                    "question": "Let's talk about your strengths. What comes naturally to you that others might find challenging? What do people tend to come to you for?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "technical_skills",
                    "follow_up_triggers": ["limited_skills", "advanced_skills", "self_taught"],
                    "priority": 1
                },
                {
                    "question": "When you're faced with a challenge or problem, how do you typically approach it? Walk me through your process.",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "problem_solving",
                    "follow_up_triggers": ["analytical", "creative", "collaborative"],
                    "priority": 1
                }
            ],

            ConversationState.PERSONALITY_PROBE: [
                {
                    "question": "Some people recharge by being around others, while others need solo time to refuel. What's your pattern? When do you feel most like yourself?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "introversion_extraversion",
                    "follow_up_triggers": ["extreme_preference", "situational", "unclear"],
                    "priority": 1
                },
                {
                    "question": "Think about a time recently when you felt really satisfied or accomplished. What were you doing, and what about that moment made it meaningful for you?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "core_motivation",
                    "follow_up_triggers": ["multiple_motivations", "unclear", "specific_examples"],
                    "priority": 1
                }
            ]
        }

    def _initialize_riasec_keywords(self) -> Dict[RIASECCategory, List[str]]:
        """Initialize RIASEC category keywords for scoring."""
        return {
            RIASECCategory.REALISTIC: [
                "hands-on", "practical", "mechanical", "building", "tools", "outdoors",
                "physical", "construction", "repair", "engineering", "technical", "tangible"
            ],
            RIASECCategory.INVESTIGATIVE: [
                "research", "analyze", "investigate", "scientific", "data", "theory",
                "logical", "experimental", "academic", "intellectual", "problem-solving", "study"
            ],
            RIASECCategory.ARTISTIC: [
                "creative", "design", "artistic", "imaginative", "aesthetic", "visual",
                "music", "writing", "innovative", "expressive", "original", "beauty"
            ],
            RIASECCategory.SOCIAL: [
                "helping", "teaching", "caring", "people", "service", "community",
                "counseling", "teamwork", "communication", "mentoring", "support", "relationships"
            ],
            RIASECCategory.ENTERPRISING: [
                "leadership", "management", "business", "sales", "persuading", "competitive",
                "entrepreneurial", "ambitious", "influence", "negotiate", "risk-taking", "profit"
            ],
            RIASECCategory.CONVENTIONAL: [
                "organized", "systematic", "detail-oriented", "procedures", "administrative",
                "structured", "accuracy", "routine", "data-entry", "planning", "methodical", "efficient"
            ]
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

        # Get available questions with priority consideration
        available_questions = self._get_available_questions_by_priority(context)

        if available_questions:
            # Select best question based on priority and context
            selected_question = self._select_best_question(available_questions, context)
            result = self._convert_to_generated_question(selected_question, context)
        else:
            # Only generate dynamic questions if we haven't reached limit and no predefined questions available
            result = self._generate_dynamic_question(context)
        
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

    def _select_best_question(self, questions: List[Dict[str, Any]], context: ConversationContext) -> Dict[str, Any]:
        """Select the best question based on current context, confidence gaps, and conversation flow."""

        # Enhanced question selection with multiple criteria
        scored_questions = []

        for question in questions:
            score = 0
            target_area = question.get("target_area", "")
            question_text = question.get("question", "")

            # 1. Priority scoring (highest weight)
            priority = question.get("priority", 3)
            score += (4 - priority) * 30  # Priority 1 = 90 points, Priority 2 = 60 points

            # 2. Context relevance scoring
            if context.response_history:
                last_response = context.response_history[-1].lower()

                # Check for topic continuity - avoid jumping topics abruptly
                question_keywords = self._extract_question_keywords(question_text)
                response_keywords = self._extract_response_keywords(last_response)

                keyword_overlap = len(set(question_keywords) & set(response_keywords))
                if keyword_overlap > 0:
                    score += keyword_overlap * 5  # Bonus for topical continuity

                # Avoid repetitive question patterns
                if any(self._questions_are_similar(question_text, prev_q)
                       for prev_q in context.question_history[-2:]):
                    score -= 20  # Penalty for similar questions

            # 3. RIASEC-based scoring - prioritize balanced coverage
            riasec_relevance = self._calculate_question_riasec_relevance(question_text)
            priority_categories = self._get_riasec_priority_categories(context)

            # Give high priority to questions that address under-explored RIASEC categories
            for category, relevance in riasec_relevance.items():
                if category in priority_categories[:2]:  # Top 2 priority categories
                    score += relevance * 40  # High bonus for priority RIASEC categories
                elif category in priority_categories:
                    score += relevance * 20  # Medium bonus for other priority categories

            # 4. Conversation flow scoring
            questions_asked = len(context.question_history)

            # Early conversation: prioritize foundational questions
            if questions_asked <= 3 and "background" in target_area:
                score += 10

            # Mid conversation: prioritize interest and skills
            elif 4 <= questions_asked <= 8 and target_area in ["work_preferences", "technical_skills", "work_style"]:
                score += 10

            # Later conversation: prioritize personality and values
            elif questions_asked > 8 and target_area in ["core_motivation", "work_environment_preferences"]:
                score += 10

            # 5. Response depth adaptation
            if context.response_history:
                last_analysis = context.session_metadata.get("last_response_completeness", 0.5)

                # If last response was shallow, ask a more engaging question
                if last_analysis < 0.4 and "motivation" in target_area:
                    score += 15

                # If responses are detailed, we can ask more specific questions
                elif last_analysis > 0.7:
                    score += 5

            scored_questions.append((question, score))

        # Sort by score (highest first) and return the best question
        scored_questions.sort(key=lambda x: x[1], reverse=True)

        # Return the highest-scoring question
        return scored_questions[0][0] if scored_questions else questions[0]

    def _extract_question_keywords(self, question_text: str) -> List[str]:
        """Extract key topics from a question for context matching."""
        # Simple keyword extraction - focus on career-relevant terms
        keywords = []
        career_terms = [
            "academic", "study", "education", "learning", "subjects",
            "work", "career", "job", "activities", "interests", "industry", "companies",
            "skills", "abilities", "talents", "problem", "solving", "teamwork", "leadership",
            "motivation", "values", "goals", "environment", "preferences", "style"
        ]

        question_lower = question_text.lower()
        for term in career_terms:
            if term in question_lower:
                keywords.append(term)

        return keywords

    def _extract_response_keywords(self, response_text: str) -> List[str]:
        """Extract topics mentioned in user responses for context matching."""
        # Extract meaningful words from user responses
        response_lower = response_text.lower()
        keywords = []

        # Look for specific topics users commonly mention
        topic_indicators = {
            "academic": ["study", "school", "college", "university", "degree", "major", "course"],
            "technical": ["computer", "software", "programming", "code", "technology", "data"],
            "creative": ["design", "art", "creative", "writing", "music", "innovation"],
            "social": ["people", "team", "help", "teach", "communicate", "social"],
            "leadership": ["lead", "manage", "organize", "coordinate", "direct"],
            "analytical": ["analysis", "research", "solve", "think", "logic", "math"]
        }

        for topic, indicators in topic_indicators.items():
            if any(indicator in response_lower for indicator in indicators):
                keywords.append(topic)

        return keywords

    def _questions_are_similar(self, question1: str, question2: str) -> bool:
        """Check if two questions are asking about similar topics."""
        if not question1 or not question2:
            return False

        q1_keywords = set(self._extract_question_keywords(question1))
        q2_keywords = set(self._extract_question_keywords(question2))

        # Consider questions similar if they share 2+ keywords
        overlap = len(q1_keywords & q2_keywords)
        return overlap >= 2

    def _convert_to_generated_question(self, question_dict: Dict[str, Any], context: ConversationContext) -> GeneratedQuestion:
        """Convert question dictionary to GeneratedQuestion object."""
        return GeneratedQuestion(
            question_text=question_dict["question"],
            question_type=QuestionType(question_dict.get("type", QuestionType.OPEN_ENDED)),
            target_area=question_dict.get("target_area", "general"),
            expected_insights=question_dict.get("expected_insights", []),
            follow_up_triggers=question_dict.get("follow_up_triggers", []),
            riasec_relevance=self._calculate_question_riasec_relevance(question_dict["question"])
        )

    def _generate_dynamic_question(self, context: ConversationContext) -> GeneratedQuestion:
        """Generate a dynamic question using LLM when pre-defined questions are exhausted."""

        # Special handling for greeting state
        if context.current_state == ConversationState.GREETING:
            return self._generate_dynamic_greeting(context)

        state_objective = self.state_machine.get_state_objective(context.current_state)
        questions_asked = len(context.question_history)
        questions_remaining = 12 - questions_asked

        # Build context from recent conversation
        recent_context = ""
        if context.response_history:
            last_response = context.response_history[-1]
            recent_context = f"\nTheir most recent response: \"{last_response}\""

        if len(context.response_history) >= 2:
            previous_response = context.response_history[-2]
            recent_context += f"\nPrevious response: \"{previous_response}\""

        # Get student's tone and energy to adapt response
        tone_context = ""
        if context.response_history:
            latest_tone = context.session_metadata.get("energy_level", "balanced")
            confidence = context.session_metadata.get("confidence_level", "moderate")
            suggested_personality = context.session_metadata.get("suggested_personality", "balanced_and_curious")

            tone_context = f"\n\nSTUDENT'S COMMUNICATION STYLE:\n- Energy: {latest_tone}\n- Confidence: {confidence}\n- Match this tone: {suggested_personality}"

        prompt = f"""You are a natural, adaptive career counselor having a real conversation with a HIGH SCHOOL STUDENT. Your responses should feel like talking to a supportive mentor, not a robot.{tone_context}

GOAL: {state_objective}

THEIR RECENT RESPONSE:{recent_context}
Questions asked: {questions_asked} of 12-15 total

YOUR APPROACH - Respond naturally with 3 parts:

1. ACKNOWLEDGE & VALIDATE (1-2 sentences)
   • If they asked a question, answer it directly and genuinely
   • React naturally to what they shared - match their energy
   • Be authentic - avoid repeating the same phrases

2. CONNECT (1 sentence)
   • Naturally link what they just said to where you're going next
   • Reference specific things they mentioned
   • Show you're actively listening

3. ASK (1-2 sentences)
   • Ask your next question about {state_objective.lower()}
   • Use their language and examples
   • Keep it conversational and high school appropriate

IMPORTANT: Do NOT include section labels like [VALIDATION], [BRIDGE], or [QUESTION] in your response. Flow naturally.

EXAMPLE (one of many possible styles):

Their response: "I love organizing events. It felt really impactful, you know?"

Your response: "Absolutely! There's something really satisfying about pulling together an event and watching it all come together. That takes real planning skills.

It sounds like you get a kick out of seeing your ideas actually happen in the real world.

When you imagine your ideal career, what kind of impact are you hoping to make? Are you thinking more short-term wins like events, or are you also drawn to longer-term projects?"

HOW TO SOUND NATURAL (Vary your language!):
• Match their enthusiasm level - if they're excited, be excited; if they're thoughtful, be thoughtful
• Use different validation phrases naturally - don't repeat yourself
• Reference specific details they mentioned
• Sound like a real person having a conversation, not following a script
• High school language level - clear and simple without being condescending

Generate your natural, conversational response:"""

        try:
            response = self.llm_wrapper.invoke([HumanMessage(content=prompt)])

            if not response or not response.content:
                raise ValueError("Empty response from LLM")

            question_text = response.content.strip()

            # Strip structural markers that should not appear in user-facing output
            question_text = self._strip_structural_markers(question_text)

            # Ensure single question only (for high school student clarity)
            question_text = self._ensure_single_question(question_text)

            # Ensure the question is valid
            if not question_text or len(question_text) < 10:
                raise ValueError("Question too short or empty")

        except Exception as e:
            logging.error(f"Dynamic question generation failed: {type(e).__name__}: {str(e)}. Using fallback question.")

            # Multiple varied fallback questions - rotate to avoid repetition
            import random
            state_name = context.current_state.name.lower()
            fallback_questions = {
                "academic_gathering": [
                    "What subject or topic in school has really clicked with you so far?",
                    "Tell me about something you're learning that genuinely interests you.",
                    "Is there a class or subject area where things just make sense to you?",
                    "What topic or area of study do you find yourself actually enjoying?",
                    "When you're learning something new, what type of content tends to grab your attention?"
                ],
                "interest_discovery": [
                    "What kinds of things do you do that make you lose track of time?",
                    "Outside of school requirements, what activities or interests pull you in?",
                    "What do you find yourself naturally drawn to when you have free time?",
                    "Is there something you do where you feel genuinely engaged and energized?",
                    "What activities or experiences have felt meaningful or satisfying to you?"
                ],
                "skills_assessment": [
                    "What's something that comes naturally to you that others might struggle with?",
                    "Is there an area where you feel particularly capable or confident?",
                    "What do people tend to come to you for help with?",
                    "What skills or abilities do you have that you feel good about?",
                    "When have you felt really competent at something? What was it?"
                ],
                "personality_probe": [
                    "How would your friends describe your personality or working style?",
                    "What kind of environment or situation brings out your best?",
                    "When do you feel most like yourself - in what kinds of settings?",
                    "What matters most to you when you're working on something?",
                    "How do you tend to approach new challenges or problems?"
                ]
            }

            # Get random question from the appropriate category
            questions_list = fallback_questions.get(state_name, [
                "What kind of career or work environment feels like it might be a good match for you?",
                "What draws you to certain types of work or activities?",
                "What are you hoping to find in a future career?",
                "When you think about your future, what matters most to you?",
                "What kind of impact or contribution would you like to make?"
            ])
            question_text = random.choice(questions_list)

        return GeneratedQuestion(
            question_text=question_text,
            question_type=QuestionType.OPEN_ENDED,
            target_area="dynamic_generation",
            expected_insights=[],
            follow_up_triggers=[],
            riasec_relevance=self._calculate_question_riasec_relevance(question_text)
        )

    def _generate_dynamic_greeting(self, context: ConversationContext) -> GeneratedQuestion:
        """Generate a dynamic, personalized greeting message using LLM."""

        session_id = context.session_metadata.get('session_id', 'unknown')
        current_time = datetime.now().strftime('%B %d, %Y')

        prompt = f"""You are a friendly career counselor starting a natural conversation with a HIGH SCHOOL STUDENT. Create a warm, genuine opening that feels like talking to a supportive mentor.

CONTEXT:
- You're talking with a high school student about their future
- Be welcoming but authentic - not overly peppy or robotic
- Make them feel comfortable and understood
- 2-3 sentences + ONE question

TASK: Write a natural opening that:
1. Introduces yourself simply as their career counselor
2. Shows genuine interest in helping them
3. Feels like a real conversation, not a survey
4. Ends with ONE engaging question

EXAMPLES OF VARIED NATURAL GREETINGS:

- "Hey! I'm here to help you explore what careers might be a good match for you. We'll talk about the things you enjoy, what you're good at, and what matters to you going forward. So what's making you think about careers right now?"

- "Hi there! I work with students to help them discover career paths that actually fit who they are. We're just going to have a conversation about your interests and strengths. What's been on your mind when you think about your future?"

- "Hello! I'm your career counselor, and I'm glad we're doing this. We'll chat about what you care about and what you're naturally drawn to. What brings you here today?"

- "Hey! I help students figure out career directions that make sense for them. We'll just talk through your interests, what you're good at, and where you might want to head. What got you interested in exploring this?"

HOW TO SOUND NATURAL:
• Friendly and approachable, but not fake-enthusiastic
• Vary your word choice - don't sound scripted
• High school appropriate - clear without being overly simple
• Make it feel like the start of a real conversation

AVOID:
- Repeating the same greeting structure
- Sounding like a bot or form letter
- Multiple questions at once
- Overly formal or academic language

Generate a natural, varied greeting:"""

        try:
            response = self.llm_wrapper.invoke([HumanMessage(content=prompt)])
            question_text = response.content.strip()

            # Strip structural markers that should not appear in user-facing output
            question_text = self._strip_structural_markers(question_text)

            # Enforce single question ending
            question_text = self._ensure_greeting_has_single_question(question_text)

            # Multiple varied fallback greetings if LLM response is empty or too short
            import random
            if not question_text or len(question_text) < 20:
                fallback_greetings = [
                    "Hey! I'm here to help you explore career paths that might be a good fit for you. We'll talk about your interests, strengths, and what matters to you. So what's got you thinking about careers right now?",
                    "Hi there! I work with students to help them figure out career directions that make sense for them. We'll just have a conversation about what you enjoy and what you're good at. What brings you here today?",
                    "Hello! I'm your career counselor, and I'm glad we're doing this. We'll chat about your interests, what comes naturally to you, and where you might want to head. What's on your mind about your future?",
                    "Hey! I help students discover career options that actually fit who they are. We'll talk through what you care about and what you're drawn to. What made you want to explore this?",
                    "Hi! I'm here to help you find career paths that could be a good match. We'll discuss your strengths, interests, and goals. So what's making you think about careers today?"
                ]
                question_text = random.choice(fallback_greetings)

        except Exception as e:
            # Multiple varied fallback greetings if LLM fails
            import random
            fallback_greetings = [
                "Hey! I'm here to help you explore career paths that might be a good fit for you. We'll talk about your interests, strengths, and what matters to you. So what's got you thinking about careers right now?",
                "Hi there! I work with students to help them figure out career directions that make sense for them. We'll just have a conversation about what you enjoy and what you're good at. What brings you here today?",
                "Hello! I'm your career counselor, and I'm glad we're doing this. We'll chat about your interests, what comes naturally to you, and where you might want to head. What's on your mind about your future?",
                "Hey! I help students discover career options that actually fit who they are. We'll talk through what you care about and what you're drawn to. What made you want to explore this?"
            ]
            question_text = random.choice(fallback_greetings)

        return GeneratedQuestion(
            question_text=question_text,
            question_type=QuestionType.OPEN_ENDED,
            target_area="initial_engagement",
            expected_insights=["engagement_level", "communication_style", "initial_career_interests"],
            follow_up_triggers=["vague", "uncertain", "multiple_interests"],
            riasec_relevance={}
        )

    def _generate_mid_conversation_summary(self, context: ConversationContext) -> GeneratedQuestion:
        """Generate a mid-conversation summary after 6-7 questions to show active listening."""

        # Gather key insights from responses so far
        responses_summary = "\n".join([
            f"Q{i+1}: {context.question_history[i]}\nA{i+1}: {context.response_history[i]}"
            for i in range(min(len(context.question_history), len(context.response_history)))
        ])

        prompt = f"""You are a GREAT career counselor talking with a HIGH SCHOOL STUDENT. You've asked 6-7 questions so far. Now PAUSE and create a summary to show you're listening.

CONVERSATION SO FAR:
{responses_summary}

YOUR TASK:
Create a MID-CONVERSATION CHECK-IN with 3 parts:

1. TRANSITION (1 sentence)
   • Simple transition to the summary
   • Example: "Let's pause for a quick sec."
   • Example: "Okay, let me make sure I'm understanding you right."

2. SUMMARY (3-5 bullet points)
   • List key things you've learned about them
   • Use THEIR words when possible
   • Keep it simple and clear
   • Example format:
     - You love [specific thing they mentioned]
     - You're good at [specific skill]
     - You value [specific value]

3. VALIDATION QUESTION (1-2 sentences)
   • Ask if you got it right
   • Give them chance to correct or add
   • Example: "Does that sound right? Did I miss anything important?"

COMPLETE EXAMPLE:

"Let's take a quick break here. Let me make sure I've got the full picture so far.

From what you've told me:
- You love working with your soccer team and being active
- You really enjoy making plans and seeing them work out
- You're a hands-on learner - you learn best by actually doing things
- You want to make a real impact and help people

Does that sound right? Is there anything important I missed or got wrong?"

LANGUAGE RULES (HIGH SCHOOL LEVEL):
✓ Use simple, everyday words
✓ Short bullet points
✓ Conversational tone
✓ Reference their specific examples

Generate the mid-conversation summary:"""

        try:
            response = self.llm_wrapper.invoke([HumanMessage(content=prompt)])

            if not response or not response.content:
                raise ValueError("Empty response from LLM")

            summary_text = response.content.strip()

            # Strip structural markers that should not appear in user-facing output
            summary_text = self._strip_structural_markers(summary_text)

            # Mark that we've done the summary
            context._summary_done = True

        except Exception as e:
            logging.error(f"Mid-conversation summary generation failed: {e}. Using fallback.")

            # Simple fallback summary
            summary_text = """Let's pause for a sec. Let me make sure I understand what you've shared:

- Your interests and strengths
- What you enjoy doing
- How you like to work

Does that sound about right? Anything important I missed?"""

            context._summary_done = True

        return GeneratedQuestion(
            question_text=summary_text,
            question_type=QuestionType.OPEN_ENDED,
            target_area="mid_conversation_summary",
            expected_insights=["validation", "clarification"],
            follow_up_triggers=[],
            riasec_relevance={}
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

        # Create analysis prompt with Chain-of-Thought reasoning
        analysis_prompt = f"""You are an expert career counselor analyzing a student's response. Use step-by-step reasoning.

QUESTION ASKED: {question.question_text}
TARGET AREA: {question.target_area}
USER RESPONSE: "{user_response}"

CONVERSATION CONTEXT:
- Current State: {context.current_state.name}
- Current RIASEC Scores: {context.riasec_scores}

FEW-SHOT EXAMPLES:
{self._format_few_shot_examples()}

ANALYSIS TASK:
Step 1: Assess response completeness and quality (0.0-1.0)
Step 2: Extract structured information relevant to career planning
Step 3: Calculate RIASEC category scores based on response content
Step 4: Identify any areas needing follow-up questions
Step 5: Classify the user's intent and sentiment

Provide analysis in this JSON format:
{{
    "completeness_score": 0.0-1.0,
    "extracted_data": {{"key": "value"}},
    "riasec_updates": {{"R": 0.0-1.0, "I": 0.0-1.0, "A": 0.0-1.0, "S": 0.0-1.0, "E": 0.0-1.0, "C": 0.0-1.0}},
    "follow_up_needed": ["area1", "area2"],
    "intent_classification": "description",
    "sentiment": "positive/neutral/negative",
    "reasoning": "step-by-step explanation"
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
                completeness_score=analysis_data.get("completeness_score", 0.5)
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

        return ResponseAnalysis(
            riasec_updates=riasec_updates,
            extracted_data={"response": user_response},
            follow_up_needed=[] if completeness_score > 0.7 else [question.target_area],
            intent_classification="general_response",
            sentiment="neutral",
            completeness_score=completeness_score
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
        Remove structural markers like [VALIDATION], [BRIDGE], [QUESTION] from LLM responses.
        These markers are used in prompts for instruction but should never appear in user-facing output.
        """
        if not text:
            return text

        # Remove all structural markers (case-insensitive)
        markers = [
            r'\[VALIDATION\]', r'\[BRIDGE\]', r'\[QUESTION\]',
            r'\[VALIDATE\]', r'\[EMPATHIZE\]', r'\[FOLLOW-UP QUESTION\]',
            r'\[SUMMARY\]', r'\[TRANSITION\]'
        ]

        cleaned_text = text
        for marker in markers:
            cleaned_text = re.sub(marker, '', cleaned_text, flags=re.IGNORECASE)

        # Clean up any extra whitespace that might result from marker removal
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Log if markers were found and removed
        if cleaned_text != text:
            logging.info(f"Stripped structural markers from response. Original length: {len(text)}, Cleaned length: {len(cleaned_text)}")

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

    def generate_follow_up_question(self, context: ConversationContext, analysis: ResponseAnalysis) -> GeneratedQuestion:
        """
        Generate a targeted follow-up question based on the analysis.
        """
        # Identify the most critical area needing follow-up
        follow_up_area = None
        if analysis.follow_up_needed:
            follow_up_area = analysis.follow_up_needed[0]
        else:
            follow_up_area = "general"

        # Extract key insights from their response
        insights_context = ""
        if hasattr(analysis, 'key_insights') and analysis.key_insights:
            insights_context = f"\nKey insights from their response: {', '.join(analysis.key_insights[:3])}"

        # Get student's tone for adaptive response
        tone_context = ""
        latest_tone = context.session_metadata.get("energy_level", "balanced")
        confidence = context.session_metadata.get("confidence_level", "moderate")
        suggested_personality = context.session_metadata.get("suggested_personality", "balanced_and_curious")

        tone_context = f"\n\nSTUDENT'S COMMUNICATION STYLE:\n- Energy: {latest_tone}\n- Confidence: {confidence}\n- Match this tone: {suggested_personality}"

        prompt = f"""You are a natural, adaptive career counselor having a real conversation with a HIGH SCHOOL STUDENT. They just shared something that needs more exploration. Dig deeper in a genuine, conversational way.{tone_context}

THEIR RESPONSE: "{context.response_history[-1] if context.response_history else 'None'}"

WHAT NEEDS MORE INFO: {follow_up_area}{insights_context}

YOUR APPROACH - Respond naturally and flow into your follow-up:

1. ACKNOWLEDGE (1 sentence)
   • React genuinely to what they shared
   • If they asked a question, answer it naturally
   • Match their tone and energy level

2. CONNECT (1 sentence)
   • Smoothly reference what they said
   • Show you're tracking the conversation
   • Use their specific words or examples

3. DIG DEEPER (1-2 sentences)
   • Ask a natural follow-up about {follow_up_area}
   • Show authentic curiosity
   • Make it feel like a real conversation, not an interrogation

IMPORTANT: Do NOT include section labels. Flow naturally from thought to thought.

EXAMPLE (one of many possible responses):

Their response: "I guess I like art. Is that useful for jobs?"

Your response: "Definitely! Creative skills are actually in demand across a lot of different fields.

It sounds like art's something you're drawn to.

What kind of creative work do you find yourself doing? Are you more into traditional stuff like drawing, or do you mess around with digital design, or something totally different?"

HOW TO SOUND NATURAL:
• Vary your responses - don't use the same validation every time
• Mirror their language style (if they're casual, be casual; if thoughtful, be thoughtful)
• Reference specific things they mentioned
• Sound genuinely interested, not like you're reading from a script
• Keep it conversational and age-appropriate

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