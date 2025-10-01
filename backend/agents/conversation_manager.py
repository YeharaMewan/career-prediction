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
from langchain_openai import ChatOpenAI

from agents.conversation_states import (
    ConversationState, ConversationContext, RIASECCategory,
    ConversationStateMachine, create_initial_conversation_context
)

# Import LangSmith configuration
from utils.langsmith_config import get_traced_run_config, log_agent_execution, setup_langsmith

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
        self.llm = ChatOpenAI(model=model, temperature=temperature)
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
                    "question": "What's your favorite subject in school and why do you like it?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "academic_background",
                    "follow_up_triggers": ["incomplete_info", "career_change", "multiple_fields"],
                    "priority": 1
                },
                {
                    "question": "Are you better at working with your hands, solving math problems, writing stories, or helping people?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "learning_style",
                    "follow_up_triggers": ["specific_methods", "challenges", "preferences"],
                    "priority": 1
                }
            ],

            ConversationState.INTEREST_DISCOVERY: [
                {
                    "question": "What kind of activities make you feel excited and engaged - like building things, helping people, creating art, or solving puzzles?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "work_preferences",
                    "follow_up_triggers": ["general_response", "multiple_activities", "unclear"],
                    "priority": 1
                },
                {
                    "question": "Do you prefer working alone or with other people, and why?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "work_style",
                    "follow_up_triggers": ["combination", "uncertain", "strong_preference"],
                    "priority": 1
                },
                {
                    "question": "What type of job setting sounds most interesting: an office, outdoors, a lab, or somewhere creative like a studio?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "industry_interests",
                    "follow_up_triggers": ["multiple_industries", "superficial_knowledge", "specific_companies"],
                    "priority": 1
                },
                {
                    "question": "What's most important to you in a job: helping others, earning good money, being creative, or having job security?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "career_values",
                    "follow_up_triggers": ["multiple_motivations", "unclear_values", "specific_impact"],
                    "priority": 1
                }
            ],

            ConversationState.SKILLS_ASSESSMENT: [
                {
                    "question": "What are you naturally good at? Think about things people compliment you on or tasks that feel easy for you.",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "technical_skills",
                    "follow_up_triggers": ["limited_skills", "advanced_skills", "self_taught"],
                    "priority": 1
                },
                {
                    "question": "When you have a problem to solve, do you like to figure it out by yourself or ask others for help?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "problem_solving",
                    "follow_up_triggers": ["analytical", "creative", "collaborative"],
                    "priority": 1
                }
            ],

            ConversationState.PERSONALITY_PROBE: [
                {
                    "question": "Do you get more energy from being around people or from having quiet time by yourself?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "introversion_extraversion",
                    "follow_up_triggers": ["extreme_preference", "situational", "unclear"],
                    "priority": 1
                },
                {
                    "question": "What makes you feel most proud: finishing a difficult project, helping someone, being creative, or leading a group?",
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

        prompt = f"""You are a friendly career counselor talking to a high school student. Generate ONE simple question that helps with career planning.

GOAL: {state_objective}

STUDENT CONTEXT:
- Current Stage: {context.current_state.name}
- Recent Questions: {context.question_history[-2:] if context.question_history else "None"}
- Student's Last Answer: {context.response_history[-1] if context.response_history else "None"}
- Questions Asked: {questions_asked} of 12 total
- Questions Remaining: {questions_remaining}

REQUIREMENTS:
1. Write ONE question only (not multiple questions)
2. Use simple words that a high school student understands
3. Keep it under 25 words
4. Ask about: {state_objective}

EXAMPLES OF GOOD QUESTIONS:
- "What do you like most about math class?"
- "Do you prefer working on projects alone or with friends?"
- "What's your favorite way to spend free time?"

Generate ONE simple question for a high school student:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])

            if not response or not response.content:
                raise ValueError("Empty response from LLM")

            question_text = response.content.strip()

            # Ensure single question only (for high school student clarity)
            question_text = self._ensure_single_question(question_text)

            # Ensure the question is valid
            if not question_text or len(question_text) < 10:
                raise ValueError("Question too short or empty")

        except Exception as e:
            logging.error(f"Dynamic question generation failed: {type(e).__name__}: {str(e)}. Using fallback question.")

            # Simple fallback questions for high school students
            state_name = context.current_state.name.lower()
            fallback_questions = {
                "academic_gathering": "What's your favorite subject in school?",
                "interest_discovery": "What activities do you enjoy most in your free time?",
                "skills_assessment": "What are you naturally good at?",
                "personality_probe": "Do you prefer working with people or working alone?"
            }
            question_text = fallback_questions.get(state_name, "What kind of career sounds interesting to you?")

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

        prompt = f"""You are a friendly AI career helper talking to a high school student. Create a simple, encouraging greeting.

CONTEXT:
- You're helping a high school student explore careers
- You'll ask simple questions about their interests and strengths
- Keep it short and friendly - no more than 50 words
- Use words a teenager understands

TASK: Write a warm greeting that:
1. Says you're an AI career helper
2. Shows you're excited to help them
3. Mentions you'll ask about their interests and what they're good at
4. Ends with EXACTLY ONE simple question (no multiple questions)

IMPORTANT: End with ONLY ONE question mark (?)

EXAMPLES:
- "Hi! I'm your AI career helper. I'm excited to help you explore different careers that might be perfect for you! I'll ask about what you enjoy and what you're good at. What interests you most about finding a career?"

- "Hello! I'm an AI that helps students discover cool career options. We'll chat about your favorite subjects, hobbies, and strengths. What made you want to learn about careers today?"

Generate a friendly, simple greeting ending with ONE question:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            question_text = response.content.strip()

            # Enforce single question ending
            question_text = self._ensure_greeting_has_single_question(question_text)

            # Simple fallback greeting if LLM response is empty or too short
            if not question_text or len(question_text) < 20:
                question_text = "Hi! I'm your AI career helper. I'm excited to help you explore different careers that might be perfect for you! What interests you most about finding a career?"

        except Exception as e:
            # Simple fallback greeting if LLM fails
            question_text = "Hello! I'm an AI that helps students discover cool career options. What made you want to learn about careers today?"

        return GeneratedQuestion(
            question_text=question_text,
            question_type=QuestionType.OPEN_ENDED,
            target_area="initial_engagement",
            expected_insights=["engagement_level", "communication_style", "initial_career_interests"],
            follow_up_triggers=["vague", "uncertain", "multiple_interests"],
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
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])

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

    def _ensure_single_question(self, question_text: str) -> str:
        """Ensure only ONE question is asked per agent message - strictly enforced."""
        if not question_text:
            return question_text

        # Remove any newlines or extra spacing
        question_text = ' '.join(question_text.split())

        # Split by question marks
        questions = question_text.split('?')

        # STRICT: Take ONLY the first question
        if len(questions) > 1 and questions[0].strip():
            first_question = questions[0].strip() + '?'

            # Log if multiple questions were detected
            if len([q for q in questions if q.strip()]) > 1:
                logging.warning(f"Multiple questions detected. Using only first: {first_question}")

            return first_question

        # If no question mark, take first sentence
        sentences = question_text.split('.')
        if sentences and sentences[0].strip():
            first_sentence = sentences[0].strip()
            # Add question mark if it seems like a question
            if any(word in first_sentence.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which', 'do you', 'are you', 'can you']):
                return first_sentence + '?'
            return first_sentence + '.'

        return question_text

    def _ensure_greeting_has_single_question(self, greeting_text: str) -> str:
        """Ensure greeting ends with exactly ONE question."""
        if not greeting_text:
            return greeting_text

        # Count question marks
        question_count = greeting_text.count('?')

        if question_count == 0:
            # No question - this is invalid, return fallback
            logging.warning("Greeting has no question, using fallback")
            return "Hi! I'm your AI career helper. What interests you most about finding a career?"

        if question_count == 1:
            # Perfect - exactly one question
            return greeting_text

        # Multiple questions - keep intro + last question only
        parts = greeting_text.split('?')
        if len(parts) >= 2:
            # Find where the last question starts
            last_question_part = parts[-2].strip()  # Second to last part (before final ?)

            # Find the last sentence before the last question
            sentences = greeting_text.split('.')
            intro_sentences = []
            for sentence in sentences:
                if '?' not in sentence and len(intro_sentences) < 2:  # Keep max 2 intro sentences
                    intro_sentences.append(sentence.strip())

            # Combine intro + last question
            intro = '. '.join(intro_sentences)
            if intro and not intro.endswith('.'):
                intro += '.'

            # Extract just the last question
            last_q_index = greeting_text.rfind('?')
            # Find start of last question (look for sentence start)
            text_before_last_q = greeting_text[:last_q_index]
            last_sentence_start = max(text_before_last_q.rfind('.'), text_before_last_q.rfind('!'), 0)
            last_question = greeting_text[last_sentence_start:last_q_index + 1].strip()
            if last_question.startswith('.') or last_question.startswith('!'):
                last_question = last_question[1:].strip()

            result = f"{intro} {last_question}"
            logging.warning(f"Multiple questions in greeting. Reduced to: {result}")
            return result

        return greeting_text

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

        return insights

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

        prompt = f"""You are conducting a professional career assessment and need to ask a contextual follow-up question based on the user's previous response.

PREVIOUS USER RESPONSE: "{context.response_history[-1] if context.response_history else 'None'}"
AREA NEEDING MORE INFORMATION: {follow_up_area}
CONVERSATION CONTEXT: Questions asked: {len(context.question_history)}, Current state: {context.current_state.name}

ANALYSIS: Based on their previous response, identify specific topics, interests, skills, or experiences they mentioned that warrant deeper exploration for career assessment purposes.

Generate a professional follow-up question that:
1. References specific details from their previous response
2. Seeks to gather more comprehensive career-relevant information
3. Uses clear, accessible language while maintaining counseling depth
4. Provides specific direction for what kind of additional detail would be helpful
5. Shows you were actively listening to their response
6. Helps build toward accurate career recommendations
7. Contains ONLY ONE question (not multiple questions)

IMPORTANT: Generate exactly ONE question with ONE question mark (?)

FOLLOW-UP QUESTION EXAMPLES:
- "You mentioned enjoying [specific activity] - what aspects of this work appeal to you most?"

- "That's interesting that you have experience with [specific skill]. Could you tell me more about how you developed this ability and what you find most rewarding about it?"

Generate ONE contextual, professional follow-up question:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])

            if not response or not response.content:
                raise ValueError("Empty response from LLM")

            question_text = response.content.strip()

            # Enforce single question
            question_text = self._ensure_single_question(question_text)

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