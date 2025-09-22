"""
Enhanced Conversation Manager for Human-in-the-Loop User Profiler Agent

This module handles intelligent question generation, response analysis,
and conversation flow control using advanced AI techniques.
"""
import json
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
    """
    confidence_updates: Dict[str, float]  # Updates to confidence scores
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
    and sophisticated intent recognition.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.state_machine = ConversationStateMachine()
        self.question_bank = self._initialize_question_bank()
        self.riasec_keywords = self._initialize_riasec_keywords()
        self.few_shot_examples = self._initialize_few_shot_examples()

    def _initialize_question_bank(self) -> Dict[ConversationState, List[Dict[str, Any]]]:
        """Initialize curated question bank for each conversation state."""
        return {
            ConversationState.GREETING: [
                # No predefined greeting - will be generated dynamically
            ],

            ConversationState.ACADEMIC_GATHERING: [
                {
                    "question": "What's your current education level, and what field are you studying or have studied?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "academic_background",
                    "follow_up_triggers": ["incomplete_info", "career_change", "multiple_fields"]
                },
                {
                    "question": "How would you describe your academic performance and what subjects do you find most engaging?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "academic_interests",
                    "follow_up_triggers": ["struggling", "excelling", "specific_subjects"]
                },
                {
                    "question": "Are you currently planning to continue your education, and if so, in what direction?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "future_education",
                    "follow_up_triggers": ["uncertain", "financial_constraints", "specific_programs"]
                }
            ],

            ConversationState.INTEREST_DISCOVERY: [
                {
                    "question": "Perfect! Now let's dive into your interests. When you imagine your ideal career, what kind of activities would you be doing day-to-day? This will help me predict which career paths would make you happiest.",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "work_preferences",
                    "follow_up_triggers": ["general_response", "multiple_activities", "unclear"]
                },
                {
                    "question": "This is important for my predictions: Do you prefer working with people, data, ideas, or hands-on projects? Your answer will help me match you to the right career categories.",
                    "type": QuestionType.MULTIPLE_CHOICE,
                    "target_area": "work_style",
                    "follow_up_triggers": ["combination", "uncertain", "strong_preference"]
                },
                {
                    "question": "What industries or fields have caught your attention recently? Understanding your industry interests helps me predict more specific career roles for you.",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "industry_interests",
                    "follow_up_triggers": ["multiple_industries", "superficial_knowledge", "specific_companies"]
                }
            ],

            ConversationState.SKILLS_ASSESSMENT: [
                {
                    "question": "What technical skills do you currently have, and which ones do you feel strongest in?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "technical_skills",
                    "follow_up_triggers": ["limited_skills", "advanced_skills", "self_taught"]
                },
                {
                    "question": "How do you typically approach problem-solving? Can you give me an example?",
                    "type": QuestionType.OPEN_ENDED,
                    "target_area": "problem_solving",
                    "follow_up_triggers": ["analytical", "creative", "collaborative"]
                },
                {
                    "question": "In group settings, do you tend to take leadership roles, contribute as a team member, or prefer working independently?",
                    "type": QuestionType.MULTIPLE_CHOICE,
                    "target_area": "teamwork_style",
                    "follow_up_triggers": ["leadership", "collaborative", "independent"]
                }
            ],

            ConversationState.PERSONALITY_PROBE: [
                {
                    "question": "Do you energize more from working with people or from having time to think and work alone?",
                    "type": QuestionType.SCALE_RATING,
                    "target_area": "introversion_extraversion",
                    "follow_up_triggers": ["extreme_preference", "situational", "unclear"]
                },
                {
                    "question": "When facing a challenge, do you prefer to analyze all options carefully or make quick decisions and adapt as you go?",
                    "type": QuestionType.MULTIPLE_CHOICE,
                    "target_area": "decision_making",
                    "follow_up_triggers": ["analytical", "intuitive", "depends_on_situation"]
                },
                {
                    "question": "What motivates you most: helping others, solving complex problems, creating something new, or achieving recognition?",
                    "type": QuestionType.MULTIPLE_CHOICE,
                    "target_area": "core_motivation",
                    "follow_up_triggers": ["multiple_motivations", "unclear", "specific_examples"]
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
                "confidence_scores": context.confidence_scores
            }
        )
        
        current_state = context.current_state

        # Check if we have pre-defined questions for this state
        state_questions = self.question_bank.get(current_state, [])

        # Filter out already asked questions
        asked_questions = set(context.question_history)
        available_questions = [
            q for q in state_questions
            if q["question"] not in asked_questions
        ]

        if available_questions:
            # Select best question based on context
            selected_question = self._select_best_question(available_questions, context)
            result = self._convert_to_generated_question(selected_question, context)
        else:
            # Generate dynamic question using LLM
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
        """Select the best question based on current context and confidence gaps."""

        # Identify areas with low confidence
        low_confidence_areas = [
            area for area, confidence in context.confidence_scores.items()
            if confidence < 0.5
        ]

        # Prioritize questions that target low-confidence areas
        for question in questions:
            if question.get("target_area") in low_confidence_areas:
                return question

        # If no specific targeting needed, return first available question
        return questions[0]

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
        low_confidence_areas = [
            area for area, confidence in context.confidence_scores.items()
            if confidence < 0.6
        ]

        prompt = f"""You are an expert career counselor conducting a student interview.

CURRENT OBJECTIVE: {state_objective}

CONVERSATION CONTEXT:
- Current State: {context.current_state.name}
- Low Confidence Areas: {low_confidence_areas}
- Previous Questions: {context.question_history[-3:] if context.question_history else "None"}
- Response History: {context.response_history[-3:] if context.response_history else "None"}

TASK: Generate ONE thoughtful question that:
1. Advances the current state objective
2. Addresses low-confidence areas
3. Uses career counseling best practices
4. Is different from previous questions

FEW-SHOT EXAMPLES:
{self._format_few_shot_examples()}

Generate a question that follows these principles:
- Be specific but not overwhelming
- Allow for open-ended responses
- Build on previous conversation
- Help assess RIASEC categories when relevant

QUESTION:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])

            if not response or not response.content:
                raise ValueError("Empty response from LLM")

            question_text = response.content.strip()

            # Ensure the question is valid
            if not question_text or len(question_text) < 10:
                raise ValueError("Question too short or empty")

        except Exception as e:
            logging.error(f"Dynamic question generation failed: {type(e).__name__}: {str(e)}. Using fallback question.")

            # Fallback question based on current state
            state_name = context.current_state.name.lower()
            fallback_questions = {
                "academic_gathering": "Could you tell me about your educational background and what subjects interest you most?",
                "interest_discovery": "What activities or types of work do you find most engaging and exciting?",
                "skills_assessment": "What skills do you feel you're strongest in, and how do you like to use them?",
                "personality_probe": "How would you describe your work style and what motivates you professionally?"
            }
            question_text = fallback_questions.get(state_name, "Could you tell me more about your career interests and goals?")

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

        prompt = f"""You are an enthusiastic AI career counselor starting a new conversation with a student.

CONTEXT:
- Today's date: {current_time}
- This is a new career discovery session
- Your goal is to help them explore their career interests and find their ideal path
- You'll gather information through intelligent conversation

TASK: Generate a warm, engaging greeting that:
1. Introduces yourself as an AI career counselor or guidance assistant
2. Expresses genuine enthusiasm about helping them discover their career path
3. Focuses on exploration and discovery rather than predictions
4. Mentions you'll ask questions about their interests, skills, and preferences
5. Ends with an engaging question to start the conversation (like asking if they're ready to begin)
6. Uses a friendly, professional tone
7. Includes appropriate emojis to make it engaging
8. Makes each greeting unique and personalized

EXAMPLES OF GOOD GREETINGS:
- "Hello! I'm your AI career counselor, and I'm thrilled to help you explore your perfect career path! 🌟"
- "Hi there! I'm an AI career guidance assistant, and I can't wait to help you discover the careers that would make you happiest! 🎯"
- "Welcome! I'm your personal AI career counselor, excited to help you find your ideal career through our conversation! 🚀"

Generate a unique, engaging greeting that feels fresh and personalized:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            question_text = response.content.strip()

            # Fallback greeting if LLM response is empty or too short
            if not question_text or len(question_text) < 50:
                question_text = "Hi! I'm your AI career counselor, and I'm excited to help you discover your ideal career path! 🎯 I'll learn about your interests, skills, and preferences through our conversation to help you explore the best career options for you. Are you ready to start this career discovery journey with me?"

        except Exception as e:
            # Fallback to default greeting if LLM fails
            question_text = "Hello! I'm your AI career counselor, and I'm here to help you discover your perfect career path! 🌟 Through our conversation, I'll learn about your interests and skills to help you explore your ideal career options. Ready to begin this exciting journey together?"

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
- Current Confidence Scores: {context.confidence_scores}
- Current RIASEC Scores: {context.riasec_scores}

FEW-SHOT EXAMPLES:
{self._format_few_shot_examples()}

ANALYSIS TASK:
Step 1: Assess response completeness and quality (0.0-1.0)
Step 2: Extract structured information relevant to career planning
Step 3: Update confidence scores for different areas based on new information
Step 4: Calculate RIASEC category scores based on response content
Step 5: Identify any areas needing follow-up questions
Step 6: Classify the user's intent and sentiment

Provide analysis in this JSON format:
{{
    "completeness_score": 0.0-1.0,
    "extracted_data": {{"key": "value"}},
    "confidence_updates": {{"area": 0.0-1.0}},
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
                confidence_updates=analysis_data.get("confidence_updates", {}),
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

        # Simple confidence scoring based on response length and specificity
        word_count = len(user_response.split())
        completeness_score = min(word_count / 20, 1.0)  # Normalize to 0-1

        confidence_updates = {
            question.target_area: completeness_score * 0.7  # Conservative scoring
        }

        return ResponseAnalysis(
            confidence_updates=confidence_updates,
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

    def update_conversation_context(self, context: ConversationContext, question: GeneratedQuestion, user_response: str, analysis: ResponseAnalysis) -> ConversationContext:
        """
        Update the conversation context with new question, response, and analysis.
        """
        # Add to history
        context.question_history.append(question.question_text)
        context.response_history.append(user_response)

        # Update confidence scores
        for area, update in analysis.confidence_updates.items():
            current = context.confidence_scores.get(area, 0.0)
            # Weighted average of current and new confidence
            context.confidence_scores[area] = (current * 0.7) + (update * 0.3)

        # Update RIASEC scores
        for category, update in analysis.riasec_updates.items():
            current = context.riasec_scores.get(category, 0.0)
            # Weighted average of current and new scores
            context.riasec_scores[category] = (current * 0.8) + (update * 0.2)

        # Update collected data
        context.collected_data.update(analysis.extracted_data)

        # Update follow-up needs
        context.follow_up_needed.extend(analysis.follow_up_needed)
        # Remove duplicates
        context.follow_up_needed = list(set(context.follow_up_needed))

        # Update session metadata
        question_type_value = question.question_type.value if question.question_type else "unknown"
        context.session_metadata.update({
            "last_question_type": question_type_value,
            "last_response_sentiment": analysis.sentiment,
            "last_response_completeness": analysis.completeness_score,
            "total_questions_asked": len(context.question_history)
        })

        return context

    def should_ask_follow_up(self, context: ConversationContext, analysis: ResponseAnalysis) -> bool:
        """
        Determine if a follow-up question should be asked before proceeding.
        """
        # Ask follow-up if:
        # 1. Response completeness is low
        # 2. There are specific follow-up needs identified
        # 3. Confidence in current area is still low

        if analysis.completeness_score < 0.5:
            return True

        if analysis.follow_up_needed:
            return True

        current_state = context.current_state
        required_confidence = self.state_machine.get_required_confidence(current_state)

        for area, min_confidence in required_confidence.items():
            if context.confidence_scores.get(area, 0.0) < min_confidence:
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
            # Find area with lowest confidence
            min_confidence = float('inf')
            for area, confidence in context.confidence_scores.items():
                if confidence < min_confidence:
                    min_confidence = confidence
                    follow_up_area = area

        prompt = f"""Generate a targeted follow-up question to clarify or expand on the user's previous response.

PREVIOUS RESPONSE: "{context.response_history[-1] if context.response_history else 'None'}"
ANALYSIS: Completeness: {analysis.completeness_score}, Sentiment: {analysis.sentiment}
AREA NEEDING CLARIFICATION: {follow_up_area}
CURRENT STATE: {context.current_state.name}

Generate a follow-up question that:
1. Politely asks for more specific information
2. Helps increase confidence in the target area
3. Is encouraging and supportive
4. Builds on what they've already shared

FOLLOW-UP QUESTION:"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])

            if not response or not response.content:
                raise ValueError("Empty response from LLM")

            question_text = response.content.strip()

            # Ensure the question is valid
            if not question_text or len(question_text) < 10:
                raise ValueError("Follow-up question too short or empty")

        except Exception as e:
            logging.error(f"Follow-up question generation failed: {type(e).__name__}: {str(e)}. Using fallback question.")

            # Fallback follow-up question
            if follow_up_area:
                question_text = f"Could you tell me more about {follow_up_area}? I'd like to better understand this aspect of your profile."
            else:
                question_text = "Could you elaborate on your previous response? I'd like to understand more about your perspective."

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
            "confidence_scores": context.confidence_scores,
            "riasec_scores": {cat.value: score for cat, score in context.riasec_scores.items()},
            "overall_confidence": sum(context.confidence_scores.values()) / len(context.confidence_scores) if context.confidence_scores else 0.0,
            "dominant_riasec": max(context.riasec_scores.items(), key=lambda x: x[1]) if context.riasec_scores else None,
            "conversation_completeness": self.state_machine.get_overall_confidence(context),
            "collected_data": context.collected_data,
            "areas_for_improvement": [area for area, conf in context.confidence_scores.items() if conf < 0.7]
        }