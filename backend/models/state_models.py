"""
State management models for the Career Planning Multi-Agent System.

This module defines the data structures and state management for agent communication
and career planning workflow, including human-in-the-loop conversation management.
"""
from typing import Dict, List, Optional, Any, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from enum import Enum


class ConversationState(Enum):
    """Conversation states for human-in-the-loop profiling."""
    GREETING = "greeting"
    ACADEMIC_GATHERING = "academic_gathering"
    INTEREST_DISCOVERY = "interest_discovery"
    SKILLS_ASSESSMENT = "skills_assessment"
    PERSONALITY_PROBE = "personality_probe"
    SUMMARIZATION = "summarization"
    CONFIRMATION = "confirmation"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING_FOR_USER = "waiting_for_user"


class RIASECCategory(Enum):
    """RIASEC model categories for career assessment."""
    REALISTIC = "R"
    INVESTIGATIVE = "I"
    ARTISTIC = "A"
    SOCIAL = "S"
    ENTERPRISING = "E"
    CONVENTIONAL = "C"


class ConversationSession(BaseModel):
    """
    Model for managing human-in-the-loop conversation sessions.
    Simplified to track question count and progress.
    """
    session_id: str = Field(..., description="Unique session identifier")
    current_conversation_state: ConversationState = Field(default=ConversationState.GREETING, description="Current conversation state")
    riasec_scores: Dict[str, float] = Field(default_factory=dict, description="RIASEC category scores (R,I,A,S,E,C)")
    question_history: List[str] = Field(default_factory=list, description="Questions asked during session")
    response_history: List[str] = Field(default_factory=list, description="User responses during session")
    collected_data: Dict[str, Any] = Field(default_factory=dict, description="Structured data collected")
    follow_up_needed: List[str] = Field(default_factory=list, description="Areas needing follow-up")
    session_metadata: Dict[str, Any] = Field(default_factory=dict, description="Session tracking metadata")
    is_active: bool = Field(default=True, description="Whether session is currently active")
    last_interaction: Optional[str] = Field(None, description="Timestamp of last interaction")
    awaiting_user_response: bool = Field(default=False, description="Whether waiting for user response")
    current_question: Optional[str] = Field(None, description="Current question awaiting response")
    questions_remaining: int = Field(default=12, description="Questions remaining (out of 12 total)")


class StudentProfile(BaseModel):
    """
    Comprehensive student profile created by User Profiler Agent.
    """
    # Basic Information
    name: Optional[str] = None
    age: Optional[int] = None
    location: Optional[str] = None
    
    # Academic Background
    current_education_level: str = Field(..., description="Current education level (e.g., High School, Bachelor's, Master's)")
    major_field: Optional[str] = Field(None, description="Current or completed major field of study")
    academic_performance: Optional[str] = Field(None, description="Academic performance level (e.g., Excellent, Good, Average)")
    institutions: List[str] = Field(default_factory=list, description="Educational institutions attended")
    
    # Skills and Competencies
    technical_skills: List[str] = Field(default_factory=list, description="Technical skills and competencies")
    soft_skills: List[str] = Field(default_factory=list, description="Soft skills and personal qualities")
    languages: List[str] = Field(default_factory=list, description="Languages spoken")
    certifications: List[str] = Field(default_factory=list, description="Professional certifications")
    
    # Interests and Preferences
    career_interests: List[str] = Field(default_factory=list, description="Areas of career interest")
    work_environment_preferences: List[str] = Field(default_factory=list, description="Preferred work environment characteristics")
    industry_preferences: List[str] = Field(default_factory=list, description="Preferred industries")
    
    # Goals and Aspirations
    short_term_goals: List[str] = Field(default_factory=list, description="Goals for next 1-2 years")
    long_term_goals: List[str] = Field(default_factory=list, description="Goals for 5-10 years")
    salary_expectations: Optional[str] = Field(None, description="Salary expectations or range")
    
    # Constraints and Limitations
    geographical_constraints: List[str] = Field(default_factory=list, description="Location preferences or limitations")
    time_constraints: List[str] = Field(default_factory=list, description="Time-related constraints")
    financial_constraints: List[str] = Field(default_factory=list, description="Financial limitations")
    
    # Analysis Results
    personality_traits: List[str] = Field(default_factory=list, description="Identified personality traits")
    learning_style: Optional[str] = Field(None, description="Preferred learning style")
    work_style: Optional[str] = Field(None, description="Preferred work style")

    # RIASEC Assessment Results
    riasec_scores: Dict[str, float] = Field(default_factory=dict, description="RIASEC category scores")
    dominant_riasec_type: Optional[str] = Field(None, description="Dominant RIASEC category")
    riasec_confidence: float = Field(default=0.0, description="Quality of RIASEC assessment (based on question distribution)")

    # Conversation-Derived Metadata
    conversation_session_id: Optional[str] = Field(None, description="Session ID that created this profile")
    areas_needing_clarification: List[str] = Field(default_factory=list, description="Areas that need more information")
    
    def to_summary(self) -> str:
        """Generate a concise summary of the student profile."""
        summary_parts = []
        
        if self.current_education_level:
            summary_parts.append(f"Education: {self.current_education_level}")
        if self.major_field:
            summary_parts.append(f"Major: {self.major_field}")
        if self.technical_skills:
            summary_parts.append(f"Technical Skills: {', '.join(self.technical_skills[:3])}")
        if self.career_interests:
            summary_parts.append(f"Interests: {', '.join(self.career_interests[:3])}")
            
        return " | ".join(summary_parts)


class CareerBlueprint(BaseModel):
    """
    Complete career blueprint including academic and skill development plans.
    """
    career_title: str = Field(..., description="Career title/position")
    career_description: str = Field(..., description="Detailed career description")
    match_score: float = Field(..., description="How well this career matches the student (0-100)")
    match_reasoning: str = Field(..., description="Explanation of why this career matches")
    
    # Plans (to be filled by specialized agents)
    academic_plan: Optional[Dict[str, Any]] = Field(None, description="Academic pathway plan")
    skill_development_plan: Optional[Dict[str, Any]] = Field(None, description="Skill development roadmap")
    
    # Career Details
    typical_responsibilities: List[str] = Field(default_factory=list)
    required_qualifications: List[str] = Field(default_factory=list)
    salary_range: Optional[str] = Field(None)
    career_progression: List[str] = Field(default_factory=list)
    
    # Market Analysis (to be filled by Future Trends Analyst)
    future_viability: Optional[Dict[str, Any]] = Field(None, description="Future market viability analysis")


class AgentState(BaseModel):
    """
    State passed between agents in the multi-agent system.
    """
    # Messages for agent communication
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    
    # Student data
    student_request: Optional[str] = Field(None, description="Original student request")
    student_profile: Optional[StudentProfile] = Field(None, description="Analyzed student profile")
    
    # Career planning data
    identified_careers: List[str] = Field(default_factory=list, description="List of identified career options")
    career_blueprints: List[CareerBlueprint] = Field(default_factory=list, description="Complete career blueprints")
    
    # Analysis results
    market_analysis: Optional[Dict[str, Any]] = Field(None, description="Future trends and market analysis")
    final_report: Optional[str] = Field(None, description="Final comprehensive report")
    
    # Workflow control
    current_step: str = Field(default="start", description="Current workflow step")
    completed_steps: List[str] = Field(default_factory=list, description="Completed workflow steps")
    
    # Session information
    session_id: Optional[str] = Field(None, description="Unique session identifier")
    timestamp: Optional[str] = Field(None, description="Request timestamp")

    # Human-in-the-loop conversation management
    conversation_session: Optional[ConversationSession] = Field(None, description="Active conversation session")
    awaiting_user_response: bool = Field(default=False, description="Whether agent is waiting for user response")
    current_question: Optional[str] = Field(None, description="Current question posed to user")
    interaction_mode: str = Field(default="batch", description="Interaction mode: 'batch' or 'interactive'")

    # User response handling
    pending_user_response: Optional[str] = Field(None, description="User response awaiting processing")
    response_processed: bool = Field(default=False, description="Whether user response has been processed")
    
    class Config:
        arbitrary_types_allowed = True


class TaskResult(BaseModel):
    """
    Standard result format for agent task completion.
    """
    agent_name: str = Field(..., description="Name of the agent that completed the task")
    task_type: str = Field(..., description="Type of task completed")
    success: bool = Field(..., description="Whether the task was completed successfully")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Task result data")
    error_message: Optional[str] = Field(None, description="Error message if task failed")
    processing_time: Optional[float] = Field(None, description="Time taken to complete the task")
    updated_state: Optional['AgentState'] = Field(None, description="Updated state after task completion")
    
    def to_message(self) -> str:
        """Convert task result to a message string."""
        if self.success:
            return f"✅ {self.agent_name} completed {self.task_type} successfully"
        else:
            return f"❌ {self.agent_name} failed {self.task_type}: {self.error_message}"