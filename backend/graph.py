"""
Simple LangGraph implementation for career planning system
"""
from typing import Dict, Any
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

# Import and setup LangSmith configuration
from utils.langsmith_config import setup_langsmith, get_traced_run_config, log_agent_execution

# Setup LangSmith for monitoring
setup_success = setup_langsmith()
if setup_success:
    print("LangSmith monitoring enabled for Career Planning System")

class AgentState(BaseModel):
    """Simple state for the career planning agent"""
    messages: list = []
    user_profile: Dict[str, Any] = {}
    career_suggestions: list = []
    next_step: str = "start"

def initialize_session(state: AgentState) -> AgentState:
    """Initialize a new career planning session"""
    log_agent_execution("career_graph", "initialize_session", "Session initialized")
    
    state.messages.append({
        "role": "assistant", 
        "content": "Welcome to the Career Planning System! Let's start by understanding your background and interests."
    })
    state.next_step = "profile_gathering"
    return state

def gather_profile(state: AgentState) -> AgentState:
    """Gather user profile information"""
    log_agent_execution("career_graph", "gather_profile", "Profile gathering started")
    
    # This would interact with the user profiler agent
    state.messages.append({
        "role": "assistant",
        "content": "Please tell me about your academic background, interests, and career goals."
    })
    state.next_step = "generate_suggestions"
    return state

def generate_career_suggestions(state: AgentState) -> AgentState:
    """Generate career suggestions based on user profile"""
    start_time = datetime.now()
    log_agent_execution("career_graph", "generate_suggestions", "Starting career suggestion generation")
    
    # This would use the main supervisor and worker agents
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    prompt = f"""
    Based on the user profile: {state.user_profile}
    And conversation: {state.messages}
    
    Generate 3-5 career suggestions with brief explanations.
    """
    
    try:
        response = llm.invoke(prompt)
        state.career_suggestions = [
            "Software Engineer - High demand in tech industry",
            "Data Scientist - Growing field with good prospects", 
            "Product Manager - Leadership role in technology"
        ]
        state.messages.append({
            "role": "assistant",
            "content": f"Based on your profile, here are some career suggestions: {', '.join(state.career_suggestions)}"
        })
        state.next_step = "end"
        
        duration = (datetime.now() - start_time).total_seconds()
        log_agent_execution("career_graph", "generate_suggestions", f"Generated {len(state.career_suggestions)} suggestions", duration)
        
    except Exception as e:
        log_agent_execution("career_graph", "generate_suggestions", f"Error: {str(e)}")
        state.messages.append({
            "role": "assistant", 
            "content": f"I'm having trouble generating suggestions right now: {str(e)}"
        })
        state.next_step = "end"
    
    return state

def should_continue(state: AgentState) -> str:
    """Determine next step in the workflow"""
    return state.next_step

# Create the workflow graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("initialize", initialize_session)
workflow.add_node("profile_gathering", gather_profile)
workflow.add_node("generate_suggestions", generate_career_suggestions)

# Add edges
workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "profile_gathering")
workflow.add_edge("profile_gathering", "generate_suggestions")
workflow.add_edge("generate_suggestions", END)

# Add conditional edges if needed
workflow.add_conditional_edges(
    "generate_suggestions",
    should_continue,
    {
        "end": END,
        "profile_gathering": "profile_gathering"
    }
)

# Create checkpointer for persistence (using memory for now)
# checkpointer = SqliteSaver.from_conn_string(":memory:")

# Compile the graph (without checkpointer for now)
app = workflow.compile()

if __name__ == "__main__":
    # Test the graph
    config = {"configurable": {"thread_id": "test"}}
    initial_state = AgentState()
    
    result = app.invoke(initial_state, config)
    print(f"Final result: {result}")