"""
Agent Handoff Tools - Communication and Delegation Utilities

This module provides tools for agent-to-agent communication and task delegation
in the hierarchical multi-agent system.
"""
from typing import Annotated, Callable, Dict, Any, Optional
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Use absolute import
from models.state_models import AgentState


def create_handoff_tool(
    *, 
    agent_name: str, 
    description: Optional[str] = None,
    task_key: str = "task_description"
) -> Callable:
    """
    Create a handoff tool for transferring control to another agent.
    
    Args:
        agent_name: Name of the target agent to hand off to
        description: Description of what this handoff tool does
        task_key: Key name for the task parameter (default: "task_description")
    
    Returns:
        A tool function that can be used by supervisor agents
    """
    tool_name = f"transfer_to_{agent_name}"
    tool_description = description or f"Transfer task to {agent_name} agent for specialized processing"
    
    @tool(tool_name, description=tool_description)
    def handoff_tool(
        task_description: Annotated[
            str,
            f"Detailed description of the task to be handled by {agent_name}. Include all relevant context and requirements."
        ],
        state: Annotated[AgentState, InjectedState],
    ) -> Command:
        """
        Hand off a task to the specified agent.
        """
        # Create the task message
        task_message = HumanMessage(content=task_description)
        
        # Create tool response message
        tool_response = AIMessage(
            content=f"✅ Task successfully transferred to {agent_name}",
            name=tool_name
        )
        
        # Update state with the handoff
        updated_state = state.copy(deep=True)
        updated_state.messages.append(tool_response)
        updated_state.current_step = f"handoff_to_{agent_name}"
        
        # Add task-specific information to state
        if task_key not in updated_state.dict():
            setattr(updated_state, task_key, task_description)
        
        return Command(
            goto=agent_name,
            update=updated_state.dict(),
            graph=Command.PARENT
        )
    
    return handoff_tool


def create_simple_handoff_tool(*, agent_name: str, description: Optional[str] = None) -> Callable:
    """
    Create a simplified handoff tool that passes the full state without modification.
    
    Args:
        agent_name: Name of the target agent
        description: Description of the handoff tool
    
    Returns:
        A simple handoff tool function
    """
    tool_name = f"delegate_to_{agent_name}"
    tool_description = description or f"Delegate current task to {agent_name} agent"
    
    @tool(tool_name, description=tool_description)
    def simple_handoff_tool(
        state: Annotated[AgentState, InjectedState],
    ) -> Command:
        """
        Simple handoff that passes the current state to another agent.
        """
        # Add handoff message to state
        handoff_message = AIMessage(
            content=f"Delegating to {agent_name} agent...",
            name=tool_name
        )
        
        updated_state = state.copy(deep=True)
        updated_state.messages.append(handoff_message)
        updated_state.current_step = f"delegated_to_{agent_name}"
        
        return Command(
            goto=agent_name,
            update=updated_state.dict(),
            graph=Command.PARENT
        )
    
    return simple_handoff_tool


def create_return_to_supervisor_tool(supervisor_name: str = "main_supervisor") -> Callable:
    """
    Create a tool for agents to return control back to their supervisor.
    
    Args:
        supervisor_name: Name of the supervisor to return to
    
    Returns:
        A return tool function
    """
    tool_name = f"return_to_{supervisor_name}"
    
    @tool(tool_name, description=f"Return control and results to {supervisor_name}")
    def return_tool(
        completion_message: Annotated[
            str,
            "Message describing the completion of the assigned task and any important results"
        ],
        state: Annotated[AgentState, InjectedState],
    ) -> Command:
        """
        Return control to the supervisor with task completion information.
        """
        # Create completion message
        completion_msg = AIMessage(
            content=f"✅ Task completed. {completion_message}",
            name=tool_name
        )
        
        updated_state = state.copy(deep=True)
        updated_state.messages.append(completion_msg)
        updated_state.current_step = f"returned_from_{state.current_step}"
        
        return Command(
            goto=supervisor_name,
            update=updated_state.dict(),
            graph=Command.PARENT
        )
    
    return return_tool


class HandoffToolFactory:
    """
    Factory class for creating various types of handoff tools.
    """
    
    @staticmethod
    def create_user_profiler_handoff() -> Callable:
        """Create handoff tool for User Profiler Agent."""
        return create_handoff_tool(
            agent_name="user_profiler",
            description="Transfer student assessment task to User Profiler Agent for comprehensive profile creation"
        )
    
    @staticmethod
    def create_career_planning_supervisor_handoff() -> Callable:
        """Create handoff tool for Career Planning Supervisor."""
        return create_handoff_tool(
            agent_name="career_planning_supervisor",
            description="Delegate career exploration and planning to Career Planning Supervisor with student profile"
        )
    
    @staticmethod
    def create_future_trends_analyst_handoff() -> Callable:
        """Create handoff tool for Future Trends Analyst."""
        return create_handoff_tool(
            agent_name="future_trends_analyst",
            description="Send career plans to Future Trends Analyst for market viability assessment"
        )
    
    @staticmethod
    def create_academic_pathway_handoff() -> Callable:
        """Create handoff tool for Academic Pathway Agent."""
        return create_handoff_tool(
            agent_name="academic_pathway",
            description="Assign academic planning task to Academic Pathway Agent for education plan creation"
        )
    
    @staticmethod
    def create_skill_development_handoff() -> Callable:
        """Create handoff tool for Skill Development Agent."""
        return create_handoff_tool(
            agent_name="skill_development",
            description="Assign skill planning task to Skill Development Agent for roadmap creation"
        )
    
    @staticmethod
    def create_main_supervisor_return() -> Callable:
        """Create return tool for returning to Main Supervisor."""
        return create_return_to_supervisor_tool("main_supervisor")
    
    @staticmethod
    def create_career_supervisor_return() -> Callable:
        """Create return tool for returning to Career Planning Supervisor."""
        return create_return_to_supervisor_tool("career_planning_supervisor")


def create_task_specific_handoff(
    agent_name: str,
    task_type: str,
    required_data: Dict[str, str],
    description: Optional[str] = None
) -> Callable:
    """
    Create a specialized handoff tool for specific task types.
    
    Args:
        agent_name: Name of the target agent
        task_type: Type of task being handed off
        required_data: Dictionary of required data fields and their descriptions
        description: Tool description
    
    Returns:
        A specialized handoff tool
    """
    tool_name = f"assign_{task_type}_to_{agent_name}"
    tool_description = description or f"Assign {task_type} task to {agent_name} with required data"
    
    @tool(tool_name, description=tool_description)
    def specialized_handoff_tool(
        state: Annotated[AgentState, InjectedState],
        **kwargs: str
    ) -> Command:
        """
        Specialized handoff with task-specific data validation.
        """
        # Validate required data
        missing_fields = []
        for field_name, field_desc in required_data.items():
            if field_name not in kwargs:
                missing_fields.append(field_name)
        
        if missing_fields:
            error_msg = f"Missing required fields for {task_type}: {', '.join(missing_fields)}"
            return Command(
                goto="error_handler",
                update={"error_message": error_msg},
                graph=Command.PARENT
            )
        
        # Create task-specific state update
        task_data = {
            "task_type": task_type,
            "task_data": kwargs,
            "assigned_agent": agent_name
        }
        
        # Create handoff message
        handoff_message = AIMessage(
            content=f"✅ {task_type} task assigned to {agent_name} with required data",
            name=tool_name
        )
        
        updated_state = state.copy(deep=True)
        updated_state.messages.append(handoff_message)
        updated_state.current_step = f"{task_type}_assigned_to_{agent_name}"
        
        # Add task data to state
        for key, value in task_data.items():
            setattr(updated_state, key, value)
        
        return Command(
            goto=agent_name,
            update=updated_state.dict(),
            graph=Command.PARENT
        )
    
    return specialized_handoff_tool


def get_standard_handoff_tools() -> Dict[str, Callable]:
    """
    Get a dictionary of all standard handoff tools for the career planning system.
    
    Returns:
        Dictionary mapping tool names to tool functions
    """
    factory = HandoffToolFactory()
    
    return {
        "user_profiler_handoff": factory.create_user_profiler_handoff(),
        "career_planning_handoff": factory.create_career_planning_supervisor_handoff(),
        "future_trends_handoff": factory.create_future_trends_analyst_handoff(),
        "academic_pathway_handoff": factory.create_academic_pathway_handoff(),
        "skill_development_handoff": factory.create_skill_development_handoff(),
        "main_supervisor_return": factory.create_main_supervisor_return(),
        "career_supervisor_return": factory.create_career_supervisor_return(),
    }