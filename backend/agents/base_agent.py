"""
Base agent classes for the Career Planning Multi-Agent System.

This module provides the foundational classes for supervisor and worker agents
in the hierarchical multi-agent architecture.
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from models.state_models import AgentState
import logging
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

# Load environment variables for LangSmith from backend folder
from dotenv import load_dotenv
import os
backend_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(backend_env_path)

# Use absolute import
from models.state_models import AgentState, TaskResult, StudentProfile

# Import LLM factory for flexible LLM provider support
from utils.llm_factory import LLMFactory


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        use_fallback: bool = True
    ):
        self.name = name
        self.description = description
        self.model = model
        self.temperature = temperature
        self.use_fallback = use_fallback
        self.logger = logging.getLogger(f"agent.{name}")

        # Initialize the LLM using factory with fallback support
        try:
            self.llm_wrapper = LLMFactory.create_llm(
                model=self.model,
                temperature=self.temperature,
                enable_fallback=self.use_fallback
            )
            self.llm = self.llm_wrapper.llm
            self.logger.info(f"✅ LLM initialized for agent '{name}' using {self.llm_wrapper.strategy.provider_name}")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM for agent '{name}': {str(e)}")
            raise

        # Agent state tracking
        self.agent_type = "base"
        self.capabilities = []
        self.tools = []
        
    @abstractmethod
    def process_task(self, state: AgentState) -> TaskResult:
        """
        Process a task and return the result.
        Must be implemented by all concrete agent classes.
        """
        pass
    
    def _create_task_result(
        self,
        task_type: str,
        success: bool,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        processing_time: Optional[float] = None,
        updated_state: Optional['AgentState'] = None
    ) -> TaskResult:
        """Helper method to create standardized task results."""
        return TaskResult(
            agent_name=self.name,
            task_type=task_type,
            success=success,
            result_data=result_data or {},
            error_message=error_message,
            processing_time=processing_time,
            updated_state=updated_state
        )
    
    def _log_task_start(self, task_type: str, details: str = ""):
        """Log the start of a task."""
        self.logger.info(f"🚀 Starting {task_type} {details}")
    
    def _log_task_completion(self, task_type: str, success: bool, details: str = ""):
        """Log the completion of a task."""
        status = "✅ Completed" if success else "❌ Failed"
        self.logger.info(f"{status} {task_type} {details}")
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about this agent."""
        info = {
            "name": self.name,
            "description": self.description,
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "model": self.model,
            "temperature": self.temperature,
            "use_fallback": self.use_fallback
        }

        # Add LLM provider info if available
        if hasattr(self, 'llm_wrapper'):
            info["llm_stats"] = self.llm_wrapper.get_stats()

        return info


class WorkerAgent(BaseAgent):
    """
    Base class for worker agents that perform specialized tasks.
    Worker agents report to supervisors and have specific specializations.
    """
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        specialization: str,
        system_prompt: str,
        tools: Optional[List[Callable]] = None,
        **kwargs
    ):
        super().__init__(name, description, **kwargs)
        self.agent_type = "worker"
        self.specialization = specialization
        self.system_prompt = system_prompt
        self.tools = tools or []
        
        # Create the react agent with tools
        self.react_agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt
        )
    
    def invoke_with_prompt(self, user_message: str) -> str:
        """
        Invoke the agent with a user message and return the response.
        """
        try:
            self._log_task_start("agent_invocation", f"with message: {user_message[:100]}...")
            
            # Create the input state
            input_state = {"messages": [HumanMessage(content=user_message)]}
            
            # Invoke the react agent
            result = self.react_agent.invoke(input_state)
            
            # Extract the final message
            if result.get("messages"):
                final_message = result["messages"][-1]
                if isinstance(final_message, AIMessage):
                    response = final_message.content
                    self._log_task_completion("agent_invocation", True, f"response length: {len(response)}")
                    return response
            
            self._log_task_completion("agent_invocation", False, "No valid response generated")
            return "I apologize, but I couldn't generate a proper response to your request."
            
        except Exception as e:
            self._log_task_completion("agent_invocation", False, f"Error: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def process_simple_task(self, task_description: str) -> TaskResult:
        """
        Process a simple task with just a description.
        """
        start_time = datetime.now()
        
        try:
            response = self.invoke_with_prompt(task_description)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return self._create_task_result(
                task_type="simple_task",
                success=True,
                result_data={"response": response},
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return self._create_task_result(
                task_type="simple_task",
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )


class SupervisorAgent(BaseAgent):
    """
    Base class for supervisor agents that coordinate and manage other agents.
    Supervisors can manage both worker agents and other supervisor agents.
    """
    
    def __init__(
        self, 
        name: str, 
        description: str,
        managed_agents: Optional[Dict[str, BaseAgent]] = None,
        orchestration_prompt: str = "",
        **kwargs
    ):
        super().__init__(name, description, **kwargs)
        self.agent_type = "supervisor"
        self.managed_agents = managed_agents or {}
        self.orchestration_prompt = orchestration_prompt
        
        # Supervisor-specific capabilities
        self.capabilities = ["agent_coordination", "task_delegation", "workflow_management"]
    
    def add_managed_agent(self, agent: BaseAgent):
        """Add an agent to be managed by this supervisor."""
        self.managed_agents[agent.name] = agent
        self.logger.info(f"➕ Added {agent.name} ({agent.agent_type}) to managed agents")
    
    def get_managed_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get a managed agent by name."""
        return self.managed_agents.get(agent_name)
    
    def list_managed_agents(self) -> List[str]:
        """Get list of managed agent names."""
        return list(self.managed_agents.keys())
    
    def delegate_task(self, agent_name: str, task: str) -> TaskResult:
        """
        Delegate a task to a specific managed agent.
        """
        self._log_task_start("task_delegation", f"to {agent_name}")
        
        agent = self.get_managed_agent(agent_name)
        if not agent:
            error_msg = f"Agent '{agent_name}' not found in managed agents"
            self._log_task_completion("task_delegation", False, error_msg)
            return self._create_task_result(
                task_type="task_delegation",
                success=False,
                error_message=error_msg
            )
        
        try:
            # Delegate the task
            if isinstance(agent, WorkerAgent):
                result = agent.process_simple_task(task)
            else:
                # For other supervisors, create a state and process
                state = AgentState(
                    messages=[HumanMessage(content=task)],
                    current_step="delegated_task"
                )
                result = agent.process_task(state)
            
            self._log_task_completion("task_delegation", result.success, f"to {agent_name}")
            return result
            
        except Exception as e:
            error_msg = f"Error delegating task to {agent_name}: {str(e)}"
            self._log_task_completion("task_delegation", False, error_msg)
            return self._create_task_result(
                task_type="task_delegation",
                success=False,
                error_message=error_msg
            )
    
    def coordinate_agents(self, tasks: Dict[str, str]) -> Dict[str, TaskResult]:
        """
        Coordinate multiple agents by assigning tasks to each.
        Returns a dictionary of agent_name -> TaskResult.
        """
        self._log_task_start("agent_coordination", f"with {len(tasks)} tasks")
        results = {}
        
        for agent_name, task in tasks.items():
            result = self.delegate_task(agent_name, task)
            results[agent_name] = result
        
        successful_tasks = sum(1 for r in results.values() if r.success)
        self._log_task_completion(
            "agent_coordination", 
            successful_tasks == len(tasks),
            f"{successful_tasks}/{len(tasks)} tasks successful"
        )
        
        return results