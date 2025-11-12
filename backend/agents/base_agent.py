"""
Base agent classes for the Career Planning Multi-Agent System.

This module provides the foundational classes for supervisor and worker agents
in the hierarchical multi-agent architecture.
"""
import os
import asyncio
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

# Import OpenTelemetry for token tracking
try:
    from opentelemetry import trace
    from tracing.setup_tracing import add_llm_token_attributes, extract_and_add_token_attributes_from_response
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None


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
            # Store the original LLM
            self._original_llm = self.llm_wrapper.llm

            # Wrap LLM with tracing if OpenTelemetry is available
            if OTEL_AVAILABLE:
                self.llm = self._create_traced_llm()
                self.logger.info(f"✅ LLM initialized with OpenTelemetry tracing for agent '{name}'")
            else:
                self.llm = self._original_llm
                self.logger.info(f"✅ LLM initialized for agent '{name}' (tracing disabled)")

            self.logger.info(f"   Provider: {self.llm_wrapper.strategy.provider_name}")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM for agent '{name}': {str(e)}")
            raise

        # Agent state tracking
        self.agent_type = "base"
        self.capabilities = []
        self.tools = []
        
    @abstractmethod
    async def process_task(self, state: AgentState) -> TaskResult:
        """
        Process a task and return the result.
        Must be implemented by all concrete agent classes.

        This is an async method to support parallel execution of multiple agents.
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

    def log_llm_token_usage_to_span(
        self,
        llm_response,
        span_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Log LLM token usage to the current OpenTelemetry span.

        This method extracts token usage from an LLM response and adds it as
        attributes to the current active span, making it visible in Jaeger UI.

        Args:
            llm_response: The LLM response object
            span_name: Optional span name if you want to create a new span

        Returns:
            Dictionary with token usage data, or None if OTel not available

        Example:
            response = self.llm.invoke(messages)
            self.log_llm_token_usage_to_span(response)
        """
        if not OTEL_AVAILABLE:
            self.logger.warning("OpenTelemetry not available - cannot log token usage to span")
            return None

        try:
            if span_name:
                # Create a new span
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(span_name) as span:
                    return extract_and_add_token_attributes_from_response(
                        span, llm_response, self.model
                    )
            else:
                # Use current span
                current_span = trace.get_current_span()
                if current_span:
                    return extract_and_add_token_attributes_from_response(
                        current_span, llm_response, self.model
                    )
                else:
                    self.logger.warning("No active span - cannot add token attributes")
                    return None

        except Exception as e:
            self.logger.warning(f"Failed to log token usage to span: {e}")
            return None

    def get_token_usage_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cumulative token usage statistics for this agent.

        Returns:
            Dictionary with token usage stats, or None if not available
        """
        if hasattr(self, 'llm_wrapper'):
            stats = self.llm_wrapper.get_stats()
            return {
                'total_input_tokens': stats.get('total_input_tokens', 0),
                'total_output_tokens': stats.get('total_output_tokens', 0),
                'total_tokens': stats.get('total_tokens', 0),
                'total_cost_usd': stats.get('total_cost_usd', 0.0),
                'usage_count': stats.get('usage_count', 0)
            }
        return None

    def _create_traced_llm(self):
        """
        Create a wrapper around the LLM that automatically traces invocations.
        This enables automatic span creation and token tracking for all LLM calls.
        """
        if not OTEL_AVAILABLE:
            return self._original_llm

        original_llm = self._original_llm
        agent_name = self.name
        model_name = self.model
        logger = self.logger

        class TracedLLM:
            """Wrapper that traces all LLM invocations"""

            def __init__(self, llm, agent_name, model_name, logger):
                self._llm = llm
                self._agent_name = agent_name
                self._model_name = model_name
                self._logger = logger

            def invoke(self, *args, **kwargs):
                """Traced invoke method"""
                tracer = trace.get_tracer(__name__)

                # Create a span for this LLM call
                span_name = f"{self._agent_name}.llm_call"
                with tracer.start_as_current_span(span_name) as span:
                    # Add request attributes
                    span.set_attribute("agent.name", self._agent_name)
                    span.set_attribute("llm.model", self._model_name)
                    span.set_attribute("llm.provider", "openai")

                    try:
                        # Call the original LLM
                        response = self._llm.invoke(*args, **kwargs)

                        # Extract and add token attributes
                        token_info = extract_and_add_token_attributes_from_response(
                            span, response, self._model_name
                        )

                        if token_info:
                            self._logger.debug(
                                f"Traced LLM call: {token_info['total_tokens']} tokens, "
                                f"${token_info['cost_usd']:.6f}"
                            )

                        span.set_attribute("llm.status", "success")
                        return response

                    except Exception as e:
                        span.set_attribute("llm.status", "error")
                        span.set_attribute("error.message", str(e))
                        span.record_exception(e)
                        self._logger.error(f"LLM call failed: {e}")
                        raise

            def __getattr__(self, name):
                """Delegate all other attributes to the original LLM"""
                return getattr(self._llm, name)

        return TracedLLM(original_llm, agent_name, model_name, logger)


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
        # Use original unwrapped LLM for LangGraph compatibility
        self.react_agent = create_react_agent(
            model=self._original_llm,
            tools=self.tools,
            prompt=self.system_prompt
        )
    
    async def invoke_with_prompt(self, user_message: str) -> str:
        """
        Invoke the agent with a user message and return the response.

        This is an async method that runs the react agent in a thread pool
        to avoid blocking the event loop.
        """
        try:
            self._log_task_start("agent_invocation", f"with message: {user_message[:100]}...")

            # Create the input state
            input_state = {"messages": [HumanMessage(content=user_message)]}

            # Invoke the react agent with OpenTelemetry tracing
            if OTEL_AVAILABLE:
                tracer = trace.get_tracer(__name__)
                span_name = f"{self.name}.react_agent_call"

                with tracer.start_as_current_span(span_name) as span:
                    # Add request attributes
                    span.set_attribute("agent.name", self.name)
                    span.set_attribute("agent.type", "worker")
                    span.set_attribute("llm.model", self.model)

                    # Invoke react agent in thread pool (non-blocking)
                    result = await asyncio.to_thread(self.react_agent.invoke, input_state)

                    # Extract token usage from result if available
                    if result.get("messages"):
                        final_message = result["messages"][-1]
                        if isinstance(final_message, AIMessage) and hasattr(final_message, 'response_metadata'):
                            # Try to extract and add token attributes
                            try:
                                token_info = extract_and_add_token_attributes_from_response(
                                    span, final_message, self.model
                                )
                                if token_info:
                                    self.logger.debug(
                                        f"React agent call: {token_info['total_tokens']} tokens, "
                                        f"${token_info['cost_usd']:.6f}"
                                    )
                            except Exception as e:
                                self.logger.debug(f"Could not extract token usage: {e}")

                    span.set_attribute("agent.status", "success")
            else:
                # No tracing available, just invoke in thread pool
                result = await asyncio.to_thread(self.react_agent.invoke, input_state)

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
    
    async def process_simple_task(self, task_description: str) -> TaskResult:
        """
        Process a simple task with just a description.

        This is an async method to support non-blocking execution.
        """
        start_time = datetime.now()

        try:
            response = await self.invoke_with_prompt(task_description)
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
    
    async def delegate_task(self, agent_name: str, task: str) -> TaskResult:
        """
        Delegate a task to a specific managed agent.

        This is an async method to support non-blocking delegation.
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
                result = await agent.process_simple_task(task)
            else:
                # For other supervisors, create a state and process
                state = AgentState(
                    messages=[HumanMessage(content=task)],
                    current_step="delegated_task"
                )
                result = await agent.process_task(state)

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
    
    async def coordinate_agents(self, tasks: Dict[str, str]) -> Dict[str, TaskResult]:
        """
        Coordinate multiple agents by assigning tasks to each.
        Returns a dictionary of agent_name -> TaskResult.

        This method executes tasks sequentially. For parallel execution,
        use coordinate_agents_parallel instead.
        """
        self._log_task_start("agent_coordination", f"with {len(tasks)} tasks")
        results = {}

        for agent_name, task in tasks.items():
            result = await self.delegate_task(agent_name, task)
            results[agent_name] = result

        successful_tasks = sum(1 for r in results.values() if r.success)
        self._log_task_completion(
            "agent_coordination",
            successful_tasks == len(tasks),
            f"{successful_tasks}/{len(tasks)} tasks successful"
        )

        return results

    async def coordinate_agents_parallel(self, tasks: Dict[str, str]) -> Dict[str, TaskResult]:
        """
        Coordinate multiple agents by assigning tasks to each IN PARALLEL.
        Returns a dictionary of agent_name -> TaskResult.

        This is the recommended method for executing independent tasks concurrently.
        """
        self._log_task_start("agent_coordination_parallel", f"with {len(tasks)} tasks")

        # Create async tasks for all agents
        async_tasks = {
            agent_name: self.delegate_task(agent_name, task)
            for agent_name, task in tasks.items()
        }

        # Execute all tasks in parallel
        results_list = await asyncio.gather(*async_tasks.values(), return_exceptions=True)

        # Map results back to agent names
        results = {}
        for agent_name, result in zip(async_tasks.keys(), results_list):
            if isinstance(result, Exception):
                # Handle exceptions from failed tasks
                results[agent_name] = self._create_task_result(
                    task_type="task_delegation",
                    success=False,
                    error_message=str(result)
                )
            else:
                results[agent_name] = result

        successful_tasks = sum(1 for r in results.values() if r.success)
        self._log_task_completion(
            "agent_coordination_parallel",
            successful_tasks == len(tasks),
            f"{successful_tasks}/{len(tasks)} tasks successful"
        )

        return results