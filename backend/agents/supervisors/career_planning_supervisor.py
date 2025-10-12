"""
Career Planning Supervisor - Orchestrates Academic Pathway and Skill Development Agents

This supervisor coordinates the career planning process by:
1. Receiving career selection from user (after career predictions)
2. Triggering both Academic Pathway and Skill Development agents IN PARALLEL
3. Collecting and returning results sequentially (academic first, then skills)

This implements the LangChain React agent pattern with handoff tools for orchestration.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

# Use absolute imports
from agents.base_agent import SupervisorAgent
from agents.workers.academic_pathway import AcademicPathwayAgent
from agents.workers.skill_development import SkillDevelopmentAgent
from models.state_models import AgentState, TaskResult, CareerBlueprint
from utils.handoff_tools import HandoffToolFactory

# Import LangSmith for monitoring
from utils.langsmith_config import get_traced_run_config, log_agent_execution


class CareerPlanningSupervisor(SupervisorAgent):
    """
    Career Planning Supervisor - Coordinates academic and skill development planning.

    This supervisor orchestrates the career planning workflow:
    1. Receives user's career choice after career predictions
    2. Triggers both Academic Pathway and Skill Development agents in parallel
    3. Returns results sequentially (academic pathway first, then skill development)

    Features:
    - React agent pattern with handoff tools
    - Parallel execution of both agents for efficiency
    - Sequential result delivery for better UX
    - Session state management
    - Integration with main supervisor workflow
    """

    def __init__(self, **kwargs):
        # Create handoff tools for worker agents
        handoff_factory = HandoffToolFactory()
        handoff_tools = [
            handoff_factory.create_academic_pathway_handoff(),
            handoff_factory.create_skill_development_handoff(),
            handoff_factory.create_main_supervisor_return()
        ]

        orchestration_prompt = self._create_orchestration_prompt()

        super().__init__(
            name="career_planning_supervisor",
            description="Supervisor coordinating academic pathway and skill development planning for chosen careers",
            orchestration_prompt=orchestration_prompt,
            **kwargs
        )

        # Create the react agent with handoff tools
        self.react_agent = create_react_agent(
            model=self.llm,
            tools=handoff_tools,
            prompt=orchestration_prompt
        )

        # Initialize managed worker agents
        self.academic_pathway_agent = AcademicPathwayAgent()
        self.skill_development_agent = SkillDevelopmentAgent()

        # Add to managed agents
        self.add_managed_agent(self.academic_pathway_agent)
        self.add_managed_agent(self.skill_development_agent)

        # Add specific capabilities
        self.capabilities.extend([
            "career_plan_orchestration",
            "parallel_agent_execution",
            "academic_planning_coordination",
            "skill_development_coordination",
            "sequential_result_delivery",
            "human_in_the_loop_career_selection"
        ])

        self.logger = logging.getLogger(f"agent.{self.name}")

    def _create_orchestration_prompt(self) -> str:
        """Create the orchestration prompt for the career planning supervisor."""
        return """You are the Career Planning Supervisor, responsible for coordinating comprehensive career planning for students.

YOUR ROLE:
You work under the Main Supervisor and manage two specialized agents:
1. Academic Pathway Agent - Creates educational roadmaps and university plans
2. Skill Development Agent - Designs skill development roadmaps and learning paths

WORKFLOW:
1. Receive the student's career choice (e.g., "I want to pursue Software Engineer")
2. Coordinate BOTH agents IN PARALLEL to create comprehensive plans
3. Deliver results sequentially:
   - First: Academic pathway plan (education roadmap)
   - Then: Skill development plan (learning roadmap)

ORCHESTRATION PRINCIPLES:
- Execute both agents simultaneously for efficiency
- Ensure comprehensive coverage of both academic and skill aspects
- Maintain consistency between academic and skill plans
- Provide actionable, detailed guidance for the chosen career
- Support the student's journey from current state to career readiness

QUALITY STANDARDS:
- Both plans must be complete and actionable
- Plans should complement each other (academic focus + skill focus)
- Consider Sri Lankan education context for academic pathways
- Include realistic timelines and resource recommendations
- Provide clear next steps for immediate action

You coordinate these agents to deliver a complete career preparation package."""

    def process_task(self, state: AgentState) -> TaskResult:
        """
        Main task processing: Orchestrate career planning for selected career.

        Expected input in state:
        - Selected career title (from user's choice)
        - Student profile
        - Career blueprint with basic information

        Returns:
        - TaskResult with both academic and skill development plans
        """
        start_time = datetime.now()
        session_id = state.session_id or "career_planning_session"

        self._log_task_start("career_planning_orchestration", f"session: {session_id}")

        # Get tracing configuration
        run_config = get_traced_run_config(
            session_type="career_planning",
            agent_name=self.name,
            session_id=session_id,
            additional_tags=["orchestration", "parallel_execution"],
            additional_metadata={
                "career_blueprints_count": len(state.career_blueprints) if state.career_blueprints else 0
            }
        )

        try:
            # Extract selected career information
            selected_career = self._extract_selected_career(state)

            if not selected_career:
                return self._create_task_result(
                    task_type="career_planning_orchestration",
                    success=False,
                    error_message="No career selection found in state"
                )

            career_title = selected_career.get("title")
            self.logger.info(f"Orchestrating career planning for: {career_title}")

            # Execute both agents IN PARALLEL
            academic_result, skill_result = self._execute_agents_in_parallel(state)

            # Check results
            if not academic_result.success:
                self.logger.error(f"Academic pathway agent failed: {academic_result.error_message}")

            if not skill_result.success:
                self.logger.error(f"Skill development agent failed: {skill_result.error_message}")

            # Update state with both results
            updated_state = self._update_state_with_results(
                state, career_title, academic_result, skill_result
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            # Log execution
            log_agent_execution(
                self.name,
                f"career_planning_{career_title}",
                f"Completed orchestration for {career_title}",
                processing_time
            )

            self._log_task_completion(
                "career_planning_orchestration",
                academic_result.success and skill_result.success,
                f"for {career_title} in {processing_time:.2f}s"
            )

            return self._create_task_result(
                task_type="career_planning_orchestration",
                success=academic_result.success and skill_result.success,
                result_data={
                    "career_title": career_title,
                    "academic_plan": academic_result.result_data.get("academic_plan") if (academic_result.success and academic_result.result_data) else None,
                    "skill_plan": skill_result.result_data.get("skill_plan") if (skill_result.success and skill_result.result_data) else None,
                    "academic_plan_summary": academic_result.result_data.get("plan_summary") if (academic_result.success and academic_result.result_data) else None,
                    "skill_plan_summary": skill_result.result_data.get("plan_summary") if (skill_result.success and skill_result.result_data) else None,
                    "academic_plan_ready": academic_result.success,
                    "skill_plan_ready": skill_result.success,
                    "execution_mode": "parallel",
                    "delivery_sequence": ["academic_pathway", "skill_development"]
                },
                processing_time=processing_time,
                updated_state=updated_state
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_task_completion("career_planning_orchestration", False, f"Error: {str(e)}")

            # Log error execution
            log_agent_execution(
                self.name,
                "career_planning_error",
                f"Failed: {str(e)}",
                processing_time
            )

            return self._create_task_result(
                task_type="career_planning_orchestration",
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )

    def _extract_selected_career(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """
        Extract the selected career from state.

        This can come from:
        1. A specific career blueprint marked as selected
        2. User's message indicating career choice
        3. The first career blueprint if only one exists
        """
        # Check if there's a career blueprint
        if state.career_blueprints:
            # If only one blueprint, use it
            if len(state.career_blueprints) == 1:
                blueprint = state.career_blueprints[0]
                return {
                    "title": blueprint.career_title,
                    "description": blueprint.career_description,
                    "blueprint": blueprint
                }

            # Otherwise, check messages for career selection
            if state.messages:
                for msg in reversed(state.messages):
                    if isinstance(msg, HumanMessage):
                        content = msg.content.lower()
                        # Look for career selection in message
                        for blueprint in state.career_blueprints:
                            if blueprint.career_title.lower() in content:
                                return {
                                    "title": blueprint.career_title,
                                    "description": blueprint.career_description,
                                    "blueprint": blueprint
                                }

        # Check student request for career mention
        if state.student_request:
            for blueprint in state.career_blueprints:
                if blueprint.career_title.lower() in state.student_request.lower():
                    return {
                        "title": blueprint.career_title,
                        "description": blueprint.career_description,
                        "blueprint": blueprint
                    }

        # Default to first blueprint if available
        if state.career_blueprints:
            blueprint = state.career_blueprints[0]
            return {
                "title": blueprint.career_title,
                "description": blueprint.career_description,
                "blueprint": blueprint
            }

        return None

    def _execute_agents_in_parallel(self, state: AgentState) -> tuple[TaskResult, TaskResult]:
        """
        Execute both Academic Pathway and Skill Development agents IN PARALLEL.

        Returns:
        - Tuple of (academic_result, skill_result)
        """
        self.logger.info("🔄 Executing agents in parallel...")

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            academic_future = executor.submit(
                self.academic_pathway_agent.process_task,
                state
            )

            skill_future = executor.submit(
                self.skill_development_agent.process_task,
                state
            )

            # Wait for both to complete
            academic_result = academic_future.result()
            skill_result = skill_future.result()

        self.logger.info("✅ Both agents completed execution")

        return academic_result, skill_result

    def _update_state_with_results(
        self,
        state: AgentState,
        career_title: str,
        academic_result: TaskResult,
        skill_result: TaskResult
    ) -> AgentState:
        """Update the state with results from both agents."""

        updated_state = state.copy(deep=True)

        # Find and update the corresponding career blueprint
        if updated_state.career_blueprints:
            for blueprint in updated_state.career_blueprints:
                if blueprint.career_title == career_title:
                    # Update with academic plan if successful
                    if academic_result.success and academic_result.result_data:
                        academic_plan = academic_result.result_data.get("academic_plan")
                        if academic_plan:
                            blueprint.academic_plan = academic_plan
                            self.logger.info(f"✅ Updated blueprint with academic plan for {career_title}")

                    # Update with skill plan if successful
                    if skill_result.success and skill_result.result_data:
                        skill_plan = skill_result.result_data.get("skill_plan")
                        if skill_plan:
                            blueprint.skill_development_plan = skill_plan
                            self.logger.info(f"✅ Updated blueprint with skill plan for {career_title}")

                    break

        # Add completion messages
        completion_message = AIMessage(
            content=f"✅ Career planning completed for {career_title} (Academic + Skills)",
            name=self.name
        )
        updated_state.messages.append(completion_message)

        return updated_state

    def orchestrate_career_planning(
        self,
        career_title: str,
        student_profile: Dict[str, Any],
        career_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to orchestrate career planning.

        Args:
            career_title: The chosen career title
            student_profile: Student profile information
            career_description: Optional career description

        Returns:
            Dictionary with both academic and skill development plans
        """
        # Create a career blueprint
        blueprint = CareerBlueprint(
            career_title=career_title,
            career_description=career_description or f"Career planning for {career_title}",
            match_score=0.0,  # Not relevant for already-selected career
            match_reasoning="User selected career"
        )

        # Create state
        state = AgentState(
            career_blueprints=[blueprint],
            session_id=f"orchestration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Process
        result = self.process_task(state)

        if result.success and result.result_data:
            return {
                "success": True,
                "career_title": career_title,
                "academic_plan": result.result_data.get("academic_plan"),
                "skill_plan": result.result_data.get("skill_plan"),
                "execution_time": result.processing_time
            }
        else:
            return {
                "success": False,
                "error": result.error_message
            }


# Example usage and testing
if __name__ == "__main__":
    from models.state_models import StudentProfile

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("🚀 Testing Career Planning Supervisor")
    print("=" * 50)

    # Create supervisor
    try:
        supervisor = CareerPlanningSupervisor()
        print("✅ Career Planning Supervisor created successfully")
    except Exception as e:
        print(f"❌ Failed to create supervisor: {e}")
        exit(1)

    # Create test data
    test_profile = StudentProfile(
        current_education_level="A/L Student",
        major_field="Physical Science Stream",
        technical_skills=["Programming", "Mathematics"],
        career_interests=["Technology", "Software Development"],
        academic_performance="Good"
    )

    test_blueprint = CareerBlueprint(
        career_title="Software Engineer",
        career_description="Design and develop software applications",
        match_score=88.0,
        match_reasoning="Strong match based on technical interests"
    )

    test_state = AgentState(
        student_profile=test_profile,
        career_blueprints=[test_blueprint],
        session_id="test_orchestration_001"
    )

    # Test orchestration
    print("\n🔄 Testing parallel orchestration...")
    try:
        result = supervisor.process_task(test_state)

        if result.success:
            print("✅ Orchestration successful!")
            print(f"Processing time: {result.processing_time:.2f}s")
            print(f"Execution mode: {result.result_data.get('execution_mode')}")
            print(f"Academic plan ready: {result.result_data.get('academic_plan_ready')}")
            print(f"Skill plan ready: {result.result_data.get('skill_plan_ready')}")
        else:
            print(f"❌ Orchestration failed: {result.error_message}")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")

    print("\n" + "=" * 50)
    print("Career Planning Supervisor test completed!")
