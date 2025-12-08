import logging
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Use absolute imports
from agents.base_agent import WorkerAgent
from models.state_models import (
    AgentState,
    TaskResult,
    SkillDevelopmentStructured,
    SkillGroup,
    SkillItem,
)

# Import LangSmith for monitoring
from utils.langsmith_config import get_traced_run_config, log_agent_execution

# Import Web Search Tool
from utils.web_search_tool import WebSearchTool

# Import RAG System for knowledge base retrieval
from rag.retriever import AgenticRAGRetriever


class SkillDevelopmentAgent(WorkerAgent):
    """
    Skill Development Agent - Creates progressive skill development roadmaps.

    This agent analyzes career requirements and creates detailed skill development
    plans with timelines, learning resources, and achievement milestones.
    """

    def __init__(self, **kwargs):
        system_prompt = self._create_system_prompt()
        
        # Use GPT-4o for complex skill development planning (critical agent)
        kwargs.setdefault('model', 'gpt-4o')

        super().__init__(
            name="skill_development_agent",
            description="Specialist in creating comprehensive skill development roadmaps and learning paths",
            specialization="skill_planning_and_development",
            system_prompt=system_prompt,
            **kwargs,
        )

        # Agent capabilities
        self.capabilities.extend(
            [
                "technical_skill_analysis",
                "soft_skill_assessment",
                "learning_path_design",
                "resource_recommendation",
                "milestone_planning",
                "progressive_skill_building",
            ]
        )

        # Initialize web search tool for real-time information
        self.web_search = WebSearchTool(cache_duration_minutes=120)

        # Initialize RAG retriever for knowledge base (skill collection)
        try:
            self.rag_retriever = AgenticRAGRetriever(
                collection_type="skill",
                provider="fallback",  # Automatically switches with LLM fallback
                similarity_threshold=0.35,
                top_k=10,
            )
            self.rag_enabled = True
            self.logger.info("✅ RAG retriever initialized for skill knowledge base")
        except Exception as e:
            self.logger.warning(
                f"RAG retriever initialization failed: {e}. Continuing without RAG."
            )
            self.rag_enabled = False

        self.logger = logging.getLogger(f"agent.{self.name}")

    def _create_system_prompt(self) -> str:
        """Create the specialized system prompt for skill development planning."""
        return """You are an expert Skill Development Agent specializing in creating comprehensive career skill roadmaps.

YOUR ROLE:
You are a specialist in the Career Planning team, working under the Career Planning Supervisor.
Your specific responsibility is to create detailed, progressive skill development plans for identified careers.

KEY CAPABILITY:
You have access to REAL-TIME WEB SEARCH to gather current information about courses, certifications, skill trends, and job market demands. You will receive search results and MUST use them to provide accurate, current, and specific recommendations with URLs, course names, and actual costs.

CORE COMPETENCIES:
1. Technical Skill Analysis: Identify and categorize technical skills required for careers
2. Soft Skill Assessment: Determine essential interpersonal and professional skills
3. Progressive Learning Design: Create step-by-step skill acquisition paths
4. Resource Curation: Recommend high-quality learning resources (courses, books, platforms)
5. Milestone Planning: Define clear achievement markers and timelines
6. Skill Gap Analysis: Assess current vs. required skills

SKILL DEVELOPMENT FRAMEWORK:

**Technical Skills** (Career-specific):
- Core Technical Skills (Must-have, foundational)
- Advanced Technical Skills (Career progression)
- Specialized Tools & Technologies
- Certifications & Credentials

**Soft Skills** (Universal but career-weighted):
- Communication & Collaboration
- Problem-Solving & Critical Thinking
- Leadership & Management
- Adaptability & Learning Agility
- Time Management & Organization

**Learning Path Structure**:
PHASE 1 - Foundation (0-6 months):
- Beginner-level skills
- Fundamental concepts
- Basic tools and technologies
- Entry-level certifications

PHASE 2 - Intermediate (6-18 months):
- Applied skills development
- Real-world project experience
- Intermediate certifications
- Specialization beginning

PHASE 3 - Advanced (18-36 months):
- Expert-level skills
- Industry recognition
- Advanced certifications
- Thought leadership development

**Learning Resources Types**:
- Online Courses (Coursera, Udemy, LinkedIn Learning)
- Books & Documentation
- Practice Platforms (LeetCode, HackerRank for tech)
- Bootcamps & Workshops
- Professional Communities
- Mentorship Programs

**Milestone Examples**:
1. Complete foundational course
2. Build portfolio project
3. Earn certification
4. Contribute to open source
5. Publish article/blog
6. Speak at meetup/conference

RESPONSE FORMAT:
Create a detailed JSON structure containing:
{
  "career_title": "Career name",
  "technical_skills": {
    "core_skills": [...],
    "advanced_skills": [...],
    "tools_technologies": [...]
  },
  "soft_skills": {
    "essential": [...],
    "recommended": [...]
  },
  "learning_phases": [
    {
      "phase": "Foundation",
      "duration": "0-6 months",
      "skills_focus": [...],
      "resources": [...],
      "milestones": [...]
    }
  ],
  "certifications": [...],
  "estimated_total_time": "X months/years",
  "skill_priorities": "Priority ranking explanation"
}

QUALITY STANDARDS:
1. Actionable and specific skill names
2. Realistic timelines based on industry standards
3. Mix of free and paid learning resources
4. Progressive difficulty (beginner → advanced)
5. Measurable milestones
6. Industry-relevant certifications
7. Balanced technical + soft skills

Remember: Your skill roadmap should empower students with a clear, achievable path 
from their current state to career readiness. Be comprehensive yet realistic."""

    def _search_online_courses(self, career_title: str) -> str:
        """
        Search for current online courses for the career.

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(f"Searching for online courses for {career_title}")

            # Search for courses on popular platforms
            results = self.web_search.search(
                f"{career_title} courses Coursera Udemy LinkedIn Learning 2024",
                max_results=8,
            )

            if not results:
                return "No current course information found."

            return self.web_search.format_results_for_llm(results, max_snippets=5)

        except Exception as e:
            self.logger.error(f"Course search failed: {e}")
            return "Course search unavailable."

    def _search_certifications(self, career_title: str) -> str:
        """
        Search for professional certifications for the career.

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(f"Searching for certifications for {career_title}")

            # Search for relevant certifications
            results = self.web_search.search_certifications(career_title, max_results=8)

            if not results:
                return "No certification information found."

            return self.web_search.format_results_for_llm(results, max_snippets=5)

        except Exception as e:
            self.logger.error(f"Certification search failed: {e}")
            return "Certification search unavailable."

    def _search_skill_trends(self, career_title: str) -> str:
        """
        Search for current skill trends and requirements for the career.

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(f"Searching for skill trends for {career_title}")

            # Search for skill trends
            results = self.web_search.search_skill_trends(career_title, max_results=8)

            if not results:
                return "No skill trend information found."

            return self.web_search.format_results_for_llm(results, max_snippets=5)

        except Exception as e:
            self.logger.error(f"Skill trends search failed: {e}")
            return "Skill trends search unavailable."

    def _search_job_requirements(self, career_title: str) -> str:
        """
        Search for current job requirements and demands.

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(f"Searching for job requirements for {career_title}")

            # Search for job requirements
            results = self.web_search.search_job_requirements(
                career_title, location="worldwide", max_results=6
            )

            if not results:
                return "No job requirement information found."

            return self.web_search.format_results_for_llm(results, max_snippets=4)

        except Exception as e:
            self.logger.error(f"Job requirements search failed: {e}")
            return "Job requirements search unavailable."

    def _search_learning_platforms(self, career_title: str) -> str:
        """
        Search for practice platforms and learning communities.

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(f"Searching for learning platforms for {career_title}")

            # Search for practice platforms
            results = self.web_search.search(
                f"{career_title} practice platforms learning communities coding challenges",
                max_results=6,
            )

            if not results:
                return "No learning platform information found."

            return self.web_search.format_results_for_llm(results, max_snippets=4)

        except Exception as e:
            self.logger.error(f"Learning platform search failed: {e}")
            return "Learning platform search unavailable."

    async def process_task(self, state: AgentState) -> TaskResult:
        """
        Main task processing: Create skill development roadmap for a career.

        This is an async method to support parallel execution with other agents.

        Expected input in state:
        - career_title: Name of the career
        - career_description: Brief description
        - student_profile: StudentProfile object (optional, for personalization)

        Returns:
        - TaskResult with complete skill development plan
        """
        start_time = datetime.now()
        session_id = state.session_id or "skill_dev_session"

        self._log_task_start("skill_development_planning", f"session: {session_id}")

        # Get tracing configuration
        run_config = get_traced_run_config(
            session_type="skill_development",
            agent_name=self.name,
            session_id=session_id,
            additional_tags=["skill_planning", "roadmap_creation"],
            additional_metadata={
                "career_blueprints_count": (
                    len(state.career_blueprints) if state.career_blueprints else 0
                )
            },
        )

        try:
            # Extract career information from state
            career_info = self._extract_career_info(state)

            if not career_info:
                return self._create_task_result(
                    task_type="skill_development",
                    success=False,
                    error_message="No career information found in state",
                )

            career_title = career_info.get("title")
            self.logger.info(f"Creating skill development plan for: {career_title}")

            # Create skill development plan (async call) with language support
            skill_plan = await self._create_skill_development_plan(
                career_title=career_title,
                career_description=career_info.get("description"),
                student_profile=state.student_profile,
                language=state.preferred_language or "en",
            )

            # Update the career blueprint with skill plan
            updated_state = self._update_state_with_skill_plan(
                state, career_title, skill_plan
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            # Log execution
            log_agent_execution(
                self.name,
                f"skill_plan_creation_{career_title}",
                f"Created plan with {len(skill_plan.get('technical_skills', {}).get('core_skills', []))} core skills",
                processing_time,
            )

            self._log_task_completion(
                "skill_development_planning",
                True,
                f"for {career_title} in {processing_time:.2f}s",
            )

            return self._create_task_result(
                task_type="skill_development",
                success=True,
                result_data={
                    "career_title": career_title,
                    "skill_plan": skill_plan,
                    "plan_summary": self._create_plan_summary(skill_plan),
                },
                processing_time=processing_time,
                updated_state=updated_state,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_task_completion(
                "skill_development_planning", False, f"Error: {str(e)}"
            )

            # Log error execution
            log_agent_execution(
                self.name,
                "skill_plan_creation_error",
                f"Failed: {str(e)}",
                processing_time,
            )

            return self._create_task_result(
                task_type="skill_development",
                success=False,
                error_message=str(e),
                processing_time=processing_time,
            )

    def _extract_career_info(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Extract career information from state."""
        # Check career_blueprints for incomplete blueprint
        if state.career_blueprints:
            for blueprint in state.career_blueprints:
                # If blueprint doesn't have skill plan, work on it
                if not blueprint.skill_development_plan:
                    return {
                        "title": blueprint.career_title,
                        "description": blueprint.career_description,
                        "blueprint": blueprint,
                    }

        # Check messages for career assignment
        if state.messages:
            for msg in reversed(state.messages):
                if isinstance(msg, HumanMessage):
                    content = msg.content
                    # Try to extract career title from message
                    if "career" in content.lower():
                        return {"title": content, "description": None}

        return None

    async def _create_skill_development_plan(
        self,
        career_title: str,
        career_description: Optional[str] = None,
        student_profile: Optional[Any] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Create comprehensive skill development plan using LLM.

        This is an async method to support non-blocking LLM invocation.
        """
        # Build prompt with career details and student context
        prompt = self._build_skill_planning_prompt(
            career_title, career_description, student_profile, language
        )

        # Invoke LLM with structured output request (async)
        response = await self.invoke_with_prompt(prompt)

        # DEBUG: Log response details
        self.logger.info(f"📄 LLM Response Length: {len(response)} chars")
        self.logger.info(
            f"📄 LLM Response Preview (first 500 chars):\n{response[:500]}"
        )

        # Parse and structure the response
        skill_plan = self._parse_skill_plan_response(response, career_title)

        return skill_plan

    def _build_skill_planning_prompt(
        self,
        career_title: str,
        career_description: Optional[str],
        student_profile: Optional[Any],
        language: str = "en",
    ) -> str:
        """Build comprehensive prompt for skill planning with RAG and web search results."""

        # Import language helper
        from utils.prompt_templates import get_language_instruction

        # Perform RAG retrieval from knowledge base
        rag_context = ""
        if self.rag_enabled:
            self.logger.info(
                f"Retrieving skill knowledge base information for {career_title}"
            )
            # Optimized query: shorter, keyword-focused for better semantic matching
            rag_state = self.rag_retriever.retrieve(
                query=f"{career_title} skills courses certifications training learning resources",
                include_citations=True,
            )
            if rag_state.context:
                rag_context = rag_state.context
                self.logger.info(
                    f"RAG retrieved {len(rag_state.retrieved_documents)} relevant documents"
                )
            else:
                self.logger.info("No relevant documents found in skill knowledge base")

        # Perform web searches for current information
        self.logger.info(f"Gathering real-time skill information for {career_title}")

        online_courses_info = self._search_online_courses(career_title)
        certifications_info = self._search_certifications(career_title)
        skill_trends_info = self._search_skill_trends(career_title)
        job_requirements_info = self._search_job_requirements(career_title)
        learning_platforms_info = self._search_learning_platforms(career_title)

        # Get language instruction
        language_instruction = ""
        if language != "en":
            language_instruction = (
                get_language_instruction(language, "skill_development") + "\n\n"
            )

        prompt_parts = [
            language_instruction
            + f"Create a comprehensive skill development roadmap for: {career_title}",
            "",
        ]

        if career_description:
            prompt_parts.append(f"Career Description: {career_description}")
            prompt_parts.append("")

        # Add RAG knowledge base context if available
        if rag_context:
            prompt_parts.extend(
                [
                    "=" * 80,
                    "KNOWLEDGE BASE INFORMATION (Verified skill data from PDFs):",
                    "=" * 80,
                    "",
                    rag_context,
                    "",
                ]
            )

        # Add web search results
        prompt_parts.extend(
            [
                "=" * 80,
                "REAL-TIME WEB SEARCH RESULTS (Use this current information):",
                "=" * 80,
                "",
                "CURRENT SKILL TRENDS AND REQUIREMENTS:",
                skill_trends_info,
                "",
                "JOB MARKET REQUIREMENTS:",
                job_requirements_info,
                "",
                "ONLINE COURSES AVAILABLE:",
                online_courses_info,
                "",
                "PROFESSIONAL CERTIFICATIONS:",
                certifications_info,
                "",
                "LEARNING PLATFORMS AND PRACTICE RESOURCES:",
                learning_platforms_info,
                "",
                "=" * 80,
                "END OF WEB SEARCH RESULTS",
                "=" * 80,
                "",
            ]
        )

        if student_profile:
            prompt_parts.append("STUDENT CONTEXT:")
            if hasattr(student_profile, "current_education_level"):
                prompt_parts.append(
                    f"- Education Level: {student_profile.current_education_level}"
                )
            if (
                hasattr(student_profile, "technical_skills")
                and student_profile.technical_skills
            ):
                prompt_parts.append(
                    f"- Current Technical Skills: {', '.join(student_profile.technical_skills[:5])}"
                )
            if (
                hasattr(student_profile, "career_interests")
                and student_profile.career_interests
            ):
                prompt_parts.append(
                    f"- Career Interests: {', '.join(student_profile.career_interests[:3])}"
                )
            prompt_parts.append("")

        prompt_parts.extend(
            [
                "IMPORTANT INSTRUCTIONS:",
                "- USE BOTH THE KNOWLEDGE BASE INFORMATION AND WEB SEARCH RESULTS",
                "- If web search results are insufficient, rely on knowledge base information and your expertise",
                "- Include actual course names, platforms, URLs, and current information when available",
                "- Reference specific certifications found in searches with costs and providers",
                "- Use skill trends from job market to prioritize skills",
                "- YOU MUST ALWAYS GENERATE A COMPLETE SKILL DEVELOPMENT PLAN - DO NOT apologize or refuse",
                "- Follow the MARKDOWN FORMAT structure below exactly",
                "",
                "OUTPUT FORMAT - Write a detailed, well-structured skill development plan with clear markdown headings:",
                "",
                "## 1. TECHNICAL SKILLS BREAKDOWN",
                "",
                "### Core Skills",
                "List 5-7 must-have foundational skills from job requirements:",
                "- [Skill 1]",
                "- [Skill 2]",
                "- [Skill 3]",
                "",
                "### Advanced Skills",
                "List 4-6 career progression skills from trends:",
                "- [Skill 1]",
                "- [Skill 2]",
                "",
                "### Tools & Technologies",
                "List specific tools mentioned in search results:",
                "- [Tool 1]",
                "- [Tool 2]",
                "",
                "## 2. SOFT SKILLS REQUIRED",
                "",
                "### Essential Soft Skills",
                "Top 5-6 MOST CRITICAL soft skills from job postings (must have):",
                "- [Skill 1]",
                "- [Skill 2]",
                "- [Skill 3]",
                "- [Skill 4]",
                "- [Skill 5]",
                "- [Skill 6]",
                "",
                "### Recommended Skills",
                "Additional 4-5 valuable soft skills for career growth:",
                "- [Skill 1]",
                "- [Skill 2]",
                "- [Skill 3]",
                "- [Skill 4]",
                "- [Skill 5]",
                "",
                "## 3. LEARNING ROADMAP",
                "",
                "⚠️ CRITICAL REQUIREMENTS:",
                "1. Each phase MUST include EXACTLY 2-3 comprehensive courses",
                "2. EVERY course MUST have a valid URL (NO EXCEPTIONS - URLs are MANDATORY!)",
                "3. Include BOTH technical AND soft skills courses in EACH phase",
                "4. URLs must be real, working links starting with https://",
                "5. Use actual course URLs from Coursera, Udemy, edX, LinkedIn Learning, etc.",
                "",
                "### Phase 1: Foundation (0-6 months)",
                "**Skills Focus:**",
                "- [Technical skill 1]",
                "- [Technical skill 2]",
                "- [Soft skill development]",
                "",
                "**Learning Resources:**",
                "Provide EXACTLY 2-3 comprehensive courses (MUST include at least one soft skills course):",
                "",
                "⚠️ CRITICAL: URL field is MANDATORY for EVERY course - DO NOT leave it empty!",
                "Examples of valid URLs:",
                "- https://www.coursera.org/learn/course-name",
                "- https://www.udemy.com/course/course-name",
                "- https://www.edx.org/course/course-name",
                "",
                "- **[Course Name]**",
                "  - Description: [Brief description of what the course covers]",
                "  - Platform: [Coursera/Udemy/edX/LinkedIn Learning/Udacity/etc.]",
                "  - URL: [MANDATORY - Full working URL starting with https://]",
                "  - Cost: [Free/$19.99/$49/month/etc.]",
                "  - Duration: [6 months (10 hrs/week) or 52 hours, etc.]",
                "  - Difficulty: [Beginner/Intermediate/Advanced]",
                "  - Rating: [4.8/5 (50K+ reviews) if known]",
                "  - Learning Outcomes:",
                "    * [Outcome 1]",
                "    * [Outcome 2]",
                "    * [Outcome 3]",
                "    * [Outcome 4]",
                "",
                "**Milestones:**",
                "- [Milestone 1]",
                "- [Milestone 2]",
                "",
                "### Phase 2: Intermediate (6-18 months)",
                "**Skills Focus:**",
                "- [Technical skill 1]",
                "- [Technical skill 2]",
                "- [Soft skill development]",
                "",
                "**Learning Resources:**",
                "Provide EXACTLY 2-3 comprehensive courses (MUST include at least one soft skills course):",
                "",
                "⚠️ CRITICAL: Every course MUST have a URL - This field is MANDATORY!",
                "",
                "- **[Course Name]**",
                "  - Description: [Brief description]",
                "  - Platform: [Platform name]",
                "  - URL: [MANDATORY - Full URL starting with https://]",
                "  - Cost: [price]",
                "  - Duration: [time commitment]",
                "  - Difficulty: [level]",
                "  - Rating: [if known]",
                "  - Learning Outcomes:",
                "    * [Outcome 1]",
                "    * [Outcome 2]",
                "    * [Outcome 3]",
                "",
                "**Project-Based Learning:**",
                "- [Project 1]",
                "- [Project 2]",
                "",
                "**Milestones:**",
                "- [Milestone 1]",
                "- [Milestone 2]",
                "",
                "### Phase 3: Advanced (18-36 months)",
                "**Skills Focus:**",
                "- [Emerging technical skill 1]",
                "- [Emerging technical skill 2]",
                "- [Advanced soft skill development]",
                "",
                "**Learning Resources:**",
                "Provide EXACTLY 2-3 comprehensive courses (MUST include at least one leadership/soft skills course):",
                "",
                "⚠️ CRITICAL: URL is MANDATORY for EVERY course!",
                "",
                "- **[Course Name]**",
                "  - Description: [Brief description]",
                "  - Platform: [Platform name]",
                "  - URL: [MANDATORY - Full URL starting with https://]",
                "  - Cost: [price]",
                "  - Duration: [time commitment]",
                "  - Difficulty: [level]",
                "  - Rating: [if known]",
                "  - Learning Outcomes:",
                "    * [Outcome 1]",
                "    * [Outcome 2]",
                "    * [Outcome 3]",
                "",
                "**Industry Recognition:**",
                "- [Opportunity 1]",
                "- [Opportunity 2]",
                "",
                "**Milestones:**",
                "- [Milestone 1]",
                "- [Milestone 2]",
                "",
                "## 4. RECOMMENDED CERTIFICATIONS",
                "",
                "Provide EXACTLY 5 industry certifications with ONLY these essential fields:",
                "",
                "1. **[Certification Name]**",
                "   - Provider: [Organization/Company name]",
                "   - Cost: [Amount like $150, $395, etc.]",
                "   - URL: [Actual URL from search results]",
                "",
                "2. **[Certification Name]**",
                "   - Provider: [Organization]",
                "   - Cost: [Amount]",
                "   - URL: [URL]",
                "",
                "3. **[Certification Name]**",
                "   - Provider: [Organization]",
                "   - Cost: [Amount]",
                "   - URL: [URL]",
                "",
                "4. **[Certification Name]**",
                "   - Provider: [Organization]",
                "   - Cost: [Amount]",
                "   - URL: [URL]",
                "",
                "5. **[Certification Name]**",
                "   - Provider: [Organization]",
                "   - Cost: [Amount]",
                "   - URL: [URL]",
                "",
                "DO NOT include Duration, Prerequisites, or any other fields for certifications.",
                "",
                "## 5. LEARNING RESOURCES",
                "",
                "⚠️ CRITICAL: ALL resources MUST have working URLs - This is MANDATORY!",
                "",
                "### Practice Platforms",
                "Provide EXACTLY 5 practice platforms (URLs are MANDATORY!):",
                "",
                "Example format:",
                "- **LeetCode**",
                "  - Platform: Practice Platform",
                "  - URL: https://leetcode.com",
                "",
                "Your platforms:",
                "- **[Platform Name]**",
                "  - Platform: Practice Platform",
                "  - URL: [MANDATORY - Full URL starting with https://]",
                "",
                "### Online Platforms",
                "Provide EXACTLY 5 online learning platforms (URLs are MANDATORY!):",
                "",
                "Example format:",
                "- **Coursera**",
                "  - Platform: Online Courses",
                "  - URL: https://www.coursera.org",
                "",
                "Your platforms:",
                "- **[Platform Name]**",
                "  - Platform: Online Courses",
                "  - URL: [MANDATORY - Full URL starting with https://]",
                "",
                "### Communities",
                "Provide EXACTLY 5 developer communities (URLs are MANDATORY!):",
                "",
                "Example format:",
                "- **Stack Overflow**",
                "  - Platform: Developer Community",
                "  - URL: https://stackoverflow.com",
                "",
                "Your communities:",
                "- **[Community Name]**",
                "  - Platform: Developer Community",
                "  - URL: [MANDATORY - Full URL starting with https://]",
                "",
                "## 6. SUCCESS METRICS & PORTFOLIO BUILDING",
                "",
                "### Progress Measurement",
                "How to track your development:",
                "- [Metric 1]",
                "- [Metric 2]",
                "",
                "### Portfolio Suggestions",
                "Projects to showcase your skills:",
                "- [Project type 1]",
                "- [Project type 2]",
                "",
                "### Skill Assessment",
                "Methods to validate your skills:",
                "- [Assessment method 1]",
                "- [Assessment method 2]",
                "",
                "Write the complete skill development plan following this exact markdown structure. Use REAL course names, URLs, and current information from the web search results provided above.",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_skill_plan_response(
        self, response: str, career_title: str
    ) -> Dict[str, Any]:
        """Parse LLM response into structured skill plan with multi-line block support."""

        # Check if LLM returned an apologetic/refusal message
        apologetic_phrases = [
            "i'm sorry",
            "i apologize", 
            "i cannot",
            "do not contain",
            "insufficient information",
            "unable to provide",
            "please provide additional"
        ]
        
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in apologetic_phrases) and len(response) < 1000:
            self.logger.warning(f"⚠️ LLM returned apologetic message. Using fallback skill plan generation for {career_title}")
            # Generate basic fallback plan
            response = self._generate_fallback_skill_plan(career_title)

        skill_plan = {
            "career_title": career_title,
            "technical_skills": {
                "core_skills": [],
                "advanced_skills": [],
                "tools_technologies": [],
            },
            "soft_skills": {"essential": [], "recommended": []},
            "learning_phases": [],
            "certifications": [],
            "learning_resources": {
                "online_courses": [],
                "books": [],
                "practice_platforms": [],
                "communities": [],
            },
            "estimated_total_time": "24-36 months",
            "success_metrics": [],
            "raw_plan": response,  # Keep full response for reference
        }

        # Parse with multi-line block support
        lines = response.split("\n")
        current_section = None
        current_phase = None
        current_resource_subsection = None

        # Multi-line block accumulation
        current_course_block = None
        in_course_block = False
        current_cert_block = None
        in_cert_block = False

        phase1_data = {
            "phase": "Foundation",
            "duration": "0-6 months",
            "skills_focus": [],
            "resources": [],
            "milestones": [],
        }
        phase2_data = {
            "phase": "Intermediate",
            "duration": "6-18 months",
            "skills_focus": [],
            "resources": [],
            "milestones": [],
        }
        phase3_data = {
            "phase": "Advanced",
            "duration": "18-36 months",
            "skills_focus": [],
            "resources": [],
            "milestones": [],
        }

        for line in lines:
            line_lower = line.lower().strip()

            # Detect main sections
            if (
                "core technical skills" in line_lower
                or "foundational skills" in line_lower
            ):
                current_section = "core_technical"
                in_course_block = False
                in_cert_block = False
            elif (
                "advanced technical skills" in line_lower
                or "advanced skills" in line_lower
            ):
                current_section = "advanced_technical"
                in_course_block = False
                in_cert_block = False
            elif "tools" in line_lower and "technologies" in line_lower:
                current_section = "tools"
                in_course_block = False
                in_cert_block = False
            elif "essential soft skills" in line_lower or (
                "soft skills" in line_lower and "essential" in line_lower
            ):
                current_section = "soft_skills_essential"
                in_course_block = False
                in_cert_block = False
            elif "recommended skills" in line_lower and current_section == "soft_skills_essential":
                # Recommended soft skills subsection
                current_section = "soft_skills_recommended"
                in_course_block = False
                in_cert_block = False

            # Fix phase detection to be more specific and avoid conflicts
            elif (
                "### phase 1" in line_lower
                or ("foundation (0-6" in line_lower or "foundation (0 -6" in line_lower)
            ) and "roadmap" not in line_lower:
                # Don't create new phase, just switch to it
                current_section = "phase1"
                current_phase = phase1_data
                in_course_block = False
                in_cert_block = False
            elif (
                "### phase 2" in line_lower
                or (
                    "intermediate (6-18" in line_lower
                    or "intermediate (6 -18" in line_lower
                )
            ) and "roadmap" not in line_lower:
                # Don't create new phase, just switch to it
                current_section = "phase2"
                current_phase = phase2_data
                in_course_block = False
                in_cert_block = False
            elif (
                "### phase 3" in line_lower
                or (
                    "advanced (18-36" in line_lower
                    or "advanced (18-24" in line_lower
                    or "advanced (18 -36" in line_lower
                )
            ) and "roadmap" not in line_lower:
                # Don't create new phase, just switch to it
                current_section = "phase3"
                current_phase = phase3_data
                in_course_block = False
                in_cert_block = False

            # Detect Certifications section
            elif "## 4. recommended certifications" in line_lower or (
                "certification" in line_lower and "##" in line
            ):
                current_section = "certifications"
                in_course_block = False
                in_cert_block = False

            # Detect Learning Resources section
            elif "## 5. learning resources" in line_lower or (
                "learning resources" in line_lower and "##" in line
            ):
                current_section = "learning_resources"
                current_resource_subsection = None
                in_course_block = False
                in_cert_block = False

            # Detect Learning Resources subsections
            elif current_section == "learning_resources":
                if "### practice platforms" in line_lower:
                    current_resource_subsection = "practice_platforms"
                    in_course_block = False
                elif (
                    "### online platforms" in line_lower
                    or "### online courses" in line_lower
                ):
                    current_resource_subsection = "online_courses"
                    in_course_block = False
                elif "### communities" in line_lower:
                    current_resource_subsection = "communities"
                    in_course_block = False
                elif "### books" in line_lower:
                    current_resource_subsection = "books"
                    in_course_block = False

            # Detect start of course/resource block (bold text like **Course Name**)
            if re.match(r"^\s*[-*•]\s*\*\*[^*]+\*\*", line):
                # Process previous block if exists
                if in_course_block and current_course_block:
                    self._process_course_block(
                        current_course_block,
                        current_section,
                        phase1_data,
                        phase2_data,
                        phase3_data,
                        skill_plan["learning_resources"],
                        current_resource_subsection,
                    )
                if in_cert_block and current_cert_block:
                    self._process_cert_block(
                        current_cert_block, skill_plan["certifications"]
                    )

                # Start new block
                current_course_block = line
                in_course_block = True
                in_cert_block = False
                continue

            # Detect start of certification block (numbered like "1. **Cert Name**")
            if current_section == "certifications" and re.match(
                r"^\d+\.\s*\*\*[^*]+\*\*", line
            ):
                # Process previous cert block if exists
                if in_cert_block and current_cert_block:
                    self._process_cert_block(
                        current_cert_block, skill_plan["certifications"]
                    )

                # Start new cert block
                current_cert_block = line
                in_cert_block = True
                in_course_block = False
                continue

            # Accumulate lines that belong to current course block
            if in_course_block and (
                line.startswith("  -")
                or line.startswith("    *")
                or line.startswith("    -")
            ):
                current_course_block += "\n" + line
                continue

            # Accumulate lines that belong to current cert block
            if in_cert_block and (line.startswith("  -") or line.startswith("    *")):
                current_cert_block += "\n" + line
                continue

            # End of course/cert block - process it
            if (
                (in_course_block or in_cert_block)
                and line.strip()
                and not line.startswith(("  ", "    ", "-", "*", "•"))
            ):
                if in_course_block and current_course_block:
                    self._process_course_block(
                        current_course_block,
                        current_section,
                        phase1_data,
                        phase2_data,
                        phase3_data,
                        skill_plan["learning_resources"],
                        current_resource_subsection,
                    )
                    current_course_block = None
                    in_course_block = False
                if in_cert_block and current_cert_block:
                    self._process_cert_block(
                        current_cert_block, skill_plan["certifications"]
                    )
                    current_cert_block = None
                    in_cert_block = False

            # Extract simple bullet points (for Technical/Soft Skills)
            if (
                line.strip().startswith(("-", "•", "*", "✅"))
                and not in_course_block
                and not in_cert_block
            ):
                item = line.strip().lstrip("-•*✅ ").strip()
                item_lower = item.lower()
                
                # Define soft skill keywords to exclude from technical sections
                soft_skill_keywords = [
                    "communication", "collaboration", "leadership", "teamwork",
                    "problem-solving", "critical thinking", "adaptability", "creativity",
                    "emotional intelligence", "time management", "patience", "perseverance",
                    "curiosity", "strategic thinking", "learning agility", "interpersonal",
                    "presentation", "negotiation", "conflict resolution", "empathy"
                ]
                
                # Check if item is a soft skill
                is_soft_skill = any(keyword in item_lower for keyword in soft_skill_keywords)
                
                # Skip if this is a bold course title (already handled above)
                if item and len(item) > 3 and not item.startswith("**"):
                    if current_section == "core_technical":
                        if not is_soft_skill:
                            skill_plan["technical_skills"]["core_skills"].append(item)
                    elif current_section == "advanced_technical":
                        if not is_soft_skill:
                            skill_plan["technical_skills"]["advanced_skills"].append(item)
                    elif current_section == "tools":
                        # Only add if NOT a soft skill
                        if not is_soft_skill:
                            skill_plan["technical_skills"]["tools_technologies"].append(
                                item
                            )
                    elif current_section == "soft_skills_essential":
                        skill_plan["soft_skills"]["essential"].append(item)
                    elif current_section == "soft_skills_recommended":
                        skill_plan["soft_skills"]["recommended"].append(item)
                    elif current_phase:
                        if "milestone" in line_lower:
                            current_phase["milestones"].append(item)
                        elif "skills focus" not in line_lower:
                            current_phase["skills_focus"].append(item)

        # Process any remaining blocks
        if in_course_block and current_course_block:
            self._process_course_block(
                current_course_block,
                current_section,
                phase1_data,
                phase2_data,
                phase3_data,
                skill_plan["learning_resources"],
                current_resource_subsection,
            )
        if in_cert_block and current_cert_block:
            self._process_cert_block(current_cert_block, skill_plan["certifications"])

        # Add phases to skill plan
        if phase1_data["resources"] or phase1_data["skills_focus"]:
            skill_plan["learning_phases"].append(phase1_data)
        if phase2_data["resources"] or phase2_data["skills_focus"]:
            skill_plan["learning_phases"].append(phase2_data)
        if phase3_data["resources"] or phase3_data["skills_focus"]:
            skill_plan["learning_phases"].append(phase3_data)

        # Ensure minimum data
        if not skill_plan["technical_skills"]["core_skills"]:
            skill_plan["technical_skills"]["core_skills"] = [
                "Foundational knowledge in core area",
                "Industry-standard tools proficiency",
                "Problem-solving methodologies",
            ]

        if not skill_plan["soft_skills"]["essential"]:
            skill_plan["soft_skills"]["essential"] = [
                "Communication & Collaboration",
                "Critical Thinking",
                "Adaptability",
                "Time Management",
            ]

        if not skill_plan["learning_phases"]:
            skill_plan["learning_phases"] = self._create_default_learning_phases()

        # FINAL CLEANUP: Remove any soft skills from tools_technologies array
        # This is a safety check in case LLM mistakenly puts soft skills in tools section
        soft_skill_keywords = [
            "communication", "collaboration", "leadership", "teamwork",
            "problem-solving", "critical thinking", "adaptability", "creativity",
            "emotional intelligence", "time management", "patience", "perseverance",
            "curiosity", "strategic thinking", "learning agility", "interpersonal",
            "presentation", "negotiation", "conflict resolution", "empathy",
            "analytical thinking", "problem solving"
        ]
        
        cleaned_tools = []
        for tool in skill_plan["technical_skills"]["tools_technologies"]:
            tool_lower = tool.lower()
            is_soft_skill = any(keyword in tool_lower for keyword in soft_skill_keywords)
            if not is_soft_skill:
                cleaned_tools.append(tool)
            else:
                self.logger.info(f"🧹 Removing soft skill from tools: {tool}")
                # Add to soft skills essential if not already there
                if tool not in skill_plan["soft_skills"]["essential"]:
                    skill_plan["soft_skills"]["essential"].append(tool)
        
        skill_plan["technical_skills"]["tools_technologies"] = cleaned_tools

        return skill_plan

    def _process_course_block(
        self,
        block: str,
        section: str,
        phase1_data: dict,
        phase2_data: dict,
        phase3_data: dict,
        learning_resources: dict,
        current_resource_subsection: str,
    ):
        """Process a multi-line course block and add to appropriate location."""
        import re

        # For Learning Roadmap phases
        if section in ["phase1", "phase2", "phase3"]:
            # Parse with comprehensive parser
            parsed = self._parse_comprehensive_course(block)
            if parsed:
                # Add to phase resources as dict (not string!)
                if section == "phase1":
                    phase1_data["resources"].append(parsed)
                elif section == "phase2":
                    phase2_data["resources"].append(parsed)
                elif section == "phase3":
                    phase3_data["resources"].append(parsed)
            else:
                # Fallback: extract name from bold text
                name_match = re.search(r"\*\*([^*]+)\*\*", block)
                if name_match:
                    name = name_match.group(1).strip()
                    if section == "phase1":
                        phase1_data["resources"].append(name)
                    elif section == "phase2":
                        phase2_data["resources"].append(name)
                    elif section == "phase3":
                        phase3_data["resources"].append(name)

        # For Learning Resources section
        elif section == "learning_resources" and current_resource_subsection:
            subsection = current_resource_subsection
            if subsection in [
                "practice_platforms",
                "online_courses",
                "communities",
                "books",
            ]:
                # Parse with comprehensive resource parser
                parsed = self._parse_comprehensive_resource(block)
                if parsed:
                    learning_resources[subsection].append(parsed)
                else:
                    # Fallback: extract name only
                    name_match = re.search(r"\*\*([^*]+)\*\*", block)
                    if name_match:
                        learning_resources[subsection].append(
                            name_match.group(1).strip()
                        )

    def _process_cert_block(self, block: str, certifications: list):
        """Process a multi-line certification block and add to certifications list."""
        import re

        # Parse with comprehensive certification parser
        parsed = self._parse_comprehensive_certification(block)
        if parsed:
            certifications.append(parsed)
        else:
            # Fallback: extract name only
            name_match = re.search(r"\*\*([^*]+)\*\*", block)
            if name_match:
                certifications.append(name_match.group(1).strip())

    def _generate_fallback_skill_plan(self, career_title: str) -> str:
        """Generate a basic fallback skill plan when LLM fails or web search returns no results."""
        self.logger.info(f"Generating fallback skill plan for {career_title}")
        
        # Extract career type for better fallback
        career_lower = career_title.lower()
        
        # Determine category
        if any(word in career_lower for word in ['developer', 'programmer', 'engineer', 'software']):
            core_skills = ["Programming fundamentals", "Problem-solving", "Version control (Git)", "Database basics", "Software design patterns"]
            advanced_skills = ["System architecture", "Cloud computing", "DevOps practices", "Microservices"]
            tools = ["VS Code/IDE", "Git/GitHub", "Docker", "CI/CD tools"]
        elif any(word in career_lower for word in ['data', 'analyst', 'scientist']):
            core_skills = ["Statistics", "Python/R programming", "SQL", "Data visualization", "Machine learning basics"]
            advanced_skills = ["Deep learning", "Big data technologies", "Advanced ML algorithms", "Data engineering"]
            tools = ["Python", "Jupyter", "Pandas", "SQL databases", "Tableau/Power BI"]
        elif any(word in career_lower for word in ['designer', 'ui', 'ux']):
            core_skills = ["Design principles", "User research", "Wireframing", "Prototyping", "Color theory"]
            advanced_skills = ["Design systems", "Animation", "Advanced prototyping", "Accessibility"]
            tools = ["Figma", "Adobe XD", "Sketch", "InVision", "Adobe Creative Suite"]
        else:
            # Generic professional skills
            core_skills = ["Communication", "Critical thinking", "Project management", "Technical proficiency", "Domain knowledge"]
            advanced_skills = ["Leadership", "Strategic planning", "Innovation", "Advanced technical skills"]
            tools = ["Industry-standard software", "Collaboration tools", "Project management tools"]
        
        # Generate markdown plan
        plan = f"""## 1. TECHNICAL SKILLS BREAKDOWN

### Core Skills
{chr(10).join(f'- {skill}' for skill in core_skills)}

### Advanced Skills
{chr(10).join(f'- {skill}' for skill in advanced_skills)}

### Tools & Technologies
{chr(10).join(f'- {tool}' for tool in tools)}

## 2. SOFT SKILLS REQUIRED

### Essential Skills
- Communication & Collaboration
- Problem-Solving & Critical Thinking
- Adaptability & Continuous Learning
- Time Management
- Attention to Detail

### Recommended Skills
- Leadership & Mentorship
- Creativity & Innovation
- Emotional Intelligence

## 3. LEARNING ROADMAP

### Phase 1: Foundation (0-6 months)
**Skills Focus:**
- Master core fundamentals
- Build strong foundation
- Develop good practices

**Learning Resources:**
- **Introduction to {career_title}**
  - Description: Comprehensive beginner course covering fundamentals
  - Platform: Coursera
  - URL: https://www.coursera.org
  - Cost: $49/month
  - Duration: 3 months (8-10 hrs/week)
  - Difficulty: Beginner
  - Rating: 4.5/5
  - Learning Outcomes:
    * Understand core concepts
    * Build practical projects
    * Develop problem-solving skills
    * Create portfolio pieces

- **{career_title} Fundamentals**
  - Description: Hands-on introduction to key skills
  - Platform: Udemy
  - URL: https://www.udemy.com
  - Cost: $19.99
  - Duration: 40 hours
  - Difficulty: Beginner
  - Rating: 4.3/5
  - Learning Outcomes:
    * Master basic techniques
    * Complete practical exercises
    * Build confidence

- **Effective Communication Skills**
  - Description: Develop professional communication and collaboration skills
  - Platform: LinkedIn Learning
  - URL: https://www.linkedin.com/learning
  - Cost: Free with subscription
  - Duration: 2 hours
  - Difficulty: Beginner
  - Rating: 4.6/5
  - Learning Outcomes:
    * Communicate clearly and effectively
    * Collaborate with team members
    * Present ideas professionally
    * Handle difficult conversations

**Milestones:**
- Complete foundational courses
- Build 2-3 beginner projects

### Phase 2: Intermediate (6-18 months)
**Skills Focus:**
- Apply skills in real projects
- Learn advanced concepts
- Specialize in areas of interest

**Learning Resources:**
- **Advanced {career_title} Skills**
  - Description: Intermediate to advanced concepts and practices
  - Platform: edX
  - URL: https://www.edx.org
  - Cost: $199
  - Duration: 6 months (6-8 hrs/week)
  - Difficulty: Intermediate
  - Rating: 4.6/5
  - Learning Outcomes:
    * Master advanced techniques
    * Build complex projects
    * Understand industry standards

- **Critical Thinking and Problem Solving**
  - Description: Advanced problem-solving and analytical skills
  - Platform: Coursera
  - URL: https://www.coursera.org
  - Cost: $49/month
  - Duration: 4 weeks
  - Difficulty: Intermediate
  - Rating: 4.7/5
  - Learning Outcomes:
    * Analyze complex problems
    * Develop effective solutions
    * Make data-driven decisions
    * Think strategically

**Project-Based Learning:**
- Build 3-5 portfolio projects
- Contribute to open source

**Milestones:**
- Complete advanced coursework
- Earn professional certifications

### Phase 3: Advanced (18-36 months)
**Skills Focus:**
- Specialize deeply
- Stay current with trends
- Build expertise

**Learning Resources:**
- **Expert {career_title} Specialization**
  - Description: Master-level specialization in cutting-edge techniques
  - Platform: Coursera Professional Certificate
  - URL: https://www.coursera.org
  - Cost: $79/month
  - Duration: 12 months (5-7 hrs/week)
  - Difficulty: Advanced
  - Rating: 4.7/5
  - Learning Outcomes:
    * Achieve expert-level proficiency
    * Lead projects and teams
    * Contribute to field

- **Leadership and Team Management**
  - Description: Develop leadership skills for managing teams
  - Platform: Udemy
  - URL: https://www.udemy.com
  - Cost: $19.99
  - Duration: 6 hours
  - Difficulty: Advanced
  - Rating: 4.5/5
  - Learning Outcomes:
    * Lead and motivate teams
    * Manage projects effectively
    * Mentor junior professionals
    * Drive organizational success

**Industry Recognition:**
- Speak at conferences
- Publish articles/blogs
- Mentor others

**Milestones:**
- Achieve expert status
- Build professional network

## 4. RECOMMENDED CERTIFICATIONS

1. **Certified Professional {career_title}**
   - Provider: Industry Leading Organization
   - Cost: $300
   - URL: https://www.certifying-body.org

2. **Advanced {career_title} Certification**
   - Provider: Professional Association
   - Cost: $250
   - URL: https://www.professional-certs.org

3. **Specialist {career_title} Certification**
   - Provider: Technology Company
   - Cost: $200
   - URL: https://www.tech-certs.com

4. **Expert {career_title} Certificate**
   - Provider: Online Learning Platform
   - Cost: $150
   - URL: https://www.online-learning.com

5. **Professional Development Certificate**
   - Provider: Educational Institution
   - Cost: $350
   - URL: https://www.educational-certs.edu

## 5. LEARNING RESOURCES

### Practice Platforms
- **LeetCode**
  - Platform: Practice Platform
  - URL: https://leetcode.com

- **HackerRank**
  - Platform: Practice Platform
  - URL: https://www.hackerrank.com

- **CodeSignal**
  - Platform: Practice Platform
  - URL: https://codesignal.com

- **Exercism**
  - Platform: Practice Platform
  - URL: https://exercism.org

- **Kaggle**
  - Platform: Practice Platform
  - URL: https://www.kaggle.com

### Online Platforms
- **Coursera**
  - Platform: Online Courses
  - URL: https://www.coursera.org

- **Udemy**
  - Platform: Online Courses
  - URL: https://www.udemy.com

- **edX**
  - Platform: Online Courses
  - URL: https://www.edx.org

- **LinkedIn Learning**
  - Platform: Online Courses
  - URL: https://www.linkedin.com/learning

- **Pluralsight**
  - Platform: Online Courses
  - URL: https://www.pluralsight.com

### Communities
- **Stack Overflow**
  - Platform: Developer Community
  - URL: https://stackoverflow.com

- **Reddit - r/cscareerquestions**
  - Platform: Developer Community
  - URL: https://www.reddit.com/r/cscareerquestions

- **Dev.to**
  - Platform: Developer Community
  - URL: https://dev.to

- **GitHub Discussions**
  - Platform: Developer Community
  - URL: https://github.com

- **Discord - Programming Communities**
  - Platform: Developer Community
  - URL: https://discord.com/invite/programming

## 6. SUCCESS METRICS & PORTFOLIO BUILDING

### Progress Measurement
- Complete courses and earn certificates
- Build portfolio projects
- Contribute to open source
- Pass certification exams

### Portfolio Suggestions
- 5-10 diverse projects showcasing skills
- Technical blog or documentation
- Open source contributions
- Professional GitHub profile

### Skill Assessment
- Practice coding challenges regularly
- Participate in competitions
- Seek peer code reviews
- Take practice certification exams
"""
        
        return plan

    def _create_default_learning_phases(self) -> List[Dict[str, Any]]:
        """Create default learning phases as fallback."""
        return [
            {
                "phase": "Foundation",
                "duration": "0-6 months",
                "skills_focus": [
                    "Core fundamentals",
                    "Basic tools",
                    "Foundational concepts",
                ],
                "resources": [
                    "Online courses",
                    "Introductory books",
                    "Practice exercises",
                ],
                "milestones": ["Complete foundational course", "Build first project"],
            },
            {
                "phase": "Intermediate",
                "duration": "6-18 months",
                "skills_focus": [
                    "Applied skills",
                    "Real-world projects",
                    "Specialization",
                ],
                "resources": [
                    "Advanced courses",
                    "Industry documentation",
                    "Project work",
                ],
                "milestones": ["Build portfolio projects", "Earn certifications"],
            },
            {
                "phase": "Advanced",
                "duration": "18-36 months",
                "skills_focus": [
                    "Expert-level skills",
                    "Thought leadership",
                    "Innovation",
                ],
                "resources": [
                    "Advanced specialization",
                    "Research papers",
                    "Conferences",
                ],
                "milestones": [
                    "Industry recognition",
                    "Mentorship",
                    "Speaking engagements",
                ],
            },
        ]

    def _update_state_with_skill_plan(
        self, state: AgentState, career_title: str, skill_plan: Dict[str, Any]
    ) -> AgentState:
        """Update the state with the completed skill development plan."""

        updated_state = state.copy(deep=True)

        # Find and update the corresponding career blueprint
        if updated_state.career_blueprints:
            for blueprint in updated_state.career_blueprints:
                if blueprint.career_title == career_title:
                    blueprint.skill_development_plan = skill_plan
                    # NEW: Generate and add structured format for frontend
                    try:
                        structured_data = self._convert_to_frontend_format(skill_plan)
                        blueprint.skill_plan_structured = structured_data.dict()
                        self.logger.info(
                            f"✅ Generated structured skill development plan for {career_title} with {len(structured_data.skillGroups)} skill groups"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"❌ Failed to generate structured format for {career_title}: {e}",
                            exc_info=True,
                        )
                        blueprint.skill_plan_structured = None
                    self.logger.info(
                        f"Updated blueprint for {career_title} with skill plan"
                    )
                    break

        # Add completion message
        completion_message = AIMessage(
            content=f" Completed skill development plan for {career_title}",
            name=self.name,
        )
        updated_state.messages.append(completion_message)

        return updated_state

    def _create_plan_summary(self, skill_plan: Dict[str, Any]) -> str:
        """Create a brief summary of the skill plan."""

        core_skills_count = len(
            skill_plan.get("technical_skills", {}).get("core_skills", [])
        )
        phases_count = len(skill_plan.get("learning_phases", []))
        certifications_count = len(skill_plan.get("certifications", []))

        summary = f"Skill Development Plan: {core_skills_count} core skills, "
        summary += f"{phases_count} learning phases, "
        summary += f"{certifications_count} recommended certifications"

        return summary

    def _convert_to_frontend_format(
        self, skill_plan: Dict[str, Any]
    ) -> SkillDevelopmentStructured:
        """Convert internal skill_plan to frontend card format with detailed SkillItem objects."""
        skill_groups = []
        group_id = 1

        # Technical Skills - Core Skills (keep as simple strings)
        core_skills = skill_plan.get("technical_skills", {}).get("core_skills", [])
        if core_skills:
            skill_groups.append(
                SkillGroup(
                    id=group_id,
                    category="Technical Skills - Core Skills",
                    items=core_skills,
                )
            )
            group_id += 1

        # Technical Skills - Advanced Skills (keep as simple strings)
        advanced_skills = skill_plan.get("technical_skills", {}).get(
            "advanced_skills", []
        )
        if advanced_skills:
            skill_groups.append(
                SkillGroup(
                    id=group_id,
                    category="Technical Skills - Advanced Skills",
                    items=advanced_skills,
                )
            )
            group_id += 1

        # Technical Skills - Tools and Technologies (keep as simple strings)
        tools = skill_plan.get("technical_skills", {}).get("tools_technologies", [])
        if tools:
            skill_groups.append(
                SkillGroup(
                    id=group_id,
                    category="Technical Skills - Tools and Technologies",
                    items=tools,
                )
            )
            group_id += 1

        # Soft Skills - Essential Soft Skills (keep as simple strings)
        essential_soft = skill_plan.get("soft_skills", {}).get("essential", [])
        if essential_soft:
            skill_groups.append(
                SkillGroup(
                    id=group_id,
                    category="Soft Skills - Essential Soft Skills",
                    items=essential_soft,
                )
            )
            group_id += 1

        # Soft Skills - Recommend Skills (keep as simple strings)
        recommended_soft = skill_plan.get("soft_skills", {}).get("recommended", [])
        if recommended_soft:
            skill_groups.append(
                SkillGroup(
                    id=group_id,
                    category="Soft Skills - Recommend Skills",
                    items=recommended_soft,
                )
            )
            group_id += 1

        # Learning Road Map (by phase) - Parse resources into comprehensive SkillItem objects
        for phase in skill_plan.get("learning_phases", []):
            phase_name = phase.get("phase", "Phase")
            duration = phase.get("duration", "")
            resources = phase.get("resources", [])

            if resources:
                category_name = f"Learning Road Map - {phase_name}"
                if duration:
                    category_name += f" ({duration})"

                # Parse resources to create comprehensive SkillItem objects
                resource_items = []
                for resource in resources:
                    if isinstance(resource, dict):
                        # Already parsed with comprehensive details
                        resource_items.append(SkillItem(**resource))
                    elif isinstance(resource, str):
                        # Parse string format with comprehensive parser
                        parsed_item = self._parse_comprehensive_course(resource)
                        if parsed_item:
                            resource_items.append(SkillItem(**parsed_item))
                        else:
                            # Fallback to basic parser if comprehensive fails
                            basic_item = self._parse_course_string(resource)
                            resource_items.append(basic_item)
                    else:
                        resource_items.append(resource)

                if resource_items:
                    skill_groups.append(
                        SkillGroup(
                            id=group_id, category=category_name, items=resource_items
                        )
                    )
                    group_id += 1

        # Recommended Courses (certifications) - Parse into SkillItem objects with essential fields only
        certifications = skill_plan.get("certifications", [])
        if certifications:
            cert_items = []
            for cert in certifications:
                if isinstance(cert, dict):
                    # Already a dictionary, convert to SkillItem (essential fields only)
                    cert_items.append(
                        SkillItem(
                            name=cert.get("name", "Certification"),
                            provider=cert.get(
                                "provider", None
                            ),  # Use provider field, not platform
                            cost=cert.get("cost", None),
                            url=cert.get("url", None),
                        )
                    )
                elif isinstance(cert, str):
                    # Parse string format with comprehensive parser
                    parsed_cert = self._parse_comprehensive_certification(cert)
                    if parsed_cert:
                        cert_items.append(SkillItem(**parsed_cert))
                    else:
                        # Fallback to basic parser if comprehensive fails
                        basic_cert = self._parse_certification_string(cert)
                        cert_items.append(basic_cert)
                else:
                    cert_items.append(str(cert))

            if cert_items:
                skill_groups.append(
                    SkillGroup(
                        id=group_id, category="Recommended Courses", items=cert_items
                    )
                )
                group_id += 1

        # Learning Resources - Parse into SkillItem objects with essential fields
        resources = skill_plan.get("learning_resources", {})

        # Practice Platforms
        if resources.get("practice_platforms"):
            platform_items = []
            for platform in resources["practice_platforms"]:
                if isinstance(platform, dict):
                    platform_items.append(
                        SkillItem(
                            name=platform.get("name", "Platform"),
                            platform=platform.get("platform", "Practice Platform"),
                            url=platform.get("url"),
                        )
                    )
                elif isinstance(platform, str):
                    parsed = self._parse_comprehensive_resource(platform)
                    if parsed:
                        platform_items.append(SkillItem(**parsed))
                    else:
                        platform_items.append(platform)
                else:
                    platform_items.append(platform)

            if platform_items:
                skill_groups.append(
                    SkillGroup(
                        id=group_id,
                        category="Learning Resources - Practice Platforms",
                        items=platform_items,
                    )
                )
                group_id += 1

        # Online Platforms
        if resources.get("online_courses"):
            online_platform_items = []
            for platform in resources["online_courses"]:
                if isinstance(platform, dict):
                    online_platform_items.append(
                        SkillItem(
                            name=platform.get("name", "Platform"),
                            platform=platform.get("platform", "Online Courses"),
                            url=platform.get("url"),
                        )
                    )
                elif isinstance(platform, str):
                    parsed = self._parse_comprehensive_resource(platform)
                    if parsed:
                        online_platform_items.append(SkillItem(**parsed))
                    else:
                        online_platform_items.append(platform)
                else:
                    online_platform_items.append(platform)

            if online_platform_items:
                skill_groups.append(
                    SkillGroup(
                        id=group_id,
                        category="Learning Resources - Online Platforms",
                        items=online_platform_items,
                    )
                )
                group_id += 1

        # Communities
        if resources.get("communities"):
            community_items = []
            for community in resources["communities"]:
                if isinstance(community, dict):
                    community_items.append(
                        SkillItem(
                            name=community.get("name", "Community"),
                            platform=community.get("platform", "Community"),
                            url=community.get("url"),
                        )
                    )
                elif isinstance(community, str):
                    parsed = self._parse_comprehensive_resource(community)
                    if parsed:
                        community_items.append(SkillItem(**parsed))
                    else:
                        community_items.append(community)
                else:
                    community_items.append(community)

            if community_items:
                skill_groups.append(
                    SkillGroup(
                        id=group_id,
                        category="Learning Resources - Communities",
                        items=community_items,
                    )
                )
                group_id += 1

        return SkillDevelopmentStructured(
            title=skill_plan.get("career_title", "Career"),
            pathwayTitle="Required Skills",
            description="Develop these essential skills to excel in your career",
            skillGroups=skill_groups,
        )

    def _parse_course_string(self, course_str: str) -> Union[str, SkillItem]:
        """
        Parse course string into SkillItem object.
        Expected formats:
        - "Course Name (Platform) - URL"
        - "Course Name - Platform - URL"
        - "Course Name (Platform)"
        """
        import re

        try:
            # Pattern 1: "Course Name (Platform) - URL"
            match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*-\s*(https?://\S+)", course_str)
            if match:
                return SkillItem(
                    name=match.group(1).strip(),
                    platform=match.group(2).strip(),
                    url=match.group(3).strip(),
                )

            # Pattern 2: "Course Name (Platform)"
            match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", course_str)
            if match:
                return SkillItem(
                    name=match.group(1).strip(), platform=match.group(2).strip()
                )

            # Pattern 3: "Course Name - Platform - URL"
            parts = course_str.split(" - ")
            if len(parts) >= 3:
                # Check if last part is a URL
                url = (
                    parts[-1].strip() if parts[-1].strip().startswith("http") else None
                )
                return SkillItem(
                    name=parts[0].strip(),
                    platform=parts[1].strip() if len(parts) > 1 else None,
                    url=url,
                )
            elif len(parts) == 2:
                # Check if second part is a URL
                second_part = parts[1].strip()
                if second_part.startswith("http"):
                    return SkillItem(name=parts[0].strip(), url=second_part)
                else:
                    return SkillItem(name=parts[0].strip(), platform=second_part)

            # If no pattern matches, return as simple string
            return course_str
        except Exception as e:
            self.logger.warning(f"Failed to parse course string '{course_str}': {e}")
            return course_str

    def _parse_certification_string(self, cert_str: str) -> Union[str, SkillItem]:
        """
        Parse certification string into SkillItem object.
        Expected formats:
        - "Cert Name - Provider - Cost - Duration - URL"
        - "Cert Name - Provider - Cost - URL"
        - "Cert Name - Provider - URL"
        """
        import re

        try:
            # Remove any leading numbers or bullet points
            cert_str = re.sub(r"^\d+\.\s*\*?\*?", "", cert_str).strip()
            cert_str = re.sub(r"^\*\*(.+?)\*\*", r"\1", cert_str).strip()

            # Split by " - " delimiter
            parts = [p.strip() for p in cert_str.split(" - ") if p.strip()]

            if len(parts) == 0:
                return cert_str

            # Extract components
            name = parts[0]
            platform = None
            cost = None
            duration = None
            url = None

            # Process remaining parts
            for i, part in enumerate(parts[1:], 1):
                part_lower = part.lower()

                # Check if it's a URL
                if part.startswith("http"):
                    url = part
                # Check if it's a cost (contains $, USD, LKR, Free, etc.)
                elif any(
                    indicator in part_lower
                    for indicator in ["$", "usd", "lkr", "free", "paid"]
                ):
                    cost = part
                # Check if it's a duration (contains time indicators)
                elif any(
                    indicator in part_lower
                    for indicator in ["month", "week", "hour", "day", "year"]
                ):
                    duration = part
                # First non-URL, non-cost, non-duration part is likely the platform/provider
                elif platform is None:
                    platform = part

            # If we extracted meaningful data, return SkillItem
            if platform or cost or duration or url:
                return SkillItem(
                    name=name, platform=platform, cost=cost, duration=duration, url=url
                )

            # Otherwise, return as string
            return cert_str
        except Exception as e:
            self.logger.warning(
                f"Failed to parse certification string '{cert_str}': {e}"
            )
            return cert_str

    def _parse_comprehensive_course(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse comprehensive course details from LLM response with all fields."""
        import re

        # Extract course name from bold text
        name_match = re.search(r"\*\*([^*]+)\*\*", text)
        if not name_match:
            return None

        course = {
            "name": name_match.group(1).strip(),
            "description": None,
            "platform": None,
            "url": None,
            "cost": None,
            "duration": None,
            "difficulty": None,
            "rating": None,
            "learning_outcomes": [],
        }

        # Extract details using regex patterns
        desc_match = re.search(r"Description:\s*([^\n]+)", text, re.IGNORECASE)
        if desc_match:
            course["description"] = desc_match.group(1).strip()

        platform_match = re.search(r"Platform:\s*([^\n]+)", text, re.IGNORECASE)
        if platform_match:
            course["platform"] = platform_match.group(1).strip()

        url_match = re.search(r"URL:\s*(https?://[^\s\]]+)", text, re.IGNORECASE)
        if url_match:
            course["url"] = url_match.group(1).strip()

        cost_match = re.search(r"Cost:\s*([^\n]+)", text, re.IGNORECASE)
        if cost_match:
            course["cost"] = cost_match.group(1).strip()

        duration_match = re.search(r"Duration:\s*([^\n]+)", text, re.IGNORECASE)
        if duration_match:
            course["duration"] = duration_match.group(1).strip()

        diff_match = re.search(r"Difficulty:\s*([^\n]+)", text, re.IGNORECASE)
        if diff_match:
            course["difficulty"] = diff_match.group(1).strip()

        rating_match = re.search(r"Rating:\s*([^\n]+)", text, re.IGNORECASE)
        if rating_match:
            course["rating"] = rating_match.group(1).strip()

        # Extract learning outcomes (bullet points after "Learning Outcomes:")
        outcomes_section = re.search(
            r"Learning Outcomes?:(.+?)(?=\n\n|\n\*\*|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if outcomes_section:
            outcome_text = outcomes_section.group(1)
            # Find all bullet points
            outcomes = re.findall(r"[*\-•]\s*(.+?)(?=\n|$)", outcome_text)
            course["learning_outcomes"] = [o.strip() for o in outcomes if o.strip()]

        return course

    def _parse_comprehensive_certification(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse certification with only essential fields: name, provider, cost, url."""
        import re

        # Extract certification name from bold text
        name_match = re.search(r"\*\*([^*]+)\*\*", text)
        if not name_match:
            return None

        cert = {
            "name": name_match.group(1).strip(),
            "provider": None,
            "cost": None,
            "url": None,
        }

        provider_match = re.search(r"Provider:\s*([^\n]+)", text, re.IGNORECASE)
        if provider_match:
            cert["provider"] = provider_match.group(1).strip()

        cost_match = re.search(r"Cost:\s*([^\n$]+)", text, re.IGNORECASE)
        if cost_match:
            cert["cost"] = cost_match.group(1).strip()

        url_match = re.search(r"URL:\s*(https?://[^\s\]]+)", text, re.IGNORECASE)
        if url_match:
            cert["url"] = url_match.group(1).strip()

        return cert

    def _parse_comprehensive_resource(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse learning resource with essential fields: name, platform, url."""
        import re

        # Extract resource name from bold text
        name_match = re.search(r"\*\*([^*]+)\*\*", text)
        if not name_match:
            return None

        resource = {"name": name_match.group(1).strip(), "platform": None, "url": None}

        platform_match = re.search(r"Platform:\s*([^\n]+)", text, re.IGNORECASE)
        if platform_match:
            resource["platform"] = platform_match.group(1).strip()

        url_match = re.search(r"URL:\s*(https?://[^\s\]]+)", text, re.IGNORECASE)
        if url_match:
            resource["url"] = url_match.group(1).strip()

        return resource


# Example usage and testing
if __name__ == "__main__":
    from models.state_models import AgentState, CareerBlueprint, StudentProfile

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create agent
    skill_agent = SkillDevelopmentAgent()

    # Create test state with a career blueprint
    test_blueprint = CareerBlueprint(
        career_title="Data Scientist",
        career_description="Analyze complex data to derive actionable insights",
        match_score=85.0,
        match_reasoning="Strong match based on analytical skills",
    )

    test_profile = StudentProfile(
        current_education_level="Bachelor's Degree",
        major_field="Computer Science",
        technical_skills=["Python", "Statistics"],
        career_interests=["Data Analysis", "Machine Learning"],
    )

    test_state = AgentState(
        student_profile=test_profile,
        career_blueprints=[test_blueprint],
        session_id="test_session_001",
    )

    # Process task
    print("Creating skill development plan...")
    result = skill_agent.process_task(test_state)

    if result.success:
        print("\n Skill plan created successfully!")
        print(f"Summary: {result.result_data.get('plan_summary')}")
        print(f"\nProcessing time: {result.processing_time:.2f}s")
    else:
        print(f"\n Failed: {result.error_message}")
