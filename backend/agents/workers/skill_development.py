import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Use absolute imports
from agents.base_agent import WorkerAgent
from models.state_models import AgentState, TaskResult

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
        
        super().__init__(
            name="skill_development_agent",
            description="Specialist in creating comprehensive skill development roadmaps and learning paths",
            specialization="skill_planning_and_development",
            system_prompt=system_prompt,
            **kwargs
        )
        
        # Agent capabilities
        self.capabilities.extend([
            "technical_skill_analysis",
            "soft_skill_assessment",
            "learning_path_design",
            "resource_recommendation",
            "milestone_planning",
            "progressive_skill_building"
        ])
        
        # Skill taxonomy database (expandable)
        self.skill_categories = self._initialize_skill_taxonomy()
        self.learning_resources = self._initialize_learning_resources()

        # Initialize web search tool for real-time information
        self.web_search = WebSearchTool(cache_duration_minutes=120)

        # Initialize RAG retriever for knowledge base (skill collection)
        try:
            self.rag_retriever = AgenticRAGRetriever(
                collection_type="skill",
                provider="fallback",  # Automatically switches with LLM fallback
                similarity_threshold=0.35,
                top_k=10
            )
            self.rag_enabled = True
            self.logger.info("✅ RAG retriever initialized for skill knowledge base")
        except Exception as e:
            self.logger.warning(f"RAG retriever initialization failed: {e}. Continuing without RAG.")
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

    def _initialize_skill_taxonomy(self) -> Dict[str, List[str]]:
        """Initialize comprehensive skill categories for different career types."""
        return {
            "software_engineering": {
                "core_technical": [
                    "Programming Languages (Python, Java, JavaScript)",
                    "Data Structures & Algorithms",
                    "Object-Oriented Programming",
                    "Version Control (Git)",
                    "Database Management (SQL, NoSQL)"
                ],
                "advanced_technical": [
                    "System Design & Architecture",
                    "Cloud Computing (AWS, Azure, GCP)",
                    "Microservices & APIs",
                    "DevOps & CI/CD",
                    "Testing & Quality Assurance"
                ],
                "tools": [
                    "IDEs (VS Code, IntelliJ)",
                    "Docker & Kubernetes",
                    "Jenkins, GitHub Actions",
                    "Monitoring tools (Prometheus, Grafana)"
                ]
            },
            "data_science": {
                "core_technical": [
                    "Statistics & Probability",
                    "Python/R Programming",
                    "Machine Learning Fundamentals",
                    "Data Wrangling (Pandas, NumPy)",
                    "Data Visualization (Matplotlib, Tableau)"
                ],
                "advanced_technical": [
                    "Deep Learning (TensorFlow, PyTorch)",
                    "Natural Language Processing",
                    "Big Data Technologies (Spark, Hadoop)",
                    "MLOps & Model Deployment",
                    "Feature Engineering"
                ],
                "tools": [
                    "Jupyter Notebooks",
                    "SQL & NoSQL Databases",
                    "Cloud ML Platforms",
                    "Git for Data Science"
                ]
            },
            "product_management": {
                "core_technical": [
                    "Product Analytics",
                    "User Research Methods",
                    "Agile/Scrum Methodologies",
                    "Wireframing & Prototyping",
                    "Data Analysis & Metrics"
                ],
                "advanced_technical": [
                    "Product Strategy & Vision",
                    "A/B Testing & Experimentation",
                    "Technical Understanding (APIs, Systems)",
                    "Product Roadmapping",
                    "Growth & Monetization"
                ],
                "tools": [
                    "JIRA, Asana, Trello",
                    "Figma, Sketch",
                    "Google Analytics, Mixpanel",
                    "SQL for Product Analytics"
                ]
            },
            "ux_ui_design": {
                "core_technical": [
                    "Design Principles & Theory",
                    "User Research & Testing",
                    "Wireframing & Prototyping",
                    "Visual Design",
                    "Information Architecture"
                ],
                "advanced_technical": [
                    "Design Systems",
                    "Interaction Design",
                    "Accessibility (WCAG)",
                    "Motion Design",
                    "Design Thinking"
                ],
                "tools": [
                    "Figma, Adobe XD",
                    "Sketch, InVision",
                    "Adobe Creative Suite",
                    "HTML/CSS basics"
                ]
            },
            "soft_skills_universal": {
                "communication": [
                    "Written Communication",
                    "Verbal Presentation",
                    "Active Listening",
                    "Technical Writing",
                    "Cross-functional Collaboration"
                ],
                "problem_solving": [
                    "Analytical Thinking",
                    "Creative Problem Solving",
                    "Root Cause Analysis",
                    "Decision Making",
                    "Systems Thinking"
                ],
                "leadership": [
                    "Team Collaboration",
                    "Mentoring & Coaching",
                    "Conflict Resolution",
                    "Project Management",
                    "Influence & Persuasion"
                ],
                "professional": [
                    "Time Management",
                    "Adaptability",
                    "Continuous Learning",
                    "Networking",
                    "Self-Management"
                ]
            }
        }

    def _initialize_learning_resources(self) -> Dict[str, List[Dict[str, str]]]:
        """Initialize learning resource database."""
        return {
            "online_platforms": [
                {"name": "Coursera", "type": "Courses", "focus": "University-level courses"},
                {"name": "Udemy", "type": "Courses", "focus": "Practical skills"},
                {"name": "LinkedIn Learning", "type": "Courses", "focus": "Professional development"},
                {"name": "Pluralsight", "type": "Courses", "focus": "Technology skills"},
                {"name": "edX", "type": "Courses", "focus": "Academic & professional"}
            ],
            "practice_platforms": [
                {"name": "LeetCode", "type": "Coding Practice", "focus": "Algorithms & DSA"},
                {"name": "HackerRank", "type": "Coding Practice", "focus": "Programming challenges"},
                {"name": "Kaggle", "type": "Data Science", "focus": "ML competitions"},
                {"name": "GitHub", "type": "Project Portfolio", "focus": "Code sharing"}
            ],
            "certification_providers": [
                {"name": "AWS Certifications", "focus": "Cloud computing"},
                {"name": "Google Professional Certificates", "focus": "Various tech domains"},
                {"name": "Microsoft Certifications", "focus": "Azure & technologies"},
                {"name": "PMI", "focus": "Project management"}
            ]
        }

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
                max_results=8
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
            results = self.web_search.search_certifications(
                career_title,
                max_results=8
            )

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
            results = self.web_search.search_skill_trends(
                career_title,
                max_results=8
            )

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
                career_title,
                location="worldwide",
                max_results=6
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
                max_results=6
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
                "career_blueprints_count": len(state.career_blueprints) if state.career_blueprints else 0
            }
        )

        try:
            # Extract career information from state
            career_info = self._extract_career_info(state)

            if not career_info:
                return self._create_task_result(
                    task_type="skill_development",
                    success=False,
                    error_message="No career information found in state"
                )

            career_title = career_info.get("title")
            self.logger.info(f"Creating skill development plan for: {career_title}")

            # Create skill development plan (async call) with language support
            skill_plan = await self._create_skill_development_plan(
                career_title=career_title,
                career_description=career_info.get("description"),
                student_profile=state.student_profile,
                language=state.preferred_language or "en"
            )

            # Update the career blueprint with skill plan
            updated_state = self._update_state_with_skill_plan(state, career_title, skill_plan)

            processing_time = (datetime.now() - start_time).total_seconds()

            # Log execution
            log_agent_execution(
                self.name,
                f"skill_plan_creation_{career_title}",
                f"Created plan with {len(skill_plan.get('technical_skills', {}).get('core_skills', []))} core skills",
                processing_time
            )

            self._log_task_completion(
                "skill_development_planning",
                True,
                f"for {career_title} in {processing_time:.2f}s"
            )

            return self._create_task_result(
                task_type="skill_development",
                success=True,
                result_data={
                    "career_title": career_title,
                    "skill_plan": skill_plan,
                    "plan_summary": self._create_plan_summary(skill_plan)
                },
                processing_time=processing_time,
                updated_state=updated_state
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_task_completion("skill_development_planning", False, f"Error: {str(e)}")

            # Log error execution
            log_agent_execution(
                self.name,
                "skill_plan_creation_error",
                f"Failed: {str(e)}",
                processing_time
            )

            return self._create_task_result(
                task_type="skill_development",
                success=False,
                error_message=str(e),
                processing_time=processing_time
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
                        "blueprint": blueprint
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
        language: str = "en"
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

        # Parse and structure the response
        skill_plan = self._parse_skill_plan_response(response, career_title)

        return skill_plan

    def _build_skill_planning_prompt(
        self,
        career_title: str,
        career_description: Optional[str],
        student_profile: Optional[Any],
        language: str = "en"
    ) -> str:
        """Build comprehensive prompt for skill planning with RAG and web search results."""

        # Import language helper
        from utils.prompt_templates import get_language_instruction

        # Perform RAG retrieval from knowledge base
        rag_context = ""
        if self.rag_enabled:
            self.logger.info(f"Retrieving skill knowledge base information for {career_title}")
            # Optimized query: shorter, keyword-focused for better semantic matching
            rag_state = self.rag_retriever.retrieve(
                query=f"{career_title} skills courses certifications training learning resources",
                include_citations=True
            )
            if rag_state.context:
                rag_context = rag_state.context
                self.logger.info(f"RAG retrieved {len(rag_state.retrieved_documents)} relevant documents")
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
            language_instruction = get_language_instruction(language, "skill_development") + "\n\n"

        prompt_parts = [
            language_instruction + f"Create a comprehensive skill development roadmap for: {career_title}",
            ""
        ]

        if career_description:
            prompt_parts.append(f"Career Description: {career_description}")
            prompt_parts.append("")

        # Add RAG knowledge base context if available
        if rag_context:
            prompt_parts.extend([
                "=" * 80,
                "KNOWLEDGE BASE INFORMATION (Verified skill data from PDFs):",
                "=" * 80,
                "",
                rag_context,
                ""
            ])

        # Add web search results
        prompt_parts.extend([
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
            ""
        ])
        
        if student_profile:
            prompt_parts.append("STUDENT CONTEXT:")
            if hasattr(student_profile, 'current_education_level'):
                prompt_parts.append(f"- Education Level: {student_profile.current_education_level}")
            if hasattr(student_profile, 'technical_skills') and student_profile.technical_skills:
                prompt_parts.append(f"- Current Technical Skills: {', '.join(student_profile.technical_skills[:5])}")
            if hasattr(student_profile, 'career_interests') and student_profile.career_interests:
                prompt_parts.append(f"- Career Interests: {', '.join(student_profile.career_interests[:3])}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "IMPORTANT INSTRUCTIONS:",
            "- USE THE REAL-TIME WEB SEARCH RESULTS PROVIDED ABOVE",
            "- Include actual course names, platforms, URLs, and current information",
            "- Reference specific certifications found in searches with costs and providers",
            "- Use skill trends from job market to prioritize skills",
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
            "### Essential Skills",
            "Top 5 soft skills from job postings:",
            "- [Skill 1]",
            "- [Skill 2]",
            "",
            "### Recommended Skills",
            "Additional 3-5 valuable soft skills:",
            "- [Skill 1]",
            "- [Skill 2]",
            "",
            "## 3. LEARNING ROADMAP",
            "",
            "### Phase 1: Foundation (0-6 months)",
            "**Skills Focus:**",
            "- [Skill 1]",
            "- [Skill 2]",
            "",
            "**Learning Resources:**",
            "- **[Course Name]** - [Platform]",
            "  - URL: [actual URL from search results]",
            "  - Cost: [price]",
            "  - Duration: [time]",
            "",
            "**Milestones:**",
            "- [Milestone 1]",
            "- [Milestone 2]",
            "",
            "**Time Investment:** [X hours per week]",
            "",
            "### Phase 2: Intermediate (6-18 months)",
            "**Skills Focus:**",
            "- [Skill 1]",
            "- [Skill 2]",
            "",
            "**Learning Resources:**",
            "- **[Course Name]** - [Platform]",
            "  - URL: [actual URL]",
            "  - Cost: [price]",
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
            "- [Emerging skill 1]",
            "- [Emerging skill 2]",
            "",
            "**Learning Resources:**",
            "- **[Advanced Course]** - [Platform]",
            "  - URL: [actual URL]",
            "  - Cost: [price]",
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
            "List certifications in priority order (most important first):",
            "",
            "1. **[Certification Name]**",
            "   - Provider: [Organization]",
            "   - Cost: [Amount]",
            "   - Duration: [Time to complete]",
            "   - URL: [Actual URL from search results]",
            "",
            "2. **[Certification Name]**",
            "   - Provider: [Organization]",
            "   - Cost: [Amount]",
            "   - Duration: [Time]",
            "   - URL: [URL]",
            "",
            "## 5. LEARNING RESOURCES",
            "",
            "### Online Courses",
            "Specific courses with platforms and URLs:",
            "- **[Course Name]** (Coursera/Udemy/etc.) - [URL]",
            "- **[Course Name]** (Platform) - [URL]",
            "",
            "### Books",
            "Top recommended books:",
            "- [Book Title] by [Author]",
            "- [Book Title] by [Author]",
            "",
            "### Practice Platforms",
            "Hands-on learning platforms:",
            "- [Platform Name]",
            "- [Platform Name]",
            "",
            "### Communities",
            "Professional communities and forums:",
            "- [Community Name]",
            "- [Forum/Group Name]",
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
            "Write the complete skill development plan following this exact markdown structure. Use REAL course names, URLs, and current information from the web search results provided above."
        ])
        
        return "\n".join(prompt_parts)

    def _parse_skill_plan_response(self, response: str, career_title: str) -> Dict[str, Any]:
        """Parse LLM response into structured skill plan."""
        
        # This is a simplified parser. In production, you might want more robust parsing
        # or request JSON output from the LLM directly
        
        skill_plan = {
            "career_title": career_title,
            "technical_skills": {
                "core_skills": [],
                "advanced_skills": [],
                "tools_technologies": []
            },
            "soft_skills": {
                "essential": [],
                "recommended": []
            },
            "learning_phases": [],
            "certifications": [],
            "learning_resources": {
                "online_courses": [],
                "books": [],
                "practice_platforms": [],
                "communities": []
            },
            "estimated_total_time": "24-36 months",
            "success_metrics": [],
            "raw_plan": response  # Keep full response for reference
        }
        
        # Extract sections using simple text parsing
        # (In production, use structured output or better parsing)
        
        lines = response.split('\n')
        current_section = None
        current_phase = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect sections
            if "core technical skills" in line_lower or "foundational skills" in line_lower:
                current_section = "core_technical"
            elif "advanced technical skills" in line_lower or "advanced skills" in line_lower:
                current_section = "advanced_technical"
            elif "tools" in line_lower and "technologies" in line_lower:
                current_section = "tools"
            elif "essential soft skills" in line_lower or "soft skills" in line_lower:
                current_section = "soft_skills_essential"
            elif "certification" in line_lower:
                current_section = "certifications"
            elif "phase 1" in line_lower or "foundation" in line_lower:
                current_section = "phase1"
                current_phase = {
                    "phase": "Foundation",
                    "duration": "0-6 months",
                    "skills_focus": [],
                    "resources": [],
                    "milestones": []
                }
            elif "phase 2" in line_lower or "intermediate" in line_lower:
                if current_phase:
                    skill_plan["learning_phases"].append(current_phase)
                current_section = "phase2"
                current_phase = {
                    "phase": "Intermediate",
                    "duration": "6-18 months",
                    "skills_focus": [],
                    "resources": [],
                    "milestones": []
                }
            elif "phase 3" in line_lower or "advanced" in line_lower:
                if current_phase:
                    skill_plan["learning_phases"].append(current_phase)
                current_section = "phase3"
                current_phase = {
                    "phase": "Advanced",
                    "duration": "18-36 months",
                    "skills_focus": [],
                    "resources": [],
                    "milestones": []
                }
            
            # Extract bullet points
            if line.strip().startswith(('-', '•', '*', '✅')):
                item = line.strip().lstrip('-•*✅ ').strip()
                if item and len(item) > 3:
                    if current_section == "core_technical":
                        skill_plan["technical_skills"]["core_skills"].append(item)
                    elif current_section == "advanced_technical":
                        skill_plan["technical_skills"]["advanced_skills"].append(item)
                    elif current_section == "tools":
                        skill_plan["technical_skills"]["tools_technologies"].append(item)
                    elif current_section == "soft_skills_essential":
                        skill_plan["soft_skills"]["essential"].append(item)
                    elif current_section == "certifications":
                        skill_plan["certifications"].append(item)
                    elif current_phase:
                        if "milestone" in line_lower:
                            current_phase["milestones"].append(item)
                        elif any(platform in line_lower for platform in ["coursera", "udemy", "book", "course"]):
                            current_phase["resources"].append(item)
                        else:
                            current_phase["skills_focus"].append(item)
        
        # Add last phase if exists
        if current_phase and current_phase not in skill_plan["learning_phases"]:
            skill_plan["learning_phases"].append(current_phase)
        
        # Ensure minimum data
        if not skill_plan["technical_skills"]["core_skills"]:
            skill_plan["technical_skills"]["core_skills"] = [
                "Foundational knowledge in core area",
                "Industry-standard tools proficiency",
                "Problem-solving methodologies"
            ]
        
        if not skill_plan["soft_skills"]["essential"]:
            skill_plan["soft_skills"]["essential"] = [
                "Communication & Collaboration",
                "Critical Thinking",
                "Adaptability",
                "Time Management"
            ]
        
        if not skill_plan["learning_phases"]:
            skill_plan["learning_phases"] = self._create_default_learning_phases()
        
        return skill_plan

    def _create_default_learning_phases(self) -> List[Dict[str, Any]]:
        """Create default learning phases as fallback."""
        return [
            {
                "phase": "Foundation",
                "duration": "0-6 months",
                "skills_focus": ["Core fundamentals", "Basic tools", "Foundational concepts"],
                "resources": ["Online courses", "Introductory books", "Practice exercises"],
                "milestones": ["Complete foundational course", "Build first project"]
            },
            {
                "phase": "Intermediate",
                "duration": "6-18 months",
                "skills_focus": ["Applied skills", "Real-world projects", "Specialization"],
                "resources": ["Advanced courses", "Industry documentation", "Project work"],
                "milestones": ["Build portfolio projects", "Earn certifications"]
            },
            {
                "phase": "Advanced",
                "duration": "18-36 months",
                "skills_focus": ["Expert-level skills", "Thought leadership", "Innovation"],
                "resources": ["Advanced specialization", "Research papers", "Conferences"],
                "milestones": ["Industry recognition", "Mentorship", "Speaking engagements"]
            }
        ]

    def _update_state_with_skill_plan(
        self,
        state: AgentState,
        career_title: str,
        skill_plan: Dict[str, Any]
    ) -> AgentState:
        """Update the state with the completed skill development plan."""
        
        updated_state = state.copy(deep=True)
        
        # Find and update the corresponding career blueprint
        if updated_state.career_blueprints:
            for blueprint in updated_state.career_blueprints:
                if blueprint.career_title == career_title:
                    blueprint.skill_development_plan = skill_plan
                    self.logger.info(f"Updated blueprint for {career_title} with skill plan")
                    break
        
        # Add completion message
        completion_message = AIMessage(
            content=f" Completed skill development plan for {career_title}",
            name=self.name
        )
        updated_state.messages.append(completion_message)
        
        return updated_state

    def _create_plan_summary(self, skill_plan: Dict[str, Any]) -> str:
        """Create a brief summary of the skill plan."""
        
        core_skills_count = len(skill_plan.get("technical_skills", {}).get("core_skills", []))
        phases_count = len(skill_plan.get("learning_phases", []))
        certifications_count = len(skill_plan.get("certifications", []))
        
        summary = f"Skill Development Plan: {core_skills_count} core skills, "
        summary += f"{phases_count} learning phases, "
        summary += f"{certifications_count} recommended certifications"
        
        return summary


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
        match_reasoning="Strong match based on analytical skills"
    )
    
    test_profile = StudentProfile(
        current_education_level="Bachelor's Degree",
        major_field="Computer Science",
        technical_skills=["Python", "Statistics"],
        career_interests=["Data Analysis", "Machine Learning"]
    )
    
    test_state = AgentState(
        student_profile=test_profile,
        career_blueprints=[test_blueprint],
        session_id="test_session_001"
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