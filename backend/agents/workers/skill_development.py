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
        
        self.logger = logging.getLogger(f"agent.{self.name}")

    def _create_system_prompt(self) -> str:
        """Create the specialized system prompt for skill development planning."""
        return """You are an expert Skill Development Agent specializing in creating comprehensive career skill roadmaps.

YOUR ROLE:
You are a specialist in the Career Planning team, working under the Career Planning Supervisor. 
Your specific responsibility is to create detailed, progressive skill development plans for identified careers.

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

    def process_task(self, state: AgentState) -> TaskResult:
        """
        Main task processing: Create skill development roadmap for a career.
        
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
            
            # Create skill development plan
            skill_plan = self._create_skill_development_plan(
                career_title=career_title,
                career_description=career_info.get("description"),
                student_profile=state.student_profile
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

    def _create_skill_development_plan(
        self,
        career_title: str,
        career_description: Optional[str] = None,
        student_profile: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive skill development plan using LLM.
        """
        # Build prompt with career details and student context
        prompt = self._build_skill_planning_prompt(
            career_title, career_description, student_profile
        )
        
        # Invoke LLM with structured output request
        response = self.invoke_with_prompt(prompt)
        
        # Parse and structure the response
        skill_plan = self._parse_skill_plan_response(response, career_title)
        
        return skill_plan

    def _build_skill_planning_prompt(
        self,
        career_title: str,
        career_description: Optional[str],
        student_profile: Optional[Any]
    ) -> str:
        """Build comprehensive prompt for skill planning."""
        
        prompt_parts = [
            f"Create a comprehensive skill development roadmap for: {career_title}",
            ""
        ]
        
        if career_description:
            prompt_parts.append(f"Career Description: {career_description}")
            prompt_parts.append("")
        
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
            "REQUIRED OUTPUT:",
            "",
            "1. TECHNICAL SKILLS BREAKDOWN:",
            "   - Core Technical Skills (5-7 must-have foundational skills)",
            "   - Advanced Technical Skills (4-6 career progression skills)",
            "   - Tools & Technologies (specific tools for this career)",
            "",
            "2. SOFT SKILLS:",
            "   - Essential soft skills (top 5 most important)",
            "   - Recommended soft skills (additional 3-5)",
            "",
            "3. LEARNING PHASES (Progressive roadmap):",
            "",
            "   PHASE 1 - FOUNDATION (0-6 months):",
            "   - Skills to focus on",
            "   - Learning resources (courses, books, platforms)",
            "   - Milestones to achieve",
            "   - Estimated time investment",
            "",
            "   PHASE 2 - INTERMEDIATE (6-18 months):",
            "   - Skills to focus on",
            "   - Learning resources",
            "   - Project-based learning suggestions",
            "   - Milestones to achieve",
            "",
            "   PHASE 3 - ADVANCED (18-36 months):",
            "   - Skills to focus on",
            "   - Advanced resources",
            "   - Industry recognition opportunities",
            "   - Milestones to achieve",
            "",
            "4. CERTIFICATIONS:",
            "   - List relevant professional certifications",
            "   - Priority order (most important first)",
            "   - Typical cost and time investment",
            "",
            "5. LEARNING RESOURCES:",
            "   - Online courses (Coursera, Udemy, etc.)",
            "   - Books (top 3-5 recommendations)",
            "   - Practice platforms",
            "   - Communities to join",
            "",
            "6. SUCCESS METRICS:",
            "   - How to measure progress",
            "   - Portfolio/project suggestions",
            "   - Skill assessment methods",
            "",
            "Make the plan:",
            "1. Specific and actionable",
            "2. Realistic with achievable timelines",
            "3. Progressive (beginner → advanced)",
            "4. Resource-rich with concrete recommendations",
            "5. Measurable with clear milestones",
            "",
            "Format your response as a detailed, structured plan."
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