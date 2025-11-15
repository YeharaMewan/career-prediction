"""
Academic Pathway Agent - Educational Roadmap Specialist

This agent creates comprehensive educational roadmaps for identified careers,
focusing on Sri Lankan and international education systems. It analyzes career
requirements and designs step-by-step academic pathways including degrees,
diplomas, certifications, and vocational training.

Key Features:
- Career level assessment (O/L, A/L, entry-level, mid-level, expert)
- Sri Lankan education system integration
- International pathway options
- Entry requirements analysis
- Institution recommendations
- Timeline and cost estimation
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import re

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Use absolute imports
from agents.base_agent import WorkerAgent
from models.state_models import AgentState, TaskResult, StudentProfile

# Import LangSmith for monitoring
from utils.langsmith_config import get_traced_run_config, log_agent_execution

# Import Web Search Tool
from utils.web_search_tool import WebSearchTool

# Import RAG System for knowledge base retrieval
from rag.retriever import AgenticRAGRetriever

# Import Prompt Templates for optimization
from utils.prompt_templates import (
    PromptTemplates,
    build_structured_messages,
    format_student_context,
    estimate_tokens
)


class AcademicPathwayAgent(WorkerAgent):
    """
    Academic Pathway Agent - Educational roadmap specialist for career preparation.
    
    This agent analyzes career requirements and creates detailed educational
    pathways tailored to Sri Lankan students with international alternatives.
    It considers the user's current academic level and provides personalized
    guidance for achieving career goals through education.
    """

    def __init__(self, **kwargs):
        # Use optimized system prompt template (650 tokens vs 1,200)
        system_prompt = PromptTemplates.academic_pathway_system()

        super().__init__(
            name="academic_pathway_agent",
            description="Specialist in creating comprehensive educational roadmaps for career preparation in Sri Lankan and international contexts",
            specialization="educational_planning_and_pathway_design",
            system_prompt=system_prompt,
            **kwargs
        )

        # Agent capabilities
        self.capabilities.extend([
            "educational_pathway_design",
            "sri_lankan_education_system_expertise",
            "international_education_options",
            "entry_requirements_analysis",
            "institution_recommendations",
            "academic_timeline_planning",
            "cost_estimation",
            "career_level_assessment"
        ])

        # Load institutions data from JSON file (instead of including in prompts)
        self.sri_lankan_institutions = self._load_institutions_data()
        self.international_options = self._initialize_international_options()
        self.career_level_matrix = self._initialize_career_level_matrix()

        # Initialize web search tool for real-time information
        self.web_search = WebSearchTool(cache_duration_minutes=120)

        # Initialize RAG retriever for knowledge base (academic collection)
        try:
            self.rag_retriever = AgenticRAGRetriever(
                collection_type="academic",
                similarity_threshold=0.35,
                top_k=10
            )
            self.rag_enabled = True
            self.logger.info("✅ RAG retriever initialized for academic knowledge base")
        except Exception as e:
            self.logger.warning(f"RAG retriever initialization failed: {e}. Continuing without RAG.")
            self.rag_enabled = False

        self.logger = logging.getLogger(f"agent.{self.name}")

        # Log token optimization
        self.logger.info(f"✅ Academic Pathway Agent initialized with optimized prompts")
        self.logger.info(f"   System prompt: ~{estimate_tokens(system_prompt)} tokens (was ~1200)")

    def _load_institutions_data(self) -> Dict[str, Any]:
        """
        Load Sri Lankan institutions data from JSON file.

        This replaces the old approach of including institution lists in prompts,
        saving ~300 tokens per prompt.

        Returns:
            Dictionary with institution data
        """
        import os

        try:
            # Get path to institutions JSON file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(os.path.dirname(current_dir))
            json_path = os.path.join(backend_dir, "prompts", "sri_lankan_institutions.json")

            # Load data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.logger.info(f"✅ Loaded institutions data from {json_path}")
            return data

        except Exception as e:
            self.logger.warning(f"⚠️ Could not load institutions data: {e}")
            self.logger.warning("   Using fallback institution data")
            # Return minimal fallback data
            return self._get_fallback_institutions()

    def _initialize_sri_lankan_institutions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive Sri Lankan education institution database."""
        return {
            "state_universities": {
                "university_of_colombo": {
                    "name": "University of Colombo",
                    "location": "Colombo",
                    "strengths": ["Medicine", "Law", "Science", "Arts", "Management"],
                    "website": "https://www.cmb.ac.lk/",
                    "entry_method": "University admission via Z-score",
                    "language": "English/Sinhala"
                },
                "university_of_peradeniya": {
                    "name": "University of Peradeniya",
                    "location": "Kandy",
                    "strengths": ["Engineering", "Medicine", "Science", "Arts", "Agriculture"],
                    "website": "https://www.pdn.ac.lk/",
                    "entry_method": "University admission via Z-score",
                    "language": "English/Sinhala"
                },
                "university_of_moratuwa": {
                    "name": "University of Moratuwa",
                    "location": "Moratuwa",
                    "strengths": ["Engineering", "IT", "Architecture", "Business"],
                    "website": "https://www.mrt.ac.lk/",
                    "entry_method": "University admission via Z-score",
                    "language": "English"
                },
                "university_of_ruhuna": {
                    "name": "University of Ruhuna",
                    "location": "Matara",
                    "strengths": ["Medicine", "Engineering", "Science", "Humanities", "Management", "Fisheries"],
                    "website": "https://www.ruh.ac.lk/",
                    "entry_method": "University admission via Z-score",
                    "language": "English/Sinhala"
                },
                "university_of_sri_jayewardenepura": {
                    "name": "University of Sri Jayewardenepura",
                    "location": "Nugegoda",
                    "strengths": ["Management", "Applied Sciences", "Humanities", "Medical Sciences"],
                    "website": "https://www.sjp.ac.lk/",
                    "entry_method": "University admission via Z-score",
                    "language": "English/Sinhala"
                },
                "open_university": {
                    "name": "Open University of Sri Lanka",
                    "location": "Multiple locations",
                    "strengths": ["Distance Learning", "Technology", "Education", "Social Sciences"],
                    "website": "https://www.ou.ac.lk/",
                    "entry_method": "Open admission for most programs",
                    "language": "English/Sinhala"
                }
            },
            "private_universities": {
                "sliit": {
                    "name": "Sri Lanka Institute of Information Technology",
                    "location": "Malabe, Colombo",
                    "strengths": ["IT", "Engineering", "Business", "Humanities"],
                    "international_partnerships": ["Liverpool John Moores", "Curtin University"],
                    "entry_method": "Private admission, A/L results",
                    "language": "English"
                },
                "nsbm": {
                    "name": "NSBM Green University",
                    "location": "Pitipana, Homagama",
                    "strengths": ["Business", "Computing", "Engineering", "Sciences"],
                    "international_partnerships": ["Plymouth University", "Victoria University"],
                    "entry_method": "Private admission",
                    "language": "English"
                },
                "iit": {
                    "name": "Informatics Institute of Technology",
                    "location": "Colombo",
                    "strengths": ["IT", "Business", "Law"],
                    "international_partnerships": ["University of Westminster", "Robert Gordon University"],
                    "entry_method": "Private admission",
                    "language": "English"
                }
            },
            "professional_institutes": {
                "ca_sri_lanka": {
                    "name": "Institute of Chartered Accountants of Sri Lanka",
                    "specialization": "Accounting and Finance",
                    "entry_requirements": ["A/L qualification", "Foundation course"],
                    "duration": "3-4 years",
                    "recognition": "International (Commonwealth countries)"
                },
                "cima": {
                    "name": "Chartered Institute of Management Accountants",
                    "specialization": "Management Accounting",
                    "entry_requirements": ["A/L qualification", "Degree exemptions available"],
                    "duration": "2-3 years",
                    "recognition": "Global"
                },
                "iesl": {
                    "name": "Institution of Engineers Sri Lanka",
                    "specialization": "Engineering",
                    "entry_requirements": ["Engineering degree", "Professional experience"],
                    "duration": "Ongoing professional development",
                    "recognition": "National and regional"
                }
            }
        }

    def _initialize_international_options(self) -> Dict[str, Dict[str, Any]]:
        """Initialize international education options database."""
        return {
            "popular_destinations": {
                "uk": {
                    "advantages": ["Quality education", "Shorter duration", "Post-study work visa"],
                    "entry_requirements": ["A/L qualifications", "IELTS 6.0-7.0", "Personal statement"],
                    "approximate_cost_usd": "25000-40000 per year",
                    "popular_universities": ["University of London", "Manchester University", "Edinburgh University"],
                    "scholarship_opportunities": ["Chevening", "Commonwealth", "University scholarships"]
                },
                "australia": {
                    "advantages": ["High quality education", "Work opportunities", "Immigration pathways"],
                    "entry_requirements": ["A/L qualifications", "IELTS 6.5-7.0", "Foundation year may be required"],
                    "approximate_cost_usd": "20000-35000 per year",
                    "popular_universities": ["University of Melbourne", "UNSW", "Monash University"],
                    "scholarship_opportunities": ["Australia Awards", "University scholarships"]
                },
                "canada": {
                    "advantages": ["Quality education", "Immigration opportunities", "Work permits"],
                    "entry_requirements": ["A/L qualifications", "IELTS 6.5", "Foundation programs available"],
                    "approximate_cost_usd": "15000-30000 per year",
                    "popular_universities": ["University of Toronto", "UBC", "McGill University"],
                    "scholarship_opportunities": ["Canadian government scholarships", "University funding"]
                },
                "usa": {
                    "advantages": ["World-class education", "Research opportunities", "Diverse programs"],
                    "entry_requirements": ["A/L qualifications", "SAT/ACT", "TOEFL/IELTS", "Essays"],
                    "approximate_cost_usd": "25000-60000 per year",
                    "popular_universities": ["State universities", "Community colleges", "Private universities"],
                    "scholarship_opportunities": ["Merit scholarships", "Need-based aid", "Athletic scholarships"]
                },
                "germany": {
                    "advantages": ["Low tuition fees", "Quality education", "Strong economy"],
                    "entry_requirements": ["A/L qualifications", "German language or English programs", "Foundation year"],
                    "approximate_cost_usd": "500-3000 per year (public universities)",
                    "popular_universities": ["TU Munich", "University of Heidelberg", "RWTH Aachen"],
                    "scholarship_opportunities": ["DAAD scholarships", "Erasmus+"]
                },
                "singapore": {
                    "advantages": ["Regional hub", "High quality", "Multicultural environment"],
                    "entry_requirements": ["Strong A/L results", "IELTS/TOEFL", "Competitive admission"],
                    "approximate_cost_usd": "20000-40000 per year",
                    "popular_universities": ["NUS", "NTU", "SMU"],
                    "scholarship_opportunities": ["Government scholarships", "University scholarships"]
                }
            },
            "online_options": {
                "advantages": ["Flexibility", "Lower cost", "Study while working"],
                "popular_providers": ["University of London Online", "Arizona State University Online", "Penn State World Campus"],
                "considerations": ["Accreditation", "Recognition in Sri Lanka", "Technology requirements"],
                "cost_range_usd": "5000-20000 per year"
            }
        }

    def _initialize_career_level_matrix(self) -> Dict[str, Dict[str, Any]]:
        """Initialize career level assessment matrix."""
        return {
            "ol_student": {
                "description": "GCE O/L student (Age 15-16)",
                "immediate_focus": "A/L subject selection",
                "timeline_to_career": "6-10 years",
                "key_decisions": ["A/L stream selection", "Career exploration", "Subject combinations"],
                "recommendations": ["Career guidance", "Subject selection counseling", "Skill development"]
            },
            "al_student": {
                "description": "GCE A/L student (Age 17-18)",
                "immediate_focus": "University admission preparation",
                "timeline_to_career": "4-8 years",
                "key_decisions": ["University selection", "Degree program choice", "Local vs international"],
                "recommendations": ["University applications", "Scholarship applications", "Gap year consideration"]
            },
            "fresh_al_graduate": {
                "description": "Recently completed A/L",
                "immediate_focus": "Higher education pathway",
                "timeline_to_career": "3-6 years",
                "key_decisions": ["Immediate university entry", "Gap year", "Foundation programs"],
                "recommendations": ["Multiple application strategies", "Foundation courses", "Work experience"]
            },
            "entry_level_professional": {
                "description": "0-2 years work experience",
                "immediate_focus": "Skill enhancement and career direction",
                "timeline_to_career": "2-5 years",
                "key_decisions": ["Part-time study", "Professional certifications", "Career change"],
                "recommendations": ["Evening/weekend programs", "Professional development", "Industry certifications"]
            },
            "mid_level_professional": {
                "description": "3-7 years work experience",
                "immediate_focus": "Career advancement and specialization",
                "timeline_to_career": "1-3 years",
                "key_decisions": ["MBA/Master's degree", "Leadership roles", "Industry specialization"],
                "recommendations": ["Executive education", "Leadership programs", "Advanced certifications"]
            },
            "senior_professional": {
                "description": "8+ years work experience",
                "immediate_focus": "Strategic career positioning",
                "timeline_to_career": "Ongoing development",
                "key_decisions": ["Executive education", "Thought leadership", "Entrepreneurship"],
                "recommendations": ["Executive MBA", "Board positions", "Consulting", "Teaching"]
            }
        }

    def _search_local_universities(self, career_title: str) -> str:
        """
        Search for Sri Lankan universities offering programs for the career.

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(f"Searching for Sri Lankan universities for {career_title}")

            # Search for universities
            results = self.web_search.search_universities(
                career_title,
                country="Sri Lanka",
                max_results=8
            )

            if not results:
                return "No current information found. Using database knowledge."

            return self.web_search.format_results_for_llm(results, max_snippets=5)

        except Exception as e:
            self.logger.error(f"University search failed: {e}")
            return "Search unavailable. Using database knowledge."

    def _search_local_scholarships(self, career_title: str) -> str:
        """
        Search for scholarships available in Sri Lanka for the career.

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(f"Searching for Sri Lankan scholarships for {career_title}")

            # Search for scholarships
            results = self.web_search.search_scholarships(
                career_title,
                country="Sri Lanka",
                max_results=6
            )

            if not results:
                return "No scholarship information found."

            return self.web_search.format_results_for_llm(results, max_snippets=4)

        except Exception as e:
            self.logger.error(f"Scholarship search failed: {e}")
            return "Scholarship search unavailable."

    def _search_international_universities(self, career_title: str, country: str = "UK") -> str:
        """
        Search for international universities offering programs for the career.

        Args:
            career_title: Career name
            country: Country to search in (UK, USA, Australia, Canada, etc.)

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(f"Searching for international universities for {career_title} in {country}")

            # Search for international universities
            results = self.web_search.search_universities(
                career_title,
                country=country,
                max_results=6
            )

            if not results:
                return f"No information found for {country}."

            return self.web_search.format_results_for_llm(results, max_snippets=4)

        except Exception as e:
            self.logger.error(f"International university search failed: {e}")
            return "International search unavailable."

    def _search_international_scholarships(self, career_title: str) -> str:
        """
        Search for international scholarship opportunities.

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(f"Searching for international scholarships for {career_title}")

            # Search for major international scholarships
            chevening_results = self.web_search.search(
                f"Chevening scholarship {career_title} Sri Lankan students",
                max_results=3
            )

            commonwealth_results = self.web_search.search(
                f"Commonwealth scholarship {career_title} Sri Lanka",
                max_results=3
            )

            # Combine results
            all_results = chevening_results + commonwealth_results

            if not all_results:
                return "No international scholarship information found."

            return self.web_search.format_results_for_llm(all_results, max_snippets=4)

        except Exception as e:
            self.logger.error(f"International scholarship search failed: {e}")
            return "International scholarship search unavailable."

    def _search_alternative_pathways(self, career_title: str) -> str:
        """
        Search for alternative educational pathways (online degrees, bootcamps, etc.).

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(f"Searching for alternative pathways for {career_title}")

            # Search for online programs and bootcamps
            results = self.web_search.search(
                f"{career_title} online degree programs bootcamps certifications",
                max_results=6
            )

            if not results:
                return "No alternative pathway information found."

            return self.web_search.format_results_for_llm(results, max_snippets=4)

        except Exception as e:
            self.logger.error(f"Alternative pathway search failed: {e}")
            return "Alternative pathway search unavailable."

    def _search_admission_requirements(self, career_title: str) -> str:
        """
        Search for typical admission requirements for the career's programs.

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(f"Searching for admission requirements for {career_title}")

            # Search for admission requirements
            results = self.web_search.search(
                f"{career_title} university admission requirements Sri Lanka A/L Z-score",
                max_results=5
            )

            if not results:
                return "No admission requirement information found."

            return self.web_search.format_results_for_llm(results, max_snippets=3)

        except Exception as e:
            self.logger.error(f"Admission requirements search failed: {e}")
            return "Admission requirements search unavailable."

    async def process_task(self, state: AgentState) -> TaskResult:
        """
        Main task processing: Create academic pathway for a career.

        This is an async method to support parallel execution with other agents.

        Expected input in state:
        - career_title: Name of the career
        - career_description: Brief description
        - student_profile: StudentProfile object for personalization

        Returns:
        - TaskResult with complete academic pathway plan
        """
        start_time = datetime.now()
        session_id = state.session_id or "academic_pathway_session"

        self._log_task_start("academic_pathway_planning", f"session: {session_id}")

        # Get tracing configuration
        run_config = get_traced_run_config(
            session_type="academic_pathway",
            agent_name=self.name,
            session_id=session_id,
            additional_tags=["educational_planning", "pathway_design"],
            additional_metadata={
                "career_blueprints_count": len(state.career_blueprints) if state.career_blueprints else 0
            }
        )

        try:
            # Extract career information from state
            career_info = self._extract_career_info(state)

            if not career_info:
                return self._create_task_result(
                    task_type="academic_pathway",
                    success=False,
                    error_message="No career information found in state"
                )

            career_title = career_info.get("title")
            self.logger.info(f"Creating academic pathway for: {career_title}")

            # Assess student's current level
            student_level_assessment = self._assess_student_level(state.student_profile)

            # Create academic pathway plan (async call)
            academic_plan = await self._create_academic_pathway_plan(
                career_title=career_title,
                career_description=career_info.get("description"),
                student_profile=state.student_profile,
                student_level=student_level_assessment
            )

            # Update the career blueprint with academic plan
            updated_state = self._update_state_with_academic_plan(state, career_title, academic_plan)

            processing_time = (datetime.now() - start_time).total_seconds()

            # Log execution
            log_agent_execution(
                self.name,
                f"academic_plan_creation_{career_title}",
                f"Created plan for {student_level_assessment['current_level']} level student",
                processing_time
            )

            self._log_task_completion(
                "academic_pathway_planning",
                True,
                f"for {career_title} in {processing_time:.2f}s"
            )

            return self._create_task_result(
                task_type="academic_pathway",
                success=True,
                result_data={
                    "career_title": career_title,
                    "academic_plan": academic_plan,
                    "student_level": student_level_assessment,
                    "plan_summary": self._create_plan_summary(academic_plan)
                },
                processing_time=processing_time,
                updated_state=updated_state
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_task_completion("academic_pathway_planning", False, f"Error: {str(e)}")

            # Log error execution
            log_agent_execution(
                self.name,
                "academic_plan_creation_error",
                f"Failed: {str(e)}",
                processing_time
            )

            return self._create_task_result(
                task_type="academic_pathway",
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )

    def _extract_career_info(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Extract career information from state."""
        # Check career_blueprints for incomplete blueprint
        if state.career_blueprints:
            for blueprint in state.career_blueprints:
                # If blueprint doesn't have academic plan, work on it
                if not blueprint.academic_plan:
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
                    if "career" in content.lower() or "academic pathway" in content.lower():
                        return {"title": content, "description": None}
        
        return None

    def _assess_student_level(self, student_profile: Optional[StudentProfile]) -> Dict[str, Any]:
        """
        Assess the student's current academic/professional level.
        """
        if not student_profile:
            return {
                "current_level": "unknown",
                "assessment_confidence": "low",
                "recommendations": ["Provide more information about current education/experience"]
            }
        
        education_level = getattr(student_profile, 'current_education_level', '').lower()
        
        # Determine level based on education
        if any(keyword in education_level for keyword in ['o/l', 'ol', 'ordinary level']):
            level_key = "ol_student"
        elif any(keyword in education_level for keyword in ['a/l', 'al', 'advanced level']):
            level_key = "al_student"
        elif any(keyword in education_level for keyword in ['bachelor', 'undergraduate', 'degree']):
            level_key = "fresh_al_graduate"  # Assuming currently studying
        elif any(keyword in education_level for keyword in ['master', 'mba', 'postgraduate']):
            level_key = "mid_level_professional"
        elif any(keyword in education_level for keyword in ['phd', 'doctorate']):
            level_key = "senior_professional"
        else:
            # Try to infer from other information
            if hasattr(student_profile, 'work_experience') and student_profile.work_experience:
                level_key = "entry_level_professional"
            elif hasattr(student_profile, 'age') and student_profile.age:
                if student_profile.age <= 16:
                    level_key = "ol_student"
                elif student_profile.age <= 19:
                    level_key = "al_student"
                else:
                    level_key = "entry_level_professional"
            else:
                level_key = "al_student"  # Default assumption
        
        level_info = self.career_level_matrix.get(level_key, self.career_level_matrix["al_student"])
        
        return {
            "current_level": level_key,
            "description": level_info["description"],
            "immediate_focus": level_info["immediate_focus"],
            "timeline_to_career": level_info["timeline_to_career"],
            "key_decisions": level_info["key_decisions"],
            "recommendations": level_info["recommendations"],
            "assessment_confidence": "high" if education_level else "medium"
        }

    async def _create_academic_pathway_plan(
        self,
        career_title: str,
        career_description: Optional[str] = None,
        student_profile: Optional[StudentProfile] = None,
        student_level: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive academic pathway plan using LLM.

        This is an async method to support non-blocking LLM invocation.
        """
        # Build prompt with career details and student context
        prompt = self._build_academic_planning_prompt(
            career_title, career_description, student_profile, student_level
        )

        # Invoke LLM with structured output request (async)
        response = await self.invoke_with_prompt(prompt)

        # Parse and structure the response
        academic_plan = self._parse_academic_plan_response(response, career_title, student_level)

        return academic_plan

    def _build_academic_planning_prompt(
        self,
        career_title: str,
        career_description: Optional[str],
        student_profile: Optional[StudentProfile],
        student_level: Optional[Dict[str, Any]]
    ) -> str:
        """Build comprehensive prompt for academic pathway planning with RAG and web search results."""

        # Perform RAG retrieval from knowledge base
        rag_context = ""
        if self.rag_enabled:
            self.logger.info(f"Retrieving knowledge base information for {career_title}")
            # Optimized query: shorter, keyword-focused for better semantic matching
            rag_state = self.rag_retriever.retrieve(
                query=f"{career_title} education degrees programs institutions training Sri Lanka",
                include_citations=True
            )
            if rag_state.context:
                rag_context = rag_state.context
                self.logger.info(f"RAG retrieved {len(rag_state.retrieved_documents)} relevant documents")
            else:
                self.logger.info("No relevant documents found in knowledge base")

        # Perform web searches for current information
        self.logger.info(f"Gathering real-time information for {career_title}")

        local_universities_info = self._search_local_universities(career_title)
        local_scholarships_info = self._search_local_scholarships(career_title)
        international_uk_info = self._search_international_universities(career_title, "UK")
        international_usa_info = self._search_international_universities(career_title, "USA")
        international_scholarships_info = self._search_international_scholarships(career_title)
        alternative_pathways_info = self._search_alternative_pathways(career_title)
        admission_requirements_info = self._search_admission_requirements(career_title)

        prompt_parts = [
            f"Create a comprehensive academic pathway plan for: {career_title}",
            ""
        ]

        if career_description:
            prompt_parts.append(f"Career Description: {career_description}")
            prompt_parts.append("")

        # Add RAG knowledge base context if available
        if rag_context:
            prompt_parts.extend([
                "=" * 80,
                "KNOWLEDGE BASE INFORMATION (Verified academic data from PDFs):",
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
            "LOCAL UNIVERSITIES AND PROGRAMS (Sri Lanka):",
            local_universities_info,
            "",
            "LOCAL SCHOLARSHIPS (Sri Lanka):",
            local_scholarships_info,
            "",
            "ADMISSION REQUIREMENTS:",
            admission_requirements_info,
            "",
            "INTERNATIONAL UNIVERSITIES (UK):",
            international_uk_info,
            "",
            "INTERNATIONAL UNIVERSITIES (USA):",
            international_usa_info,
            "",
            "INTERNATIONAL SCHOLARSHIPS:",
            international_scholarships_info,
            "",
            "ALTERNATIVE PATHWAYS:",
            alternative_pathways_info,
            "",
            "=" * 80,
            "END OF WEB SEARCH RESULTS",
            "=" * 80,
            ""
        ])
        
        if student_level:
            prompt_parts.append("STUDENT LEVEL ASSESSMENT:")
            prompt_parts.append(f"- Current Level: {student_level.get('description', 'Unknown')}")
            prompt_parts.append(f"- Timeline to Career: {student_level.get('timeline_to_career', 'Unknown')}")
            prompt_parts.append(f"- Immediate Focus: {student_level.get('immediate_focus', 'Career preparation')}")
            prompt_parts.append("")
        
        if student_profile:
            prompt_parts.append("STUDENT CONTEXT:")
            if hasattr(student_profile, 'current_education_level'):
                prompt_parts.append(f"- Education Level: {student_profile.current_education_level}")
            if hasattr(student_profile, 'major_field') and student_profile.major_field:
                prompt_parts.append(f"- Current/Previous Field: {student_profile.major_field}")
            if hasattr(student_profile, 'career_interests') and student_profile.career_interests:
                prompt_parts.append(f"- Career Interests: {', '.join(student_profile.career_interests[:3])}")
            if hasattr(student_profile, 'academic_performance') and student_profile.academic_performance:
                prompt_parts.append(f"- Academic Performance: {student_profile.academic_performance}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "IMPORTANT INSTRUCTIONS:",
            "- USE THE REAL-TIME WEB SEARCH RESULTS PROVIDED ABOVE",
            "- Include actual university names, programs, URLs, and current costs from search results",
            "- Reference specific scholarships and opportunities found in searches",
            "- Follow the STRUCTURE: Local Pathways → International Pathways → Other Relevant Information",
            "- Provide AT LEAST 5-8 specific institutions per category (Sri Lankan state, private, international)",
            "- Include SPECIFIC degree program names (e.g., 'BSc (Hons) in Computer Science' not 'related degree')",
            "",
            "OUTPUT FORMAT - Write a detailed, well-structured plan with clear section headings:",
            "",
            "## 1. STUDENT ASSESSMENT",
            "- **Current Level:** [Specify the student's level]",
            "- **Timeline to Career:** [Realistic timeframe]",
            "",
            "## 2. LOCAL PATHWAYS (Sri Lankan Options)",
            "",
            "### A. STATE UNIVERSITIES",
            "List 5-8 state universities with specific programs:",
            "- **[University Name]** - [Specific Program Name]",
            "  - Program: [Full degree/diploma name]",
            "  - Duration: [X years]",
            "  - Entry Requirements: [Z-score, A/L subjects]",
            "  - Approximate Cost: [LKR amount]",
            "  - Application Period: [Specific months]",
            "  - Website: [URL if available]",
            "",
            "### B. PRIVATE UNIVERSITIES/INSTITUTES",
            "List 5-8 private institutions with specific programs:",
            "- **[Institution Name]** - [Specific Program Name]",
            "  - Program: [Full degree/diploma name]",
            "  - Duration: [X years]",
            "  - Entry Requirements: [Specific requirements]",
            "  - Approximate Cost: [LKR amount per year]",
            "  - International Partnerships: [Partner universities if any]",
            "  - Website: [URL if available]",
            "",
            "### C. PROFESSIONAL INSTITUTES",
            "List relevant professional certifications:",
            "- **[Institute Name]** - [Qualification Name]",
            "  - Program: [Full certification name]",
            "  - Duration: [X years/months]",
            "  - Entry Requirements: [Requirements]",
            "  - Approximate Cost: [LKR amount]",
            "",
            "## 3. INTERNATIONAL PATHWAY OPTIONS",
            "",
            "### United Kingdom",
            "List 3-5 universities:",
            "- **[University Name]** - [Program Name]",
            "  - Program: [Specific degree name]",
            "  - Duration: [X years]",
            "  - Entry Requirements: [A/L, IELTS scores]",
            "  - Approximate Cost: [$X,XXX-$X,XXX per year]",
            "  - Scholarships: [Specific scholarship names]",
            "",
            "### United States",
            "List 3-5 universities:",
            "- **[University Name]** - [Program Name]",
            "  - Program: [Specific degree name]",
            "  - Duration: [X years]",
            "  - Entry Requirements: [SAT/ACT, TOEFL/IELTS]",
            "  - Approximate Cost: [$X,XXX-$X,XXX per year]",
            "  - Scholarships: [Available scholarships]",
            "",
            "### Other Countries",
            "List options from Australia, Canada, Germany, or Singapore with similar details.",
            "",
            "## 4. STEP-BY-STEP IMPLEMENTATION PLAN",
            "",
            "### PHASE 1 - IMMEDIATE PREPARATION (Months 1-6)",
            "- [Specific action 1]",
            "- [Specific action 2]",
            "- [Specific action 3]",
            "",
            "### PHASE 2 - APPLICATION PERIOD (Months 6-12)",
            "- [Specific action 1]",
            "- [Specific action 2]",
            "- [Specific action 3]",
            "",
            "### PHASE 3 - STUDY PERIOD (Years 1-4)",
            "- [Milestone 1]",
            "- [Milestone 2]",
            "- [Milestone 3]",
            "",
            "## 5. FINANCIAL PLANNING",
            "- **Total Estimated Cost (Local):** LKR [amount] for [X] years",
            "- **Total Estimated Cost (International):** $[amount] for [X] years",
            "",
            "**Funding Options:**",
            "- [Specific scholarship 1] - [Brief details]",
            "- [Specific scholarship 2] - [Brief details]",
            "- [Loan option 1]",
            "- [Other funding sources]",
            "",
            "## 6. ALTERNATIVE PATHWAYS",
            "- **Online Programs:** [List 2-3 specific online degree programs with names and providers]",
            "- **Bootcamps:** [List relevant bootcamps/intensive programs]",
            "- **Bridge Programs:** [List foundation/bridge courses]",
            "",
            "## 7. IMMEDIATE NEXT STEPS",
            "1. [Most urgent action with timeline]",
            "2. [Second action with timeline]",
            "3. [Third action with timeline]",
            "4. [Fourth action]",
            "5. [Fifth action]",
            "",
            "CRITICAL REQUIREMENTS:",
            "1. USE WEB SEARCH RESULTS - Extract and include specific names, programs, and URLs",
            "2. PROVIDE SPECIFIC DETAILS - No generic terms like 'related program' or 'various universities'",
            "3. INCLUDE COSTS - Actual LKR and USD amounts, not ranges like 'varies'",
            "4. LOCAL FIRST - Prioritize Sri Lankan options with the most detail",
            "5. MULTIPLE OPTIONS - At least 5-8 institutions per major category",
            "6. ACTIONABLE - Each step should be specific and executable",
            "",
            "Write the complete plan following this exact structure."
        ])
        
        return "\n".join(prompt_parts)

    def _parse_academic_plan_response(self, response: str, career_title: str, student_level: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse LLM response into structured academic plan with enhanced extraction."""

        academic_plan = {
            "career_title": career_title,
            "student_assessment": {
                "current_level": student_level.get("current_level", "unknown") if student_level else "unknown",
                "academic_background": "To be determined from assessment",
                "recommended_timeline": student_level.get("timeline_to_career", "3-6 years") if student_level else "3-6 years"
            },
            "pathway_options": [],
            "step_by_step_plan": [],
            "financial_planning": {
                "total_estimated_cost_lkr": "To be determined",
                "breakdown": {},
                "funding_options": [],
                "cost_saving_tips": []
            },
            "alternative_pathways": [],
            "next_immediate_steps": [],
            "raw_plan": response  # Keep full response
        }

        # Parse response sections more intelligently
        lines = response.split('\n')
        current_section = None
        current_pathway = None
        current_phase = None
        current_institution_data = {}

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            line_clean = line.strip()

            # Detect major section headers (markdown style)
            if line_clean.startswith('##'):
                # Main sections
                if "student assessment" in line_lower:
                    current_section = "assessment"
                elif "local pathway" in line_lower or "sri lankan" in line_lower:
                    current_section = "local_pathways"
                    if not current_pathway:
                        current_pathway = {
                            "pathway_type": "Local (Sri Lankan)",
                            "education_level": "Undergraduate/Postgraduate",
                            "sri_lankan_options": [],
                            "international_options": []
                        }
                elif "international pathway" in line_lower:
                    current_section = "international_pathways"
                    # Save current pathway before starting international
                    if current_pathway and current_pathway["sri_lankan_options"]:
                        academic_plan["pathway_options"].append(current_pathway)
                        current_pathway = {
                            "pathway_type": "International",
                            "education_level": "Undergraduate/Postgraduate",
                            "sri_lankan_options": [],
                            "international_options": []
                        }
                elif "implementation plan" in line_lower or "step-by-step" in line_lower:
                    current_section = "implementation"
                    # Save pathways before moving to implementation
                    if current_pathway:
                        academic_plan["pathway_options"].append(current_pathway)
                        current_pathway = None
                elif "financial planning" in line_lower:
                    current_section = "financial"
                elif "alternative pathway" in line_lower:
                    current_section = "alternatives"
                elif "immediate" in line_lower and "step" in line_lower:
                    current_section = "immediate_steps"
                elif "next step" in line_lower:
                    current_section = "immediate_steps"

            # Detect subsections (### headers)
            elif line_clean.startswith('###'):
                if "state universit" in line_lower:
                    current_section = "state_universities"
                elif "private" in line_lower and ("universit" in line_lower or "institute" in line_lower):
                    current_section = "private_universities"
                elif "professional institute" in line_lower:
                    current_section = "professional_institutes"
                elif "united kingdom" in line_lower or "uk" == line_lower:
                    current_section = "international_uk"
                elif "united states" in line_lower or "usa" in line_lower:
                    current_section = "international_usa"
                elif "australia" in line_lower:
                    current_section = "international_australia"
                elif "phase 1" in line_lower:
                    current_phase = {
                        "phase": "Immediate Preparation",
                        "timeframe": self._extract_timeframe(line_clean),
                        "actions": [],
                        "milestones": [],
                        "key_decisions": []
                    }
                elif "phase 2" in line_lower:
                    if current_phase:
                        academic_plan["step_by_step_plan"].append(current_phase)
                    current_phase = {
                        "phase": "Application Period",
                        "timeframe": self._extract_timeframe(line_clean),
                        "actions": [],
                        "milestones": [],
                        "key_decisions": []
                    }
                elif "phase 3" in line_lower:
                    if current_phase:
                        academic_plan["step_by_step_plan"].append(current_phase)
                    current_phase = {
                        "phase": "Study Period",
                        "timeframe": self._extract_timeframe(line_clean),
                        "actions": [],
                        "milestones": [],
                        "key_decisions": []
                    }

            # Extract structured institution data
            if current_section in ["state_universities", "private_universities", "professional_institutes"]:
                # Look for institution entries starting with bold markers or dashes
                if line_clean.startswith(('- **', '* **', '• **')) or (line_clean.startswith('-') and '**' in line_clean):
                    # Extract institution and program from line
                    inst_info = self._extract_institution_details(line_clean, lines[i:min(i+10, len(lines))])
                    if inst_info and current_pathway:
                        current_pathway["sri_lankan_options"].append(inst_info)

            elif current_section in ["international_uk", "international_usa", "international_australia"]:
                if line_clean.startswith(('- **', '* **', '• **')) or (line_clean.startswith('-') and '**' in line_clean):
                    intl_info = self._extract_international_details(line_clean, lines[i:min(i+10, len(lines))], current_section)
                    if intl_info and current_pathway:
                        current_pathway["international_options"].append(intl_info)

            # Extract bullet points
            if line_clean.startswith(('-', '•', '*', '1.', '2.', '3.', '4.', '5.')):
                item = re.sub(r'^[-•*\d\.\s]+', '', line_clean).strip()
                if item and len(item) > 5:
                    if current_section == "assessment":
                        if "level" in line_lower:
                            academic_plan["student_assessment"]["current_level"] = item
                        elif "timeline" in line_lower:
                            academic_plan["student_assessment"]["recommended_timeline"] = item
                    elif current_section == "financial":
                        if "scholarship" in line_lower or "funding" in line_lower or "loan" in line_lower:
                            academic_plan["financial_planning"]["funding_options"].append(item)
                        elif "lkr" in line_lower or "estimated cost" in line_lower:
                            if "local" in line_lower:
                                academic_plan["financial_planning"]["total_estimated_cost_lkr"] = item
                    elif current_section == "alternatives":
                        academic_plan["alternative_pathways"].append({
                            "pathway_description": item,
                            "advantages": [],
                            "considerations": []
                        })
                    elif current_section == "immediate_steps":
                        academic_plan["next_immediate_steps"].append(item)
                    elif current_phase:
                        current_phase["actions"].append(item)

        # Add remaining data
        if current_pathway and (current_pathway["sri_lankan_options"] or current_pathway["international_options"]):
            academic_plan["pathway_options"].append(current_pathway)
        if current_phase:
            academic_plan["step_by_step_plan"].append(current_phase)

        # Ensure minimum viable data
        if not academic_plan["pathway_options"]:
            academic_plan["pathway_options"] = self._create_default_pathways(career_title)

        if not academic_plan["next_immediate_steps"]:
            academic_plan["next_immediate_steps"] = [
                f"Research specific degree programs for {career_title} at Sri Lankan universities",
                "Check A/L subject requirements for relevant programs",
                "Explore scholarship and funding options (local and international)",
                "Contact university admission offices for up-to-date information",
                "Prepare required documents (transcripts, certificates, recommendations)"
            ]

        return academic_plan

    def _extract_timeframe(self, text: str) -> str:
        """Extract timeframe from phase header."""
        # Look for patterns like (Months 1-6) or (Next 6 months)
        match = re.search(r'\((.*?)\)', text)
        if match:
            return match.group(1)
        return "Variable duration"

    def _extract_institution_details(self, line: str, following_lines: List[str]) -> Optional[Dict[str, Any]]:
        """Extract institution details from markdown-formatted text."""
        # Extract institution name and program from bold text
        inst_match = re.search(r'\*\*(.*?)\*\*(?:\s*-\s*(.*))?', line)
        if not inst_match:
            return None

        institution_name = inst_match.group(1).strip()
        program_name = inst_match.group(2).strip() if inst_match.group(2) else "Relevant program"

        # Extract details from following indented lines
        details = {
            "institution_name": institution_name,
            "program_name": program_name,
            "duration": "3-4 years",
            "entry_requirements": [],
            "approximate_cost": "Contact institution for details",
            "application_timeline": "Check university website",
            "additional_notes": ""
        }

        for follow_line in following_lines[:8]:  # Check next 8 lines for details
            follow_clean = follow_line.strip()
            follow_lower = follow_clean.lower()

            if follow_clean.startswith(('- **', '  - ', '    - ')):
                # Extract field and value
                if "program:" in follow_lower:
                    details["program_name"] = self._extract_field_value(follow_clean, "program")
                elif "duration:" in follow_lower:
                    details["duration"] = self._extract_field_value(follow_clean, "duration")
                elif "entry requirements:" in follow_lower or "requirements:" in follow_lower:
                    req_value = self._extract_field_value(follow_clean, "entry requirements", "requirements")
                    if req_value:
                        details["entry_requirements"] = [r.strip() for r in req_value.split(',')]
                elif "cost:" in follow_lower or "approximate cost:" in follow_lower:
                    details["approximate_cost"] = self._extract_field_value(follow_clean, "cost", "approximate cost")
                elif "application" in follow_lower:
                    details["application_timeline"] = self._extract_field_value(follow_clean, "application")
                elif "website:" in follow_lower or "url:" in follow_lower:
                    details["additional_notes"] = f"Website: {self._extract_field_value(follow_clean, 'website', 'url')}"

        return details

    def _extract_international_details(self, line: str, following_lines: List[str], section: str) -> Optional[Dict[str, Any]]:
        """Extract international institution details."""
        inst_match = re.search(r'\*\*(.*?)\*\*(?:\s*-\s*(.*))?', line)
        if not inst_match:
            return None

        institution_name = inst_match.group(1).strip()
        program_name = inst_match.group(2).strip() if inst_match.group(2) else "Relevant program"

        # Determine country
        country = "International"
        if "uk" in section:
            country = "United Kingdom"
        elif "usa" in section:
            country = "United States"
        elif "australia" in section:
            country = "Australia"

        details = {
            "country": country,
            "institution_examples": [institution_name],
            "program_type": program_name,
            "duration": "3-4 years",
            "entry_requirements": [],
            "approximate_cost": "Contact institution for details",
            "scholarship_opportunities": [],
            "notes": ""
        }

        for follow_line in following_lines[:8]:
            follow_clean = follow_line.strip()
            follow_lower = follow_clean.lower()

            if follow_clean.startswith(('- **', '  - ', '    - ')):
                if "program:" in follow_lower:
                    details["program_type"] = self._extract_field_value(follow_clean, "program")
                elif "duration:" in follow_lower:
                    details["duration"] = self._extract_field_value(follow_clean, "duration")
                elif "entry requirements:" in follow_lower or "requirements:" in follow_lower:
                    req_value = self._extract_field_value(follow_clean, "entry requirements", "requirements")
                    if req_value:
                        details["entry_requirements"] = [r.strip() for r in req_value.split(',')]
                elif "cost:" in follow_lower or "approximate cost:" in follow_lower:
                    details["approximate_cost"] = self._extract_field_value(follow_clean, "cost", "approximate cost")
                elif "scholarship" in follow_lower:
                    sch_value = self._extract_field_value(follow_clean, "scholarship")
                    if sch_value:
                        details["scholarship_opportunities"] = [s.strip() for s in sch_value.split(',')]

        return details

    def _extract_field_value(self, text: str, *field_names) -> str:
        """Extract value after field name (e.g., 'Duration: 4 years' -> '4 years')."""
        text_lower = text.lower()
        for field_name in field_names:
            field_lower = field_name.lower()
            if field_lower in text_lower:
                # Find position of field name
                pos = text_lower.find(field_lower)
                # Extract everything after the colon
                after_field = text[pos + len(field_name):].strip()
                # Remove leading colon and whitespace
                after_field = re.sub(r'^[:\s\*\-]+', '', after_field).strip()
                return after_field
        return ""

    def _parse_institution_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse institution information from text."""
        # Simple extraction - in production, use more sophisticated parsing
        return {
            "institution_name": text[:50],  # Truncate for summary
            "program_name": "Related degree program",
            "duration": "3-4 years",
            "entry_requirements": ["A/L qualification", "Z-score requirement"],
            "approximate_cost": "LKR 50,000 - 500,000 per year",
            "application_timeline": "Apply during A/L results release",
            "additional_notes": "Check current admission requirements"
        }

    def _parse_international_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse international education information from text."""
        return {
            "country": "Various countries",
            "institution_examples": ["Top universities in destination country"],
            "program_type": "Bachelor's/Master's degree",
            "duration": "3-4 years",
            "entry_requirements": ["A/L qualification", "English proficiency", "Application essays"],
            "approximate_cost": "$15,000-$40,000 per year",
            "scholarship_opportunities": ["Government scholarships", "University funding"],
            "notes": "Research specific country requirements"
        }

    def _create_default_pathways(self, career_title: str) -> List[Dict[str, Any]]:
        """Create default academic pathways with specific Sri Lankan examples as fallback."""

        # Map career titles to relevant programs - provide realistic examples
        career_lower = career_title.lower()

        # Determine relevant programs based on career type
        if any(term in career_lower for term in ["software", "computer", "it", "technology", "data", "developer", "programming"]):
            state_programs = [
                ("University of Colombo", "BSc (Hons) in Computer Science", "4 years", "Z-score 1.8+, Maths stream"),
                ("University of Moratuwa", "BSc Engineering (Computer Science & Engineering)", "4 years", "Z-score 2.0+, Maths stream"),
                ("University of Kelaniya", "BSc (Hons) in Software Engineering", "4 years", "Z-score 1.7+, Maths stream"),
                ("University of Sri Jayewardenepura", "BSc (Hons) in Information Technology", "4 years", "Z-score 1.6+, Maths/Bio stream"),
            ]
            private_programs = [
                ("SLIIT", "BSc (Hons) in Information Technology", "4 years", "LKR 800,000-1,200,000 per year"),
                ("NSBM Green University", "BSc (Hons) in Computer Science", "3-4 years", "LKR 750,000-1,000,000 per year"),
                ("IIT (Informatics Institute)", "BSc (Hons) in Computing (Westminster)", "3 years", "LKR 900,000-1,400,000 per year"),
                ("APIIT", "BSc (Hons) in Computer Science (Staffordshire)", "3-4 years", "LKR 850,000-1,300,000 per year"),
            ]
        elif any(term in career_lower for term in ["engineer", "mechanical", "electrical", "civil"]):
            state_programs = [
                ("University of Moratuwa", "BSc Engineering (Relevant specialization)", "4 years", "Z-score 1.9+, Maths stream"),
                ("University of Peradeniya", "BSc Engineering (Relevant specialization)", "4 years", "Z-score 1.8+, Maths stream"),
                ("University of Ruhuna", "BSc Engineering (Relevant specialization)", "4 years", "Z-score 1.7+, Maths stream"),
            ]
            private_programs = [
                ("SLIIT", "BSc (Hons) in Engineering (Relevant specialization)", "4 years", "LKR 900,000-1,400,000 per year"),
                ("NSBM Green University", "BSc (Hons) in Engineering", "4 years", "LKR 850,000-1,250,000 per year"),
            ]
        elif any(term in career_lower for term in ["business", "management", "marketing", "finance", "accounting"]):
            state_programs = [
                ("University of Sri Jayewardenepura", "BSc in Business Administration", "3-4 years", "Z-score 1.5+, Commerce/Arts"),
                ("University of Kelaniya", "BSc in Management Studies & Commerce", "3-4 years", "Z-score 1.4+, Commerce/Arts"),
                ("University of Colombo", "BBA/BSc in Management", "3-4 years", "Z-score 1.6+, Commerce/Arts"),
            ]
            private_programs = [
                ("NSBM Green University", "BSc (Hons) in Business Management", "3-4 years", "LKR 600,000-900,000 per year"),
                ("SLIIT", "BSc (Hons) in Business Administration", "3-4 years", "LKR 650,000-950,000 per year"),
                ("IIT", "BSc (Hons) in Business Management (Westminster)", "3 years", "LKR 700,000-1,000,000 per year"),
            ]
        elif any(term in career_lower for term in ["design", "art", "creative", "graphic", "animation"]):
            state_programs = [
                ("University of the Visual & Performing Arts", "BA in Design", "4 years", "Portfolio + A/L qualification"),
                ("University of Moratuwa", "BSc in Architecture", "5 years", "Aptitude test + Z-score 1.5+"),
            ]
            private_programs = [
                ("SLIIT", "BSc (Hons) in Multimedia & Web Design", "3-4 years", "LKR 700,000-1,000,000 per year"),
                ("NSBM Green University", "BSc (Hons) in Graphic Design", "3-4 years", "LKR 650,000-900,000 per year"),
                ("AOD (Academy of Design)", "Higher Diploma in Graphic Design", "2-3 years", "LKR 500,000-800,000 per year"),
            ]
        else:
            # Generic fallback for other careers
            state_programs = [
                ("University of Colombo", f"Relevant degree program for {career_title}", "3-4 years", "Z-score varies by program"),
                ("University of Peradeniya", f"Relevant degree program", "3-4 years", "Z-score varies by program"),
                ("University of Kelaniya", f"Relevant degree program", "3-4 years", "Z-score varies by program"),
            ]
            private_programs = [
                ("SLIIT", f"Relevant degree program for {career_title}", "3-4 years", "LKR 700,000-1,200,000 per year"),
                ("NSBM Green University", f"Relevant degree program", "3-4 years", "LKR 650,000-1,000,000 per year"),
            ]

        # Build Sri Lankan options from the selected programs
        sri_lankan_options = []

        for univ, program, duration, req_or_cost in state_programs:
            sri_lankan_options.append({
                "institution_name": univ,
                "program_name": program,
                "duration": duration,
                "entry_requirements": [req_or_cost, "A/L with relevant subjects"],
                "approximate_cost": "LKR 30,000-100,000 per year (state university)",
                "application_timeline": "Apply during university admission period (August-September)",
                "additional_notes": "Highly competitive admission via UGC"
            })

        for univ, program, duration, cost in private_programs:
            sri_lankan_options.append({
                "institution_name": univ,
                "program_name": program,
                "duration": duration,
                "entry_requirements": ["A/L qualification (3 passes minimum)", "Interview or entrance test"],
                "approximate_cost": cost,
                "application_timeline": "Multiple intake periods throughout the year",
                "additional_notes": "Flexible admission, international degree partnerships available"
            })

        return [
            {
                "pathway_type": "Sri Lankan & International",
                "education_level": "Undergraduate",
                "sri_lankan_options": sri_lankan_options,
                "international_options": [
                    {
                        "country": "United Kingdom",
                        "institution_examples": ["University of London", "Manchester Metropolitan University", "Coventry University", "Plymouth University"],
                        "program_type": f"BSc/BA (Hons) in related field to {career_title}",
                        "duration": "3 years (full-time)",
                        "entry_requirements": ["A/L qualification with good grades", "IELTS 6.0-6.5 overall", "Personal statement"],
                        "approximate_cost": "$22,000-35,000 per year (tuition + living)",
                        "scholarship_opportunities": ["Chevening Scholarship", "Commonwealth Scholarships", "University merit scholarships"],
                        "notes": "Shorter duration than US, post-study work visa available for 2 years"
                    },
                    {
                        "country": "United States",
                        "institution_examples": ["State Universities (e.g., Arizona State, Penn State)", "Community Colleges with transfer pathways"],
                        "program_type": f"Bachelor's degree in related field to {career_title}",
                        "duration": "4 years (full-time)",
                        "entry_requirements": ["A/L qualification", "SAT/ACT scores", "TOEFL (80+) or IELTS (6.5+)", "Essays and recommendations"],
                        "approximate_cost": "$25,000-50,000 per year (varies widely)",
                        "scholarship_opportunities": ["University scholarships (merit/need-based)", "Fulbright Program", "Athletic scholarships"],
                        "notes": "Many universities offer financial aid to international students"
                    },
                    {
                        "country": "Australia",
                        "institution_examples": ["Curtin University", "Victoria University", "Monash University", "University of Melbourne"],
                        "program_type": f"Bachelor degree in related field",
                        "duration": "3-4 years",
                        "entry_requirements": ["A/L qualification or foundation year", "IELTS 6.5-7.0", "May require bridging courses"],
                        "approximate_cost": "$20,000-35,000 per year (AUD)",
                        "scholarship_opportunities": ["Australia Awards", "University scholarships", "Destination Australia"],
                        "notes": "Post-study work rights available, pathway to permanent residence"
                    }
                ]
            }
        ]

    def _update_state_with_academic_plan(
        self,
        state: AgentState,
        career_title: str,
        academic_plan: Dict[str, Any]
    ) -> AgentState:
        """Update the state with the completed academic plan."""
        
        updated_state = state.copy(deep=True)
        
        # Find and update the corresponding career blueprint
        if updated_state.career_blueprints:
            for blueprint in updated_state.career_blueprints:
                if blueprint.career_title == career_title:
                    blueprint.academic_plan = academic_plan
                    self.logger.info(f"Updated blueprint for {career_title} with academic plan")
                    break
        
        # Add completion message
        completion_message = AIMessage(
            content=f"✅ Completed academic pathway plan for {career_title}",
            name=self.name
        )
        updated_state.messages.append(completion_message)
        
        return updated_state

    def _create_plan_summary(self, academic_plan: Dict[str, Any]) -> str:
        """Create a brief summary of the academic plan."""
        
        pathways_count = len(academic_plan.get("pathway_options", []))
        phases_count = len(academic_plan.get("step_by_step_plan", []))
        student_level = academic_plan.get("student_assessment", {}).get("current_level", "unknown")
        
        summary = f"Academic Pathway Plan: {pathways_count} pathway options, "
        summary += f"{phases_count} implementation phases for {student_level} level student"
        
        return summary


# Example usage and testing function
def test_academic_pathway_agent():
    """Test function for the Academic Pathway Agent."""
    import sys
    import os
    
    # Add the backend directory to the path
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    backend_parent = os.path.dirname(os.path.dirname(backend_dir))
    if backend_parent not in sys.path:
        sys.path.append(backend_parent)
    
    from models.state_models import AgentState, CareerBlueprint, StudentProfile
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🚀 Testing Academic Pathway Agent")
    print("=" * 50)
    
    # Create agent
    try:
        academic_agent = AcademicPathwayAgent()
        print("✅ Agent created successfully")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return
    
    # Create test state with career blueprint
    test_blueprint = CareerBlueprint(
        career_title="Software Engineer",
        career_description="Design and develop software applications and systems",
        match_score=88.0,
        match_reasoning="Strong match based on technical interests and analytical skills"
    )
    
    test_profile = StudentProfile(
        current_education_level="A/L Student",
        major_field="Physical Science Stream",
        technical_skills=["Basic Programming", "Mathematics"],
        career_interests=["Technology", "Software Development", "Problem Solving"],
        academic_performance="Good"
    )
    
    test_state = AgentState(
        student_profile=test_profile,
        career_blueprints=[test_blueprint],
        session_id="test_academic_session_001"
    )
    
    # Process task
    print("\n📚 Creating academic pathway plan...")
    try:
        result = academic_agent.process_task(test_state)
        
        if result.success:
            print("✅ Academic pathway created successfully!")
            print(f"Summary: {result.result_data.get('plan_summary')}")
            print(f"Student Level: {result.result_data.get('student_level', {}).get('current_level')}")
            print(f"Processing time: {result.processing_time:.2f}s")
            
            # Print some details from the plan
            academic_plan = result.result_data.get('academic_plan', {})
            pathways = academic_plan.get('pathway_options', [])
            if pathways:
                print(f"\n📊 Generated {len(pathways)} pathway options:")
                for i, pathway in enumerate(pathways, 1):
                    print(f"  {i}. {pathway.get('pathway_type', 'Unknown')} pathway")
                    sri_lankan = pathway.get('sri_lankan_options', [])
                    if sri_lankan:
                        print(f"     - {len(sri_lankan)} Sri Lankan options")
                    international = pathway.get('international_options', [])
                    if international:
                        print(f"     - {len(international)} international options")
            
            next_steps = academic_plan.get('next_immediate_steps', [])
            if next_steps:
                print(f"\n🎯 Next immediate steps ({len(next_steps)}):")
                for step in next_steps[:3]:
                    print(f"  - {step}")
        else:
            print(f"❌ Failed: {result.error_message}")
    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

    print("\n" + "=" * 50)
    print("Academic Pathway Agent test completed!")


if __name__ == "__main__":
    test_academic_pathway_agent()