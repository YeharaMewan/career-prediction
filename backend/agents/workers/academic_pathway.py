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
from models.state_models import (
    AgentState,
    TaskResult,
    StudentProfile,
    AcademicPathwayStructured,
    AcademicPathwaySection,
    AcademicPathwaySubsection,
    AcademicCourseCard,
)

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
    estimate_tokens,
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
            **kwargs,
        )

        # IMPORTANT: Override LLM with higher max_tokens for comprehensive output
        # Academic pathway requires detailed output (30+ international universities, local pathways, etc.)
        # Default max_tokens (4096) causes truncation, missing International Pathway section
        from langchain_openai import ChatOpenAI

        self._original_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            timeout=60,
            max_retries=2,
            max_tokens=16384,  # Increase from default 4096 to support full output
        )

        # Recreate react agent with new LLM
        from langgraph.prebuilt import create_react_agent

        self.react_agent = create_react_agent(
            model=self._original_llm, tools=self.tools, prompt=self.system_prompt
        )

        self.logger.info(
            f"✅ Academic Pathway Agent initialized with max_tokens=16384 for comprehensive output"
        )

        # Agent capabilities
        self.capabilities.extend(
            [
                "educational_pathway_design",
                "sri_lankan_education_system_expertise",
                "international_education_options",
                "entry_requirements_analysis",
                "institution_recommendations",
                "academic_timeline_planning",
                "cost_estimation",
                "career_level_assessment",
            ]
        )

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
                provider="fallback",  # Automatically switches with LLM fallback
                similarity_threshold=0.35,
                top_k=10,
            )
            self.rag_enabled = True
            self.logger.info("✅ RAG retriever initialized for academic knowledge base")
        except Exception as e:
            self.logger.warning(
                f"RAG retriever initialization failed: {e}. Continuing without RAG."
            )
            self.rag_enabled = False

        self.logger = logging.getLogger(f"agent.{self.name}")

        # Log token optimization
        self.logger.info(
            f"✅ Academic Pathway Agent initialized with optimized prompts"
        )
        self.logger.info(
            f"   System prompt: ~{estimate_tokens(system_prompt)} tokens (was ~1200)"
        )

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
            json_path = os.path.join(
                backend_dir, "prompts", "sri_lankan_institutions.json"
            )

            # Load data
            with open(json_path, "r", encoding="utf-8") as f:
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
                    "language": "English/Sinhala",
                },
                "university_of_peradeniya": {
                    "name": "University of Peradeniya",
                    "location": "Kandy",
                    "strengths": [
                        "Engineering",
                        "Medicine",
                        "Science",
                        "Arts",
                        "Agriculture",
                    ],
                    "website": "https://www.pdn.ac.lk/",
                    "entry_method": "University admission via Z-score",
                    "language": "English/Sinhala",
                },
                "university_of_moratuwa": {
                    "name": "University of Moratuwa",
                    "location": "Moratuwa",
                    "strengths": ["Engineering", "IT", "Architecture", "Business"],
                    "website": "https://www.mrt.ac.lk/",
                    "entry_method": "University admission via Z-score",
                    "language": "English",
                },
                "university_of_ruhuna": {
                    "name": "University of Ruhuna",
                    "location": "Matara",
                    "strengths": [
                        "Medicine",
                        "Engineering",
                        "Science",
                        "Humanities",
                        "Management",
                        "Fisheries",
                    ],
                    "website": "https://www.ruh.ac.lk/",
                    "entry_method": "University admission via Z-score",
                    "language": "English/Sinhala",
                },
                "university_of_sri_jayewardenepura": {
                    "name": "University of Sri Jayewardenepura",
                    "location": "Nugegoda",
                    "strengths": [
                        "Management",
                        "Applied Sciences",
                        "Humanities",
                        "Medical Sciences",
                    ],
                    "website": "https://www.sjp.ac.lk/",
                    "entry_method": "University admission via Z-score",
                    "language": "English/Sinhala",
                },
                "open_university": {
                    "name": "Open University of Sri Lanka",
                    "location": "Multiple locations",
                    "strengths": [
                        "Distance Learning",
                        "Technology",
                        "Education",
                        "Social Sciences",
                    ],
                    "website": "https://www.ou.ac.lk/",
                    "entry_method": "Open admission for most programs",
                    "language": "English/Sinhala",
                },
            },
            "private_universities": {
                "sliit": {
                    "name": "Sri Lanka Institute of Information Technology",
                    "location": "Malabe, Colombo",
                    "strengths": ["IT", "Engineering", "Business", "Humanities"],
                    "international_partnerships": [
                        "Liverpool John Moores",
                        "Curtin University",
                    ],
                    "entry_method": "Private admission, A/L results",
                    "language": "English",
                },
                "nsbm": {
                    "name": "NSBM Green University",
                    "location": "Pitipana, Homagama",
                    "strengths": ["Business", "Computing", "Engineering", "Sciences"],
                    "international_partnerships": [
                        "Plymouth University",
                        "Victoria University",
                    ],
                    "entry_method": "Private admission",
                    "language": "English",
                },
                "iit": {
                    "name": "Informatics Institute of Technology",
                    "location": "Colombo",
                    "strengths": ["IT", "Business", "Law"],
                    "international_partnerships": [
                        "University of Westminster",
                        "Robert Gordon University",
                    ],
                    "entry_method": "Private admission",
                    "language": "English",
                },
            },
            "professional_institutes": {
                "ca_sri_lanka": {
                    "name": "Institute of Chartered Accountants of Sri Lanka",
                    "specialization": "Accounting and Finance",
                    "entry_requirements": ["A/L qualification", "Foundation course"],
                    "duration": "3-4 years",
                    "recognition": "International (Commonwealth countries)",
                },
                "cima": {
                    "name": "Chartered Institute of Management Accountants",
                    "specialization": "Management Accounting",
                    "entry_requirements": [
                        "A/L qualification",
                        "Degree exemptions available",
                    ],
                    "duration": "2-3 years",
                    "recognition": "Global",
                },
                "iesl": {
                    "name": "Institution of Engineers Sri Lanka",
                    "specialization": "Engineering",
                    "entry_requirements": [
                        "Engineering degree",
                        "Professional experience",
                    ],
                    "duration": "Ongoing professional development",
                    "recognition": "National and regional",
                },
            },
        }

    def _initialize_international_options(self) -> Dict[str, Dict[str, Any]]:
        """Initialize international education options database."""
        return {
            "popular_destinations": {
                "uk": {
                    "advantages": [
                        "Quality education",
                        "Shorter duration",
                        "Post-study work visa",
                    ],
                    "entry_requirements": [
                        "A/L qualifications",
                        "IELTS 6.0-7.0",
                        "Personal statement",
                    ],
                    "approximate_cost_usd": "25000-40000 per year",
                    "popular_universities": [
                        "University of London",
                        "Manchester University",
                        "Edinburgh University",
                    ],
                    "scholarship_opportunities": [
                        "Chevening",
                        "Commonwealth",
                        "University scholarships",
                    ],
                },
                "australia": {
                    "advantages": [
                        "High quality education",
                        "Work opportunities",
                        "Immigration pathways",
                    ],
                    "entry_requirements": [
                        "A/L qualifications",
                        "IELTS 6.5-7.0",
                        "Foundation year may be required",
                    ],
                    "approximate_cost_usd": "20000-35000 per year",
                    "popular_universities": [
                        "University of Melbourne",
                        "UNSW",
                        "Monash University",
                    ],
                    "scholarship_opportunities": [
                        "Australia Awards",
                        "University scholarships",
                    ],
                },
                "canada": {
                    "advantages": [
                        "Quality education",
                        "Immigration opportunities",
                        "Work permits",
                    ],
                    "entry_requirements": [
                        "A/L qualifications",
                        "IELTS 6.5",
                        "Foundation programs available",
                    ],
                    "approximate_cost_usd": "15000-30000 per year",
                    "popular_universities": [
                        "University of Toronto",
                        "UBC",
                        "McGill University",
                    ],
                    "scholarship_opportunities": [
                        "Canadian government scholarships",
                        "University funding",
                    ],
                },
                "usa": {
                    "advantages": [
                        "World-class education",
                        "Research opportunities",
                        "Diverse programs",
                    ],
                    "entry_requirements": [
                        "A/L qualifications",
                        "SAT/ACT",
                        "TOEFL/IELTS",
                        "Essays",
                    ],
                    "approximate_cost_usd": "25000-60000 per year",
                    "popular_universities": [
                        "State universities",
                        "Community colleges",
                        "Private universities",
                    ],
                    "scholarship_opportunities": [
                        "Merit scholarships",
                        "Need-based aid",
                        "Athletic scholarships",
                    ],
                },
                "germany": {
                    "advantages": [
                        "Low tuition fees",
                        "Quality education",
                        "Strong economy",
                    ],
                    "entry_requirements": [
                        "A/L qualifications",
                        "German language or English programs",
                        "Foundation year",
                    ],
                    "approximate_cost_usd": "500-3000 per year (public universities)",
                    "popular_universities": [
                        "TU Munich",
                        "University of Heidelberg",
                        "RWTH Aachen",
                    ],
                    "scholarship_opportunities": ["DAAD scholarships", "Erasmus+"],
                },
                "singapore": {
                    "advantages": [
                        "Regional hub",
                        "High quality",
                        "Multicultural environment",
                    ],
                    "entry_requirements": [
                        "Strong A/L results",
                        "IELTS/TOEFL",
                        "Competitive admission",
                    ],
                    "approximate_cost_usd": "20000-40000 per year",
                    "popular_universities": ["NUS", "NTU", "SMU"],
                    "scholarship_opportunities": [
                        "Government scholarships",
                        "University scholarships",
                    ],
                },
            },
            "online_options": {
                "advantages": ["Flexibility", "Lower cost", "Study while working"],
                "popular_providers": [
                    "University of London Online",
                    "Arizona State University Online",
                    "Penn State World Campus",
                ],
                "considerations": [
                    "Accreditation",
                    "Recognition in Sri Lanka",
                    "Technology requirements",
                ],
                "cost_range_usd": "5000-20000 per year",
            },
        }

    def _initialize_career_level_matrix(self) -> Dict[str, Dict[str, Any]]:
        """Initialize career level assessment matrix."""
        return {
            "ol_student": {
                "description": "GCE O/L student (Age 15-16)",
                "immediate_focus": "A/L subject selection",
                "timeline_to_career": "6-10 years",
                "key_decisions": [
                    "A/L stream selection",
                    "Career exploration",
                    "Subject combinations",
                ],
                "recommendations": [
                    "Career guidance",
                    "Subject selection counseling",
                    "Skill development",
                ],
            },
            "al_student": {
                "description": "GCE A/L student (Age 17-18)",
                "immediate_focus": "University admission preparation",
                "timeline_to_career": "4-8 years",
                "key_decisions": [
                    "University selection",
                    "Degree program choice",
                    "Local vs international",
                ],
                "recommendations": [
                    "University applications",
                    "Scholarship applications",
                    "Gap year consideration",
                ],
            },
            "fresh_al_graduate": {
                "description": "Recently completed A/L",
                "immediate_focus": "Higher education pathway",
                "timeline_to_career": "3-6 years",
                "key_decisions": [
                    "Immediate university entry",
                    "Gap year",
                    "Foundation programs",
                ],
                "recommendations": [
                    "Multiple application strategies",
                    "Foundation courses",
                    "Work experience",
                ],
            },
            "entry_level_professional": {
                "description": "0-2 years work experience",
                "immediate_focus": "Skill enhancement and career direction",
                "timeline_to_career": "2-5 years",
                "key_decisions": [
                    "Part-time study",
                    "Professional certifications",
                    "Career change",
                ],
                "recommendations": [
                    "Evening/weekend programs",
                    "Professional development",
                    "Industry certifications",
                ],
            },
            "mid_level_professional": {
                "description": "3-7 years work experience",
                "immediate_focus": "Career advancement and specialization",
                "timeline_to_career": "1-3 years",
                "key_decisions": [
                    "MBA/Master's degree",
                    "Leadership roles",
                    "Industry specialization",
                ],
                "recommendations": [
                    "Executive education",
                    "Leadership programs",
                    "Advanced certifications",
                ],
            },
            "senior_professional": {
                "description": "8+ years work experience",
                "immediate_focus": "Strategic career positioning",
                "timeline_to_career": "Ongoing development",
                "key_decisions": [
                    "Executive education",
                    "Thought leadership",
                    "Entrepreneurship",
                ],
                "recommendations": [
                    "Executive MBA",
                    "Board positions",
                    "Consulting",
                    "Teaching",
                ],
            },
        }

    def _search_local_universities(self, career_title: str) -> str:
        """
        Search for Sri Lankan universities offering programs for the career.

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(
                f"Searching for Sri Lankan universities for {career_title}"
            )

            # Search for universities
            results = self.web_search.search_universities(
                career_title, country="Sri Lanka", max_results=8
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
            self.logger.info(
                f"Searching for Sri Lankan scholarships for {career_title}"
            )

            # Search for scholarships
            results = self.web_search.search_scholarships(
                career_title, country="Sri Lanka", max_results=6
            )

            if not results:
                return "No scholarship information found."

            return self.web_search.format_results_for_llm(results, max_snippets=4)

        except Exception as e:
            self.logger.error(f"Scholarship search failed: {e}")
            return "Scholarship search unavailable."

    def _search_international_universities(
        self, career_title: str, country: str = "UK"
    ) -> str:
        """
        Search for international universities offering programs for the career.
        Uses both RAG (vector database) and web search for comprehensive results.

        Args:
            career_title: Career name
            country: Country to search in (UK, USA, Australia, Canada, etc.)

        Returns:
            Formatted string with search results (5-10 universities)
        """
        try:
            self.logger.info(
                f"Searching for international universities for {career_title} in {country}"
            )

            results_text = []

            # 1. First try RAG retriever (vector database) for international university data
            if self.rag_enabled:
                try:
                    rag_query = f"{career_title} {country} university programs admission requirements costs scholarships"
                    self.logger.info(f"   RAG search: {rag_query}")

                    # Retrieve using correct method signature
                    rag_state = self.rag_retriever.retrieve(
                        query=rag_query,
                        force_retrieval=True,  # Force retrieval for international universities
                        filter_metadata=None,
                        include_citations=True,
                    )

                    if (
                        rag_state.retrieved_documents
                        and len(rag_state.retrieved_documents) > 0
                    ):
                        docs = rag_state.retrieved_documents
                        self.logger.info(
                            f"   ✅ RAG found {len(docs)} documents for {country}"
                        )

                        # Format RAG results - use the formatted context directly
                        if rag_state.context:
                            rag_text = (
                                f"\n=== KNOWLEDGE BASE DATA FOR {country.upper()} ===\n"
                            )
                            rag_text += rag_state.context
                            results_text.append(rag_text)
                            self.logger.info(
                                f"   ✅ Added RAG results for {country}: {len(rag_text)} chars"
                            )
                        else:
                            self.logger.warning(f"   ⚠️ RAG context empty for {country}")
                    else:
                        self.logger.warning(f"   ⚠️ No RAG results for {country}")

                except Exception as e:
                    self.logger.error(f"   ❌ RAG search failed for {country}: {e}")

            # 2. Also use web search as additional source
            try:
                web_results = self.web_search.search_universities(
                    career_title,
                    country=country,
                    max_results=10,  # Increased from 6 to 10
                )

                if web_results:
                    self.logger.info(
                        f"   ✅ Web search found {len(web_results)} results for {country}"
                    )
                    web_text = f"\n=== WEB SEARCH RESULTS FOR {country.upper()} ===\n"
                    web_text += self.web_search.format_results_for_llm(
                        web_results, max_snippets=6
                    )
                    results_text.append(web_text)
                else:
                    self.logger.warning(f"   ⚠️ No web results for {country}")

            except Exception as e:
                self.logger.error(f"   ❌ Web search failed for {country}: {e}")

            # Combine results
            if results_text:
                combined = "\n".join(results_text)
                self.logger.info(
                    f"   ✅ Total combined results for {country}: {len(combined)} chars"
                )
                return combined
            else:
                return f"No information found for {country}. Please provide at least 5 universities with details."

        except Exception as e:
            self.logger.error(f"International university search failed: {e}")
            return f"Search unavailable for {country}. Please provide at least 5 universities with details."

    def _search_international_scholarships(self, career_title: str) -> str:
        """
        Search for international scholarship opportunities.

        Returns:
            Formatted string with search results
        """
        try:
            self.logger.info(
                f"Searching for international scholarships for {career_title}"
            )

            # Search for major international scholarships
            chevening_results = self.web_search.search(
                f"Chevening scholarship {career_title} Sri Lankan students",
                max_results=3,
            )

            commonwealth_results = self.web_search.search(
                f"Commonwealth scholarship {career_title} Sri Lanka", max_results=3
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
                max_results=6,
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
                max_results=5,
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
                    task_type="academic_pathway",
                    success=False,
                    error_message="No career information found in state",
                )

            career_title = career_info.get("title")
            self.logger.info(f"Creating academic pathway for: {career_title}")

            # Assess student's current level
            student_level_assessment = self._assess_student_level(state.student_profile)

            # Create academic pathway plan (async call) with language support
            academic_plan = await self._create_academic_pathway_plan(
                career_title=career_title,
                career_description=career_info.get("description"),
                student_profile=state.student_profile,
                student_level=student_level_assessment,
                language=state.preferred_language or "en",
            )

            # Update the career blueprint with academic plan
            updated_state = self._update_state_with_academic_plan(
                state, career_title, academic_plan
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            # Log execution
            log_agent_execution(
                self.name,
                f"academic_plan_creation_{career_title}",
                f"Created plan for {student_level_assessment['current_level']} level student",
                processing_time,
            )

            self._log_task_completion(
                "academic_pathway_planning",
                True,
                f"for {career_title} in {processing_time:.2f}s",
            )

            return self._create_task_result(
                task_type="academic_pathway",
                success=True,
                result_data={
                    "career_title": career_title,
                    "academic_plan": academic_plan,
                    "student_level": student_level_assessment,
                    "plan_summary": self._create_plan_summary(academic_plan),
                },
                processing_time=processing_time,
                updated_state=updated_state,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._log_task_completion(
                "academic_pathway_planning", False, f"Error: {str(e)}"
            )

            # Log error execution
            log_agent_execution(
                self.name,
                "academic_plan_creation_error",
                f"Failed: {str(e)}",
                processing_time,
            )

            return self._create_task_result(
                task_type="academic_pathway",
                success=False,
                error_message=str(e),
                processing_time=processing_time,
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
                        "blueprint": blueprint,
                    }

        # Check messages for career assignment
        if state.messages:
            for msg in reversed(state.messages):
                if isinstance(msg, HumanMessage):
                    content = msg.content
                    # Try to extract career title from message
                    if (
                        "career" in content.lower()
                        or "academic pathway" in content.lower()
                    ):
                        return {"title": content, "description": None}

        return None

    def _assess_student_level(
        self, student_profile: Optional[StudentProfile]
    ) -> Dict[str, Any]:
        """
        Assess the student's current academic/professional level.
        """
        if not student_profile:
            return {
                "current_level": "unknown",
                "assessment_confidence": "low",
                "recommendations": [
                    "Provide more information about current education/experience"
                ],
            }

        education_level = getattr(
            student_profile, "current_education_level", ""
        ).lower()

        # Determine level based on education
        if any(
            keyword in education_level for keyword in ["o/l", "ol", "ordinary level"]
        ):
            level_key = "ol_student"
        elif any(
            keyword in education_level for keyword in ["a/l", "al", "advanced level"]
        ):
            level_key = "al_student"
        elif any(
            keyword in education_level
            for keyword in ["bachelor", "undergraduate", "degree"]
        ):
            level_key = "fresh_al_graduate"  # Assuming currently studying
        elif any(
            keyword in education_level for keyword in ["master", "mba", "postgraduate"]
        ):
            level_key = "mid_level_professional"
        elif any(keyword in education_level for keyword in ["phd", "doctorate"]):
            level_key = "senior_professional"
        else:
            # Try to infer from other information
            if (
                hasattr(student_profile, "work_experience")
                and student_profile.work_experience
            ):
                level_key = "entry_level_professional"
            elif hasattr(student_profile, "age") and student_profile.age:
                if student_profile.age <= 16:
                    level_key = "ol_student"
                elif student_profile.age <= 19:
                    level_key = "al_student"
                else:
                    level_key = "entry_level_professional"
            else:
                level_key = "al_student"  # Default assumption

        level_info = self.career_level_matrix.get(
            level_key, self.career_level_matrix["al_student"]
        )

        return {
            "current_level": level_key,
            "description": level_info["description"],
            "immediate_focus": level_info["immediate_focus"],
            "timeline_to_career": level_info["timeline_to_career"],
            "key_decisions": level_info["key_decisions"],
            "recommendations": level_info["recommendations"],
            "assessment_confidence": "high" if education_level else "medium",
        }

    async def _create_academic_pathway_plan(
        self,
        career_title: str,
        career_description: Optional[str] = None,
        student_profile: Optional[StudentProfile] = None,
        student_level: Optional[Dict[str, Any]] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Create comprehensive academic pathway plan using LLM.

        This is an async method to support non-blocking LLM invocation.
        """
        # Build prompt with career details and student context
        prompt = self._build_academic_planning_prompt(
            career_title, career_description, student_profile, student_level, language
        )

        # Invoke LLM with structured output request (async)
        response = await self.invoke_with_prompt(prompt)

        # DEBUG: Log LLM response
        response_content = response if isinstance(response, str) else response
        self.logger.info(f"📄 LLM Response Length: {len(response_content)} chars")
        self.logger.info(
            f"📄 LLM Response Preview (first 500 chars):\n{response_content[:500]}"
        )

        # Parse and structure the response
        academic_plan = self._parse_academic_plan_response(
            response, career_title, student_level
        )

        return academic_plan

    def _build_academic_planning_prompt(
        self,
        career_title: str,
        career_description: Optional[str],
        student_profile: Optional[StudentProfile],
        student_level: Optional[Dict[str, Any]],
        language: str = "en",
    ) -> str:
        """Build comprehensive prompt for academic pathway planning with RAG and web search results."""

        # Import language helper
        from utils.prompt_templates import get_language_instruction

        # Perform RAG retrieval from knowledge base
        rag_context = ""
        if self.rag_enabled:
            self.logger.info(
                f"Retrieving knowledge base information for {career_title}"
            )
            # Optimized query: shorter, keyword-focused for better semantic matching
            rag_state = self.rag_retriever.retrieve(
                query=f"{career_title} education degrees programs institutions training Sri Lanka",
                include_citations=True,
            )
            if rag_state.context:
                rag_context = rag_state.context
                self.logger.info(
                    f"✅ RAG retrieved {len(rag_state.retrieved_documents)} relevant documents, {len(rag_context)} chars of context"
                )
            else:
                self.logger.warning(
                    "⚠️ RAG retrieval returned no context - check vector database population"
                )

        # Perform web searches for current information
        self.logger.info(f"Gathering real-time information for {career_title}")

        local_universities_info = self._search_local_universities(career_title)
        local_scholarships_info = self._search_local_scholarships(career_title)
        international_uk_info = self._search_international_universities(
            career_title, "UK"
        )
        international_usa_info = self._search_international_universities(
            career_title, "USA"
        )
        international_australia_info = self._search_international_universities(
            career_title, "Australia"
        )
        international_canada_info = self._search_international_universities(
            career_title, "Canada"
        )
        international_germany_info = self._search_international_universities(
            career_title, "Germany"
        )
        international_nz_info = self._search_international_universities(
            career_title, "New Zealand"
        )
        international_singapore_info = self._search_international_universities(
            career_title, "Singapore"
        )
        international_netherlands_info = self._search_international_universities(
            career_title, "Netherlands"
        )
        international_ireland_info = self._search_international_universities(
            career_title, "Ireland"
        )
        international_sweden_info = self._search_international_universities(
            career_title, "Sweden"
        )
        international_scholarships_info = self._search_international_scholarships(
            career_title
        )
        alternative_pathways_info = self._search_alternative_pathways(career_title)
        admission_requirements_info = self._search_admission_requirements(career_title)

        # Get language instruction
        language_instruction = ""
        if language != "en":
            language_instruction = (
                get_language_instruction(language, "academic_pathway") + "\n\n"
            )

        prompt_parts = [
            language_instruction
            + f"Create a comprehensive academic pathway plan for: {career_title}",
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
                    "KNOWLEDGE BASE INFORMATION (Verified academic data from PDFs):",
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
                "INTERNATIONAL UNIVERSITIES (Australia):",
                international_australia_info,
                "",
                "INTERNATIONAL UNIVERSITIES (Canada):",
                international_canada_info,
                "",
                "INTERNATIONAL UNIVERSITIES (Germany):",
                international_germany_info,
                "",
                "INTERNATIONAL UNIVERSITIES (New Zealand):",
                international_nz_info,
                "",
                "INTERNATIONAL UNIVERSITIES (Singapore):",
                international_singapore_info,
                "",
                "INTERNATIONAL UNIVERSITIES (Netherlands):",
                international_netherlands_info,
                "",
                "INTERNATIONAL UNIVERSITIES (Ireland):",
                international_ireland_info,
                "",
                "INTERNATIONAL UNIVERSITIES (Sweden):",
                international_sweden_info,
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
                "",
            ]
        )

        if student_level:
            prompt_parts.append("STUDENT LEVEL ASSESSMENT:")
            prompt_parts.append(
                f"- Current Level: {student_level.get('description', 'Unknown')}"
            )
            prompt_parts.append(
                f"- Timeline to Career: {student_level.get('timeline_to_career', 'Unknown')}"
            )
            prompt_parts.append(
                f"- Immediate Focus: {student_level.get('immediate_focus', 'Career preparation')}"
            )
            prompt_parts.append("")

        if student_profile:
            prompt_parts.append("STUDENT CONTEXT:")
            if hasattr(student_profile, "current_education_level"):
                prompt_parts.append(
                    f"- Education Level: {student_profile.current_education_level}"
                )
            if hasattr(student_profile, "major_field") and student_profile.major_field:
                prompt_parts.append(
                    f"- Current/Previous Field: {student_profile.major_field}"
                )
            if (
                hasattr(student_profile, "career_interests")
                and student_profile.career_interests
            ):
                prompt_parts.append(
                    f"- Career Interests: {', '.join(student_profile.career_interests[:3])}"
                )
            if (
                hasattr(student_profile, "academic_performance")
                and student_profile.academic_performance
            ):
                prompt_parts.append(
                    f"- Academic Performance: {student_profile.academic_performance}"
                )
            prompt_parts.append("")

        prompt_parts.extend(
            [
                "IMPORTANT INSTRUCTIONS:",
                "- **PRIORITIZE SRI LANKAN UNIVERSITIES** - Local options should be the most detailed and extensive",
                "- USE THE REAL-TIME WEB SEARCH RESULTS PROVIDED ABOVE",
                "- USE THE KNOWLEDGE BASE (RAG) DATA - Extract Sri Lankan university information from the PDF documents provided",
                "- Include actual university names, programs, URLs, and current costs from search results",
                "- Reference specific scholarships and opportunities found in searches",
                "- Follow the STRUCTURE: Local Pathways (MOST DETAILED) → International Pathways → Other Relevant Information",
                "- Provide AT LEAST 5-10 specific Sri Lankan government universities (aim for 10 if possible, minimum 5) AND 5-8 private universities",
                "- Provide AT LEAST 3 international universities PER COUNTRY (minimum 3 per country shown)",
                "- Include SPECIFIC degree program names (e.g., 'BSc (Hons) in Computer Science' not 'related degree')",
                "- For Sri Lankan universities, include admission requirements, application deadlines, and specific program details",
                "- ALWAYS include official website URLs for ALL universities and programs",
                "",
                "OUTPUT FORMAT - Write a detailed, well-structured plan with clear section headings:",
                "",
                "## 1. LOCAL PATHWAYS (Sri Lankan Options)",
                "",
                "### A. GOVERNMENT UNIVERSITIES (FREE EDUCATION)",
                "List 5-10 government/state universities with specific programs (aim for 10 if possible):",
                "**NOTE: Government universities in Sri Lanka offer FREE education. Students only pay minimal registration/exam fees (LKR 30,000-100,000 per year).**",
                "",
                "- **[University Name]** - [Specific Program Name]",
                "  - Program: [Full degree/diploma name]",
                "  - Duration: [X years]",
                "  - Entry Requirements: [A/L subjects and specific requirements]",
                "  - Website: [Official university URL - REQUIRED]",
                "",
                "### B. PRIVATE UNIVERSITIES (PAID EDUCATION)",
                "List 5-8 private institutions with specific programs:",
                "",
                "- **[Institution Name]** - [Specific Program Name]",
                "  - Program: [Full degree/diploma name]",
                "  - Duration: [X years]",
                "  - Entry Requirements: [Specific requirements]",
                "  - Cost per Year: LKR [amount] per year",
                "  - Total Program Cost: LKR [amount] for [X] years",
                "  - International Partnerships: [Partner universities if any]",
                "  - Website: [Official institution URL - REQUIRED]",
                "",
                "**FINANCIAL PLANNING FOR PRIVATE UNIVERSITIES:**",
                "- Payment Options: Most private universities offer semester or annual installment plans",
                "- Institution Scholarships: Merit-based and need-based scholarships available at most institutions",
                "- Loan Options: Bank loans and student financing options available through major banks",
                "- Part-time Work: Many institutions offer on-campus or nearby part-time work opportunities",
                "",
                "### C. PROFESSIONAL INSTITUTES",
                "List relevant professional certifications:",
                "- **[Institute Name]** - [Qualification Name]",
                "  - Program: [Full certification name]",
                "  - Duration: [X years/months]",
                "  - Entry Requirements: [Requirements]",
                "  - Approximate Cost: [LKR amount]",
                "",
                "## 2. INTERNATIONAL PATHWAY OPTIONS",
                "",
                "**CRITICAL: USE THE KNOWLEDGE BASE (RAG) AND WEB SEARCH RESULTS PROVIDED ABOVE**",
                "**REQUIREMENT: Provide 3 universities PER COUNTRY (exactly 3 for each country)**",
                "**DO NOT use generic or placeholder university names. Extract actual universities from the search results.**",
                "**Each university MUST have: Name, Program, Duration, Cost, Website URL**",
                "",
                "### United Kingdom",
                "**REQUIRED: List exactly 3 universities FROM THE SEARCH RESULTS:**",
                "",
                "For EACH university, provide:",
                "- **[Actual University Name]** - [Actual Program Name]",
                "  - Program: [Specific degree name - e.g., 'BSc Computer Science', 'MSc Data Science']",
                "  - Duration: [X years - e.g., '3 years undergraduate', '1 year postgraduate']",
                "  - Entry Requirements: [A/L grades, IELTS 6.0-6.5, specific subjects]",
                "  - Approximate Cost: [£X,XXX-£X,XXX per year - tuition only or including living]",
                "  - Scholarships: [Specific scholarship names if available from search]",
                "  - Website: [Official university URL from search - REQUIRED]",
                "",
                "### United States",
                "**REQUIRED: List exactly 3 universities FROM THE SEARCH RESULTS:**",
                "",
                "For EACH university, provide:",
                "- **[Actual University Name]** - [Actual Program Name]",
                "  - Program: [Specific degree name]",
                "  - Duration: [X years]",
                "  - Entry Requirements: [SAT/ACT scores, TOEFL 80+/IELTS 6.5+, GPA requirements]",
                "  - Approximate Cost: [$X,XXX-$X,XXX per year]",
                "  - Scholarships: [Available scholarships from search]",
                "  - Website: [Official university URL from search - REQUIRED]",
                "",
                "### Australia",
                "**REQUIRED: List exactly 3 universities FROM THE SEARCH RESULTS:**",
                "",
                "For EACH university, provide:",
                "- **[Actual University Name]** - [Actual Program Name]",
                "  - Program: [Specific degree name]",
                "  - Duration: [X years]",
                "  - Entry Requirements: [A/L or IB, IELTS 6.5-7.0]",
                "  - Approximate Cost: [AUD $X,XXX-$X,XXX per year]",
                "  - Scholarships: [Available scholarships]",
                "  - Website: [Official university URL from search - REQUIRED]",
                "",
                "### Canada",
                "**REQUIRED: List exactly 3 universities FROM THE SEARCH RESULTS:**",
                "",
                "For EACH university, provide:",
                "- **[Actual University Name]** - [Actual Program Name]",
                "  - Program: [Specific degree name]",
                "  - Duration: [X years]",
                "  - Entry Requirements: [High school grades, IELTS/TOEFL scores]",
                "  - Approximate Cost: [CAD $X,XXX-$X,XXX per year]",
                "  - Scholarships: [Available scholarships]",
                "  - Website: [Official university URL from search - REQUIRED]",
                "",
                "### Germany",
                "**REQUIRED: List exactly 3 universities FROM THE SEARCH RESULTS:**",
                "",
                "For EACH university, provide:",
                "- **[Actual University Name]** - [Actual Program Name]",
                "  - Program: [Specific degree name]",
                "  - Duration: [X years]",
                "  - Entry Requirements: [High school grades, German language (if required), IELTS/TOEFL]",
                "  - Approximate Cost: [€X,XXX-€X,XXX per year or FREE for public universities]",
                "  - Scholarships: [Available scholarships - DAAD, etc.]",
                "  - Website: [Official university URL from search - REQUIRED]",
                "",
                "### New Zealand",
                "**REQUIRED: List exactly 3 universities FROM THE SEARCH RESULTS:**",
                "",
                "For EACH university, provide:",
                "- **[Actual University Name]** - [Actual Program Name]",
                "  - Program: [Specific degree name]",
                "  - Duration: [X years]",
                "  - Entry Requirements: [A/L or IB, IELTS 6.0-6.5]",
                "  - Approximate Cost: [NZD $X,XXX-$X,XXX per year]",
                "  - Scholarships: [Available scholarships]",
                "  - Website: [Official university URL from search - REQUIRED]",
                "",
                "### Singapore",
                "**REQUIRED: List exactly 3 universities FROM THE SEARCH RESULTS:**",
                "",
                "For EACH university, provide:",
                "- **[Actual University Name]** - [Actual Program Name]",
                "  - Program: [Specific degree name]",
                "  - Duration: [X years]",
                "  - Entry Requirements: [A/L grades, IELTS 6.5-7.0]",
                "  - Approximate Cost: [SGD $X,XXX-$X,XXX per year]",
                "  - Scholarships: [Available scholarships - Ministry scholarships, etc.]",
                "  - Website: [Official university URL from search - REQUIRED]",
                "",
                "### Netherlands",
                "**REQUIRED: List exactly 3 universities FROM THE SEARCH RESULTS:**",
                "",
                "For EACH university, provide:",
                "- **[Actual University Name]** - [Actual Program Name]",
                "  - Program: [Specific degree name]",
                "  - Duration: [X years]",
                "  - Entry Requirements: [High school diploma, IELTS 6.0-6.5]",
                "  - Approximate Cost: [€X,XXX-€X,XXX per year]",
                "  - Scholarships: [Available scholarships - Holland Scholarship, etc.]",
                "  - Website: [Official university URL from search - REQUIRED]",
                "",
                "### Ireland",
                "**REQUIRED: List exactly 3 universities FROM THE SEARCH RESULTS:**",
                "",
                "For EACH university, provide:",
                "- **[Actual University Name]** - [Actual Program Name]",
                "  - Program: [Specific degree name]",
                "  - Duration: [X years]",
                "  - Entry Requirements: [Leaving Certificate or A/L, IELTS 6.0-6.5]",
                "  - Approximate Cost: [€X,XXX-€X,XXX per year]",
                "  - Scholarships: [Available scholarships]",
                "  - Website: [Official university URL from search - REQUIRED]",
                "",
                "### Sweden",
                "**REQUIRED: List exactly 3 universities FROM THE SEARCH RESULTS:**",
                "",
                "For EACH university, provide:",
                "- **[Actual University Name]** - [Actual Program Name]",
                "  - Program: [Specific degree name]",
                "  - Duration: [X years]",
                "  - Entry Requirements: [High school diploma, IELTS 6.5-7.0]",
                "  - Approximate Cost: [SEK XXX,XXX per year or FREE for EU/EEA students]",
                "  - Scholarships: [Available scholarships - Swedish Institute, etc.]",
                "  - Website: [Official university URL from search - REQUIRED]",
                "",
                "## 3. STEP-BY-STEP IMPLEMENTATION PLAN",
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
                "## 4. ALTERNATIVE PATHWAYS",
                "- Online Programs: [List 2-3 specific online degree programs with names, providers, and URLs]",
                "- Bootcamps: [List relevant bootcamps/intensive programs with URLs]",
                "- Bridge Programs: [List foundation/bridge courses with URLs]",
                "",
                f"## 5. IMMEDIATE NEXT STEPS (For {student_level.get('description', 'Current Student') if student_level else 'Current Student'})",
                "",
                "### A. REQUIRED EXAMS AND TESTS",
                "List specific exams needed with registration timelines:",
                "- **IELTS/TOEFL** (if applying internationally):",
                "  - Registration: [Month/timeline]",
                "  - Test dates: [Available dates]",
                "  - Preparation time needed: 3-6 months",
                "  - Cost: [LKR amount]",
                "- **SAT/ACT** (for US universities):",
                "  - Registration deadlines: [Dates]",
                "  - Test dates: [Dates]",
                "- **Local Entrance Exams** (Sri Lankan universities):",
                "  - University-specific exams: [Names and dates]",
                "  - Registration periods: [Timelines]",
                "",
                "### B. DOCUMENT PREPARATION CHECKLIST",
                "Essential documents to prepare now:",
                "1. **Academic Transcripts**",
                "   - Request certified copies from current institution",
                "   - Timeline: 2-4 weeks processing",
                "2. **Recommendation Letters**",
                "   - Identify 2-3 teachers/supervisors",
                "   - Request at least 1 month before deadline",
                "3. **Personal Statement/Essay**",
                "   - Draft and review (allow 3-4 weeks)",
                "   - Specific to each university application",
                "4. **Portfolio** (if applicable to {career_title})",
                "   - Compile relevant projects and work samples",
                "5. **Identification Documents**",
                "   - Passport (if applying internationally)",
                "   - National ID",
                "   - Recent photographs",
                "",
                "### C. ACTION TIMELINE (Next 6 Months)",
                "Month-by-month breakdown:",
                "1. **Month 1-2:** [Specific actions based on student level]",
                "2. **Month 3-4:** [Application preparation actions]",
                "3. **Month 5-6:** [Final steps and submissions]",
                "",
                "Write the complete plan following this exact structure. Give maximum detail to Sri Lankan universities.",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_academic_plan_response(
        self, response: str, career_title: str, student_level: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse LLM response into structured academic plan with enhanced extraction."""

        # DEBUG: Log parsing start
        self.logger.info(
            f"🔍 Starting to parse academic plan, response length: {len(response)}"
        )

        academic_plan = {
            "career_title": career_title,
            "student_assessment": {
                "current_level": (
                    student_level.get("current_level", "unknown")
                    if student_level
                    else "unknown"
                ),
                "academic_background": "To be determined from assessment",
                "recommended_timeline": (
                    student_level.get("timeline_to_career", "3-6 years")
                    if student_level
                    else "3-6 years"
                ),
            },
            "pathway_options": [],
            "step_by_step_plan": [],
            "financial_planning": {
                "total_estimated_cost_lkr": "To be determined",
                "breakdown": {},
                "funding_options": [],
                "cost_saving_tips": [],
            },
            "alternative_pathways": [],
            "next_immediate_steps": [],
            "raw_plan": response,  # Keep full response
        }

        # Parse response sections more intelligently
        lines = response.split("\n")
        current_section = None
        current_pathway = None
        current_phase = None
        current_institution_data = {}

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            line_clean = line.strip()

            # Detect major section headers (markdown style)
            if line_clean.startswith("##"):
                # Main sections
                if "student assessment" in line_lower:
                    current_section = "assessment"
                elif any(
                    keyword in line_lower
                    for keyword in ["## 1", "local pathway", "sri lankan", "sri lanka"]
                ):
                    current_section = "local_pathways"
                    if not current_pathway:
                        current_pathway = {
                            "pathway_type": "Local (Sri Lankan)",
                            "education_level": "Undergraduate/Postgraduate",
                            "government_universities": [],
                            "private_universities": [],
                            "professional_institutes": [],
                            "international_options": [],
                        }
                elif any(
                    keyword in line_lower
                    for keyword in [
                        "## 2",
                        "international pathway",
                        "overseas",
                        "abroad",
                    ]
                ):
                    self.logger.info(
                        f"🔍 DETECTED INTERNATIONAL PATHWAY SECTION: {line_clean}"
                    )
                    current_section = "international_pathways"
                    # Save current pathway before starting international
                    if current_pathway and (
                        current_pathway["government_universities"]
                        or current_pathway["private_universities"]
                        or current_pathway["professional_institutes"]
                    ):
                        academic_plan["pathway_options"].append(current_pathway)
                    
                    # CRITICAL FIX: Always create new pathway for international section
                    current_pathway = {
                        "pathway_type": "International",
                        "education_level": "Undergraduate/Postgraduate",
                        "government_universities": [],
                        "private_universities": [],
                        "professional_institutes": [],
                        "international_options": [],
                    }
                    self.logger.info(f"✅ Created new International pathway container")
                elif (
                    "implementation plan" in line_lower or "step-by-step" in line_lower
                ):
                    current_section = "implementation"
                    # Save pathways before moving to implementation
                    if current_pathway and (
                        current_pathway["government_universities"]
                        or current_pathway["private_universities"]
                        or current_pathway["professional_institutes"]
                        or current_pathway["international_options"]
                    ):
                        self.logger.debug(
                            f"💾 Saving pathway before implementation section: {len(current_pathway['government_universities'])} gov unis, {len(current_pathway['private_universities'])} private unis, {len(current_pathway['professional_institutes'])} prof institutes, {len(current_pathway['international_options'])} intl options"
                        )
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
            elif line_clean.startswith("###"):
                if any(
                    keyword in line_lower
                    for keyword in [
                        "government universit",
                        "state universit",
                        "free education",
                        "public universit",
                        "### a",
                    ]
                ):
                    current_section = "government_universities"
                elif any(
                    keyword in line_lower
                    for keyword in [
                        "private universit",
                        "private institute",
                        "paid education",
                        "### b",
                    ]
                ) or (
                    ("private" in line_lower or "paid" in line_lower)
                    and ("universit" in line_lower or "institute" in line_lower)
                ):
                    current_section = "private_universities"
                elif any(
                    keyword in line_lower
                    for keyword in [
                        "professional institute",
                        "professional qualifications",
                        "### c",
                    ]
                ):
                    current_section = "professional_institutes"
                elif "united kingdom" in line_lower or "uk" == line_lower:
                    self.logger.info(f"🔍 DETECTED UK SUBSECTION: {line_clean}")
                    current_section = "international_uk"
                elif "united states" in line_lower or "usa" in line_lower:
                    self.logger.info(f"🔍 DETECTED USA SUBSECTION: {line_clean}")
                    current_section = "international_usa"
                elif "australia" in line_lower:
                    self.logger.info(f"🔍 DETECTED AUSTRALIA SUBSECTION: {line_clean}")
                    current_section = "international_australia"
                elif "canada" in line_lower:
                    self.logger.info(f"🔍 DETECTED CANADA SUBSECTION: {line_clean}")
                    current_section = "international_canada"
                elif "germany" in line_lower:
                    self.logger.info(f"🔍 DETECTED GERMANY SUBSECTION: {line_clean}")
                    current_section = "international_germany"
                elif "new zealand" in line_lower:
                    self.logger.info(
                        f"🔍 DETECTED NEW ZEALAND SUBSECTION: {line_clean}"
                    )
                    current_section = "international_nz"
                elif "singapore" in line_lower:
                    self.logger.info(f"🔍 DETECTED SINGAPORE SUBSECTION: {line_clean}")
                    current_section = "international_singapore"
                elif "netherlands" in line_lower:
                    self.logger.info(
                        f"🔍 DETECTED NETHERLANDS SUBSECTION: {line_clean}"
                    )
                    current_section = "international_netherlands"
                elif "ireland" in line_lower:
                    self.logger.info(f"🔍 DETECTED IRELAND SUBSECTION: {line_clean}")
                    current_section = "international_ireland"
                elif "sweden" in line_lower:
                    self.logger.info(f"🔍 DETECTED SWEDEN SUBSECTION: {line_clean}")
                    current_section = "international_sweden"
                elif "phase 1" in line_lower:
                    current_phase = {
                        "phase": "Immediate Preparation",
                        "timeframe": self._extract_timeframe(line_clean),
                        "actions": [],
                        "milestones": [],
                        "key_decisions": [],
                    }
                elif "phase 2" in line_lower:
                    if current_phase:
                        academic_plan["step_by_step_plan"].append(current_phase)
                    current_phase = {
                        "phase": "Application Period",
                        "timeframe": self._extract_timeframe(line_clean),
                        "actions": [],
                        "milestones": [],
                        "key_decisions": [],
                    }
                elif "phase 3" in line_lower:
                    if current_phase:
                        academic_plan["step_by_step_plan"].append(current_phase)
                    current_phase = {
                        "phase": "Study Period",
                        "timeframe": self._extract_timeframe(line_clean),
                        "actions": [],
                        "milestones": [],
                        "key_decisions": [],
                    }

            # Extract structured institution data
            if current_section in [
                "government_universities",
                "private_universities",
                "professional_institutes",
            ]:
                # Look for institution entries - more flexible matching to handle extra spaces and different bullets
                line_stripped = line_clean.lstrip()
                if (
                    line_stripped.startswith(("**", "- **", "* **", "• **", "  - **"))
                    or (line_clean.startswith("-") and "**" in line_clean)
                    or (
                        "**" in line_clean
                        and any(
                            line_clean.startswith(prefix)
                            for prefix in ["- ", "* ", "• ", "  -"]
                        )
                    )
                ):
                    # DEBUG LOGGING
                    self.logger.debug(f"📌 Found institution line: {line_clean[:100]}")
                    self.logger.debug(f"   current_section: {current_section}")
                    self.logger.debug(
                        f"   current_pathway exists: {current_pathway is not None}"
                    )

                    # Extract institution and program from line
                    inst_info = self._extract_institution_details(
                        line_clean,
                        lines[i : min(i + 10, len(lines))],
                        is_private=(current_section == "private_universities"),
                    )

                    # DEBUG LOGGING
                    self.logger.debug(
                        f"   inst_info extracted: {inst_info is not None}"
                    )
                    if inst_info:
                        self.logger.debug(
                            f"   Institution: {inst_info.get('institution_name', 'UNKNOWN')}"
                        )

                    if inst_info and current_pathway:
                        # Add to appropriate array based on section
                        if current_section == "government_universities":
                            current_pathway["government_universities"].append(inst_info)
                            self.logger.debug(
                                f"   ✅ Added to government_universities (count: {len(current_pathway['government_universities'])})"
                            )
                        elif current_section == "private_universities":
                            current_pathway["private_universities"].append(inst_info)
                        elif current_section == "professional_institutes":
                            current_pathway["professional_institutes"].append(inst_info)
                    elif current_pathway and "**" in line_clean:
                        # Fallback: Create basic institution from bold text
                        name_match = re.search(r"\*\*(.+?)\*\*", line_clean)
                        if name_match:
                            inst_name = name_match.group(1)
                            basic_institution = {
                                "institution_name": inst_name,
                                "program_name": "Relevant program for chosen career",
                                "duration": "3-4 years",
                                "additional_notes": line_clean.replace("- ", "")
                                .replace("* ", "")
                                .replace("• ", ""),
                            }
                            if current_section == "government_universities":
                                current_pathway["government_universities"].append(
                                    basic_institution
                                )
                            elif current_section == "private_universities":
                                current_pathway["private_universities"].append(
                                    basic_institution
                                )
                            elif current_section == "professional_institutes":
                                current_pathway["professional_institutes"].append(
                                    basic_institution
                                )
                            self.logger.warning(
                                f"⚠️ Used fallback extraction for: {inst_name}"
                            )

            elif current_section in [
                "international_uk",
                "international_usa",
                "international_australia",
                "international_canada",
                "international_germany",
                "international_nz",
                "international_singapore",
                "international_netherlands",
                "international_ireland",
                "international_sweden",
            ]:
                # ENHANCED PARSING: Handle multiple LLM output formats
                # Formats: numbered lists (1.), bullets (-, *, •), plain bold (**)
                line_stripped = line_clean.lstrip()
                has_bold = "**" in line_clean and line_clean.count("**") >= 2
                starts_bullet = line_stripped.startswith(("- ", "* ", "• "))
                starts_number = bool(re.match(r"^\d+\.\s+", line_stripped))
                starts_bold = line_stripped.startswith("**")

                # CRITICAL: More flexible detection
                is_university_entry = has_bold and (
                    starts_bullet or starts_number or starts_bold or 
                    # Also accept if bold text appears near start of line
                    line_clean.find("**") < 5
                )

                if is_university_entry:
                    self.logger.info(
                        f"🔍 [{current_section}] Found university entry: {line_clean[:80]}"
                    )
                    
                    # CRITICAL FIX: Ensure current_pathway exists
                    if not current_pathway:
                        self.logger.warning(f"⚠️ current_pathway was None, creating new one")
                        current_pathway = {
                            "pathway_type": "International",
                            "education_level": "Undergraduate/Postgraduate",
                            "government_universities": [],
                            "private_universities": [],
                            "professional_institutes": [],
                            "international_options": [],
                        }
                    
                    intl_info = self._extract_international_details(
                        line_clean, lines[i : min(i + 15, len(lines))], current_section
                    )
                    if intl_info:
                        unis = intl_info.get("institution_examples", ["Unknown"])
                        self.logger.info(
                            f"✅ Extracted international uni: {unis[0] if unis else 'Unknown'}"
                        )
                        current_pathway["international_options"].append(intl_info)
                    else:
                        self.logger.warning(
                            f"⚠️ Extraction failed for: {line_clean[:80]}"
                        )

            # Extract bullet points
            if line_clean.startswith(("-", "•", "*", "1.", "2.", "3.", "4.", "5.")):
                item = re.sub(r"^[-•*\d\.\s]+", "", line_clean).strip()
                if item and len(item) > 5:
                    if current_section == "assessment":
                        if "level" in line_lower:
                            academic_plan["student_assessment"]["current_level"] = item
                        elif "timeline" in line_lower:
                            academic_plan["student_assessment"][
                                "recommended_timeline"
                            ] = item
                    elif current_section == "alternatives":
                        # Enhanced parsing for alternative pathways with title, provider, duration, URL
                        # Skip lines that are just category headers or duplicates
                        if re.match(
                            r"^(Online Programs?|Bootcamps?|Bridge Programs?|Certifications?):\s*$",
                            item,
                            re.IGNORECASE,
                        ):
                            # Skip standalone category headers
                            continue

                        # Skip if this looks like just a provider line or website line
                        if item.lower().startswith(
                            ("provider:", "website:", "duration:", "http")
                        ):
                            continue

                        pathway_name = ""
                        provider = ""
                        duration = "Varies"
                        url = ""
                        description = ""

                        # Extract URL from markdown link format [text](url) or plain URL
                        url_match = re.search(r"\[([^\]]+)\]\(([^\)]+)\)", item)
                        if url_match:
                            # If URL is in markdown format, it might be the provider or just the link
                            link_text = url_match.group(1)
                            url = url_match.group(2)
                            item = item.replace(url_match.group(0), "").strip()
                            # If link text looks like a provider name, use it
                            if not any(
                                word in link_text.lower()
                                for word in ["click", "here", "visit", "website"]
                            ):
                                if not item or len(item) < len(link_text):
                                    # Markdown link was probably the main title
                                    pathway_name = link_text
                        else:
                            # Look for plain URL at the end
                            plain_url = re.search(r"https?://[^\s]+$", item)
                            if plain_url:
                                url = plain_url.group(0)
                                item = item.replace(url, "").strip()

                        # Parse category prefix (e.g., "Online Programs:", "Bootcamps:")
                        category_match = re.match(
                            r"^(Online Programs?|Bootcamps?|Bridge Programs?|Certifications?):\s*(.+)$",
                            item,
                            re.IGNORECASE,
                        )
                        if category_match:
                            description = category_match.group(1)
                            item = category_match.group(2).strip()

                        # Extract title, provider, duration from remaining text
                        # Try to split by common separators (-, |, or parentheses for provider)

                        # Check for provider in parentheses first
                        provider_match = re.search(
                            r"\(Provider:\s*([^)]+)\)|\(([^)]+University[^)]*)\)",
                            item,
                            re.IGNORECASE,
                        )
                        if provider_match:
                            provider = (
                                provider_match.group(1) or provider_match.group(2)
                            ).strip()
                            item = item.replace(provider_match.group(0), "").strip()

                        # Check for duration pattern
                        duration_match = re.search(
                            r"\(?(Duration|Time):\s*([^)]+)\)?", item, re.IGNORECASE
                        )
                        if duration_match:
                            duration = duration_match.group(2).strip()
                            item = item.replace(duration_match.group(0), "").strip()

                        # Split remaining by dash or pipe
                        parts = [
                            p.strip()
                            for p in re.split(r"\s*[-–—]\s*|\s*\|\s*", item)
                            if p.strip() and len(p.strip()) > 2
                        ]

                        if not pathway_name and parts:
                            pathway_name = parts[0]
                            if len(parts) > 1 and not provider:
                                # Second part might be provider
                                if not any(
                                    word in parts[1].lower()
                                    for word in ["month", "year", "week", "varies"]
                                ):
                                    provider = parts[1]

                        # Skip if we couldn't extract a meaningful name
                        if not pathway_name or len(pathway_name) < 5:
                            continue

                        # Check if this entry already exists (to avoid duplicates)
                        duplicate = False
                        for existing in academic_plan["alternative_pathways"]:
                            if (
                                existing.get("pathway_name", "").lower()
                                == pathway_name.lower()
                            ):
                                # Update existing entry with any new info
                                if url and not existing.get("url"):
                                    existing["url"] = url
                                if provider and not existing.get("provider"):
                                    existing["provider"] = provider
                                duplicate = True
                                break

                        if not duplicate:
                            academic_plan["alternative_pathways"].append(
                                {
                                    "pathway_name": pathway_name,
                                    "pathway_description": description,
                                    "provider": provider,
                                    "duration": duration,
                                    "url": url,
                                    "advantages": [],
                                    "considerations": [],
                                }
                            )
                    elif current_section == "immediate_steps":
                        academic_plan["next_immediate_steps"].append(item)
                    elif current_phase:
                        current_phase["actions"].append(item)

        # Add remaining data
        if current_pathway and (
            current_pathway["government_universities"]
            or current_pathway["private_universities"]
            or current_pathway["international_options"]
        ):
            academic_plan["pathway_options"].append(current_pathway)
        if current_phase:
            academic_plan["step_by_step_plan"].append(current_phase)

        # Ensure minimum viable data
        if not academic_plan["pathway_options"]:
            academic_plan["pathway_options"] = self._create_default_pathways(
                career_title
            )

        if not academic_plan["next_immediate_steps"]:
            academic_plan["next_immediate_steps"] = [
                f"Research specific degree programs for {career_title} at Sri Lankan universities",
                "Check A/L subject requirements for relevant programs",
                "Explore scholarship and funding options (local and international)",
                "Contact university admission offices for up-to-date information",
                "Prepare required documents (transcripts, certificates, recommendations)",
            ]

        # DEBUG: Log parsing results
        self.logger.info(f"✅ Parsing complete:")
        self.logger.info(
            f"   - pathway_options count: {len(academic_plan.get('pathway_options', []))}"
        )

        # Count total international options across all pathways
        total_intl = sum(
            len(p.get("international_options", []))
            for p in academic_plan.get("pathway_options", [])
        )
        self.logger.info(f"   - TOTAL INTERNATIONAL OPTIONS: {total_intl}")

        for i, pathway in enumerate(academic_plan.get("pathway_options", [])):
            self.logger.info(f"   - Pathway {i}: type={pathway.get('pathway_type')}")
            self.logger.info(
                f"     - gov unis: {len(pathway.get('government_universities', []))}"
            )
            # Log first institution name if exists
            if pathway.get("government_universities"):
                self.logger.info(
                    f"       Example: {pathway['government_universities'][0].get('institution_name', 'N/A')}"
                )

            self.logger.info(
                f"     - private unis: {len(pathway.get('private_universities', []))}"
            )
            if pathway.get("private_universities"):
                self.logger.info(
                    f"       Example: {pathway['private_universities'][0].get('institution_name', 'N/A')}"
                )

            self.logger.info(
                f"     - prof institutes: {len(pathway.get('professional_institutes', []))}"
            )
            if pathway.get("professional_institutes"):
                self.logger.info(
                    f"       Example: {pathway['professional_institutes'][0].get('institution_name', 'N/A')}"
                )

            self.logger.info(
                f"     - intl options: {len(pathway.get('international_options', []))}"
            )
            if pathway.get("international_options"):
                self.logger.info(
                    f"       Example: {pathway['international_options'][0].get('institution_name', 'N/A')}"
                )
        self.logger.info(
            f"   - alternative_pathways count: {len(academic_plan.get('alternative_pathways', []))}"
        )

        return academic_plan

    def _extract_timeframe(self, text: str) -> str:
        """Extract timeframe from phase header."""
        # Look for patterns like (Months 1-6) or (Next 6 months)
        match = re.search(r"\((.*?)\)", text)
        if match:
            return match.group(1)
        return "Variable duration"

    def _extract_institution_details(
        self, line: str, following_lines: List[str], is_private: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Extract institution details from markdown-formatted text."""
        # Extract institution name and program from bold text
        inst_match = re.search(r"\*\*(.*?)\*\*(?:\s*-\s*(.*))?", line)

        if not inst_match:
            self.logger.warning(
                f"❌ Primary regex failed to extract institution from line: {line[:100]}"
            )
            # Try simpler pattern - just extract text between **
            simple_match = re.search(r"\*\*([^*]+)\*\*", line)
            if simple_match:
                institution_name = simple_match.group(1).strip()
                program_name = "Relevant program"
                self.logger.debug(
                    f"✅ Fallback extraction succeeded: {institution_name}"
                )
            else:
                self.logger.warning(
                    f"❌ Fallback extraction also failed for line: {line[:100]}"
                )
                return None
        else:
            institution_name = inst_match.group(1).strip()
            program_name = (
                inst_match.group(2).strip()
                if inst_match.group(2)
                else "Relevant program"
            )

        # Extract details from following indented lines
        details = {
            "institution_name": institution_name,
            "program_name": program_name,
            "duration": "3-4 years",
            "entry_requirements": [],
            "approximate_cost": "Contact institution for details",
            "application_timeline": "Check university website",
            "additional_notes": "",
            "is_free": not is_private,  # Government universities are free
        }

        # For private universities, add financial planning structure
        if is_private:
            details["financial_planning"] = {
                "total_cost": "To be determined",
                "payment_options": [],
                "institution_scholarships": [],
                "loan_options": [],
                "part_time_work_opportunities": [],
            }

        for follow_line in following_lines[:8]:  # Check next 8 lines for details
            follow_clean = follow_line.strip()
            follow_lower = follow_clean.lower()

            if follow_clean.startswith(("- **", "  - ", "    - ")):
                # Extract field and value
                if "program:" in follow_lower:
                    details["program_name"] = self._extract_field_value(
                        follow_clean, "program"
                    )
                elif "duration:" in follow_lower:
                    details["duration"] = self._extract_field_value(
                        follow_clean, "duration"
                    )
                elif (
                    "entry requirements:" in follow_lower
                    or "requirements:" in follow_lower
                ):
                    req_value = self._extract_field_value(
                        follow_clean, "entry requirements", "requirements"
                    )
                    if req_value:
                        details["entry_requirements"] = [
                            r.strip() for r in req_value.split(",")
                        ]
                elif "cost:" in follow_lower or "approximate cost:" in follow_lower:
                    details["approximate_cost"] = self._extract_field_value(
                        follow_clean, "cost", "approximate cost"
                    )
                elif "application" in follow_lower:
                    details["application_timeline"] = self._extract_field_value(
                        follow_clean, "application"
                    )
                elif "website:" in follow_lower or "url:" in follow_lower:
                    details["additional_notes"] = (
                        f"Website: {self._extract_field_value(follow_clean, 'website', 'url')}"
                    )

                # Extract financial planning for private universities
                elif is_private and "financial planning" in follow_lower:
                    # This indicates start of financial planning section for this institution
                    pass
                elif is_private and details.get("financial_planning"):
                    if "total cost:" in follow_lower:
                        details["financial_planning"]["total_cost"] = (
                            self._extract_field_value(follow_clean, "total cost")
                        )
                    elif "payment option" in follow_lower:
                        details["financial_planning"]["payment_options"].append(
                            self._extract_field_value(follow_clean, "payment option")
                        )
                    elif (
                        "scholarship" in follow_lower
                        and "institution scholarship" in follow_lower
                    ):
                        details["financial_planning"][
                            "institution_scholarships"
                        ].append(
                            self._extract_field_value(
                                follow_clean, "institution scholarship", "scholarship"
                            )
                        )
                    elif "loan" in follow_lower:
                        details["financial_planning"]["loan_options"].append(
                            self._extract_field_value(follow_clean, "loan")
                        )
                    elif (
                        "part-time work" in follow_lower
                        or "part time work" in follow_lower
                    ):
                        details["financial_planning"][
                            "part_time_work_opportunities"
                        ].append(
                            self._extract_field_value(
                                follow_clean, "part-time work", "part time work"
                            )
                        )

        return details

    def _extract_international_details(
        self, line: str, following_lines: List[str], section: str
    ) -> Optional[Dict[str, Any]]:
        """Extract international institution details with enhanced parsing."""
        # ENHANCED: Handle multiple formats
        # Format 1: **University Name** - Program Name
        # Format 2: 1. **University Name** - Program Name
        # Format 3: - **University Name**
        
        # Remove leading bullets/numbers
        line_clean = re.sub(r"^[\s\-\*•\d\.]+", "", line).strip()
        
        # Extract institution name from bold text
        inst_match = re.search(r"\*\*(.+?)\*\*", line_clean)
        if not inst_match:
            self.logger.warning(f"❌ Could not extract institution from: {line[:80]}")
            return None

        institution_name = inst_match.group(1).strip()
        
        # Extract program name (text after the bold text and optional dash)
        program_name = "Relevant program"
        remaining_text = line_clean[inst_match.end():].strip()
        if remaining_text.startswith("-"):
            remaining_text = remaining_text[1:].strip()
        if remaining_text and len(remaining_text) > 3:
            program_name = remaining_text

        # Determine country
        country = "International"
        if "uk" in section:
            country = "United Kingdom"
        elif "usa" in section:
            country = "United States"
        elif "australia" in section:
            country = "Australia"
        elif "canada" in section:
            country = "Canada"
        elif "germany" in section:
            country = "Germany"
        elif "nz" in section or "new_zealand" in section:
            country = "New Zealand"
        elif "singapore" in section:
            country = "Singapore"
        elif "netherlands" in section:
            country = "Netherlands"
        elif "ireland" in section:
            country = "Ireland"
        elif "sweden" in section:
            country = "Sweden"

        details = {
            "country": country,
            "institution_examples": [institution_name],
            "program_type": program_name,
            "duration": "3-4 years",
            "entry_requirements": [],
            "approximate_cost": "Contact institution for details",
            "scholarship_opportunities": [],
            "notes": "",
        }

        # ENHANCED: Parse following lines for additional details
        for follow_line in following_lines[:10]:
            follow_clean = follow_line.strip()
            follow_lower = follow_clean.lower()

            # Skip empty lines or section headers
            if not follow_clean or follow_clean.startswith("##"):
                continue
            
            # Break if we hit another institution (starts with bold)
            if "**" in follow_clean and follow_clean.count("**") >= 2:
                break

            # Extract details from indented bullet points or fields
            if follow_clean.startswith(("- ", "  - ", "    - ", "* ", "• ")):
                # Remove leading bullets
                field_text = re.sub(r"^[\s\-\*•]+", "", follow_clean).strip()
                
                if "program:" in follow_lower or "program type:" in follow_lower:
                    details["program_type"] = self._extract_field_value(
                        field_text, "program", "program type"
                    )
                elif "duration:" in follow_lower:
                    details["duration"] = self._extract_field_value(
                        field_text, "duration"
                    )
                elif "entry requirements:" in follow_lower or "requirements:" in follow_lower:
                    req_value = self._extract_field_value(
                        field_text, "entry requirements", "requirements"
                    )
                    if req_value:
                        details["entry_requirements"] = [
                            r.strip() for r in req_value.split(",") if r.strip()
                        ]
                elif "cost:" in follow_lower or "approximate cost:" in follow_lower or "tuition:" in follow_lower:
                    details["approximate_cost"] = self._extract_field_value(
                        field_text, "cost", "approximate cost", "tuition"
                    )
                elif "scholarship" in follow_lower:
                    sch_value = self._extract_field_value(field_text, "scholarship")
                    if sch_value:
                        details["scholarship_opportunities"] = [
                            s.strip() for s in sch_value.split(",") if s.strip()
                        ]
                elif "note:" in follow_lower or "notes:" in follow_lower:
                    details["notes"] = self._extract_field_value(field_text, "note", "notes")

        self.logger.info(f"✅ Extracted {country} university: {institution_name}")
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
                after_field = text[pos + len(field_name) :].strip()
                # Remove leading colon and whitespace
                after_field = re.sub(r"^[:\s\*\-]+", "", after_field).strip()
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
            "additional_notes": "Check current admission requirements",
        }

    def _parse_international_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse international education information from text."""
        return {
            "country": "Various countries",
            "institution_examples": ["Top universities in destination country"],
            "program_type": "Bachelor's/Master's degree",
            "duration": "3-4 years",
            "entry_requirements": [
                "A/L qualification",
                "English proficiency",
                "Application essays",
            ],
            "approximate_cost": "$15,000-$40,000 per year",
            "scholarship_opportunities": [
                "Government scholarships",
                "University funding",
            ],
            "notes": "Research specific country requirements",
        }

    def _create_default_pathways(self, career_title: str) -> List[Dict[str, Any]]:
        """Create default academic pathways with specific Sri Lankan examples as fallback."""

        # Map career titles to relevant programs - provide realistic examples
        career_lower = career_title.lower()

        # Determine relevant programs based on career type
        if any(
            term in career_lower
            for term in [
                "software",
                "computer",
                "it",
                "technology",
                "data",
                "developer",
                "programming",
            ]
        ):
            state_programs = [
                (
                    "University of Colombo",
                    "BSc (Hons) in Computer Science",
                    "4 years",
                    "Z-score 1.8+, Maths stream",
                    "https://science.cmb.ac.lk",
                ),
                (
                    "University of Moratuwa",
                    "BSc Engineering (Computer Science & Engineering)",
                    "4 years",
                    "Z-score 2.0+, Maths stream",
                    "https://www.mrt.ac.lk",
                ),
                (
                    "University of Kelaniya",
                    "BSc (Hons) in Software Engineering",
                    "4 years",
                    "Z-score 1.7+, Maths stream",
                    "https://www.kln.ac.lk",
                ),
                (
                    "University of Sri Jayewardenepura",
                    "BSc (Hons) in Information Technology",
                    "4 years",
                    "Z-score 1.6+, Maths/Bio stream",
                    "https://www.sjp.ac.lk",
                ),
                (
                    "University of Ruhuna",
                    "BSc (Hons) in Computer Science",
                    "4 years",
                    "Z-score 1.6+, Maths stream",
                    "https://www.ruh.ac.lk",
                ),
                (
                    "University of Peradeniya",
                    "BSc (Hons) in Computer Engineering",
                    "4 years",
                    "Z-score 1.9+, Maths stream",
                    "https://www.pdn.ac.lk",
                ),
                (
                    "Sabaragamuwa University",
                    "BSc (Hons) in Computing and Information Systems",
                    "4 years",
                    "Z-score 1.5+, Maths/Bio stream",
                    "https://www.sab.ac.lk",
                ),
                (
                    "Rajarata University",
                    "BSc (Hons) in Information Technology",
                    "4 years",
                    "Z-score 1.5+, Maths stream",
                    "https://www.rjt.ac.lk",
                ),
                (
                    "Eastern University",
                    "BSc (Hons) in Information Technology",
                    "4 years",
                    "Z-score 1.4+, Maths stream",
                    "https://www.esn.ac.lk",
                ),
                (
                    "Wayamba University",
                    "BSc (Hons) in Software Engineering",
                    "4 years",
                    "Z-score 1.5+, Maths stream",
                    "https://www.wyb.ac.lk",
                ),
            ]
            private_programs = [
                (
                    "SLIIT",
                    "BSc (Hons) in Information Technology",
                    "4 years",
                    "LKR 800,000-1,200,000 per year",
                    "https://www.sliit.lk",
                ),
                (
                    "NSBM Green University",
                    "BSc (Hons) in Computer Science",
                    "3-4 years",
                    "LKR 750,000-1,000,000 per year",
                    "https://www.nsbm.ac.lk",
                ),
                (
                    "IIT (Informatics Institute)",
                    "BSc (Hons) in Computing (Westminster)",
                    "3 years",
                    "LKR 900,000-1,400,000 per year",
                    "https://www.iit.ac.lk",
                ),
                (
                    "APIIT",
                    "BSc (Hons) in Computer Science (Staffordshire)",
                    "3-4 years",
                    "LKR 850,000-1,300,000 per year",
                    "https://www.apiit.lk",
                ),
            ]
        elif any(
            term in career_lower
            for term in ["engineer", "mechanical", "electrical", "civil"]
        ):
            state_programs = [
                (
                    "University of Moratuwa",
                    "BSc Engineering (Relevant specialization)",
                    "4 years",
                    "Z-score 1.9+, Maths stream",
                ),
                (
                    "University of Peradeniya",
                    "BSc Engineering (Relevant specialization)",
                    "4 years",
                    "Z-score 1.8+, Maths stream",
                ),
                (
                    "University of Ruhuna",
                    "BSc Engineering (Relevant specialization)",
                    "4 years",
                    "Z-score 1.7+, Maths stream",
                ),
            ]
            private_programs = [
                (
                    "SLIIT",
                    "BSc (Hons) in Engineering (Relevant specialization)",
                    "4 years",
                    "LKR 900,000-1,400,000 per year",
                ),
                (
                    "NSBM Green University",
                    "BSc (Hons) in Engineering",
                    "4 years",
                    "LKR 850,000-1,250,000 per year",
                ),
            ]
        elif any(
            term in career_lower
            for term in ["business", "management", "marketing", "finance", "accounting"]
        ):
            state_programs = [
                (
                    "University of Sri Jayewardenepura",
                    "BSc in Business Administration",
                    "3-4 years",
                    "Z-score 1.5+, Commerce/Arts",
                ),
                (
                    "University of Kelaniya",
                    "BSc in Management Studies & Commerce",
                    "3-4 years",
                    "Z-score 1.4+, Commerce/Arts",
                ),
                (
                    "University of Colombo",
                    "BBA/BSc in Management",
                    "3-4 years",
                    "Z-score 1.6+, Commerce/Arts",
                ),
            ]
            private_programs = [
                (
                    "NSBM Green University",
                    "BSc (Hons) in Business Management",
                    "3-4 years",
                    "LKR 600,000-900,000 per year",
                ),
                (
                    "SLIIT",
                    "BSc (Hons) in Business Administration",
                    "3-4 years",
                    "LKR 650,000-950,000 per year",
                ),
                (
                    "IIT",
                    "BSc (Hons) in Business Management (Westminster)",
                    "3 years",
                    "LKR 700,000-1,000,000 per year",
                ),
            ]
        elif any(
            term in career_lower
            for term in ["design", "art", "creative", "graphic", "animation"]
        ):
            state_programs = [
                (
                    "University of the Visual & Performing Arts",
                    "BA in Design",
                    "4 years",
                    "Portfolio + A/L qualification",
                ),
                (
                    "University of Moratuwa",
                    "BSc in Architecture",
                    "5 years",
                    "Aptitude test + Z-score 1.5+",
                ),
            ]
            private_programs = [
                (
                    "SLIIT",
                    "BSc (Hons) in Multimedia & Web Design",
                    "3-4 years",
                    "LKR 700,000-1,000,000 per year",
                ),
                (
                    "NSBM Green University",
                    "BSc (Hons) in Graphic Design",
                    "3-4 years",
                    "LKR 650,000-900,000 per year",
                ),
                (
                    "AOD (Academy of Design)",
                    "Higher Diploma in Graphic Design",
                    "2-3 years",
                    "LKR 500,000-800,000 per year",
                ),
            ]
        else:
            # Generic fallback for other careers
            state_programs = [
                (
                    "University of Colombo",
                    f"Relevant degree program for {career_title}",
                    "3-4 years",
                    "Z-score varies by program",
                ),
                (
                    "University of Peradeniya",
                    f"Relevant degree program",
                    "3-4 years",
                    "Z-score varies by program",
                ),
                (
                    "University of Kelaniya",
                    f"Relevant degree program",
                    "3-4 years",
                    "Z-score varies by program",
                ),
            ]
            private_programs = [
                (
                    "SLIIT",
                    f"Relevant degree program for {career_title}",
                    "3-4 years",
                    "LKR 700,000-1,200,000 per year",
                ),
                (
                    "NSBM Green University",
                    f"Relevant degree program",
                    "3-4 years",
                    "LKR 650,000-1,000,000 per year",
                ),
            ]

        # Build separate lists for government, private, and professional institutions
        government_universities = []
        for univ, program, duration, req_or_cost, url in state_programs:
            government_universities.append(
                {
                    "institution_name": univ,
                    "program_name": program,
                    "duration": duration,
                    "entry_requirements": [req_or_cost, "A/L with relevant subjects"],
                    "approximate_cost": "LKR 30,000-100,000 per year (state university)",
                    "application_timeline": "Apply during university admission period (August-September)",
                    "website_url": url,
                    "additional_notes": "Highly competitive admission via UGC",
                }
            )

        private_universities = []
        for univ, program, duration, cost, url in private_programs:
            private_universities.append(
                {
                    "institution_name": univ,
                    "program_name": program,
                    "duration": duration,
                    "entry_requirements": [
                        "A/L qualification (3 passes minimum)",
                        "Interview or entrance test",
                    ],
                    "approximate_cost": cost,
                    "application_timeline": "Multiple intake periods throughout the year",
                    "website_url": url,
                    "additional_notes": "Flexible admission, international degree partnerships available",
                }
            )

        # Add professional institutes (common alternatives)
        professional_institutes = [
            {
                "institution_name": "BCS (British Computer Society)",
                "program_name": "Higher Education Qualifications in IT",
                "duration": "1-2 years",
                "entry_requirements": [
                    "O/L passes (minimum)",
                    "Can progress without A/Ls",
                ],
                "approximate_cost": "LKR 200,000-400,000 per year",
                "additional_notes": "Internationally recognized IT qualifications, pathway to degree programs",
            },
            {
                "institution_name": "CIMA (Chartered Institute of Management Accountants)",
                "program_name": "Professional Accounting Qualification",
                "duration": "2-3 years",
                "entry_requirements": ["A/L or equivalent", "Can start after O/L"],
                "approximate_cost": "LKR 300,000-500,000 total",
                "additional_notes": "Globally recognized accounting qualification",
            },
        ]

        return [
            {
                "pathway_type": "Local (Sri Lankan)",
                "education_level": "Undergraduate",
                "government_universities": government_universities,
                "private_universities": private_universities,
                "professional_institutes": professional_institutes,
                "international_options": [],  # Empty - AI will populate from web search results
            }
        ]

    def _update_state_with_academic_plan(
        self, state: AgentState, career_title: str, academic_plan: Dict[str, Any]
    ) -> AgentState:
        """Update the state with the completed academic plan."""

        updated_state = state.copy(deep=True)

        # Find and update the corresponding career blueprint
        if updated_state.career_blueprints:
            for blueprint in updated_state.career_blueprints:
                if blueprint.career_title == career_title:
                    blueprint.academic_plan = academic_plan
                    # NEW: Generate and add structured format for frontend
                    try:
                        structured_data = self._convert_to_frontend_format(
                            academic_plan
                        )
                        blueprint.academic_plan_structured = structured_data.dict()
                        self.logger.info(
                            f"✅ Generated structured academic pathway for {career_title} with {len(structured_data.sections)} sections"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"❌ Failed to generate structured format for {career_title}: {e}",
                            exc_info=True,
                        )
                        blueprint.academic_plan_structured = None
                    self.logger.info(
                        f"Updated blueprint for {career_title} with academic plan"
                    )
                    break

        # Add completion message
        completion_message = AIMessage(
            content=f"✅ Completed academic pathway plan for {career_title}",
            name=self.name,
        )
        updated_state.messages.append(completion_message)

        return updated_state

    def _create_plan_summary(self, academic_plan: Dict[str, Any]) -> str:
        """Create a brief summary of the academic plan."""

        pathways_count = len(academic_plan.get("pathway_options", []))
        phases_count = len(academic_plan.get("step_by_step_plan", []))
        student_level = academic_plan.get("student_assessment", {}).get(
            "current_level", "unknown"
        )

        summary = f"Academic Pathway Plan: {pathways_count} pathway options, "
        summary += (
            f"{phases_count} implementation phases for {student_level} level student"
        )

        return summary

    def _convert_to_frontend_format(
        self, academic_plan: Dict[str, Any]
    ) -> AcademicPathwayStructured:
        """Convert internal academic_plan to frontend card format with detailed fields."""
        # DEBUG: Log conversion start
        self.logger.info(f"🎨 Converting to frontend format")

        sections = []
        card_id = 1

        # Extract pathway options from academic_plan
        pathway_options = academic_plan.get("pathway_options", [])
        self.logger.info(f"   - pathway_options to process: {len(pathway_options)}")

        for pathway in pathway_options:
            pathway_type = pathway.get("pathway_type", "")

            # LOCAL PATHWAYS Section
            if "Local" in pathway_type or "Sri Lankan" in pathway_type:
                local_subsections = []

                # Government Universities Subsection
                gov_universities = pathway.get("government_universities", [])
                if gov_universities:
                    gov_cards = []
                    for inst in gov_universities:
                        gov_cards.append(
                            AcademicCourseCard(
                                id=card_id,
                                name=inst.get("institution_name", ""),
                                program_name=inst.get("program_name", ""),
                                duration=inst.get("duration", "4 years"),
                                website_url=inst.get("website_url", ""),
                                additional_notes=inst.get("additional_notes", ""),
                            )
                        )
                        card_id += 1

                    local_subsections.append(
                        AcademicPathwaySubsection(
                            subsectionTitle="Government Universities", cards=gov_cards
                        )
                    )

                # Private Universities Subsection
                private_universities = pathway.get("private_universities", [])
                if private_universities:
                    private_cards = []
                    for inst in private_universities:
                        # Extract financial planning if available for calculating total cost
                        fin_plan = inst.get("financial_planning", {})

                        private_cards.append(
                            AcademicCourseCard(
                                id=card_id,
                                name=inst.get("institution_name", ""),
                                program_name=inst.get("program_name", ""),
                                duration=inst.get("duration", "3 years"),
                                cost_per_year=inst.get("approximate_cost", ""),
                                total_cost=(
                                    fin_plan.get("total_cost", "") if fin_plan else None
                                ),
                                international_partnerships=inst.get(
                                    "international_partnerships", []
                                ),
                                website_url=inst.get("website_url", ""),
                                additional_notes=inst.get("additional_notes", ""),
                            )
                        )
                        card_id += 1

                    local_subsections.append(
                        AcademicPathwaySubsection(
                            subsectionTitle="Private Universities", cards=private_cards
                        )
                    )

                # Professional Institutes Subsection
                prof_institutes = pathway.get("professional_institutes", [])
                if prof_institutes:
                    prof_cards = []
                    for inst in prof_institutes:
                        prof_cards.append(
                            AcademicCourseCard(
                                id=card_id,
                                name=inst.get("institution_name", ""),
                                program_name=inst.get("program_name", ""),
                                duration=inst.get("duration", "1 year"),
                                entry_requirements=inst.get("entry_requirements", []),
                                cost_per_year=inst.get("approximate_cost", ""),
                                website_url=inst.get("website_url", ""),
                                additional_notes=inst.get("additional_notes", ""),
                            )
                        )
                        card_id += 1

                    local_subsections.append(
                        AcademicPathwaySubsection(
                            subsectionTitle="Professional Institutes", cards=prof_cards
                        )
                    )

                if local_subsections:
                    sections.append(
                        AcademicPathwaySection(
                            sectionTitle="Local Pathways", subsections=local_subsections
                        )
                    )

            # INTERNATIONAL PATHWAYS Section
            international_options = pathway.get("international_options", [])
            if international_options:
                # Group by country and create separate cards for each university
                # Store country-level metadata separately
                countries = {}
                for intl_option in international_options:
                    country = intl_option.get("country", "International")
                    if country not in countries:
                        countries[country] = {
                            "cards": [],
                            "metadata": {
                                "program_type": intl_option.get("program_type", ""),
                                "duration": intl_option.get("duration", ""),
                                "cost": intl_option.get("approximate_cost", ""),
                                "scholarships": intl_option.get(
                                    "scholarship_opportunities", []
                                ),
                                "notes": intl_option.get("notes", ""),
                            },
                        }

                    # Extract institution names and URLs from pipe-separated format
                    institution_examples = intl_option.get("institution_examples", [])

                    # Create a separate card for each university with country-level data
                    for inst in institution_examples:
                        institution_name = inst
                        website_url = None

                        if "|" in inst:
                            # New format: "University Name|https://url"
                            parts = inst.split("|", 1)
                            institution_name = parts[0].strip()
                            if len(parts) > 1:
                                website_url = parts[1].strip()
                        else:
                            # Old format: just the name
                            institution_name = inst.strip()

                        # Each university card gets the country-level data
                        countries[country]["cards"].append(
                            AcademicCourseCard(
                                id=card_id,
                                name=institution_name,
                                program_name=intl_option.get("program_type", ""),
                                duration=intl_option.get("duration", ""),
                                cost_per_year=intl_option.get("approximate_cost", ""),
                                scholarships=intl_option.get(
                                    "scholarship_opportunities", []
                                ),
                                website_url=website_url,
                                additional_notes=intl_option.get("notes", ""),
                            )
                        )
                        card_id += 1

                international_subsections = []
                for country, data in countries.items():
                    international_subsections.append(
                        AcademicPathwaySubsection(
                            subsectionTitle=country, cards=data["cards"]
                        )
                    )

                sections.append(
                    AcademicPathwaySection(
                        sectionTitle="International Pathway",
                        subsections=international_subsections,
                    )
                )

        # ALTERNATIVE PATHWAYS Section
        alternative_pathways = academic_plan.get("alternative_pathways", [])
        if alternative_pathways:
            alt_cards = []
            for alt in alternative_pathways:
                # Extract title (pathway_name or first part of description)
                alt_name = alt.get("pathway_name", "") or alt.get(
                    "pathway_description", "Alternative Option"
                )

                # Extract provider if available
                provider = alt.get("provider", "")

                # Extract duration
                duration = alt.get("duration", "Varies")

                # Extract URL
                website_url = alt.get("url", "")

                # Build program_name from provider if available
                program_name = f"Provider: {provider}" if provider else ""

                # Build additional notes from description and advantages
                notes_parts = []
                if alt.get("pathway_description"):
                    notes_parts.append(alt["pathway_description"])
                if alt.get("advantages"):
                    notes_parts.append("Advantages: " + ", ".join(alt["advantages"]))
                additional_notes = " | ".join(notes_parts) if notes_parts else ""

                alt_cards.append(
                    AcademicCourseCard(
                        id=card_id,
                        name=alt_name,
                        program_name=program_name,
                        duration=duration,
                        cost_per_year="",
                        website_url=website_url,
                        additional_notes=additional_notes,
                    )
                )
                card_id += 1

            sections.append(
                AcademicPathwaySection(
                    sectionTitle="Alternative Pathways",
                    subsections=[
                        AcademicPathwaySubsection(
                            subsectionTitle="Alternative Options", cards=alt_cards
                        )
                    ],
                )
            )

        # DEBUG: Log conversion results
        self.logger.info(f"✅ Conversion complete: {len(sections)} sections created")
        for section in sections:
            self.logger.info(
                f"   - Section: {section.sectionTitle} ({len(section.subsections)} subsections)"
            )

        return AcademicPathwayStructured(
            title=academic_plan.get("career_title", "Career"),
            pathwayTitle="Academic Pathway",
            description="Build a strong foundation in computer science and software development",
            sections=sections,
        )

    def _extract_website_from_notes(self, notes: str) -> Optional[str]:
        """Extract website URL from notes field"""
        if not notes:
            return None
        # Look for "Website: <url>" pattern
        match = re.search(r"Website:\s*(https?://[^\s]+)", notes)
        if match:
            return match.group(1)
        # Look for any URL
        match = re.search(r"(https?://[^\s]+)", notes)
        return match.group(1) if match else None

    def _format_institution_details(self, inst: Dict, is_free: bool) -> str:
        """Format institution data into condensed details string."""
        parts = []

        # Entry requirements
        if inst.get("entry_requirements"):
            reqs = inst["entry_requirements"]
            if isinstance(reqs, list):
                parts.append(f"Entry: {', '.join(reqs[:2])}")
            else:
                parts.append(f"Entry: {reqs}")

        # Cost
        if is_free:
            parts.append("Cost: Free (Government funded)")
        elif inst.get("approximate_cost"):
            parts.append(f"Cost: {inst['approximate_cost']}")

        # Timeline
        if inst.get("application_timeline"):
            parts.append(f"Apply: {inst['application_timeline']}")

        # Scholarships
        if inst.get("scholarships") or inst.get("financial_planning", {}).get(
            "institution_scholarships"
        ):
            parts.append("Scholarships: Available")

        return ". ".join(parts)

    def _format_international_details(self, intl: Dict) -> str:
        """Format international option details."""
        parts = []

        if intl.get("entry_requirements"):
            reqs = intl["entry_requirements"]
            if isinstance(reqs, list):
                parts.append(f"Entry: {', '.join(reqs[:2])}")
            else:
                parts.append(f"Entry: {reqs}")

        if intl.get("approximate_cost"):
            parts.append(f"Cost: {intl['approximate_cost']}")

        if intl.get("scholarship_opportunities"):
            schols = intl["scholarship_opportunities"]
            if isinstance(schols, list):
                parts.append(f"Scholarships: {', '.join(schols[:2])}")
            else:
                parts.append(f"Scholarships: {schols}")

        if intl.get("notes"):
            parts.append(intl["notes"][:60])

        return ". ".join(parts)


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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
        match_reasoning="Strong match based on technical interests and analytical skills",
    )

    test_profile = StudentProfile(
        current_education_level="A/L Student",
        major_field="Physical Science Stream",
        technical_skills=["Basic Programming", "Mathematics"],
        career_interests=["Technology", "Software Development", "Problem Solving"],
        academic_performance="Good",
    )

    test_state = AgentState(
        student_profile=test_profile,
        career_blueprints=[test_blueprint],
        session_id="test_academic_session_001",
    )

    # Process task
    print("\n📚 Creating academic pathway plan...")
    try:
        result = academic_agent.process_task(test_state)

        if result.success:
            print("✅ Academic pathway created successfully!")
            print(f"Summary: {result.result_data.get('plan_summary')}")
            print(
                f"Student Level: {result.result_data.get('student_level', {}).get('current_level')}"
            )
            print(f"Processing time: {result.processing_time:.2f}s")

            # Print some details from the plan
            academic_plan = result.result_data.get("academic_plan", {})
            pathways = academic_plan.get("pathway_options", [])
            if pathways:
                print(f"\n📊 Generated {len(pathways)} pathway options:")
                for i, pathway in enumerate(pathways, 1):
                    print(f"  {i}. {pathway.get('pathway_type', 'Unknown')} pathway")
                    sri_lankan = pathway.get("sri_lankan_options", [])
                    if sri_lankan:
                        print(f"     - {len(sri_lankan)} Sri Lankan options")
                    international = pathway.get("international_options", [])
                    if international:
                        print(f"     - {len(international)} international options")

            next_steps = academic_plan.get("next_immediate_steps", [])
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
