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


class AcademicPathwayAgent(WorkerAgent):
    """
    Academic Pathway Agent - Educational roadmap specialist for career preparation.
    
    This agent analyzes career requirements and creates detailed educational
    pathways tailored to Sri Lankan students with international alternatives.
    It considers the user's current academic level and provides personalized
    guidance for achieving career goals through education.
    """

    def __init__(self, **kwargs):
        system_prompt = self._create_system_prompt()
        
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
        
        # Initialize education databases
        self.sri_lankan_institutions = self._initialize_sri_lankan_institutions()
        self.international_options = self._initialize_international_options()
        self.career_level_matrix = self._initialize_career_level_matrix()
        
        self.logger = logging.getLogger(f"agent.{self.name}")

    def _create_system_prompt(self) -> str:
        """Create the specialized system prompt for academic pathway planning."""
        return """You are an expert Academic Pathway Agent specializing in educational roadmap design for Sri Lankan students pursuing various careers.

YOUR ROLE:
You are a specialist in the Career Planning team, working under the Career Planning Supervisor. 
Your specific responsibility is to create detailed, actionable educational pathways for identified careers, with expertise in both Sri Lankan and international education systems.

CORE COMPETENCIES:
1. Career Level Assessment: Determine student's current academic/professional level
2. Sri Lankan Education System: Deep knowledge of local universities, institutes, and pathways
3. International Education: Global university options and pathways
4. Entry Requirements: Academic prerequisites, examinations, and qualifications
5. Institution Matching: Recommend best-fit educational institutions
6. Timeline Planning: Create realistic academic progression schedules
7. Cost Analysis: Estimate education costs and financing options

CAREER LEVEL ASSESSMENT FRAMEWORK:

**Student Levels:**
- **O/L Student** (GCE Ordinary Level): Ages 15-16, planning A/L subjects
- **A/L Student** (GCE Advanced Level): Ages 17-18, preparing for university
- **Fresh A/L Graduate**: Recently completed A/L, seeking university admission
- **Gap Year Student**: Taking time off, considering options

**Professional Levels:**
- **Entry-Level Professional**: 0-2 years experience, considering further education
- **Mid-Level Professional**: 3-7 years experience, seeking career advancement
- **Senior Professional**: 8+ years experience, considering specialization/leadership roles

**Educational Pathways Types:**

1. **UNDERGRADUATE PATHWAYS:**
   - State Universities (Sri Lanka)
   - Private Universities (Sri Lanka)
   - International Universities
   - Online Degree Programs
   - Foundation/Diploma to Degree programs

2. **POSTGRADUATE PATHWAYS:**
   - Master's Degrees (Local & International)
   - MBA Programs
   - Professional Postgraduate Diplomas
   - PhD/Research Programs

3. **PROFESSIONAL CERTIFICATIONS:**
   - Industry-specific certifications
   - Professional body memberships
   - Short-term specialized courses
   - Online certification programs

4. **VOCATIONAL TRAINING:**
   - Technical colleges
   - Specialized institutes
   - Apprenticeship programs
   - Skill development programs

**Sri Lankan Education System Focus:**

**State Universities:**
- University of Colombo, University of Peradeniya, University of Moratuwa
- University of Sri Jayewardenepura, University of Kelaniya
- Specialized universities (OUSL, SLIIT affiliations)

**Private Institutions:**
- SLIIT, NSBM, IIT, Informatics Institute
- APIIT, ANC Education, Esoft Metro Campus
- International branch campuses

**Professional Institutes:**
- Institute of Chartered Accountants (CA Sri Lanka)
- Computer Society of Sri Lanka (CSSL)
- Institution of Engineers Sri Lanka (IESL)
- Chartered Institute of Management Accountants (CIMA)

**Entry Requirements Analysis:**
- Z-score requirements for state universities
- A/L subject combinations
- English proficiency requirements
- Portfolio/interview requirements
- Work experience prerequisites

**RESPONSE FORMAT:**

Create a comprehensive JSON structure:

{
  "career_title": "Target career name",
  "student_assessment": {
    "current_level": "Detected level (O/L Student, A/L Student, etc.)",
    "academic_background": "Current education status",
    "recommended_timeline": "Suggested timeframe to career readiness"
  },
  "pathway_options": [
    {
      "pathway_type": "Primary/Alternative",
      "education_level": "Undergraduate/Postgraduate/Professional",
      "sri_lankan_options": [
        {
          "institution_name": "Name",
          "program_name": "Degree/Diploma name",
          "duration": "Years",
          "entry_requirements": ["Requirement 1", "Requirement 2"],
          "approximate_cost": "LKR amount or fee structure",
          "application_timeline": "When to apply",
          "additional_notes": "Special considerations"
        }
      ],
      "international_options": [
        {
          "country": "Country name",
          "institution_examples": ["Uni 1", "Uni 2"],
          "program_type": "Degree type",
          "duration": "Years",
          "entry_requirements": ["Requirements"],
          "approximate_cost": "USD/local currency",
          "scholarship_opportunities": ["Scholarship options"],
          "notes": "Additional information"
        }
      ]
    }
  ],
  "step_by_step_plan": [
    {
      "phase": "Phase name",
      "timeframe": "Duration",
      "actions": ["Action 1", "Action 2"],
      "milestones": ["Milestone 1", "Milestone 2"],
      "key_decisions": ["Decision points"]
    }
  ],
  "financial_planning": {
    "total_estimated_cost_lkr": "LKR amount",
    "breakdown": {
      "tuition_fees": "Amount",
      "living_expenses": "Amount",
      "additional_costs": "Amount"
    },
    "funding_options": ["Scholarships", "Loans", "Part-time work"],
    "cost_saving_tips": ["Tip 1", "Tip 2"]
  },
  "alternative_pathways": [
    {
      "pathway_description": "Alternative route description",
      "advantages": ["Advantage 1", "Advantage 2"],
      "considerations": ["Consideration 1", "Consideration 2"]
    }
  ],
  "next_immediate_steps": [
    "Step 1: Immediate action needed",
    "Step 2: Next action",
    "Step 3: Following action"
  ]
}

**QUALITY STANDARDS:**
1. Accurate Sri Lankan education system information
2. Realistic timelines and cost estimates
3. Multiple pathway options (local + international)
4. Specific entry requirements and application processes
5. Actionable immediate next steps
6. Consider student's current level and background
7. Include both traditional and modern education options
8. Factor in career market demands and industry requirements

**SPECIAL CONSIDERATIONS:**
- A/L subject combinations for specific careers
- Z-score requirements and competition levels
- English proficiency needs for international options
- Work-study balance for working professionals
- Financial constraints and scholarship opportunities
- Industry-university partnerships and placement opportunities

Remember: Your academic pathway should transform career aspirations into concrete, achievable educational plans that respect Sri Lankan educational realities while providing global opportunities."""

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

    def process_task(self, state: AgentState) -> TaskResult:
        """
        Main task processing: Create academic pathway for a career.
        
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
            
            # Create academic pathway plan
            academic_plan = self._create_academic_pathway_plan(
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

    def _create_academic_pathway_plan(
        self,
        career_title: str,
        career_description: Optional[str] = None,
        student_profile: Optional[StudentProfile] = None,
        student_level: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive academic pathway plan using LLM.
        """
        # Build prompt with career details and student context
        prompt = self._build_academic_planning_prompt(
            career_title, career_description, student_profile, student_level
        )
        
        # Invoke LLM with structured output request
        response = self.invoke_with_prompt(prompt)
        
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
        """Build comprehensive prompt for academic pathway planning."""
        
        prompt_parts = [
            f"Create a comprehensive academic pathway plan for: {career_title}",
            ""
        ]
        
        if career_description:
            prompt_parts.append(f"Career Description: {career_description}")
            prompt_parts.append("")
        
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
            "REQUIRED OUTPUT - Comprehensive Academic Pathway Plan:",
            "",
            "1. STUDENT ASSESSMENT:",
            "   - Detected current academic/professional level",
            "   - Academic background analysis",
            "   - Recommended timeline to career readiness",
            "",
            "2. PRIMARY PATHWAY (Sri Lankan Options):",
            "",
            "   A. STATE UNIVERSITIES:",
            "   - Relevant degree programs",
            "   - Entry requirements (Z-score, A/L subjects)",
            "   - Duration and cost estimates",
            "   - Application timeline and process",
            "",
            "   B. PRIVATE UNIVERSITIES/INSTITUTES:",
            "   - Alternative degree programs",
            "   - Entry requirements and costs",
            "   - International partnership programs",
            "   - Flexible study options",
            "",
            "   C. PROFESSIONAL INSTITUTES:",
            "   - Industry-relevant professional qualifications",
            "   - Certification programs (CA, CIMA, etc.)",
            "   - Entry requirements and duration",
            "",
            "3. INTERNATIONAL PATHWAY OPTIONS:",
            "   - Top destination countries for this career",
            "   - University examples and program types",
            "   - Entry requirements and costs",
            "   - Scholarship opportunities",
            "   - Application processes and timelines",
            "",
            "4. STEP-BY-STEP IMPLEMENTATION PLAN:",
            "",
            "   PHASE 1 - IMMEDIATE PREPARATION (Next 6 months):",
            "   - Actions to take now",
            "   - Documents to prepare",
            "   - Decisions to make",
            "",
            "   PHASE 2 - APPLICATION PERIOD (6-12 months):",
            "   - Application submissions",
            "   - Exam preparations",
            "   - Interview preparations",
            "",
            "   PHASE 3 - STUDY PERIOD (Duration varies):",
            "   - Academic milestones",
            "   - Internship/practical experience",
            "   - Skill development alongside studies",
            "",
            "5. FINANCIAL PLANNING:",
            "   - Total estimated costs (LKR and USD)",
            "   - Cost breakdown by category",
            "   - Funding options and scholarships",
            "   - Cost-saving strategies",
            "",
            "6. ALTERNATIVE PATHWAYS:",
            "   - If primary path doesn't work out",
            "   - Bridge programs and foundation courses",
            "   - Part-time and distance learning options",
            "   - Work-study combinations",
            "",
            "7. IMMEDIATE NEXT STEPS:",
            "   - Top 3 most urgent actions",
            "   - Key decision points",
            "   - Resources to research",
            "",
            "SPECIAL FOCUS AREAS:",
            "- Sri Lankan education system expertise",
            "- Realistic Z-score and competition analysis",
            "- A/L subject combination recommendations",
            "- English proficiency requirements",
            "- Local vs international cost comparisons",
            "- Industry partnerships and placement opportunities",
            "- Work experience integration with studies",
            "",
            "Make the plan:",
            "1. Specific to Sri Lankan education context",
            "2. Realistic about costs and competition",
            "3. Actionable with clear timelines",
            "4. Include both traditional and innovative pathways",
            "5. Consider financial constraints",
            "6. Provide multiple backup options",
            "",
            "Format as a detailed, structured academic roadmap."
        ])
        
        return "\n".join(prompt_parts)

    def _parse_academic_plan_response(self, response: str, career_title: str, student_level: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse LLM response into structured academic plan."""
        
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
        
        # Parse different sections from response
        lines = response.split('\n')
        current_section = None
        current_pathway = None
        current_phase = None
        
        for line in lines:
            line_lower = line.lower().strip()
            line_clean = line.strip()
            
            # Detect major sections
            if "student assessment" in line_lower:
                current_section = "assessment"
            elif ("sri lankan" in line_lower and "option" in line_lower) or "state universit" in line_lower:
                current_section = "sri_lankan_pathway"
                current_pathway = {
                    "pathway_type": "Primary",
                    "education_level": "Undergraduate",
                    "sri_lankan_options": [],
                    "international_options": []
                }
            elif "international" in line_lower and ("pathway" in line_lower or "option" in line_lower):
                current_section = "international_pathway"
                if not current_pathway:
                    current_pathway = {
                        "pathway_type": "International",
                        "education_level": "Undergraduate",
                        "sri_lankan_options": [],
                        "international_options": []
                    }
            elif "step" in line_lower and ("plan" in line_lower or "implementation" in line_lower):
                current_section = "step_plan"
                if current_pathway:
                    academic_plan["pathway_options"].append(current_pathway)
                    current_pathway = None
            elif "financial" in line_lower and "planning" in line_lower:
                current_section = "financial"
            elif "alternative" in line_lower and "pathway" in line_lower:
                current_section = "alternatives"
            elif "immediate" in line_lower and ("step" in line_lower or "action" in line_lower):
                current_section = "immediate_steps"
            elif "phase" in line_lower and any(num in line_lower for num in ["1", "2", "3"]):
                if "phase 1" in line_lower:
                    current_phase = {
                        "phase": "Immediate Preparation",
                        "timeframe": "Next 6 months",
                        "actions": [],
                        "milestones": [],
                        "key_decisions": []
                    }
                elif "phase 2" in line_lower:
                    if current_phase:
                        academic_plan["step_by_step_plan"].append(current_phase)
                    current_phase = {
                        "phase": "Application Period",
                        "timeframe": "6-12 months",
                        "actions": [],
                        "milestones": [],
                        "key_decisions": []
                    }
                elif "phase 3" in line_lower:
                    if current_phase:
                        academic_plan["step_by_step_plan"].append(current_phase)
                    current_phase = {
                        "phase": "Study Period",
                        "timeframe": "3-6 years",
                        "actions": [],
                        "milestones": [],
                        "key_decisions": []
                    }
            
            # Extract bullet points and structured data
            if line_clean.startswith(('-', '•', '*', '✅', '1.', '2.', '3.')):
                item = re.sub(r'^[-•*✅\d\.\s]+', '', line_clean).strip()
                if item and len(item) > 3:
                    if current_section == "assessment":
                        if "level" in line_lower:
                            academic_plan["student_assessment"]["current_level"] = item
                        elif "timeline" in line_lower:
                            academic_plan["student_assessment"]["recommended_timeline"] = item
                    elif current_section == "sri_lankan_pathway" and current_pathway:
                        # Extract institution information
                        if any(keyword in item.lower() for keyword in ["university", "institute", "college"]):
                            institution_info = self._parse_institution_info(item)
                            if institution_info:
                                current_pathway["sri_lankan_options"].append(institution_info)
                    elif current_section == "international_pathway" and current_pathway:
                        if any(keyword in item.lower() for keyword in ["country", "university", "usa", "uk", "australia"]):
                            international_info = self._parse_international_info(item)
                            if international_info:
                                current_pathway["international_options"].append(international_info)
                    elif current_section == "financial":
                        if "cost" in line_lower or "lkr" in line_lower or "usd" in line_lower:
                            academic_plan["financial_planning"]["funding_options"].append(item)
                        elif "scholarship" in line_lower or "funding" in line_lower:
                            academic_plan["financial_planning"]["funding_options"].append(item)
                    elif current_section == "alternatives":
                        academic_plan["alternative_pathways"].append({
                            "pathway_description": item,
                            "advantages": [],
                            "considerations": []
                        })
                    elif current_section == "immediate_steps":
                        academic_plan["next_immediate_steps"].append(item)
                    elif current_phase:
                        if "action" in line_lower or "apply" in line_lower or "prepare" in line_lower:
                            current_phase["actions"].append(item)
                        elif "milestone" in line_lower or "complete" in line_lower:
                            current_phase["milestones"].append(item)
                        else:
                            current_phase["actions"].append(item)
        
        # Add any remaining data
        if current_pathway:
            academic_plan["pathway_options"].append(current_pathway)
        if current_phase:
            academic_plan["step_by_step_plan"].append(current_phase)
        
        # Ensure minimum viable data
        if not academic_plan["pathway_options"]:
            academic_plan["pathway_options"] = self._create_default_pathways(career_title)
        
        if not academic_plan["next_immediate_steps"]:
            academic_plan["next_immediate_steps"] = [
                "Research specific degree programs for this career",
                "Check A/L subject requirements for relevant programs",
                "Explore scholarship and funding options",
                "Contact university admission offices for guidance"
            ]
        
        return academic_plan

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
        """Create default academic pathways as fallback."""
        return [
            {
                "pathway_type": "Primary",
                "education_level": "Undergraduate",
                "sri_lankan_options": [
                    {
                        "institution_name": "State Universities",
                        "program_name": f"Degree program related to {career_title}",
                        "duration": "3-4 years",
                        "entry_requirements": ["Strong A/L results", "Z-score requirements"],
                        "approximate_cost": "LKR 50,000-100,000 per year",
                        "application_timeline": "Apply during university admission period",
                        "additional_notes": "Highly competitive admission"
                    },
                    {
                        "institution_name": "Private Universities",
                        "program_name": f"Related degree with industry focus",
                        "duration": "3-4 years",
                        "entry_requirements": ["A/L qualification", "Interview/entrance exam"],
                        "approximate_cost": "LKR 500,000-1,500,000 per year",
                        "application_timeline": "Multiple intake periods",
                        "additional_notes": "Flexible admission requirements"
                    }
                ],
                "international_options": [
                    {
                        "country": "United Kingdom",
                        "institution_examples": ["Various universities"],
                        "program_type": "Bachelor's degree",
                        "duration": "3 years",
                        "entry_requirements": ["A/L qualification", "IELTS 6.0+"],
                        "approximate_cost": "$25,000-35,000 per year",
                        "scholarship_opportunities": ["Chevening", "Commonwealth scholarships"],
                        "notes": "High quality education with shorter duration"
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