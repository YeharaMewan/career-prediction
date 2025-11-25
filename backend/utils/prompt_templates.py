"""
Optimized Prompt Templates System for Career Planning Multi-Agent System

This module provides centralized, token-optimized prompt templates to reduce
redundancy and improve maintainability across all agents.

Token Reduction Strategy:
- Replace verbose role descriptions with concise templates
- Centralize common instructions (quality standards, formatting)
- Use structured system+user message format instead of long prompts
- Eliminate repetitive examples and redundant context

Expected Savings: ~40% token reduction (2,835 tokens) across all prompts
"""

from typing import Dict, List, Optional, Any
import json


# =============================================================================
# SYSTEM ROLES - Concise agent role definitions
# =============================================================================

SYSTEM_ROLES = {
    "academic_pathway": {
        "role": "Academic pathway specialist for Sri Lankan students",
        "expertise": "Educational roadmaps, universities, degrees, scholarships",
        "focus": "Sri Lankan + international options, entry requirements, timelines"
    },

    "skill_development": {
        "role": "Skill development specialist for career preparation",
        "expertise": "Technical/soft skills, online courses, certifications, learning paths",
        "focus": "Progressive roadmaps (Foundation → Intermediate → Advanced)"
    },

    "career_counselor": {
        "role": "Career counselor for high school students",
        "expertise": "Career assessment, RIASEC framework, student guidance",
        "focus": "Simple language, encouragement, personalized questions"
    },

    "response_analyst": {
        "role": "Career response analyst",
        "expertise": "RIASEC scoring, intent classification, pattern recognition",
        "focus": "Accurate scoring, structured JSON output"
    },

    "career_predictor": {
        "role": "Career prediction specialist",
        "expertise": "Career matching, profile analysis, multi-career recommendations",
        "focus": "RIASEC-based predictions, match scoring, reasoning"
    },

    "main_supervisor": {
        "role": "Career planning system orchestrator",
        "expertise": "Agent coordination, workflow management, session control",
        "focus": "Efficient delegation, quality assurance, user experience"
    },

    "career_planning_supervisor": {
        "role": "Academic + skill planning coordinator",
        "expertise": "Parallel agent orchestration, comprehensive career plans",
        "focus": "Coordinated academic and skill development pathways"
    }
}


# =============================================================================
# OUTPUT SCHEMAS - Structured JSON templates
# =============================================================================

OUTPUT_SCHEMAS = {
    "academic_plan": {
        "type": "object",
        "required_fields": [
            "education_level_required",
            "primary_pathway",
            "alternative_pathways",
            "institutions",
            "timeline",
            "estimated_cost"
        ],
        "example": {
            "education_level_required": "Bachelor's Degree",
            "primary_pathway": {"degree": "...", "institution": "...", "duration": "..."},
            "institutions": [{"name": "...", "type": "...", "location": "...", "url": "..."}],
            "timeline": "4 years",
            "estimated_cost": {"local": "...", "international": "..."}
        }
    },

    "skill_plan": {
        "type": "object",
        "required_fields": [
            "technical_skills",
            "soft_skills",
            "learning_roadmap",
            "certifications",
            "resources"
        ],
        "example": {
            "technical_skills": {"core": ["..."], "advanced": ["..."]},
            "learning_roadmap": [
                {"phase": "Foundation", "duration": "...", "skills": ["..."]}
            ],
            "certifications": [{"name": "...", "provider": "...", "priority": "..."}],
            "resources": [{"platform": "...", "course": "...", "url": "..."}]
        }
    },

    "riasec_scores": {
        "type": "object",
        "required_fields": ["R", "I", "A", "S", "E", "C"],
        "format": {"R": 0.8, "I": 0.6, "A": 0.3, "S": 0.5, "E": 0.4, "C": 0.7},
        "note": "Scores between 0.0-1.0"
    },

    "career_prediction": {
        "type": "array",
        "items": {
            "career_title": "string",
            "match_score": "float (0-100)",
            "match_reasoning": "string",
            "riasec_alignment": "object"
        },
        "example": [{
            "career_title": "Software Engineer",
            "match_score": 88.5,
            "match_reasoning": "Strong technical skills, analytical thinking",
            "riasec_alignment": {"R": 0.5, "I": 0.9, "A": 0.3}
        }]
    }
}


# =============================================================================
# QUALITY STANDARDS - Shared quality guidelines (used across all agents)
# =============================================================================

QUALITY_STANDARDS = """Quality Requirements:
- Use provided web search results (include URLs)
- Provide specific, actionable recommendations
- Include realistic timelines and cost estimates
- Consider Sri Lankan context when relevant
- Output structured JSON matching schema"""


# =============================================================================
# FORMATTING RULES - Common output formatting instructions
# =============================================================================

FORMATTING_RULES = {
    "json_output": "Return valid JSON only. No markdown, no code blocks.",
    "url_inclusion": "Include URLs for all institutions, courses, resources.",
    "sri_lankan_focus": "Prioritize Sri Lankan options, provide international alternatives.",
    "web_search_usage": "Use real-time web search results provided in context."
}


# =============================================================================
# LANGUAGE SUPPORT - Multi-language prompt generation
# =============================================================================

def get_language_instruction(language: str, agent_role: str = "general") -> str:
    """
    Get language-specific instruction for LLM prompts.

    Args:
        language: "en" (English) or "si" (Sinhala)
        agent_role: Agent type for specialized instructions (general, career_counselor,
                   academic_pathway, skill_development, response_analyst, career_predictor)

    Returns:
        Language instruction string to prepend to prompts
    """
    if language == "si":
        instructions = {
            "general": """LANGUAGE: Sinhala (සිංහල)
Generate ALL content in natural, conversational Sinhala.
Use Sinhala script for all text.""",

            "career_counselor": """LANGUAGE: Sinhala (සිංහල)
Generate questions and responses in natural Sinhala.
Use simple language suitable for high school students.
Write entirely in Sinhala script (සිංහල අකුරු).
Be warm, encouraging, and conversational.""",

            "academic_pathway": """LANGUAGE: Sinhala (සිංහල)
Provide academic plans with Sinhala descriptions.
JSON keys remain in English, but all text values must be in Sinhala.
Transliterate institution names to Sinhala when appropriate.
Example: {"degree": "තොරතුරු තාක්ෂණ උපාධිය", "duration": "අවුරුදු 4"}""",

            "skill_development": """LANGUAGE: Sinhala (සිංහල)
Provide skill plans with Sinhala descriptions.
JSON keys remain in English, but all text values must be in Sinhala.
Course names can be transliterated or kept in English if widely known.
Example: {"skill": "Python Programming", "description": "ක්‍රමලේඛන භාෂාව"}""",

            "response_analyst": """LANGUAGE: Sinhala (සිංහල)
JSON keys remain in English, but "intent" field should be in Sinhala.
Example: {"scores": {...}, "intent": "තාක්ෂණය සහ ප්‍රශ්න විසඳීම ගැන උනන්දුවක්"}""",

            "career_predictor": """LANGUAGE: Sinhala (සිංහල)
Career titles can remain in English or be translated (e.g., "Software Engineer" or "මෘදුකාංග ඉංජිනේරු").
All descriptions and reasoning must be in natural Sinhala.
JSON keys remain in English."""
        }
        return instructions.get(agent_role, instructions["general"])

    # Default: English (no special instruction needed)
    return "Language: English. Provide all content in English."


# =============================================================================
# TEMPLATE BUILDER FUNCTIONS
# =============================================================================

def build_system_prompt(role_key: str, additional_context: Optional[str] = None) -> str:
    """
    Build optimized system prompt from role definition.

    Args:
        role_key: Key from SYSTEM_ROLES dict
        additional_context: Optional extra context

    Returns:
        Concise system prompt string
    """
    if role_key not in SYSTEM_ROLES:
        raise ValueError(f"Unknown role: {role_key}")

    role_def = SYSTEM_ROLES[role_key]

    prompt_parts = [
        f"Role: {role_def['role']}",
        f"Expertise: {role_def['expertise']}",
        f"Focus: {role_def['focus']}"
    ]

    if additional_context:
        prompt_parts.append(additional_context)

    return "\n".join(prompt_parts)


def build_task_prompt(
    task_description: str,
    context_data: Dict[str, Any],
    output_schema_key: Optional[str] = None,
    include_quality_standards: bool = True,
    web_search_results: Optional[str] = None,
    language: str = "en",
    agent_role: str = "general"
) -> str:
    """
    Build optimized task prompt with context and output requirements.

    Args:
        task_description: Brief task description
        context_data: Key context variables (student profile, career info, etc.)
        output_schema_key: Key from OUTPUT_SCHEMAS dict
        include_quality_standards: Whether to include quality guidelines
        web_search_results: Optional web search results to include
        language: Response language ("en" or "si")
        agent_role: Agent role for language-specific instructions

    Returns:
        Optimized task prompt string
    """
    prompt_parts = []

    # Add language instruction first if not English
    if language != "en":
        prompt_parts.append(get_language_instruction(language, agent_role))
        prompt_parts.append("")

    prompt_parts.append(task_description)
    prompt_parts.append("")

    # Add context
    if context_data:
        prompt_parts.append("Context:")
        for key, value in context_data.items():
            if value:  # Only include non-empty values
                prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append("")

    # Add web search results if provided
    if web_search_results:
        prompt_parts.append("=== REAL-TIME WEB SEARCH RESULTS ===")
        prompt_parts.append(web_search_results)
        prompt_parts.append("")

    # Add output schema if specified
    if output_schema_key and output_schema_key in OUTPUT_SCHEMAS:
        schema = OUTPUT_SCHEMAS[output_schema_key]
        prompt_parts.append(f"Output: JSON with fields: {', '.join(schema['required_fields'])}")
        prompt_parts.append("")

    # Add quality standards if requested
    if include_quality_standards:
        prompt_parts.append(QUALITY_STANDARDS)

    return "\n".join(prompt_parts)


def build_structured_messages(
    role_key: str,
    task_description: str,
    context_data: Dict[str, Any],
    output_schema_key: Optional[str] = None,
    **kwargs
) -> List[Dict[str, str]]:
    """
    Build structured message list (system + user) for LLM invocation.

    This is the recommended approach for optimal token usage.

    Args:
        role_key: Agent role from SYSTEM_ROLES
        task_description: Task description
        context_data: Context variables
        output_schema_key: Output schema key
        **kwargs: Additional args for build_task_prompt

    Returns:
        List of message dicts: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    system_message = build_system_prompt(role_key)
    user_message = build_task_prompt(task_description, context_data, output_schema_key, **kwargs)

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]


# =============================================================================
# SPECIALIZED PROMPTS - Pre-built templates for common operations
# =============================================================================

class PromptTemplates:
    """Collection of specialized prompt templates for common operations"""

    @staticmethod
    def academic_pathway_system() -> str:
        """Optimized system prompt for Academic Pathway Agent"""
        return """Role: Academic pathway specialist for Sri Lankan students
Expertise: Educational roadmaps, universities, degrees, scholarships, entry requirements
Focus: Sri Lankan + international options, realistic timelines, cost estimates

Career Levels: O/L → A/L → Diploma → Bachelor's → Master's → PhD
Education Types: University degrees, technical diplomas, vocational training, online courses
Institutions: Use web search for current Sri Lankan universities, international options

Output: Structured academic plan with pathways, institutions (with URLs), timeline, costs"""

    @staticmethod
    def skill_development_system() -> str:
        """Optimized system prompt for Skill Development Agent"""
        return """Role: Skill development specialist for career preparation
Expertise: Technical/soft skills, online courses, certifications, learning platforms
Focus: Progressive roadmaps (Foundation → Intermediate → Advanced)

Skill Types: Technical (programming, tools, frameworks), Soft (communication, leadership)
Learning Phases: Foundation (basics), Intermediate (application), Advanced (mastery)
Resources: Use web search for current courses (Coursera, Udemy, etc.), certifications

Output: Structured skill plan with learning roadmap, certifications, resources (with URLs)"""

    @staticmethod
    def career_prediction_system() -> str:
        """Optimized system prompt for career prediction"""
        return """Role: Career prediction specialist
Expertise: Career matching, RIASEC analysis, profile assessment
Focus: Top 5 careers with match scores and reasoning

Method: Analyze RIASEC scores + interests + skills + academic level
Output: JSON array of 5 careers with: title, match_score (0-100), reasoning, riasec_alignment"""

    @staticmethod
    def question_generation() -> Dict[str, str]:
        """Template for dynamic question generation"""
        return {
            "system": "Career counselor. Generate ONE simple question for high school student. Clear, encouraging language.",
            "user_template": "Goal: {objective}\nContext: {student_context}\nRIASEC focus: {category}\n\nQuestion:"
        }

    @staticmethod
    def response_analysis() -> Dict[str, str]:
        """Template for response analysis"""
        return {
            "system": "Analyze student response. Score RIASEC categories (0.0-1.0). Return JSON.",
            "user_template": "Question: {question}\nCategory: {category}\nResponse: {response}\n\nOutput: {\"scores\": {\"R\": 0.0-1.0, \"I\": 0.0-1.0, ...}, \"intent\": \"...\"}"
        }

    @staticmethod
    def follow_up_question() -> Dict[str, str]:
        """Template for follow-up question generation"""
        return {
            "system": "Generate contextual follow-up question based on previous response. Natural, conversational.",
            "user_template": "Previous: {previous_response}\nTopic: {topic}\n\nFollow-up question:"
        }


# =============================================================================
# TOKEN COUNTING UTILITIES
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text (rough approximation).

    Rule of thumb: ~4 characters per token for English text

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4


def compare_prompts(old_prompt: str, new_prompt: str) -> Dict[str, Any]:
    """
    Compare token usage between old and new prompts.

    Args:
        old_prompt: Original prompt
        new_prompt: Optimized prompt

    Returns:
        Dict with comparison stats
    """
    old_tokens = estimate_tokens(old_prompt)
    new_tokens = estimate_tokens(new_prompt)
    savings = old_tokens - new_tokens
    savings_pct = (savings / old_tokens * 100) if old_tokens > 0 else 0

    return {
        "old_tokens": old_tokens,
        "new_tokens": new_tokens,
        "savings": savings,
        "savings_pct": f"{savings_pct:.1f}%",
        "reduction_factor": f"{old_tokens/new_tokens:.2f}x" if new_tokens > 0 else "N/A"
    }


# =============================================================================
# CONTEXT HELPERS - Smart context injection
# =============================================================================

def format_student_context(student_profile: Any, max_length: int = 200) -> str:
    """
    Format student profile into concise context string.

    Args:
        student_profile: StudentProfile object
        max_length: Maximum character length

    Returns:
        Concise context string
    """
    if not student_profile:
        return "No profile"

    parts = []

    if hasattr(student_profile, 'current_education_level'):
        parts.append(f"Level: {student_profile.current_education_level}")

    if hasattr(student_profile, 'major_field') and student_profile.major_field:
        parts.append(f"Field: {student_profile.major_field}")

    if hasattr(student_profile, 'career_interests') and student_profile.career_interests:
        interests = student_profile.career_interests[:3]  # Top 3
        parts.append(f"Interests: {', '.join(interests)}")

    if hasattr(student_profile, 'technical_skills') and student_profile.technical_skills:
        skills = student_profile.technical_skills[:3]  # Top 3
        parts.append(f"Skills: {', '.join(skills)}")

    context = " | ".join(parts)

    # Truncate if too long
    if len(context) > max_length:
        context = context[:max_length-3] + "..."

    return context


def format_riasec_scores(scores: Dict[str, float], top_n: int = 3) -> str:
    """
    Format RIASEC scores into concise string.

    Args:
        scores: Dict of RIASEC scores
        top_n: Number of top categories to include

    Returns:
        Concise RIASEC string (e.g., "I:0.9, R:0.7, E:0.6")
    """
    if not scores:
        return "No scores"

    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Take top N
    top_scores = sorted_scores[:top_n]

    # Format
    return ", ".join([f"{cat}:{score:.1f}" for cat, score in top_scores])


# =============================================================================
# VALIDATION
# =============================================================================

def validate_schema(data: Dict[str, Any], schema_key: str) -> bool:
    """
    Validate data against output schema.

    Args:
        data: Data to validate
        schema_key: Schema key from OUTPUT_SCHEMAS

    Returns:
        True if valid, False otherwise
    """
    if schema_key not in OUTPUT_SCHEMAS:
        return False

    schema = OUTPUT_SCHEMAS[schema_key]
    required_fields = schema.get("required_fields", [])

    # Check all required fields present
    for field in required_fields:
        if field not in data:
            return False

    return True


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'SYSTEM_ROLES',
    'OUTPUT_SCHEMAS',
    'QUALITY_STANDARDS',
    'FORMATTING_RULES',

    # Builder functions
    'build_system_prompt',
    'build_task_prompt',
    'build_structured_messages',

    # Language support
    'get_language_instruction',

    # Template class
    'PromptTemplates',

    # Utilities
    'estimate_tokens',
    'compare_prompts',
    'format_student_context',
    'format_riasec_scores',
    'validate_schema'
]


# =============================================================================
# USAGE EXAMPLES (for documentation)
# =============================================================================

if __name__ == "__main__":
    print("=== Prompt Templates System ===\n")

    # Example 1: Build system prompt
    print("1. System Prompt Example:")
    sys_prompt = build_system_prompt("academic_pathway")
    print(sys_prompt)
    print(f"Tokens: ~{estimate_tokens(sys_prompt)}\n")

    # Example 2: Build task prompt
    print("2. Task Prompt Example:")
    task_prompt = build_task_prompt(
        "Create academic pathway for Software Engineer",
        context_data={
            "student_level": "A/L Student",
            "interests": "Technology, Programming",
            "budget": "Moderate"
        },
        output_schema_key="academic_plan"
    )
    print(task_prompt[:200] + "...")
    print(f"Tokens: ~{estimate_tokens(task_prompt)}\n")

    # Example 3: Build structured messages
    print("3. Structured Messages Example:")
    messages = build_structured_messages(
        "career_counselor",
        "Generate a question about career interests",
        {"riasec_focus": "Investigative", "previous_answers": "Likes science"}
    )
    print(f"System: {messages[0]['content']}")
    print(f"User: {messages[1]['content'][:100]}...")
    total_tokens = sum(estimate_tokens(m['content']) for m in messages)
    print(f"Total tokens: ~{total_tokens}\n")

    # Example 4: Token comparison
    print("4. Token Comparison Example:")
    old_prompt = """You are an expert Academic Pathway Agent specializing in educational roadmap design for Sri Lankan students pursuing various careers.

YOUR ROLE:
You are a specialist in the Career Planning team, working under the Career Planning Supervisor.
Your specific responsibility is to create detailed, actionable educational pathways for identified careers, with expertise in both Sri Lankan and international education systems.
""" + "..." * 100  # Simulating long prompt

    new_prompt = PromptTemplates.academic_pathway_system()
    comparison = compare_prompts(old_prompt, new_prompt)
    print(f"Old: {comparison['old_tokens']} tokens")
    print(f"New: {comparison['new_tokens']} tokens")
    print(f"Savings: {comparison['savings']} tokens ({comparison['savings_pct']})")
    print(f"Reduction: {comparison['reduction_factor']}")
