"""
Personality Analyzer Module

Analyzes user response patterns to infer personality traits based on
communication style, thinking patterns, and emotional expression.
Uses rule-based heuristics for efficient, cost-effective analysis.
"""

from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass, field


class PersonalityDimension(Enum):
    """
    Five-dimension personality model for career profiling.
    Each dimension represents a spectrum of traits.
    """
    # Communication Style
    EXPRESSIVE = "expressive"  # Long, detailed responses
    RESERVED = "reserved"  # Short, concise responses

    # Thinking Style
    ANALYTICAL = "analytical"  # Logical, structured reasoning
    CREATIVE = "creative"  # Imaginative, unconventional thinking

    # Detail Orientation
    DETAIL_ORIENTED = "detail_oriented"  # Mentions specifics, examples
    BIG_PICTURE = "big_picture"  # Focuses on concepts, themes

    # Emotional Expression
    EMOTIONALLY_EXPRESSIVE = "emotionally_expressive"  # Uses emotion words, exclamations
    EMOTIONALLY_NEUTRAL = "emotionally_neutral"  # Factual, neutral tone

    # Decision Making
    DECISIVE = "decisive"  # Clear preferences, confident statements
    EXPLORATORY = "exploratory"  # Open-ended, considering options


class PersonalityAnalyzer:
    """
    Analyzes response patterns to infer personality traits.
    Tracks cumulative evidence across all responses.

    Uses rule-based heuristics:
    - Word count for communication style
    - Keyword matching for thinking style
    - Example usage for detail orientation
    - Emotion words for emotional expression
    - Certainty language for decision making
    """

    def __init__(self):
        """Initialize analyzer with zero trait scores."""
        self.trait_scores: Dict[PersonalityDimension, float] = {
            trait: 0.0 for trait in PersonalityDimension
        }
        self.evidence_count: int = 0

    def analyze_response(self, user_response: str) -> Dict[PersonalityDimension, float]:
        """
        Analyze a single response for personality signals.

        Args:
            user_response: The user's text response

        Returns:
            Dict mapping PersonalityDimension to incremental scores (0-1)
        """
        analysis = {}
        response_lower = user_response.lower()
        word_count = len(user_response.split())

        # 1. Communication Style (Response Length)
        if word_count > 40:
            analysis[PersonalityDimension.EXPRESSIVE] = 0.3
        elif word_count < 15:
            analysis[PersonalityDimension.RESERVED] = 0.3

        # 2. Thinking Style (Keywords)
        analytical_keywords = ["because", "therefore", "analyze", "consider", "think",
                              "reason", "logical", "evaluate", "assess"]
        creative_keywords = ["creative", "imagine", "design", "innovative", "unique",
                           "artistic", "original", "inventive"]

        analytical_score = sum(1 for kw in analytical_keywords if kw in response_lower)
        creative_score = sum(1 for kw in creative_keywords if kw in response_lower)

        if analytical_score > creative_score and analytical_score > 0:
            analysis[PersonalityDimension.ANALYTICAL] = min(analytical_score * 0.15, 0.4)
        elif creative_score > analytical_score and creative_score > 0:
            analysis[PersonalityDimension.CREATIVE] = min(creative_score * 0.15, 0.4)

        # 3. Detail Orientation (Examples, Specifics)
        detail_indicators = ["for example", "specifically", "such as", "like when",
                           "including", "for instance", "in particular"]
        has_examples = any(indicator in response_lower for indicator in detail_indicators)

        if has_examples:
            analysis[PersonalityDimension.DETAIL_ORIENTED] = 0.3
        elif word_count > 30 and not has_examples:
            # Long responses without examples suggest big-picture thinking
            analysis[PersonalityDimension.BIG_PICTURE] = 0.2

        # 4. Emotional Expression (Emotion words, punctuation)
        emotion_words = ["love", "enjoy", "excited", "passionate", "hate", "dislike",
                        "frustrated", "happy", "sad", "angry", "thrilled", "delighted"]
        exclamation_count = user_response.count("!")

        emotion_score = sum(1 for word in emotion_words if word in response_lower)
        emotion_score += exclamation_count * 0.5

        if emotion_score >= 2:
            analysis[PersonalityDimension.EMOTIONALLY_EXPRESSIVE] = min(emotion_score * 0.15, 0.4)
        elif emotion_score == 0 and word_count > 20:
            # Long factual responses without emotion
            analysis[PersonalityDimension.EMOTIONALLY_NEUTRAL] = 0.2

        # 5. Decision Making (Certainty language)
        decisive_markers = ["definitely", "absolutely", "certainly", "i know", "for sure",
                          "without a doubt", "clearly", "obviously"]
        exploratory_markers = ["maybe", "possibly", "considering", "exploring", "not sure",
                             "might", "could", "perhaps", "thinking about"]

        decisive_count = sum(1 for marker in decisive_markers if marker in response_lower)
        exploratory_count = sum(1 for marker in exploratory_markers if marker in response_lower)

        if decisive_count > exploratory_count and decisive_count > 0:
            analysis[PersonalityDimension.DECISIVE] = min(decisive_count * 0.2, 0.4)
        elif exploratory_count > decisive_count and exploratory_count > 0:
            analysis[PersonalityDimension.EXPLORATORY] = min(exploratory_count * 0.2, 0.4)

        return analysis

    def update_cumulative_scores(self, incremental_analysis: Dict[PersonalityDimension, float]):
        """
        Update cumulative trait scores with incremental evidence.
        Uses weighted averaging to stabilize over time.

        Args:
            incremental_analysis: New trait scores from current response
        """
        self.evidence_count += 1
        # Decreasing weight for new evidence as we gather more data
        weight = min(0.3, 1.0 / self.evidence_count)

        for trait, score in incremental_analysis.items():
            current = self.trait_scores[trait]
            # Weighted update: mostly preserve existing, add new evidence
            self.trait_scores[trait] = (current * (1 - weight)) + (score * weight)

    def get_dominant_traits(self, threshold: float = 0.3) -> List[str]:
        """
        Get personality traits that have strong evidence (score >= threshold).

        Args:
            threshold: Minimum score for trait inclusion (0-1)

        Returns:
            List of trait names (strings) with scores above threshold
        """
        return [
            trait.value
            for trait, score in self.trait_scores.items()
            if score >= threshold
        ]

    def get_confidence_score(self) -> float:
        """
        Get overall confidence in personality assessment.

        Returns:
            Confidence score (0-1) based on evidence count and trait strengths
        """
        if self.evidence_count == 0:
            return 0.0

        # Confidence increases with evidence, maxing at 10 responses
        evidence_confidence = min(self.evidence_count / 10.0, 1.0)

        # Confidence also based on trait score strengths
        max_trait_score = max(self.trait_scores.values()) if self.trait_scores else 0.0

        # Combined confidence
        return (evidence_confidence * 0.6) + (max_trait_score * 0.4)

    def get_trait_summary(self) -> str:
        """
        Get a human-readable summary of detected personality traits.

        Returns:
            String summarizing dominant traits
        """
        dominant = self.get_dominant_traits(threshold=0.25)

        if not dominant:
            return "Personality profile developing..."

        # Group by dimension for better readability
        communication = [t for t in dominant if t in ["expressive", "reserved"]]
        thinking = [t for t in dominant if t in ["analytical", "creative"]]
        detail = [t for t in dominant if t in ["detail_oriented", "big_picture"]]
        emotion = [t for t in dominant if t in ["emotionally_expressive", "emotionally_neutral"]]
        decision = [t for t in dominant if t in ["decisive", "exploratory"]]

        summary_parts = []
        if communication:
            summary_parts.append(f"Communication: {communication[0].replace('_', ' ').title()}")
        if thinking:
            summary_parts.append(f"Thinking: {thinking[0].replace('_', ' ').title()}")
        if detail:
            summary_parts.append(f"Detail Focus: {detail[0].replace('_', ' ').title()}")
        if emotion:
            summary_parts.append(f"Expression: {emotion[0].replace('_', ' ').title()}")
        if decision:
            summary_parts.append(f"Decision Style: {decision[0].replace('_', ' ').title()}")

        return " | ".join(summary_parts)
