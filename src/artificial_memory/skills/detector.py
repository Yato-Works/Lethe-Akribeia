"""Skill Detector - Query Analysis for Skill Routing.

Provides lightweight, deterministic detection of when specialized skills
should be invoked based on query patterns, categories, and content analysis.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from artificial_memory.core.ir.structured import StructuredIR
from .base import MemorySkill, SkillRegistry


# ============================================================
# LoCoMo Category Keywords (for fast category detection)
# ============================================================
LOCOMO_CATEGORY_KEYWORDS = {
    1: [  # Multi-hop
        "both", "each", "every", "all of", "combine", "together", "and also",
        "in what ways", "what ways", "how many", "list", "which", "who",
        "participate", "attend", "both of", "and", "or", "multiple",
        "different", "various", "several", "between", "across", "sessions",
        "speakers", "both", "either", "each person", "everyone"
    ],
    2: [  # Temporal
        "when", "what date", "what day", "what month", "what year", "what time",
        "how long", "how many days", "how many weeks", "how many months",
        "how many years", "before", "after", "between", "since", "ago",
        "last", "next", "this", "yesterday", "tomorrow", "weekend", "week",
        "month", "year", "day", "friday", "monday", "tuesday", "wednesday",
        "thursday", "saturday", "sunday", "january", "february", "march",
        "april", "may", "june", "july", "august", "september", "october",
        "november", "december", "order", "first", "second", "third", "last",
        "earliest", "latest", "earlier", "later", "before", "after",
        "passed since", "have passed", "had passed", "ago did", "weeks ago",
        "months ago", "years ago", "how old", "order from", "chronological",
        "who graduated", "who first", "who second", "who third"
    ],
    3: [  # Open-domain
        "would", "likely", "probably", "prefer", "interested", "enjoy",
        "personality", "trait", "character", "political", "religious",
        "member of", "ally", "support", "oppose", "collect", "collecting",
        "fan of", "fan", "like", "dislike", "hate", "love", "enjoy",
        "would she", "would he", "would they", "national park",
        "dr. seuss", "four seasons", "lgbtq", "lgbt"
    ],
    4: [  # Single-hop
        "what", "who", "where", "which", "how many", "how much",
        "what color", "what kind", "what type", "what brand",
        "what did", "who did", "when did", "where did",
        "why did", "how did", "what was", "who was",
        "what is", "who is", "where is", "how is",
        "directly", "exactly", "specific", "concisely"
    ],
    5: [  # Adversarial
        "never", "didn't", "did not", "never happened", "false",
        "didn't say", "did not say", "not mentioned", "not in",
        "wrong", "incorrect", "false premise", "bait", "trap"
    ],
}

# LongMemEval Question Type Keywords
LME_TYPE_KEYWORDS = {
    "temporal-reasoning": [
        "when", "what date", "how long", "how many days", "how many weeks",
        "how many months", "order", "first", "last", "earliest", "latest",
        "before", "after", "between", "passed since", "ago", "chronological"
    ],
    "multi-session": [
        "total", "sum", "across", "all sessions", "both sessions",
        "each session", "how many times", "count", "aggregate",
        "difference", "more", "less", "faster", "slower"
    ],
    "knowledge-update": [
        "current", "latest", "now", "updated", "changed", "previous",
        "before", "earlier", "former", "initially", "originally"
    ],
    "single-session-user": [
        "what did i", "what was my", "what is my", "what was the",
        "i said", "i told", "i mentioned", "i asked", "i bought",
        "i went", "i visited", "i ate", "i read", "i watched"
    ],
    "single-session-assistant": [
        "what did you", "what was the assistant", "assistant said",
        "assistant told", "assistant mentioned", "assistant recommended",
        "assistant suggested", "assistant provided", "assistant gave"
    ],
    "single-session-preference": [
        "recommend", "suggest", "prefer", "like", "enjoy", "favorite",
        "based on", "tailor", "personalized", "my preference"
    ],
    "abstention": [
        "not enough", "not mentioned", "cannot determine", "unknown",
        "not provided", "not specified", "not stated"
    ],
    "counting": [
        "how many", "count", "list all", "enumerate", "total number",
        "number of", "distinct", "unique", "how many times", "aggregate", "sum"
    ],
}


class SkillDetector:
    """Lightweight deterministic detector for skill routing.
    
    Analyzes queries using keyword matching, regex patterns, and category
    information to determine which skills should be invoked.
    """
    
    def __init__(self, registry: Optional["SkillRegistry"] = None):
        self.registry = registry
        # Compile LoCoMo category patterns for fast matching
        self._cat_patterns = {}
        for cat, keywords in LOCOMO_CATEGORY_KEYWORDS.items():
            pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
            self._cat_patterns[cat] = re.compile(pattern, re.IGNORECASE)
        
        # Compile LME type patterns
        self._lme_patterns = {}
        for qtype, keywords in LME_TYPE_KEYWORDS.items():
            pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
            self._lme_patterns[qtype] = re.compile(pattern, re.IGNORECASE)
    
    def detect_category(self, question: str) -> list[int]:
        """Detect LoCoMo categories from question text.
        
        Returns list of matching category IDs (1-5), sorted by match strength.
        """
        q_lower = question.lower()
        scores = {}
        for cat, pattern in self._cat_patterns.items():
            matches = len(pattern.findall(q_lower))
            if matches > 0:
                scores[cat] = matches
        return sorted(scores.keys(), key=lambda c: -scores[c])
    
    def detect_lme_type(self, question: str) -> list[str]:
        """Detect LongMemEval question types from question text."""
        q_lower = question.lower()
        scores = {}
        for qtype, pattern in self._lme_patterns.items():
            matches = len(pattern.findall(q_lower))
            if matches > 0:
                scores[qtype] = matches
        return sorted(scores.keys(), key=lambda t: -scores[t])
    
    def is_temporal_query(self, question: str) -> bool:
        """Quick check if question is temporal in nature."""
        return bool(self._cat_patterns[2].search(question.lower()))
    
    def is_multi_hop_query(self, question: str) -> bool:
        """Quick check if question requires multi-hop reasoning."""
        return bool(self._cat_patterns[1].search(question.lower()))
    
    def is_abstention_query(self, question: str) -> bool:
        """Quick check if question is likely unanswerable (abstention)."""
        return bool(self._cat_patterns[5].search(question.lower()) or
                   self._lme_patterns["abstention"].search(question.lower()))
    
    def find_applicable_skills(self, question: str, records: Sequence[StructuredIR],
                               category: Optional[int] = None,
                               question_type: Optional[str] = None,
                               registry: Optional["SkillRegistry"] = None) -> list[MemorySkill]:
        """Find all skills applicable to the query."""
        if registry is None:
            from .base import get_global_registry
            registry = get_global_registry()
        
        # Use registry's built-in logic
        skills = registry.find_skills(question, records, category, question_type)
        
        # Add auto-detection based on question content
        if category is None:
            detected_cats = self.detect_category(question)
            if detected_cats:
                # Re-run with first detected category
                skills = registry.find_skills(question, records, detected_cats[0], question_type)
        
        if question_type is None:
            detected_types = self.detect_lme_type(question)
            if detected_types:
                skills = registry.find_skills(question, records, category, detected_types[0])
        
        return skills


# Global detector instance
_detector_instance: Optional[SkillDetector] = None


def get_skill_detector(registry: Optional["SkillRegistry"] = None) -> SkillDetector:
    """Get or create the global SkillDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = SkillDetector(registry)
    return _detector_instance


# Convenience functions
def detect_category(question: str) -> list[int]:
    """Quick category detection."""
    return get_skill_detector().detect_category(question)


def detect_lme_type(question: str) -> list[str]:
    """Quick LME type detection."""
    return get_skill_detector().detect_lme_type(question)


def is_temporal_query(question: str) -> bool:
    """Quick temporal query check."""
    return get_skill_detector().is_temporal_query(question)


def is_multi_hop_query(question: str) -> bool:
    """Quick multi-hop query check."""
    return get_skill_detector().is_multi_hop_query(question)


def find_skills_for_query(question: str, records: Sequence[StructuredIR],
                          category: Optional[int] = None,
                          question_type: Optional[str] = None) -> list[MemorySkill]:
    """Find all applicable skills for a query."""
    return get_skill_detector().find_applicable_skills(question, records, category, question_type)


__all__ = [
    "SkillDetector",
    "LOCOMO_CATEGORY_KEYWORDS",
    "LME_TYPE_KEYWORDS",
    "get_skill_detector",
    "detect_category",
    "detect_lme_type",
    "is_temporal_query",
    "is_multi_hop_query",
    "is_abstention_query",
    "find_skills_for_query",
]