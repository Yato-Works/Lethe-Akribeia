"""
Query Analyzer: Deterministic Feature Extractor and Type Router for Lethe-Akribeia V4.

Inspects raw question text ONLY (no metadata, no gold labels) to extract linguistic,
temporal, aggregation, and conversational features, routing the query to the
appropriate Cognitive Compilation pathway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryType(str, Enum):
    FACTUAL_USER = "single-session-user"
    FACTUAL_ASSISTANT = "single-session-assistant"
    TEMPORAL = "temporal-reasoning"
    AGGREGATION = "multi-session"
    STATE_EVOLUTION = "knowledge-update"
    PREFERENCE = "single-session-preference"


@dataclass(frozen=True)
class QueryFeatures:
    """Linguistic and semantic features extracted deterministically from question text."""
    raw_query: str
    normalized_query: str
    
    # Temporal features
    has_relative_time: bool = False       # "how many days/weeks/months ago", "since"
    has_temporal_ordering: bool = False    # "which happened first", "before", "after"
    has_duration: bool = False             # "how long did it take", "how many hours"
    
    # Aggregation features
    has_counting: bool = False             # "how many [plural nouns]", "total number of"
    has_arithmetic_diff: bool = False      # "difference in price", "how much faster", "how much more"
    has_total_sum: bool = False            # "total cost", "in total"
    
    # State evolution features
    has_state_transition: bool = False     # "currently", "now", "previously", "used to", "switched"
    has_recent_vs_past: bool = False       # "latest", "most recent", "former", "initial"
    
    # Preference features
    has_recommendation_request: bool = False  # "can you recommend", "can you suggest", "what should i"
    has_preference_target: bool = False       # "would i prefer", "my taste"
    
    # Assistant recall features
    has_assistant_attribution: bool = False   # "you told me", "you recommended", "in our previous conversation"
    has_list_ordinal: bool = False            # "first item", "second", "last recommendation"


class QueryAnalyzer:
    """Deterministic question feature extractor and router."""

    # Precompiled regex patterns
    _RE_RELATIVE_TIME = re.compile(
        r"\b(how many (days|weeks|months|years) ago|how long ago|since (i|we|my)|passed between|how much time passed)\b",
        re.I
    )
    _RE_TEMPORAL_ORDER = re.compile(
        r"\b(which (item|event|activity|project|group|movie|book|place|trip) did i (do|join|start|finish|visit|buy|purchase|attend|see) first|"
        r"which happened first|order from first to last|earlier or later|before or after|"
        r"which did i .* (first|earlier))\b",
        re.I
    )
    _RE_DURATION = re.compile(
        r"\b(how long (did it take|did i spend|have i been)|how many (hours|minutes|days) did i spend)\b",
        re.I
    )

    _RE_COUNTING = re.compile(
        r"\b(how many (different )?[a-z]+ (did i|have i|do i|are there)|total number of|how many times did i)\b",
        re.I
    )
    _RE_ARITHMETIC_DIFF = re.compile(
        r"\b(difference in (price|cost|time|age|speed|weight)|how much (more|less|faster|slower|cheaper|expensive)|compare the)\b",
        re.I
    )
    _RE_TOTAL_SUM = re.compile(
        r"\b(total (cost|price|amount|time|expense|distance)|in total|combined cost)\b",
        re.I
    )

    _RE_STATE_TRANSITION = re.compile(
        r"\b(currently|right now|at present|previously|used to|no longer|switched to|changed my|updated my|former stance|prior to)\b",
        re.I
    )
    _RE_RECENT_VS_PAST = re.compile(
        r"\b(most recent|latest|former|initially|original|newest)\b",
        re.I
    )

    _RE_RECOMMENDATION = re.compile(
        r"\b(can you (recommend|suggest|give me)|what should i (try|get|buy|visit|watch|read)|"
        r"recommend (some|a|an)|suggest (some|a|an)|ideas for|options for)\b",
        re.I
    )
    _RE_PREFERENCE = re.compile(
        r"\b(would i prefer|align with my (taste|preference|style)|my favorite type of)\b",
        re.I
    )

    _RE_ASSISTANT_ATTRIBUTION = re.compile(
        r"\b(you (told|recommended|suggested|mentioned|shared|advised|gave me)|"
        r"in our (previous|earlier|past) conversation|what was that [a-z]+ you (told|recommended|mentioned))\b",
        re.I
    )
    _RE_LIST_ORDINAL = re.compile(
        r"\b(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final) "
        r"(item|option|recommendation|suggestion|place|movie|tip|step|recipe)\b",
        re.I
    )

    def extract_features(self, query: str) -> QueryFeatures:
        """Extract all deterministic features from question text."""
        norm = re.sub(r"\s+", " ", query).strip()

        return QueryFeatures(
            raw_query=query,
            normalized_query=norm,
            has_relative_time=bool(self._RE_RELATIVE_TIME.search(norm)),
            has_temporal_ordering=bool(self._RE_TEMPORAL_ORDER.search(norm)),
            has_duration=bool(self._RE_DURATION.search(norm)),
            has_counting=bool(self._RE_COUNTING.search(norm)),
            has_arithmetic_diff=bool(self._RE_ARITHMETIC_DIFF.search(norm)),
            has_total_sum=bool(self._RE_TOTAL_SUM.search(norm)),
            has_state_transition=bool(self._RE_STATE_TRANSITION.search(norm)),
            has_recent_vs_past=bool(self._RE_RECENT_VS_PAST.search(norm)),
            has_recommendation_request=bool(self._RE_RECOMMENDATION.search(norm)),
            has_preference_target=bool(self._RE_PREFERENCE.search(norm)),
            has_assistant_attribution=bool(self._RE_ASSISTANT_ATTRIBUTION.search(norm)),
            has_list_ordinal=bool(self._RE_LIST_ORDINAL.search(norm)),
        )

    def route(self, query: str) -> QueryType:
        """Deterministically route a query to its target QueryType."""
        feat = self.extract_features(query)
        q_lower = feat.normalized_query.lower()

        # Priority 1: Assistant recall (explicit mention of assistant's past remarks)
        if feat.has_assistant_attribution or feat.has_list_ordinal:
            return QueryType.FACTUAL_ASSISTANT

        # Priority 2: Preference (explicit request for recommendations / suggestions)
        if feat.has_recommendation_request or feat.has_preference_target:
            return QueryType.PREFERENCE

        # Priority 3: State evolution (knowledge-update: queries about state transitions over time)
        if feat.has_state_transition or feat.has_recent_vs_past:
            # Distinguish from pure temporal difference (e.g. "how many days ago did I switch")
            if not feat.has_relative_time:
                return QueryType.STATE_EVOLUTION

        # Priority 4: Temporal reasoning (dates, durations, relative time, ordering)
        if feat.has_relative_time or feat.has_temporal_ordering:
            return QueryType.TEMPORAL
        if any(w in q_lower for w in ["what date", "which day", "on what date", "what time of year", "how many days passed", "how many weeks passed"]):
            return QueryType.TEMPORAL

        # Priority 5: Aggregation & Multi-session (counting, sums, comparisons across sessions)
        if feat.has_counting or feat.has_arithmetic_diff or feat.has_total_sum:
            return QueryType.AGGREGATION

        # Priority 6: Default Factual User memory
        return QueryType.FACTUAL_USER
