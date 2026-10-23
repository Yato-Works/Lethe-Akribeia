"""Universal Evidence Scoring Engine for AM Apex (Phase X.1).

Implements multi-dimensional evidence scoring:
    S(e, q) = w_l * S_lexical
            + w_s * S_semantic
            + w_e * S_entity
            + w_a * S_actor
            + w_t * S_temporal
            + w_p * S_provenance

Decouples open-domain conversation retrieval from domain-specific engineering
priors. Vocabulary tables (concept synonym expansions, actor roster, rare
named-entity targets) are *data*, not code: they are loaded once, at import
time, from the audited ontology plane (artificial_memory/ontology/data/).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from artificial_memory.core.ir.memory_types import ApexMemoryUnit
from artificial_memory.ontology import ontology as _ontology


@dataclass
class EvidenceScoreWeights:
    """Weights for the multi-dimensional evidence scoring formula."""
    w_lexical: float = 1.0
    w_semantic: float = 1.0
    w_entity: float = 1.5
    w_actor: float = 2.0
    w_temporal: float = 1.5
    w_provenance: float = 0.5


@dataclass
class EvidenceScoreBreakdown:
    """Detailed score breakdown for ablation and diagnostic logging."""
    total_score: float
    s_lexical: float
    s_semantic: float
    s_entity: float
    s_actor: float
    s_temporal: float
    s_provenance: float


class UniversalEvidenceScorer:
    """Computes multi-dimensional relevance scores between queries and memory units."""

    STOP_WORDS = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "did", "does", "do", "was", "were", "is", "are", "am", "have", "has", "had",
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
        "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "can", "will", "just", "should", "now", "likely", "would",
        "could", "might", "must", "her", "his", "she", "he", "they", "their", "them",
        "it", "its", "you", "your", "my", "our", "we", "us", "also",
        "many", "much", "this", "that", "these", "those", "year", "years",
        "day", "days", "week", "weeks", "month", "months", "time", "times",
        "different", "various", "distinct", "type", "types", "kind", "kinds",
        "sort", "sorts", "piece", "pieces", "item", "items",
    }

    # Concept synonym expansions loaded from the ontology data plane
    # (concept_synonyms.yaml). Kept as a class attribute for backward
    # compatibility with historical consumers (entity_aggregator,
    # budgeted_compiler, coverage_selector, per_session_sampler).
    SEMANTIC_EXPANSIONS: dict[str, list[str]] = dict(_ontology.mapping("concept_synonyms"))

    # Person roster for actor binding / speaker filtering (actor_roster.yaml).
    KNOWN_ACTORS: list[str] = list(_ontology.list("actor_roster"))

    # Rare named-entity lookup targets (synonym_expansion_targets.yaml).
    _NAMED_TARGETS: frozenset[str] = frozenset(_ontology.list("synonym_expansion_targets", key="targets"))

    @staticmethod
    def stem_word(w: str) -> str:
        """Fast deterministic English suffix stemmer."""
        w = w.lower()
        if len(w) <= 3:
            return w
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        for _ in range(2):
            stripped = False
            for suffix in ["s", "es", "ed", "ing", "tion", "ment", "ly"]:
                if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                    w = w[:-len(suffix)]
                    stripped = True
                    break
            if not stripped:
                break
        return w

    def extract_content_words(self, text: str) -> list[str]:
        """Extract meaningful content words (excluding stop words and punctuation)."""
        words = re.findall(r"\b[a-zA-Z0-9_\-\']+\b", text.lower())
        return [w for w in words if len(w) > 2 and w not in self.STOP_WORDS]

    def extract_target_actor(self, query: str) -> str | None:
        """Detect human subject/actor in query."""
        q_lower = query.lower()
        for actor in self.KNOWN_ACTORS:
            if re.search(rf"\b{actor}\b", q_lower):
                return actor
        return None

    def extract_entities(self, query: str) -> list[str]:
        """Extract named entities or quoted phrases from query."""
        entities = []
        # 1. Quoted terms: "Nothing is Impossible"
        for q in re.findall(r'"([^"]+)"', query):
            entities.append(q.lower())
        # 2. Capitalized multi-word proper nouns or specific names
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        for c in caps:
            c_low = c.lower()
            if c_low not in self.STOP_WORDS and c_low not in self.KNOWN_ACTORS:
                entities.append(c_low)
        # 3. Known key names & entities (ontology: synonym_expansion_targets)
        for k in self._NAMED_TARGETS:
            if k in query.lower():
                entities.append(k)
        return list(set(entities))

    def compute_score(
        self,
        query: str,
        unit: ApexMemoryUnit,
        weights: EvidenceScoreWeights | None = None,
        doc_frequencies: dict[str, int] | None = None,
        total_docs: int = 400,
    ) -> EvidenceScoreBreakdown:
        """Compute the full multi-dimensional evidence score for a single memory unit."""
        if weights is None:
            weights = EvidenceScoreWeights()

        q_lower = query.lower()
        content_lower = unit.ir.raw_content.lower()
        content_words = self.extract_content_words(query)

        # 1. S_lexical: Content Word Match with IDF Weighting & Stemming
        s_lexical = 0.0
        matched_words = 0
        content_tokens = re.findall(r"\b[a-zA-Z0-9_\-\']+\b", content_lower)
        content_stems = set(self.stem_word(w) for w in content_tokens)

        for w in content_words:
            w_stem = self.stem_word(w)
            if w in content_lower or (w.endswith("s") and len(w) > 3 and w[:-1] in content_lower) or (w + "s" in content_lower):
                matched_words += 1
                # Rare words (lower doc frequency) get much higher weight
                if doc_frequencies and w in doc_frequencies:
                    df = doc_frequencies[w]
                    idf = 1.0 + (total_docs / (df + 1))
                else:
                    idf = 5.0 + min(len(w), 8) * 0.5
                s_lexical += idf
            elif w_stem in content_stems:
                # Stem overlap (e.g. camped <-> camping, books <-> book)
                matched_words += 0.8
                s_lexical += 6.0

        # Bonus for high proportion of matched query words
        if content_words:
            coverage_ratio = matched_words / len(content_words)
            s_lexical += coverage_ratio * 15.0

        # 2. S_actor: Subject & Speaker Binding
        s_actor = 0.0
        target_actor = self.extract_target_actor(query)
        if target_actor:
            u_speaker = (unit.ir.source or "").lower()
            # Did the actor speak this turn?
            if u_speaker == target_actor:
                s_actor += 15.0
                # 1st-person statement by target actor
                if any(p in content_lower for p in ["i ", "i'm", "my ", "me ", "we "]):
                    s_actor += 10.0
            elif target_actor in content_lower:
                # Turn talks about the target actor
                s_actor += 12.0
            elif u_speaker in self.KNOWN_ACTORS and u_speaker != target_actor:
                # Another actor speaking about themselves
                if any(p in content_lower for p in ["i ", "i'm", "my "]):
                    s_actor -= 8.0

        # 3. S_entity: Named Entity & Specific Term Alignment (with Proper Noun Boost)
        s_entity = 0.0
        entities = self.extract_entities(query)
        for ent in entities:
            if ent in content_lower:
                # Super-charge multi-word or rare proper nouns (ontology: synonym_expansion_targets)
                if len(ent.split()) > 1 or ent in self._NAMED_TARGETS:
                    s_entity += 40.0
                else:
                    s_entity += 20.0
            else:
                # Partial entity word match
                ent_words = ent.split()
                if len(ent_words) > 1 and all(ew in content_lower for ew in ent_words):
                    s_entity += 25.0

        # 4. S_temporal: Temporal Scope Alignment
        s_temporal = 0.0
        ref_time = (unit.ir.time_scope or "") + " " + content_lower
        # Check for year / month matches
        years = re.findall(r"\b(202\d)\b", query)
        for y in years:
            if y in ref_time:
                s_temporal += 15.0

        months = ["january", "february", "march", "april", "may", "june",
                  "july", "august", "september", "october", "november", "december"]
        for m in months:
            if m in q_lower and m in ref_time:
                s_temporal += 12.0

        if any(w in q_lower for w in ["yesterday", "recent", "recently", "last week", "last month"]):
            if any(w in content_lower for w in ["yesterday", "recently", "last week", "last month", "ago"]):
                s_temporal += 10.0

        # 5. S_semantic: Concept & Synonym Expansions
        s_semantic = 0.0
        for concept, expansions in self.SEMANTIC_EXPANSIONS.items():
            if concept in q_lower:
                for exp in expansions:
                    if exp in content_lower:
                        s_semantic += 15.0
                        break

        # 6. S_provenance: Direct assertion vs vague mention
        s_provenance = 0.0
        if unit.ir.relation.value in ["asserts", "prefers", "located_at"]:
            s_provenance += 5.0
        # Turns containing concrete numbers / dates
        if re.search(r"\b\d+\b", content_lower):
            s_provenance += 3.0

        # Weighted Total
        total = (
            weights.w_lexical * s_lexical
            + weights.w_semantic * s_semantic
            + weights.w_entity * s_entity
            + weights.w_actor * s_actor
            + weights.w_temporal * s_temporal
            + weights.w_provenance * s_provenance
        )

        return EvidenceScoreBreakdown(
            total_score=total,
            s_lexical=s_lexical,
            s_semantic=s_semantic,
            s_entity=s_entity,
            s_actor=s_actor,
            s_temporal=s_temporal,
            s_provenance=s_provenance,
        )
