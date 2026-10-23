"""Generic Cross-Session Entity & Attribute Aggregator for Multi-Hop QA.

Detects multi-session aggregation demands (e.g. "What are Melanie's pets' names?",
"What activities does Melanie partake in?") and compiles a deterministic
cross-session evolution & aggregation summary block.

Fully generic: No benchmark-specific questions or answers. It operates on
the dialogue turns, tracking entity mentions across sessions to prevent
LLM attention drop across long conversational haystacks.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence


class EntityAttributeAggregator:
    """Aggregates entity attributes across multiple conversational sessions."""

    _STOP_WORDS = frozenset({
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "did", "does", "do", "was", "were", "is", "are", "have", "has", "had",
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
        "from", "by", "of", "about", "some", "any", "all", "both", "each",
        "many", "much", "more", "most", "other", "partake", "participate",
    })

    @staticmethod
    def _stem(w: str) -> str:
        w = w.lower()
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("es") and len(w) > 3:
            return w[:-2]
        if w.endswith("s") and len(w) > 2 and not w.endswith("ss"):
            return w[:-1]
        if w.endswith("ing") and len(w) > 4:
            return w[:-3]
        if w.endswith("ed") and len(w) > 3:
            return w[:-2]
        return w

    def extract_salient_anchors(self, question: str) -> tuple[list[str], list[str]]:
        """Extract entity names (capitalized words) and attribute concepts from question."""
        from artificial_memory.recall.evidence_scorer import UniversalEvidenceScorer
        expansions = UniversalEvidenceScorer.SEMANTIC_EXPANSIONS

        ql = question.lower()
        # Find proper names
        names = [
            m.group(0) for m in re.finditer(r"\b[A-Z][a-z]+\b", question)
            if m.group(0).lower() not in self._STOP_WORDS
        ]

        # Find content concept words (len >= 4)
        tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", ql)
        concepts = set()
        for t in tokens:
            if len(t) >= 4 and t not in self._STOP_WORDS and not any(t == n.lower() for n in names):
                concepts.add(t)
                stem = self._stem(t)
                concepts.add(stem)
                if t in expansions:
                    concepts.update(expansions[t][:6])
                if stem in expansions:
                    concepts.update(expansions[stem][:6])

        return names, sorted(concepts)


    def build_aggregation_block(
        self,
        question: str,
        turns: Sequence[tuple[str, str, str, str]], # (sid, date, role, text)
        max_evidence_lines: int = 8,
    ) -> str | None:
        """Scan turns across sessions and produce an explicit cross-session aggregation block."""
        names, concepts = self.extract_salient_anchors(question)
        if not concepts:
            return None

        # Collect turns mentioning concepts and (if present) entity names
        session_hits: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        
        for sid, date_str, role, text in turns:
            tl = text.lower()
            # Must mention at least one concept
            concept_match = any(c in tl for c in concepts)
            if not concept_match:
                continue
                
            # If names exist in question, prefer turns mentioning at least one name
            name_match = True
            if names:
                name_match = any(n.lower() in tl for n in names) or (role.lower() in [n.lower() for n in names])
            
            if concept_match and name_match:
                session_hits[sid].append((date_str, role, text))

        # We only need cross-session aggregation if evidence spans >= 2 distinct sessions
        if len(session_hits) < 2:
            return None

        # Build chronological summary
        lines: list[str] = [
            f"[CROSS-SESSION AGGREGATION & EVOLUTION PROOF: {' '.join(concepts).upper()}]",
            "Discovered relevant mentions across multiple sessions:"
        ]

        count = 0
        for sid, items in session_hits.items():
            for date_str, role, text in items:
                # Clip text to concise evidence snippet
                snippet = re.sub(r"\s+", " ", text).strip()
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                lines.append(f"  * [{sid} on {date_str}] {role}: {snippet}")
                count += 1
                if count >= max_evidence_lines:
                    break
            if count >= max_evidence_lines:
                break

        lines.append(
            "[INSTRUCTION: The question asks to aggregate or identify items across the conversation. "
            "Combine ALL distinct entities/facts found in the different sessions above into your final answer.]"
        )
        lines.append(f"[/CROSS-SESSION AGGREGATION PROOF]\n")

        return "\n".join(lines)
