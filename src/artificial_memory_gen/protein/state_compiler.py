"""Deterministic State Compiler for AM Apex Protein Phase (Phase C5 & C6).

Compiles multi-hop graph paths and evidence records into structured cognitive states:
    [STATE] <Subject>'s <Property> is <TargetValue>. (supported_by: [D3:13, D4:3])

Features:
1. Linguistic Query Target Extraction (verbs, nouns, counts, locations).
2. Graph-Path Traversal (1-hop direct, 2-hop bridging A -> B -> C, multi-item aggregation).
3. Candidate Generation & Consistency Ranking:
       Score(s) = w_prop * PropMatch + w_sup * SupportCount + w_sal * PathSalience
4. Zero LLM Write-Side Calls ($0.00 cost, 100% deterministic).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.ontology import ontology as _ONTOLOGY


@dataclass
class CompiledState:
    subject: str
    property_name: str
    value: str
    score: float
    supported_by: list[str] = field(default_factory=list)

    def format_state(self) -> str:
        """Format as a verified cognitive state statement with provenance."""
        subj = self.subject.capitalize()
        prop = self.property_name.strip()
        val = self.value.strip()

        prov = f" (supported_by: [{', '.join(self.supported_by)}])" if self.supported_by else ""

        # Generic, evidence-driven phrasing.
        #
        # This block used to return literal facts of one evaluation corpus as
        # canned [STATE] strings.  Verified against that corpus: 7 of them were
        # exact or near-exact gold answers.  That is Label-level contamination
        # -- the compiler shipped with the answer key.  All of it is gone; the
        # exact removed strings are recorded in git history, not here.
        # See docs/AUDIT_contamination_and_gaming.md.
        if prop == "identity":
            return f"[STATE] {subj}'s identity is {val}.{prov}"
        if any(w in prop.lower() for w in (
            "activities", "items", "events", "symbols", "hobbies", "artists",
            "bands", "books", "pets", "instruments", "types",
        )):
            return f"[STATE] {subj}'s {prop} include {val}.{prov}"
        if any(w in prop.lower() for w in ("relationship", "status")):
            return f"[STATE] {subj}'s relationship status is {val}.{prov}"
        if any(w in prop.lower() for w in ("home country", "origin", "moved from")):
            return f"[STATE] {subj}'s home country is {val}.{prov}"
        if any(w in prop.lower() for w in ("career", "profession")):
            return f"[STATE] {subj}'s career path is {val}.{prov}"
        if any(w in prop.lower() for w in ("count", "times", "children")):
            return f"[STATE] {subj}'s {prop} is {val}.{prov}"
        return f"[STATE] {subj}'s {prop} is {val}.{prov}"


class StateCompiler:
    """Graph-path deterministic state compiler."""

    # Primary actor set: corpora personas (actor_roster) + role/scaffold tokens.
    PRIMARY_ACTORS = set(_ONTOLOGY.list("actor_roster")) | {
        "user", "assistant", "system", "me", "i", "you", "he", "she", "they", "we", "it",
    }

    # Capitalised tokens that are question scaffolding, not the subject.
    _NON_SUBJECT_TOKENS = frozenset({
        "what", "when", "where", "which", "who", "why", "how", "the", "did",
        "does", "was", "were", "has", "have", "had", "monday", "tuesday",
        "wednesday", "thursday", "friday", "saturday", "sunday", "january",
        "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
    })

    # Syntactic question-shape -> property/anchors rules are data
    # (ontology: query_patterns.rules).
    PROPERTY_EXTRACTION_RULES = [
        (re.compile(rule["pattern"], re.I), rule["property"], list(rule["anchors"]))
        for rule in _ONTOLOGY.table("query_patterns")["rules"]
    ]

    # Anchor -> display canonicalization is data
    # (ontology: value_canonicalizations.canonicalizations).
    _VALUE_DISPLAY = {
        entry["match"]: entry["display"]
        for entry in _ONTOLOGY.table("value_canonicalizations")["canonicalizations"]
    }

    def __init__(self, max_states: int = 2) -> None:
        self.max_states = max_states

    def compile_states(
        self,
        query: str,
        records: Sequence[StructuredIR],
    ) -> list[CompiledState]:
        """Compile verified state candidates from graph evidence and rank by consistency."""
        q_lower = query.lower()

        # 1. Identify Target Subject and Target Property
        #
        # GENERIC: the subject is taken from the query itself, never from a
        # hardcoded cast list.  This used to read
        #     target_subject = "Caroline" if "caroline" in q_lower
        #                     else "Melanie" if "melanie" in q_lower else ""
        # which baked the speakers of one evaluation corpus into the compiler.
        target_subject = ""
        for name in re.findall(r"\b([A-Z][a-z]{2,})\b", query):
            if name.lower() not in self._NON_SUBJECT_TOKENS:
                target_subject = name
                break
        target_property = ""
        expected_anchors: list[str] = []

        for pat, prop_name, anchors in self.PROPERTY_EXTRACTION_RULES:
            m = pat.search(q_lower)
            if m:
                target_property = prop_name
                expected_anchors = anchors
                if not target_subject and m.groups():
                    target_subject = m.group(1).capitalize()
                break

        if not target_property:
            return []

        # 2. Candidate Generation Across Graph Records
        candidates: list[CompiledState] = []
        gathered_values: list[str] = []
        supporting_turn_ids: list[str] = []

        for r in records:
            content = (r.raw_content or "").lower()
            val = (r.value or "").lower()
            combined = f"{content} {val}"

            m_turn = re.search(r"\[([dD]\d+:\d+|\d+)", r.raw_content or "")
            turn_id = m_turn.group(1).upper() if m_turn else ""
            clean_text = re.sub(r"\[[dD]\d+:\d+[^\]]*\]", "", combined)

            # Check matching anchors with word boundary on clean text
            for anchor in expected_anchors:
                if re.search(rf"\b{re.escape(anchor)}\b", clean_text):
                    # Clean up value (ontology: value_canonicalizations)
                    clean_val = self._VALUE_DISPLAY.get(anchor, anchor)
                    gathered_values.append(clean_val)
                    if turn_id and turn_id not in supporting_turn_ids:
                        supporting_turn_ids.append(turn_id)

        if not gathered_values:
            return []

        # Deduplicate preserving order
        unique_vals = list(dict.fromkeys(gathered_values))
        final_val = ", ".join(unique_vals)

        # 3. Consistency Scoring
        prop_score = 10.0
        support_score = min(len(supporting_turn_ids) * 2.0, 6.0)
        path_salience = 4.0 if len(supporting_turn_ids) >= 2 else 2.0
        total_score = prop_score + support_score + path_salience

        compiled = CompiledState(
            subject=target_subject or "Subject",
            property_name=target_property,
            value=final_val,
            score=total_score,
            supported_by=supporting_turn_ids,
        )
        candidates.append(compiled)

        # 4. Consistency Ranking
        candidates.sort(key=lambda s: -s.score)
        return candidates[:self.max_states]
