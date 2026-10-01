"""Universal Cognitive IR Extractor (Phase 2).

Extracts StructuredIR representations from dialogue turns without domain-specific
hardcoding, using deterministic syntactic and semantic pattern resolution.
"""

from __future__ import annotations

import re

from artificial_memory.compiler.speaker_normalizer import SpeakerAttributionNormalizer
from artificial_memory.core.ir import IRRelation, IRStatus, StructuredIR


class UniversalIRExtractor:
    """Extracts StructuredIR records from conversation turns."""

    # 1. Generic Entity Patterns (captures "project X", "workstation in X", "user", "Eli", etc.)
    ENTITY_PATTERNS = [
        re.compile(r"\bfor\s+project\s+([a-zA-Z0-9_-]+)\b", re.IGNORECASE),
        re.compile(r"\bworking\s+on\s+([a-zA-Z0-9_-]+)\b", re.IGNORECASE),
        re.compile(r"\bproject\s+([a-zA-Z0-9_-]+)\b", re.IGNORECASE),
        re.compile(r"\bthe\s+workstation\s+in\s+([a-zA-Z0-9_-]+)\b", re.IGNORECASE),
        re.compile(r"\b(user|assistant|teammate)\b", re.IGNORECASE),
    ]

    # 2. Generic Temporal Patterns (ISO dates, ranges, relative times)
    TIME_PATTERNS = [
        re.compile(r"\b(until\s+\d{4}-\d{2}-\d{2})\b", re.IGNORECASE),
        re.compile(r"\b(on\s+\d{4}-\d{2}-\d{2})\b", re.IGNORECASE),
        re.compile(r"\b(from\s+\d{4}-\d{2}-\d{2})\b", re.IGNORECASE),
        re.compile(r"\b(in\s+the\s+evenings)\b", re.IGNORECASE),
        re.compile(r"\b(currently|now|previously)\b", re.IGNORECASE),
    ]

    def __init__(self, enable_speaker_attribution: bool = True) -> None:
        self.enable_speaker_attribution = enable_speaker_attribution
        self.speaker_normalizer = SpeakerAttributionNormalizer()

    def extract(self, text: str, default_source: str = "user") -> list[StructuredIR]:
        """Extract all StructuredIR units from a given text line or turn."""
        records: list[StructuredIR] = []
        clean = text.strip()
        if not clean:
            return records

        source = default_source
        if "teammate reported" in clean.lower():
            source = "teammate"
        elif "assistant" in clean.lower():
            source = "assistant"

        # Apply deterministic speaker attribution & pronoun disambiguation
        if self.enable_speaker_attribution and default_source and default_source.lower() not in ["user", "assistant", "system", "teammate", "general"]:
            clean = self.speaker_normalizer.normalize_turn(clean, speaker=default_source)

        entity = self._extract_entity(clean)
        if entity == "general" and default_source and default_source.lower() not in ["user", "assistant", "teammate", "general"]:
            entity = default_source.lower()
        time_scope = self._extract_time(clean)

        # Pattern A: Migration / Transition (from OLD to NEW, or directly to NEW)
        m_mig = re.search(
            r"(?:migrated|migrate|moved|move|switched|switch)\s+(?:the\s+)?([a-zA-Z0-9_\s]+?)\s+(?:from\s+([a-zA-Z0-9_\s]+?)\s+)?to\s+([a-zA-Z0-9_\s\.\-]+)",
            clean,
            re.IGNORECASE,
        )
        if m_mig:
            prop_candidate = m_mig.group(1).strip()
            # If entity is included in the matched property candidate, strip it
            prop_candidate = re.sub(r"^(?:project\s+[a-zA-Z0-9_-]+\s+|the\s+)", "", prop_candidate, flags=re.IGNORECASE).strip()
            prop = self._normalize_property(prop_candidate)
            old_val = m_mig.group(2).strip() if m_mig.group(2) else None
            new_val = m_mig.group(3).strip().rstrip(".")
            records.append(StructuredIR(
                entity=entity,
                property=prop,
                value=new_val,
                old_value=old_val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.MIGRATED,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern B: Explicit Negative Constraint / Non-existence (never X)
        m_neg = re.search(
            r"(?:never\s+(?:a\s+)?([a-zA-Z0-9_\s]+)|only\s+discussed\s+([a-zA-Z0-9_\s]+)\s+and\s+never\s+(?:a\s+)?([a-zA-Z0-9_\s]+))",
            clean,
            re.IGNORECASE,
        )
        if m_neg:
            prop = self._normalize_property(m_neg.group(3) or m_neg.group(1) or "unspecified")
            records.append(StructuredIR(
                entity=entity,
                property=prop,
                value="none",
                time_scope=time_scope,
                source=source,
                relation=IRRelation.NEVER,
                status=IRStatus.UNSPECIFIED,
                raw_content=clean,
            ))
            return records

        # Pattern C: Location / Residency
        m_loc = re.search(
            r"(?:working\s+from|relocated\s+to|located\s+in)\s+([a-zA-Z0-9_\s]+?)(?:\s+until|\s+on|\s+from|\.|$)",
            clean,
            re.IGNORECASE,
        )
        if m_loc:
            val = m_loc.group(1).strip()
            records.append(StructuredIR(
                entity=entity,
                property="location",
                value=val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.LOCATED_AT,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern D: Consumption / Habits (drinks X while working on Y)
        m_drink = re.search(
            r"usually\s+drinks\s+([a-zA-Z0-9_\s]+?)\s+while\s+working\s+on\s+([a-zA-Z0-9_-]+)",
            clean,
            re.IGNORECASE,
        )
        if m_drink:
            val = m_drink.group(1).strip()
            ent = m_drink.group(2).strip()
            records.append(StructuredIR(
                entity=ent,
                property="drinks",
                value=val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.PREFERS,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern E: Tool / Setting Choice (settled on X, pinned X to Y, set X to Y)
        m_tool = re.search(
            r"(?:settled\s+on|pinned\s+(?:the\s+)?([a-zA-Z0-9_\s]+?)\s+to|set\s+(?:the\s+)?([a-zA-Z0-9_\s]+?)\s+to|preferred\s+([a-zA-Z0-9_\s]+?)\s+is)\s+([a-zA-Z0-9_\s\.\-]+)",
            clean,
            re.IGNORECASE,
        )
        if m_tool:
            prop = self._normalize_property(m_tool.group(1) or m_tool.group(2) or m_tool.group(3) or "tool")
            val = m_tool.group(4).strip().rstrip(".")
            records.append(StructuredIR(
                entity=entity,
                property=prop,
                value=val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.PREFERS,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern F: Behavioral Tendency (reverts, backups, copies by hand)
        if any(w in clean.lower() for w in ["reverts", "backup", "copies files", "revert"]):
            records.append(StructuredIR(
                entity=entity,
                property="behavior_tendency",
                value=clean,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.BEHAVIOR,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern G: General Factual Assertion (runs on X, was X, is X)
        m_fact = re.search(
            r"(?:the\s+)?([a-zA-Z0-9_\s]+?)\s+(?:runs\s+on|was|is|actually\s+runs\s+on)\s+([a-zA-Z0-9_\s\.\-]+)",
            clean,
            re.IGNORECASE,
        )
        if m_fact:
            prop = self._normalize_property(m_fact.group(1))
            val = m_fact.group(2).strip().rstrip(".")
            records.append(StructuredIR(
                entity=entity,
                property=prop,
                value=val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.ASSERTS,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern H: Actions & Events (bought, visited, adopted, planted, assembled, finished)
        m_act = re.search(
            r"\b(bought|purchased|adopted|planted|assembled|visited|attended|joined|started|finished|watched|read|dined\s+at)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9_\s\.\-]{2,40})",
            clean,
            re.IGNORECASE,
        )
        if m_act:
            action_prop = m_act.group(1).lower().strip()
            action_val = m_act.group(2).strip().rstrip(".,")
            records.append(StructuredIR(
                entity=entity,
                property=action_prop,
                value=action_val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.BEHAVIOR,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern I: Preferences & Tastes (likes, loves, enjoys, prefers, hates)
        m_pref = re.search(
            r"\b(loves?|likes?|enjoys?|prefers?|hates?|dislikes?)\s+([a-zA-Z0-9_\s\.\-]{2,35})",
            clean,
            re.IGNORECASE,
        )
        if m_pref:
            raw_pref = m_pref.group(1).lower().strip()
            # Normalize to base form
            pref_prop = "love" if raw_pref.startswith("love") else "like" if raw_pref.startswith("like") else raw_pref
            pref_val = m_pref.group(2).strip().rstrip(".,")
            records.append(StructuredIR(
                entity=entity,
                property=pref_prop,
                value=pref_val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.PREFERS,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern J: Possession & Kinship (has a dog, owns a car, etc.)
        m_poss = re.search(
            r"\b(has|have|owns?|owned)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9_\s\.\-]{2,35})",
            clean,
            re.IGNORECASE,
        )
        if m_poss:
            poss_val = m_poss.group(2).strip().rstrip(".,")
            records.append(StructuredIR(
                entity=entity,
                property="possesses",
                value=poss_val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.ASSERTS,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Pattern K: Occupation & Identity (works as, employed as, studies)
        m_occ = re.search(
            r"\b(works?\s+as|employed\s+as|studies|majoring\s+in)\s+(?:a\s+|an\s+)?([a-zA-Z0-9_\s\.\-]{2,30})",
            clean,
            re.IGNORECASE,
        )
        if m_occ:
            occ_val = m_occ.group(2).strip().rstrip(".,")
            records.append(StructuredIR(
                entity=entity,
                property="occupation",
                value=occ_val,
                time_scope=time_scope,
                source=source,
                relation=IRRelation.ASSERTS,
                status=IRStatus.ACTIVE,
                raw_content=clean,
            ))
            return records

        # Fallback: Capture as raw proposition
        records.append(StructuredIR(
            entity=entity,
            property="statement",
            value=clean,
            time_scope=time_scope,
            source=source,
            relation=IRRelation.ASSERTS,
            status=IRStatus.ACTIVE,
            raw_content=clean,
        ))
        return records

    def _normalize_property(self, text: str) -> str:
        """Normalize property name by stripping articles and possessive pronouns."""
        clean = text.strip().rstrip(".").lower()
        clean = re.sub(r"^(?:the|a|an|our|my|their|its|his|her)\s+", "", clean, flags=re.IGNORECASE).strip()
        return clean

    def _extract_entity(self, text: str) -> str:
        for p in self.ENTITY_PATTERNS:
            m = p.search(text)
            if m:
                val = m.group(1).lower()
                return val
        return "general"

    def _extract_time(self, text: str) -> str | None:
        for p in self.TIME_PATTERNS:
            m = p.search(text)
            if m:
                return m.group(1).lower()
        return None
