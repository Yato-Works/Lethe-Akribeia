"""Derivation Scaffolding Engine for Minimum Sufficient Context (MSC).

Bridging the 24.5% Reasoning Gap:
Small Reader models (1.5B - 7B) fail on multi-session joins, temporal deltas,
and knowledge updates even when all evidence is present in context (Gold-Context Ceiling: 78.6% fail).
The DerivationScaffolder synthesizes deterministic co-processor outputs into
an explicit, salient derivation guide ([DERIVATION SCAFFOLD: ...]) so the Reader
can directly extract the grounded answer without complex mental arithmetic.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from artificial_memory.core.ir.memory_types import ApexMemoryUnit, QueryIntent


class DerivationScaffolder:
    """Produces explicit deduction scaffolds from deterministic engine artifacts."""

    @staticmethod
    def scaffold_temporal(grounding_text: str | None, query: str) -> str | None:
        """Produce a salient derivation note from a temporal calculation or ordering certificate."""
        if not grounding_text:
            return None

        # 1. Temporal calculation / duration
        m_calc = re.search(r"\[Temporal Calculation:\s*([^\]]+)\]", grounding_text)
        if m_calc:
            detail = m_calc.group(1).strip()
            return f"[DERIVATION SCAFFOLD - TEMPORAL CALCULATION]: Verified timeline deduction: {detail}"

        # 2. Temporal ordering
        m_ord = re.search(r"\[Temporal Ordering:\s*([^\]]+)\]", grounding_text)
        if m_ord:
            detail = m_ord.group(1).strip()
            return f"[DERIVATION SCAFFOLD - CHRONOLOGICAL ORDER]: Verified chronological deduction: {detail}"

        return None

    @staticmethod
    def scaffold_state_update(state_resolution: Any | None) -> str | None:
        """Produce a salient note when an entity property has been updated/superseded."""
        if not state_resolution:
            return None

        cert = getattr(state_resolution, "grounding_certificate", "") or str(state_resolution)
        if "superseded" in cert.lower() or "updated" in cert.lower() or "conflict" in cert.lower():
            return f"[DERIVATION SCAFFOLD - CURRENT ACTIVE STATE]: {cert}"
        return None

    @staticmethod
    def scaffold_aggregation(
        query: str,
        intent: QueryIntent,
        selected_units: Sequence[ApexMemoryUnit],
    ) -> str | None:
        """Produce an explicit cross-session aggregation summary when counting items or summing quantities."""
        if intent != QueryIntent.AGGREGATION_QUERY:
            return None

        ql = query.lower()
        # Find numeric quantities mentioned in distinct sessions
        session_evidence: dict[str, list[str]] = {}
        for u in selected_units:
            raw = u.ir.raw_content
            m_sess = re.search(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+[^\]]+)?\]", raw)
            sid = m_sess.group(1) if m_sess else "s0"

            # Clean raw line to extract key assertion
            body = re.sub(r"^\[.*?\]\s*(?:user|assistant)?:\s*", "", raw).strip()
            # If body has numbers, amounts, or distinct events
            if re.search(r"\b\d+\b", body) or "$" in body or any(w in body.lower() for w in ["one", "two", "three", "four", "five", "six", "first", "second", "third"]):
                if sid not in session_evidence:
                    session_evidence[sid] = []
                session_evidence[sid].append(body)

        if len(session_evidence) >= 2:
            items_found = []
            for sid, texts in session_evidence.items():
                items_found.append(f"({sid}): {texts[0][:80]}")
            evidence_summary = " | ".join(items_found[:5])
            return f"[DERIVATION SCAFFOLD - MULTI-SESSION AGGREGATION]: Evidence identified across {len(session_evidence)} distinct sessions -> {evidence_summary}"

        return None
