"""Evidence Compiler v2 with Dynamic Marginal Utility & Incremental Requirement Gain.

Evaluates candidates based on:
M(b | S) = w_closure * ΔR(b | S) + w_anc * A_effective(b | S) + w_nov * N(b | S) + w_prov * P(b)
where ΔR(b | S) = R(S ∪ {b}) - R(S) is strictly recomputed with respect to the
already admitted context S, completely eliminating duplicate requirement inflation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence, Any

from artificial_memory.compiler.closure.types import (
    EvidenceClosureResult,
    ClosureCandidate,
    EvidenceRequirement,
    RequirementStatus,
)


def _tokenize_words(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z0-9_+/'-]+\b", text.lower()))


@dataclass(frozen=True)
class CompiledContextV2:
    """Compiled dialogue context ready for Reader ingestion."""
    formatted_text: str
    admitted_candidates: tuple[ClosureCandidate, ...]
    admitted_turn_ids: tuple[str, ...]
    total_estimated_tokens: int
    resolved_requirements: tuple[str, ...]
    unresolved_requirements: tuple[str, ...]
    sessions_represented: tuple[str, ...]


class EvidenceCompilerV2:
    """Dynamic Marginal Utility Knapsack Compiler (Write LLM = 0)."""

    def __init__(
        self,
        w_closure: float = 8.0,
        w_anchor: float = 1.0,
        w_novelty: float = 3.0,
        w_provenance: float = 1.5,
    ) -> None:
        self.w_closure = w_closure
        self.w_anchor = w_anchor
        self.w_novelty = w_novelty
        self.w_provenance = w_provenance

    def compile(
        self,
        closure_result: EvidenceClosureResult,
        token_budget: int | None = None,
    ) -> CompiledContextV2:
        budget = token_budget or closure_result.query_plan.recommended_budget
        query_words = _tokenize_words(closure_result.query_plan.query_text)

        # Requirement-to-turns mapping from closure evaluation
        req_supporting_turns: dict[str, set[str]] = {}
        for req in closure_result.requirement_details:
            req_supporting_turns[req.requirement_id] = set(req.supporting_turn_ids)

        admitted: list[ClosureCandidate] = []
        admitted_tids: set[str] = set()
        resolved_reqs: set[str] = set()
        words_in_context: set[str] = set()
        session_counts: dict[str, int] = {}
        accumulated_tokens = 0

        remaining = list(closure_result.admitted_candidates)

        while remaining:
            best_cand: ClosureCandidate | None = None
            best_density = -1.0
            best_marginal = 0.0
            best_tokens = 0

            for cand in remaining:
                cand_tokens = int(len(cand.content.split()) * 1.3) + 20
                if accumulated_tokens + cand_tokens > budget:
                    continue

                # 1. Dynamic incremental requirement gain: ΔR(b | S) = R(S ∪ {b}) - R(S)
                newly_resolved_count = 0
                for req_id, supporting_tids in req_supporting_turns.items():
                    if req_id not in resolved_reqs and cand.turn_id in supporting_tids:
                        newly_resolved_count += 1
                delta_r = float(newly_resolved_count)

                # 2. Dynamic session-adaptive anchor score: A_effective(b | S)
                k_session = session_counts.get(cand.session_id, 0)
                base_score = cand.rrf_score * 100.0  # scale RRF to comparable anchor magnitude

                cand_words = _tokenize_words(cand.content)
                has_novel_query_word = bool(query_words.intersection(cand_words) - words_in_context)

                if base_score >= 8.0 or has_novel_query_word:
                    # Protect depth for high-confidence evidence or new query terms
                    a_effective = base_score
                else:
                    # Apply session repetition damping
                    a_effective = base_score / (1.0 + 0.8 * k_session)

                # Exploration bonus for unrepresented sessions
                if k_session == 0:
                    a_effective += 4.0

                # 3. Dynamic novelty term: N(b | S)
                novel_query_tokens = len(query_words.intersection(cand_words) - words_in_context)
                novelty_score = float(novel_query_tokens)

                # 4. Provenance certifier bonus
                prov_score = 1.0 if cand.speaker and cand.session_date != "None" else 0.2

                # Total dynamic marginal utility M(b | S)
                m_value = (
                    self.w_closure * delta_r
                    + self.w_anchor * a_effective
                    + self.w_novelty * novelty_score
                    + self.w_provenance * prov_score
                )

                # Sublinear token penalty density
                density = m_value / (max(60, cand_tokens) ** 0.15)

                if density > best_density:
                    best_density = density
                    best_cand = cand
                    best_marginal = m_value
                    best_tokens = cand_tokens

            if best_cand is None:
                # No more candidates fit the remaining token budget
                break

            # Admit the candidate
            admitted.append(best_cand)
            admitted_tids.add(best_cand.turn_id)
            accumulated_tokens += best_tokens
            session_counts[best_cand.session_id] = session_counts.get(best_cand.session_id, 0) + 1

            # Update words in context
            words_in_context.update(_tokenize_words(best_cand.content))

            # Update resolved requirements dynamically
            for req_id, supporting_tids in req_supporting_turns.items():
                if best_cand.turn_id in supporting_tids:
                    resolved_reqs.add(req_id)

            remaining.remove(best_cand)

        # Sort admitted context chronologically by session and turn order
        admitted.sort(key=lambda c: (c.session_id, c.turn_id))

        # Format context with session headers and turn attribution
        formatted_lines = []
        current_sid = None
        for c in admitted:
            if c.session_id != current_sid:
                current_sid = c.session_id
                date_str = f" on {c.session_date}" if c.session_date and c.session_date != "None" else ""
                formatted_lines.append(f"\n[{c.session_id}{date_str}]")
            formatted_lines.append(f"[{c.turn_id}] {c.speaker}: {c.content}")

        formatted_context = "\n".join(formatted_lines).strip()

        all_req_ids = {r.requirement_id for r in closure_result.requirement_details}
        unresolved_reqs = tuple(sorted(all_req_ids - resolved_reqs))

        return CompiledContextV2(
            formatted_text=formatted_context,
            admitted_candidates=tuple(admitted),
            admitted_turn_ids=tuple(c.turn_id for c in admitted),
            total_estimated_tokens=accumulated_tokens,
            resolved_requirements=tuple(sorted(resolved_reqs)),
            unresolved_requirements=unresolved_reqs,
            sessions_represented=tuple(sorted(session_counts.keys())),
        )
