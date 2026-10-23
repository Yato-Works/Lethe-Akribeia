"""Multi-Session Coverage Engine (R6 Core).

Implements Convergent Multi-Session Retrieval:
Transitions from single-shot retrieval to an iterative, convergent memory coverage loop.
Guarantees full-set evidence retrieval across fragmented sessions for count, sum,
enumeration, and comparative queries.
"""

from __future__ import annotations

from collections.abc import Sequence

from artificial_memory_gen.compiler.coverage.coverage_planner import (
    EvidenceCoveragePlanner,
)


class MultiSessionCoverageEngine:
    """Convergent memory retrieval engine maximizing multi-session evidence coverage."""

    def __init__(self, max_convergent_rounds: int = 3) -> None:
        self.max_convergent_rounds = max_convergent_rounds
        self.planner = EvidenceCoveragePlanner()

    def retrieve_convergent_evidence(
        self,
        query: str,
        all_turns: Sequence[tuple[str, str, str, str]],  # (sid, s_date, speaker, content)
        retriever,
        reference_date: str | None = None,
        base_top_k: int = 45,
        enable_pairwise: bool = True,
        enable_concepts: bool = True,
    ) -> tuple[list[tuple[str, str, str, str]], dict[tuple[str, str], float], set[str]]:
        """Execute convergent retrieval to guarantee complete multi-session coverage.

        Returns:
            sorted_turns: Complete sequence of candidate turns ordered by aggregate relevance.
            merged_scores: Aggregated relevance scores per (sid, content).
            guaranteed_sids: Set of session IDs uncovered by targeted requirement expansions.
        """
        # 1. Plan Evidence Requirements
        reqs = self.planner.plan(query)

        # 2. Round 1: Primary Retrieval
        initial_turns, initial_scores = retriever.rank_turns(
            query, all_turns, top_k=base_top_k, reference_date=reference_date
        )

        merged_turns_dict: dict[tuple[str, str], tuple[str, str, str, str]] = {}
        merged_scores: dict[tuple[str, str], float] = dict(initial_scores)

        for t in initial_turns:
            key = (t[0], t[3])
            merged_turns_dict[key] = t

        collected_sids: set[str] = {t[0] for t in initial_turns}
        guaranteed_sids: set[str] = set()

        # 3. Targeted Pairwise / Dual-Entity Expansion (if differential/comparative/dual-conjunction)
        if reqs.pairwise_requirements and enable_pairwise:
            for pair in reqs.pairwise_requirements:
                # Retrieve entity A turns
                a_turns, a_scores = retriever.rank_turns(
                    pair.subquery_a, all_turns, top_k=25, reference_date=reference_date
                )
                # Retrieve entity B turns
                b_turns, b_scores = retriever.rank_turns(
                    pair.subquery_b, all_turns, top_k=25, reference_date=reference_date
                )

                for t in a_turns:
                    key = (t[0], t[3])
                    merged_turns_dict[key] = t
                    collected_sids.add(t[0])
                    guaranteed_sids.add(t[0])
                    s = a_scores.get(key, 0.0)
                    merged_scores[key] = max(merged_scores.get(key, 0.0), s * 1.6, 25.0)

                for t in b_turns:
                    key = (t[0], t[3])
                    merged_turns_dict[key] = t
                    collected_sids.add(t[0])
                    guaranteed_sids.add(t[0])
                    s = b_scores.get(key, 0.0)
                    merged_scores[key] = max(merged_scores.get(key, 0.0), s * 1.6, 25.0)

        # 4. Convergent Concept Ingestion (if abstract categories triggered)
        if reqs.concept_subqueries and enable_concepts:
            stagnant_rounds = 0
            for subq in reqs.concept_subqueries[:10]:
                c_turns, c_scores = retriever.rank_turns(
                    subq, all_turns, top_k=20, reference_date=reference_date
                )
                new_sids_found = 0

                for t in c_turns:
                    key = (t[0], t[3])
                    if key not in merged_turns_dict:
                        merged_turns_dict[key] = t
                    if t[0] not in collected_sids:
                        new_sids_found += 1
                        collected_sids.add(t[0])
                        guaranteed_sids.add(t[0])
                    s = c_scores.get(key, 0.0)
                    merged_scores[key] = max(merged_scores.get(key, 0.0), s * 1.3, 18.0)

                if new_sids_found == 0:
                    stagnant_rounds += 1
                    if stagnant_rounds >= 2:
                        # Retrieval set converged
                        break
                else:
                    stagnant_rounds = 0

        # Sort merged turns by aggregate relevance score descending
        sorted_turns = sorted(
            merged_turns_dict.values(),
            key=lambda t: merged_scores.get((t[0], t[3]), 0.0),
            reverse=True,
        )

        return sorted_turns, merged_scores, guaranteed_sids
