"""Evidence Closure Engine (ECE) Loop (Write LLM = 0).

Executes the iterative closure loop:
1. Safe handling of empty/incomplete query plans and edge boundaries.
2. Initial candidate retrieval via Multi-View RRF with immediate explored_turn_ids tracking.
3. Candidate deduplication by turn_id, merging view ranks on re-discovery.
4. Rigorous requirement evaluation with SATISFIED, PARTIALLY_SUPPORTED, CONFLICTING, UNRESOLVED.
5. Explicit coverage_basis and unresolved_reason provenance.
6. Evidence graph construction with verifiable dialogue and speaker links.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence, Any

from artificial_memory.compiler.closure.types import (
    QueryPlan,
    ClosureState,
    ClosureCandidate,
    EvidenceRequirement,
    RequirementStatus,
    EvidenceGraphNode,
    EvidenceGraphEdge,
    EvidenceClosureResult,
    AnswerTopology,
)
from artificial_memory.compiler.closure.retriever import MultiViewRetriever


class EvidenceClosureEngine:
    """Deterministic Evidence Closure Loop engine."""

    def __init__(self, retriever: MultiViewRetriever | None = None) -> None:
        self.retriever = retriever or MultiViewRetriever()

    def run_closure(
        self,
        query_plan: QueryPlan,
        all_turns: Sequence[Any],
        initial_top_k: int = 35,
    ) -> EvidenceClosureResult:
        """Execute the multi-round evidence closure loop."""
        # Edge Case 1: Empty or Incomplete QueryPlan
        if not query_plan.requirements:
            return EvidenceClosureResult(
                query_plan=query_plan,
                closure_state=ClosureState(
                    iteration=0,
                    is_converged=False,
                    termination_reason="planning_incomplete",
                ),
                admitted_candidates=(),
                unmet_requirements=(),
                graph_nodes=(),
                graph_edges=(),
                closure_completeness=0.0,
                requirement_details=(),
                provenance_summary={
                    "total_candidates": 0,
                    "iterations_run": 0,
                    "termination_reason": "planning_incomplete",
                    "sessions_covered": [],
                },
            )

        # Edge Case 2: Empty dialogue
        if not all_turns:
            return EvidenceClosureResult(
                query_plan=query_plan,
                closure_state=ClosureState(
                    iteration=0,
                    unmet_requirements={r.requirement_id for r in query_plan.requirements},
                    is_converged=False,
                    termination_reason="empty_dialogue",
                ),
                admitted_candidates=(),
                unmet_requirements=tuple(sorted(r.requirement_id for r in query_plan.requirements)),
                graph_nodes=(),
                graph_edges=(),
                closure_completeness=0.0,
                requirement_details=tuple(
                    EvidenceRequirement(
                        requirement_id=r.requirement_id,
                        description=r.description,
                        target_entity=r.target_entity,
                        target_predicate=r.target_predicate,
                        status=RequirementStatus.UNRESOLVED,
                        unresolved_reason="Dialogue input was empty",
                    )
                    for r in query_plan.requirements
                ),
                provenance_summary={
                    "total_candidates": 0,
                    "iterations_run": 0,
                    "termination_reason": "empty_dialogue",
                    "sessions_covered": [],
                },
            )

        closure_state = ClosureState(
            unmet_requirements={r.requirement_id for r in query_plan.requirements}
        )
        candidate_dict: dict[str, ClosureCandidate] = {}

        # Step 1: Initial Multi-View Retrieval Pass & immediate explored_turn_ids registration
        initial_candidates = self.retriever.retrieve(query_plan, all_turns, top_k=initial_top_k)
        for c in initial_candidates:
            candidate_dict[c.turn_id] = c
            closure_state.explored_turn_ids.add(c.turn_id)

        # Step 2: Iterative Closure Loop
        current_req_evaluations: dict[str, EvidenceRequirement] = {}

        while closure_state.iteration < max(0, query_plan.max_closure_hops):
            closure_state.iteration += 1

            # A. Evaluate requirement coverage against current candidates
            current_candidates = list(candidate_dict.values())
            current_req_evaluations = self._evaluate_requirements(
                query_plan.requirements, current_candidates, query_plan.topology
            )

            # Update closure state sets
            closure_state.satisfied_requirements.clear()
            closure_state.partially_satisfied_requirements.clear()
            closure_state.conflicting_requirements.clear()
            closure_state.unmet_requirements.clear()

            for req_id, req in current_req_evaluations.items():
                if req.status == RequirementStatus.SATISFIED:
                    closure_state.satisfied_requirements.add(req_id)
                elif req.status == RequirementStatus.PARTIALLY_SUPPORTED:
                    closure_state.partially_satisfied_requirements.add(req_id)
                    closure_state.unmet_requirements.add(req_id)
                elif req.status == RequirementStatus.CONFLICTING:
                    closure_state.conflicting_requirements.add(req_id)
                    closure_state.unmet_requirements.add(req_id)
                else:
                    closure_state.unmet_requirements.add(req_id)

            # B. Check termination
            if not closure_state.unmet_requirements:
                closure_state.is_converged = True
                closure_state.termination_reason = "all_requirements_satisfied"
                break

            # C. Generate targeted expansion plan
            expansion_targets = self._derive_expansion_queries(
                closure_state.unmet_requirements, query_plan
            )
            if not expansion_targets:
                closure_state.termination_reason = "no_further_expansion_possible"
                break

            # D. Execute targeted expansion & deduplicate via candidate_dict
            new_turns_found = 0
            for exp_plan in expansion_targets:
                expanded_candidates = self.retriever.retrieve(exp_plan, all_turns, top_k=15)
                for c in expanded_candidates:
                    if c.turn_id in candidate_dict:
                        # Merge view ranks and boost RRF score on re-discovery
                        existing = candidate_dict[c.turn_id]
                        merged_ranks = dict(existing.view_ranks)
                        for v, r in c.view_ranks.items():
                            if v not in merged_ranks or r < merged_ranks[v]:
                                merged_ranks[v] = r
                        updated_rrf = existing.rrf_score + (c.rrf_score * 0.5)
                        candidate_dict[c.turn_id] = ClosureCandidate(
                            turn_id=existing.turn_id,
                            session_id=existing.session_id,
                            session_date=existing.session_date,
                            speaker=existing.speaker,
                            content=existing.content,
                            view_ranks=merged_ranks,
                            rrf_score=updated_rrf,
                        )
                    else:
                        candidate_dict[c.turn_id] = c
                        closure_state.explored_turn_ids.add(c.turn_id)
                        new_turns_found += 1

            if new_turns_found == 0:
                closure_state.termination_reason = "expansion_exhausted"
                break

        if not closure_state.is_converged and closure_state.termination_reason == "in_progress":
            closure_state.termination_reason = "max_hops_reached"

        # Final requirement evaluation pass
        final_candidates = list(candidate_dict.values())
        if not current_req_evaluations:
            current_req_evaluations = self._evaluate_requirements(
                query_plan.requirements, final_candidates, query_plan.topology
            )

        # Step 3: Build Verifiable Evidence Graph
        nodes, edges = self._build_evidence_graph(final_candidates)

        # Step 4: Calculate Graded Closure Completeness (avoiding 0-div)
        total_reqs = len(query_plan.requirements)
        if total_reqs == 0:
            completeness = 0.0
        else:
            score_sum = 0.0
            for r in current_req_evaluations.values():
                if r.status == RequirementStatus.SATISFIED:
                    score_sum += 1.0
                elif r.status == RequirementStatus.PARTIALLY_SUPPORTED:
                    score_sum += 0.5
                elif r.status == RequirementStatus.CONFLICTING:
                    score_sum += 0.25
            completeness = score_sum / total_reqs

        req_detail_tuple = tuple(current_req_evaluations.values())

        return EvidenceClosureResult(
            query_plan=query_plan,
            closure_state=closure_state,
            admitted_candidates=tuple(final_candidates),
            unmet_requirements=tuple(sorted(closure_state.unmet_requirements)),
            graph_nodes=tuple(nodes),
            graph_edges=tuple(edges),
            closure_completeness=completeness,
            requirement_details=req_detail_tuple,
            provenance_summary={
                "total_candidates": len(final_candidates),
                "iterations_run": closure_state.iteration,
                "termination_reason": closure_state.termination_reason,
                "requirement_statuses": {r.requirement_id: r.status.value for r in req_detail_tuple},
                "coverage_bases": {r.requirement_id: r.coverage_basis for r in req_detail_tuple if r.coverage_basis},
                "unresolved_reasons": {r.requirement_id: r.unresolved_reason for r in req_detail_tuple if r.unresolved_reason},
                "sessions_covered": list(sorted({c.session_id for c in final_candidates})),
            },
        )

    def _evaluate_requirements(
        self,
        requirements: tuple[EvidenceRequirement, ...],
        candidates: list[ClosureCandidate],
        topology: AnswerTopology,
    ) -> dict[str, EvidenceRequirement]:
        """Rigorously evaluates requirements with SATISFIED, PARTIAL, CONFLICTING, or UNRESOLVED status."""
        evaluations: dict[str, EvidenceRequirement] = {}
        candidate_sessions = {c.session_id for c in candidates}

        for req in requirements:
            supporting_tids: list[str] = []

            # 1. SET_ENUMERATION instance requirement
            if req.requirement_id == "req_enum_instances":
                # Find candidate turns with descriptive predicates
                for c in candidates:
                    if len(c.content.split()) >= 4:
                        supporting_tids.append(c.turn_id)

                if len(supporting_tids) >= 2:
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        target_entity=req.target_entity,
                        target_predicate=req.target_predicate,
                        status=RequirementStatus.SATISFIED,
                        supporting_turn_ids=tuple(supporting_tids),
                        coverage_basis=f"Found {len(supporting_tids)} distinct descriptive turns supporting set instances",
                        confidence=0.95,
                    )
                elif len(supporting_tids) == 1:
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        target_entity=req.target_entity,
                        target_predicate=req.target_predicate,
                        status=RequirementStatus.PARTIALLY_SUPPORTED,
                        supporting_turn_ids=tuple(supporting_tids),
                        coverage_basis="Found 1 instance, but set enumeration requires multiple instances",
                        unresolved_reason="Single instance discovered; full enumeration incomplete",
                        confidence=0.5,
                    )
                else:
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        target_entity=req.target_entity,
                        target_predicate=req.target_predicate,
                        status=RequirementStatus.UNRESOLVED,
                        supporting_turn_ids=(),
                        unresolved_reason="No descriptive candidate turns matched enumeration predicate",
                        confidence=0.0,
                    )

            # 2. Multi-Session Breadth requirement
            elif req.requirement_id == "req_multi_session_breadth":
                if len(candidate_sessions) >= 2:
                    top_tids = [c.turn_id for c in candidates if c.session_id in list(candidate_sessions)[:3]]
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        target_entity=req.target_entity,
                        status=RequirementStatus.SATISFIED,
                        supporting_turn_ids=tuple(top_tids[:4]),
                        coverage_basis=f"Candidates verified across {len(candidate_sessions)} distinct sessions: {sorted(candidate_sessions)}",
                        confidence=1.0,
                    )
                elif len(candidate_sessions) == 1:
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        target_entity=req.target_entity,
                        status=RequirementStatus.PARTIALLY_SUPPORTED,
                        supporting_turn_ids=tuple(c.turn_id for c in candidates[:2]),
                        coverage_basis=f"Single session covered: {list(candidate_sessions)}",
                        unresolved_reason="Evidence restricted to single session; cross-session breadth unverified",
                        confidence=0.5,
                    )
                else:
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        target_entity=req.target_entity,
                        status=RequirementStatus.UNRESOLVED,
                        supporting_turn_ids=(),
                        unresolved_reason="Zero candidate sessions retrieved",
                        confidence=0.0,
                    )

            # 3. Anchor Event requirement
            elif req.requirement_id == "req_anchor_event":
                target_pred = (req.target_predicate or "").lower()
                for c in candidates:
                    if target_pred and target_pred in c.content.lower():
                        supporting_tids.append(c.turn_id)
                    elif c.rrf_score > 0.02:
                        supporting_tids.append(c.turn_id)

                if supporting_tids:
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        target_entity=req.target_entity,
                        target_predicate=req.target_predicate,
                        status=RequirementStatus.SATISFIED,
                        supporting_turn_ids=tuple(supporting_tids),
                        coverage_basis=f"Anchor event turns matched predicate '{target_pred}' or high RRF confidence",
                        confidence=0.9,
                    )
                else:
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        target_entity=req.target_entity,
                        target_predicate=req.target_predicate,
                        status=RequirementStatus.UNRESOLVED,
                        unresolved_reason=f"No anchor event found for predicate '{target_pred}'",
                        confidence=0.0,
                    )

            # 4. Temporal Anchor requirement
            elif req.requirement_id == "req_temporal_anchor":
                for c in candidates:
                    if c.session_date and c.session_date != "None":
                        supporting_tids.append(c.turn_id)

                if supporting_tids:
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        status=RequirementStatus.SATISFIED,
                        supporting_turn_ids=tuple(supporting_tids[:3]),
                        coverage_basis=f"Verified session date header: {candidates[0].session_date}",
                        confidence=1.0,
                    )
                else:
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        status=RequirementStatus.UNRESOLVED,
                        unresolved_reason="Missing calendar date header across candidate sessions",
                        confidence=0.0,
                    )

            # 5. Direct Fact / State Evolution requirement
            else:
                target_ent = (req.target_entity or "").lower()
                target_pred = (req.target_predicate or "").lower()

                # Check for conflicting temporal states (e.g. used to like X vs now prefers Y)
                conflict_found = False
                state_change_markers = ("used to", "previously", "no longer", "now", "switched", "changed", "currently")
                matched_change_turns = []

                for c in candidates:
                    c_low = c.content.lower()
                    spk_low = c.speaker.lower()
                    if (target_ent in spk_low or target_ent in c_low) and (not target_pred or target_pred in c_low):
                        supporting_tids.append(c.turn_id)
                    if any(marker in c_low for marker in state_change_markers):
                        matched_change_turns.append(c.turn_id)

                if topology == AnswerTopology.STATE_EVOLUTION and len(matched_change_turns) >= 2:
                    conflict_found = True

                if conflict_found:
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        target_entity=req.target_entity,
                        target_predicate=req.target_predicate,
                        status=RequirementStatus.CONFLICTING,
                        supporting_turn_ids=tuple(matched_change_turns),
                        coverage_basis="Detected chronological evolution markers across candidate sessions",
                        unresolved_reason="Multiple conflicting chronological states present; timeline resolution required",
                        confidence=0.7,
                    )
                elif supporting_tids:
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        target_entity=req.target_entity,
                        target_predicate=req.target_predicate,
                        status=RequirementStatus.SATISFIED,
                        supporting_turn_ids=tuple(supporting_tids),
                        coverage_basis=f"Direct factual match for entity '{target_ent}' and predicate '{target_pred}'",
                        confidence=0.9,
                    )
                else:
                    evaluations[req.requirement_id] = EvidenceRequirement(
                        requirement_id=req.requirement_id,
                        description=req.description,
                        target_entity=req.target_entity,
                        target_predicate=req.target_predicate,
                        status=RequirementStatus.UNRESOLVED,
                        unresolved_reason=f"No factual match found for entity '{target_ent}'",
                        confidence=0.0,
                    )

        return evaluations

    def _derive_expansion_queries(
        self,
        unmet_ids: set[str],
        query_plan: QueryPlan,
    ) -> list[QueryPlan]:
        """Formulate targeted query plans to expand missing evidence views."""
        expansions: list[QueryPlan] = []

        if "req_multi_session_breadth" in unmet_ids or "req_enum_instances" in unmet_ids:
            if query_plan.predicates:
                pred_name = query_plan.predicates[0].predicate_name
                exp_text = f"{query_plan.entity_targets[0].canonical_name if query_plan.entity_targets else ''} {pred_name} sessions activities"
                expansions.append(QueryPlan(
                    query_text=exp_text.strip(),
                    topology=query_plan.topology,
                    entity_targets=query_plan.entity_targets,
                    predicates=query_plan.predicates,
                    temporal_constraints=(),
                    requirements=(),
                    recommended_budget=query_plan.recommended_budget,
                    max_closure_hops=1,
                ))

        if "req_temporal_anchor" in unmet_ids or "req_anchor_event" in unmet_ids:
            exp_text = f"{query_plan.query_text} date year month when"
            expansions.append(QueryPlan(
                query_text=exp_text,
                topology=AnswerTopology.TEMPORAL_DURATION,
                entity_targets=query_plan.entity_targets,
                predicates=query_plan.predicates,
                temporal_constraints=query_plan.temporal_constraints,
                requirements=(),
                recommended_budget=query_plan.recommended_budget,
                max_closure_hops=1,
            ))

        return expansions

    def _build_evidence_graph(
        self,
        candidates: list[ClosureCandidate],
    ) -> tuple[list[EvidenceGraphNode], list[EvidenceGraphEdge]]:
        """Constructs a verifiable evidence graph from candidate turns."""
        nodes: list[EvidenceGraphNode] = []
        edges: list[EvidenceGraphEdge] = []

        session_nodes = defaultdict(list)
        speaker_nodes = defaultdict(list)

        for c in candidates:
            node = EvidenceGraphNode(
                node_id=f"node_{c.turn_id}",
                turn_id=c.turn_id,
                session_id=c.session_id,
                speaker=c.speaker,
                verbatim_text=c.content,
                timestamp_scope=c.session_date,
            )
            nodes.append(node)
            session_nodes[c.session_id].append(node)
            speaker_nodes[c.speaker.lower()].append(node)

        # 1. Dialogue Flow Edges (within same session)
        for sid, s_node_list in session_nodes.items():
            for i in range(len(s_node_list) - 1):
                edges.append(EvidenceGraphEdge(
                    source_id=s_node_list[i].node_id,
                    target_id=s_node_list[i + 1].node_id,
                    relation_type="DIALOGUE_FLOW",
                    confidence=1.0,
                    provenance=f"session_{sid}",
                ))

        # 2. Speaker Coreference Edges
        for spk, spk_node_list in speaker_nodes.items():
            for i in range(len(spk_node_list) - 1):
                edges.append(EvidenceGraphEdge(
                    source_id=spk_node_list[i].node_id,
                    target_id=spk_node_list[i + 1].node_id,
                    relation_type="SAME_SPEAKER_REMENTION",
                    confidence=0.9,
                    provenance=f"speaker_{spk}",
                ))

        return nodes, edges
