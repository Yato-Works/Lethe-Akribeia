"""R6 Phase-1 feature extraction (deterministic, LLM-free).

Feature groups
  - plan demands       : execution planner capability demand vector + triggers
  - topology           : structural properties of retrieved memory records
  - candidate          : number of candidate turns / sessions retained
  - conflict           : competing / superseded values (contradiction proxy)
  - cert               : deterministic co-processor certificate presence
  - temporal complexity: date variance + chronos trigger + temporal demand
  - evidence coverage  : query-word coverage of selected evidence
  - anchor distribution: top-18 anchor session spread + top1 anchor score
  - budget utilization : tokens actually selected / budget limit
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SEP = re.compile(r"[\s,;:\[\]{}]+")


@dataclass
class R6Features:
    direct_fact_demand: float
    temporal_demand: float
    aggregation_demand: float
    state_evolution_demand: float
    preference_demand: float
    assistant_recall_demand: float
    dominant_mode: str
    recommended_budget: int
    requires_chronos_diff: bool
    requires_event_aggregation: bool
    requires_state_timeline: bool
    requires_profile_grounding: bool
    total_records: int
    session_count: int
    date_count: int
    has_date_variance: bool
    has_superseded_or_migrated: bool
    competing_values_count: int
    has_assistant_records: bool
    has_user_records: bool
    candidate_count: int
    candidate_sessions: int
    conflict: bool
    cert_present: bool
    temporal_complexity: int
    evidence_coverage: float
    anchor_session_count: int
    top1_anchor_score: float
    budget_util: float
    question_type: str
    is_aggregation: bool
    correct: bool
    reader_model: str
    predicted: str


def _sep_split(q: str) -> list[str]:
    return _SEP.split(q.lower())


def _query_words(q: str) -> set[str]:
    return set(_sep_split(q))


def _coverage(compiled, q: str) -> float:
    if compiled is None:
        return 0.0
    words = _query_words(q)
    if not words:
        return 0.0
    return float(compiled.utility_score) / len(words)


def _session_ids(records) -> set[str]:
    sids: set[str] = set()
    for r in records:
        m = UniversalIRExtractor._SESSION_EXTRACT.search(r.raw_content or "")
        if m:
            sids.add(m.group(1))
    return sids


def _top_anchor_info(check_scores: dict, max_sessions: int = 18) -> tuple[int, float]:
    ids = list(dict.fromkeys(check_scores.keys()))[:max_sessions]
    return len(ids), max(check_scores.values(), default=0.0)

def extract_r6_features(
    item,
    q: str,
    plan,
    bundle,
    compiled,
    candidate_turns,
    candidate_records,
    anchor_scores: dict,
    is_aggregation: bool,
    correct: bool,
    reader_model: str = "",
    predicted: str = "",
):
    """Build the deterministic feature row for one question.

    All inputs are cheap pipeline objects; no LLM is involved.
    """
    t = plan.topology
    ret = R6Features(
        # Plan / demand vector
        direct_fact_demand=plan.direct_fact_demand,
        temporal_demand=plan.temporal_demand,
        aggregation_demand=plan.aggregation_demand,
        state_evolution_demand=plan.state_evolution_demand,
        preference_demand=plan.preference_demand,
        assistant_recall_demand=plan.assistant_recall_demand,
        dominant_mode=plan.dominant_mode,
        recommended_budget=plan.recommended_budget,
        requires_chronos_diff=plan.requires_chronos_diff,
        requires_event_aggregation=plan.requires_event_aggregation,
        requires_state_timeline=plan.requires_state_timeline,
        requires_profile_grounding=plan.requires_profile_grounding,
        # Topology
        total_records=t.total_records,
        session_count=t.session_count,
        date_count=t.date_count,
        has_date_variance=t.has_date_variance,
        has_superseded_or_migrated=t.has_superseded_or_migrated,
        competing_values_count=t.competing_values_count,
        has_assistant_records=t.has_assistant_records,
        has_user_records=t.has_user_records,
        # Candidate count
        candidate_count=len(candidate_turns),
        candidate_sessions=len({t[0] for t in candidate_turns}),
        # Conflict
        conflict=(t.competing_values_count > 0) or bool(t.has_superseded_or_migrated),
        # Cert
        cert_present=bool(bundle.deterministic_answer_override),
        # Temporal complexity (sum of binary flags)
        temporal_complexity=int(t.has_date_variance)
        + int(plan.requires_chronos_diff)
        + int(plan.temporal_demand >= 0.7),
        # Evidence coverage
        evidence_coverage=_coverage(compiled, q),
        # Anchor / episodic distribution
        anchor_session_count=_top_anchor_info(anchor_scores)[0],
        top1_anchor_score=_top_anchor_info(anchor_scores)[1],
        # Budget utilization
        budget_util=(compiled.total_tokens / compiled.budget_limit) if compiled and compiled.budget_limit else 0.0,
        # Question-level
        question_type=item.get("question_type", ""),
        is_aggregation=is_aggregation,
        correct=correct,
        reader_model=reader_model,
        predicted=predicted,
    )
    return ret


def feature_dict(f: R6Features) -> dict:
    """Flatten -> JSON-serializable dict."""
    return {
        "direct_fact_demand": f.direct_fact_demand,
        "temporal_demand": f.temporal_demand,
        "aggregation_demand": f.aggregation_demand,
        "state_evolution_demand": f.state_evolution_demand,
        "preference_demand": f.preference_demand,
        "assistant_recall_demand": f.assistant_recall_demand,
        "dominant_mode": f.dominant_mode,
        "recommended_budget": f.recommended_budget,
        "requires_chronos_diff": f.requires_chronos_diff,
        "requires_event_aggregation": f.requires_event_aggregation,
        "requires_state_timeline": f.requires_state_timeline,
        "requires_profile_grounding": f.requires_profile_grounding,
        "total_records": f.total_records,
        "session_count": f.session_count,
        "date_count": f.date_count,
        "has_date_variance": f.has_date_variance,
        "has_superseded_or_migrated": f.has_superseded_or_migrated,
        "competing_values_count": f.competing_values_count,
        "has_assistant_records": f.has_assistant_records,
        "has_user_records": f.has_user_records,
        "candidate_count": f.candidate_count,
        "candidate_sessions": f.candidate_sessions,
        "conflict": f.conflict,
        "cert_present": f.cert_present,
        "temporal_complexity": f.temporal_complexity,
        "evidence_coverage": round(f.evidence_coverage, 4),
        "anchor_session_count": f.anchor_session_count,
        "top1_anchor_score": round(f.top1_anchor_score, 4),
        "budget_util": round(f.budget_util, 4),
        "question_type": f.question_type,
        "is_aggregation": f.is_aggregation,
        "correct": f.correct,
        "reader_model": f.reader_model,
        "predicted": f.predicted,
    }
