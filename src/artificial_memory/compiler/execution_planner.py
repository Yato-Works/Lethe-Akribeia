"""
Cognitive Execution Planner: Memory-Conditioned Routing & Demand Vector for Lethe-Akribeia V4.

Analyzes BOTH the question semantics AND the topology of retrieved memory records
to produce a continuous Cognitive Demand Vector and an Adaptive Token Budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from artificial_memory.core.ir.structured import IRRelation, IRStatus, StructuredIR


@dataclass(frozen=True)
class MemoryTopology:
    """Structural properties of candidate memory records retrieved for a question."""
    total_records: int
    unique_sessions: set[str]
    unique_dates: set[str]
    session_count: int
    date_count: int
    has_date_variance: bool
    has_superseded_or_migrated: bool
    competing_values_count: int
    has_assistant_records: bool
    has_user_records: bool
    # candidate_tokens was removed: the pool-scaling that consumed it measured
    # as dead code (0 of 1,986 LoCoMo pools reach the point where the ratio
    # engages), and no caller ever set it.  See the note at the removal site.


@dataclass(frozen=True)
class ExecutionPlan:
    """Continuous capability demand vector and execution strategy."""
    # Capability Demands in [0.0, 1.0]
    direct_fact_demand: float
    temporal_demand: float
    aggregation_demand: float
    state_evolution_demand: float
    preference_demand: float
    assistant_recall_demand: float

    # Adaptive Budget in tokens
    recommended_budget: int

    # Deterministic Co-processor Triggers
    requires_chronos_diff: bool
    requires_event_aggregation: bool
    requires_state_timeline: bool
    requires_profile_grounding: bool

    # Diagnostics
    dominant_mode: str
    topology: MemoryTopology


class CognitiveExecutionPlanner:
    """Memory-Conditioned Cognitive Execution Planner."""

    # Question regex patterns
    _RE_RELATIVE_TIME = re.compile(
        r"\b(how many (days|weeks|months|years) ago|how long ago|since (i|we|my)|passed between|how much time passed)\b",
        re.I
    )
    _RE_TEMPORAL_ORDER = re.compile(
        r"\b(which|who)\b.*?\b(first|most recently|last|earlier|later)\b|"
        r"\b(order from first to last|earlier or later|before or after|which happened first)\b",
        re.I
    )
    _RE_DATE_QUERY = re.compile(
        r"\b(what date|which day|on what date|what time of year|which year|which month)\b",
        re.I
    )
    _RE_COUNTING = re.compile(
        r"\b(how many (different )?(\w+(?:\s+\w+){0,3})\s+(did i|have i|do i|are there|were|can i count|did we)|total number of|how many times did i)\b",
        re.I
    )
    _RE_ARITHMETIC_DIFF = re.compile(
        r"\b(difference in (price|cost|time|age|speed|weight)|how much (more|less|faster|slower|cheaper|expensive)|compare the)\b",
        re.I
    )
    _RE_TOTAL_SUM = re.compile(
        r"\b(total (cost|price|amount|time|expense|distance|spending|hours)|in total|combined (cost|price|total))\b",
        re.I
    )
    _RE_STATE_TRANSITION = re.compile(
        r"\b(currently|right now|at present|previously|used to|no longer|switched to|changed my|updated my|former stance|prior to)\b",
        re.I
    )
    _RE_RECOMMENDATION = re.compile(
        r"\b(can you (recommend|suggest|give me)|what should i (try|get|buy|visit|watch|read|serve|make|cook)|"
        r"recommend (some|a|an)|suggest (some|a|an)|ideas (for|on)|options for|"
        r"any tips|any advice|any suggestions|how can i|tips for|advice on|stuck with|"
        r"what would you recommend|do you have any (tips|ideas|suggestions|advice))\b",
        re.I
    )
    _RE_PREFERENCE = re.compile(
        r"\b(would i prefer|align with my (taste|preference|style)|my favorite type of)\b",
        re.I
    )
    _RE_ASSISTANT_REF = re.compile(
        r"\b(you (told|recommended|suggested|mentioned|shared|advised|gave|said|wrote|created|provided|listed|crafted|composed|designed|specified)|"
        r"(in|from|about|through|during) our (previous|earlier|past) (conversation|chat|discussion|interaction)|"
        r"our (previous|earlier|past) (conversation|chat|discussion|interaction)|"
        r"what was that [a-z]+ you (told|recommended|mentioned|suggested)|"
        r"can you remind me|remind me what|remind me how|remind me of|remind me who|remind me which|remind me where|remind me\b|"
        r"trying to (recall|remember)|"
        r"did you say|did you tell|did you recommend|did you suggest)\b",
        re.I
    )
    _RE_ORDINAL = re.compile(
        r"\b(\d+(st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final)\s+[a-z]+\b",
        re.I
    )

    _SESSION_EXTRACT = re.compile(r"\[([a-zA-Z0-9_-]+)(?:\s+on\s+([^\]]+))?\]")

    def analyze_topology(self, records: Sequence[StructuredIR]) -> MemoryTopology:
        """Inspect the structural distribution of candidate memory records."""
        sessions = set()
        dates = set()
        has_superseded = False
        has_assistant = False
        has_user = False

        prop_to_values: dict[str, set[str]] = {}

        for r in records:
            # Session & Date extraction
            m = self._SESSION_EXTRACT.search(r.raw_content)
            if m:
                sid = m.group(1)
                sessions.add(sid)
                if m.group(2):
                    dates.add(m.group(2).strip())
            elif r.time_scope:
                dates.add(r.time_scope)

            # Speaker
            src = getattr(r, "source", "user").lower()
            if "assistant" in src or "assistant:" in r.raw_content.lower():
                has_assistant = True
            else:
                has_user = True

            # Lifecycle status & migrations
            if r.status == IRStatus.SUPERSEDED or r.relation == IRRelation.MIGRATED or r.old_value:
                has_superseded = True

            # Property conflicts
            prop_key = f"{r.entity}::{r.property}".lower().strip(":")
            if prop_key and r.value:
                prop_to_values.setdefault(prop_key, set()).add(r.value.lower().strip())

        competing_props = sum(1 for vals in prop_to_values.values() if len(vals) > 1)

        return MemoryTopology(
            total_records=len(records),
            unique_sessions=sessions,
            unique_dates=dates,
            session_count=len(sessions),
            date_count=len(dates),
            has_date_variance=len(dates) > 1,
            has_superseded_or_migrated=has_superseded,
            competing_values_count=competing_props,
            has_assistant_records=has_assistant,
            has_user_records=has_user,
        )

    def plan(self, query: str, records: Sequence[StructuredIR], reference_date: str | None = None) -> ExecutionPlan:
        """Condition execution plan on both query features and retrieved memory topology."""
        topology = self.analyze_topology(records)
        q_norm = re.sub(r"\s+", " ", query).strip()

        # 1. Base Query Signals [0.0, 1.0]
        q_has_relative_time = bool(self._RE_RELATIVE_TIME.search(q_norm))
        q_has_temporal_order = bool(self._RE_TEMPORAL_ORDER.search(q_norm))
        q_has_date_query = bool(self._RE_DATE_QUERY.search(q_norm))
        # Specific event/trip constraint or collection progress update is NOT an aggregation count
        is_single_event_spec = bool(re.search(r"\b(before the \d+|earlier (trip|visit|session|event|flight)|on the \d+/\d+)\b", q_norm, re.I))
        is_collection_progress = bool(re.search(r"\b(in my collection|to my collection|my collection)\b", q_norm, re.I))

        q_has_counting = bool(self._RE_COUNTING.search(q_norm)) and not is_single_event_spec and not is_collection_progress
        q_has_diff = bool(self._RE_ARITHMETIC_DIFF.search(q_norm))
        q_has_sum = bool(self._RE_TOTAL_SUM.search(q_norm))
        q_has_state_trans = bool(self._RE_STATE_TRANSITION.search(q_norm)) or is_collection_progress
        q_has_rec = bool(self._RE_RECOMMENDATION.search(q_norm))
        q_has_pref = bool(self._RE_PREFERENCE.search(q_norm))
        q_has_asst_ref = bool(self._RE_ASSISTANT_REF.search(q_norm))
        q_has_ordinal = bool(self._RE_ORDINAL.search(q_norm))

        # 2. Demand Vector Synthesis (Memory-Conditioned)
        
        # Assistant recall: strong query signal + assistant records in memory
        assistant_demand = 0.0
        if q_has_asst_ref or q_has_ordinal:
            assistant_demand = 0.95 if topology.has_assistant_records else 0.80
        elif topology.has_assistant_records and not topology.has_user_records:
            assistant_demand = 0.60

        # Preference: recommendations / tastes
        preference_demand = 0.0
        if q_has_rec or q_has_pref:
            preference_demand = 0.95

        # Temporal demand: questions about time differences or ordering + date variance
        temporal_demand = 0.0
        if q_has_relative_time or q_has_temporal_order or q_has_date_query:
            temporal_demand = 0.90
            if topology.has_date_variance or reference_date:
                temporal_demand = 0.98

        # State evolution: state transitions + evidence of superseded values across dates
        state_demand = 0.0
        if q_has_state_trans:
            state_demand = 0.75
            if topology.competing_values_count > 0 or topology.has_superseded_or_migrated:
                state_demand = 0.95
        elif topology.has_superseded_or_migrated or topology.competing_values_count > 0:
            if not q_has_counting and not q_has_relative_time:
                state_demand = 0.70

        # Aggregation demand: counting/sums/diffs across multiple sessions
        agg_demand = 0.0
        if q_has_counting or q_has_diff or q_has_sum:
            # Crucial Memory Condition:
            # If records only come from 1 session, multi-session aggregation is unneeded; it's a local fact.
            if topology.session_count > 1:
                agg_demand = 0.95
            elif topology.session_count == 1:
                agg_demand = 0.40  # Local count, single fact
            else:
                agg_demand = 0.80

        # Direct fact demand: inversely related to complex co-processor needs
        max_complex = max(temporal_demand, agg_demand, state_demand, preference_demand, assistant_demand)
        direct_fact_demand = max(0.1, 1.0 - max_complex)

        # 3. Determine Dominant Mode
        modes = [
            ("assistant_recall", assistant_demand),
            ("preference", preference_demand),
            ("temporal", temporal_demand),
            ("state_evolution", state_demand),
            ("aggregation", agg_demand),
            ("direct_fact", direct_fact_demand),
        ]
        dominant_mode = max(modes, key=lambda x: x[1])[0]

        # 4. Adaptive Token Budget
        # Fact: 256-384 tok
        # State: 384-512 tok
        # Temporal: 384-600 tok
        # Aggregation: 512-800 tok
        # Assistant: 512-800 tok
        # Preference: 400-600 tok
        budget_map = {
            "direct_fact": 550,
            "temporal": 550,
            "state_evolution": 600,
            "preference": 600,
            "assistant_recall": 750,
            "aggregation": 850,
        }
        recommended_budget = budget_map.get(dominant_mode, 600)
        if (agg_demand >= 0.5 or dominant_mode == "aggregation") and topology.session_count >= 3:
            recommended_budget = max(recommended_budget, 900)

        # CANDIDATE-POOL SCALING -- REMOVED, measured as dead code.
        #
        # This block scaled the per-mode budget by topology.candidate_tokens.
        # Measured over all 1,986 LoCoMo questions: 0 pools reach the point
        # where the ratio engages (floor 1400 / ratio 0.12 = 11667 tokens;
        # real pools run 30-2417, median 1611), so it produced the constant
        # 1400 for every question and moved nothing.
        #
        # It was also unreachable twice over: no caller sets candidate_tokens
        # (the LoCoMo harness calls _scaled_budget() in the harness itself and
        # never populates the field), and an earlier version read
        # topology.total_records, which callers pass as an empty list.
        #
        # The +3.5pt credited to this in 58b1c0b came from the OTHER half of
        # that commit -- _scaled_budget() raising the floor in the harness --
        # not from this block.  Recorded so the attribution is not repeated.

        # 5. Co-processor Triggers
        requires_chronos = temporal_demand >= 0.7
        requires_aggregator = agg_demand >= 0.7
        requires_state_timeline = state_demand >= 0.7
        requires_profile_grounding = preference_demand >= 0.7

        return ExecutionPlan(
            direct_fact_demand=round(direct_fact_demand, 3),
            temporal_demand=round(temporal_demand, 3),
            aggregation_demand=round(agg_demand, 3),
            state_evolution_demand=round(state_demand, 3),
            preference_demand=round(preference_demand, 3),
            assistant_recall_demand=round(assistant_demand, 3),
            recommended_budget=recommended_budget,
            requires_chronos_diff=requires_chronos,
            requires_event_aggregation=requires_aggregator,
            requires_state_timeline=requires_state_timeline,
            requires_profile_grounding=requires_profile_grounding,
            dominant_mode=dominant_mode,
            topology=topology,
        )
