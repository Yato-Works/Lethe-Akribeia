"""
P2.2 Ablation Experiment: Evaluate CognitiveExecutionPlanner on LongMemEval Dev 250.

Fast, realistic pipeline:
  1. Fast Lexical / Keyword Filter selects Top Candidate Turns from Haystack (~20 turns).
  2. UniversalIRExtractor extracts StructuredIR from candidate turns only.
  3. CognitiveExecutionPlanner analyzes Question + Candidate Memory Topology.
  4. Evaluates alignment against LongMemEval true categories.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"
SPLIT_PATH = REPO / "benchmark_results" / "splits" / "lme_dev_qids.json"

sys.path.insert(0, str(REPO / "src"))

from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.core.ir.structured import StructuredIR

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "i", "you", "my", "your", "we", "our",
    "what", "where", "when", "which", "who", "whom", "how", "did", "do", "does", "have",
    "has", "had", "in", "on", "at", "to", "for", "with", "from", "by", "about", "of", "and",
    "or", "not", "so", "can", "could", "would", "should", "me", "that", "this", "it"
})


def fast_retrieve_candidate_turns(query: str, haystack_sessions: list, haystack_dates: list, haystack_sids: list, top_k: int = 25) -> list[tuple[str, str, str, str]]:
    """Fast initial retrieval of candidate dialogue turns matching query keywords."""
    q_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower())) - _STOP_WORDS
    if not q_words:
        q_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower()))

    scored_turns = []
    
    for s_idx, session in enumerate(haystack_sessions):
        s_date = haystack_dates[s_idx] if s_idx < len(haystack_dates) else ""
        sid = haystack_sids[s_idx] if s_idx < len(haystack_sids) else f"S{s_idx+1}"

        for turn in session:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            c_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", content.lower()))
            overlap = len(q_words & c_words)
            if overlap > 0:
                scored_turns.append((overlap, sid, s_date, role, content))

    scored_turns.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [(sid, s_date, role, content) for _, sid, s_date, role, content in scored_turns[:top_k]]
    return top_candidates


def main():
    with open(SPLIT_PATH, encoding="utf-8") as f:
        dev_qids = set(json.load(f))

    with open(DATASET_PATH, encoding="utf-8") as f:
        all_items = json.load(f)

    dev_items = [it for it in all_items if it["question_id"] in dev_qids]

    extractor = UniversalIRExtractor()
    planner = CognitiveExecutionPlanner()

    mode_to_category = {
        "direct_fact": "single-session-user",
        "assistant_recall": "single-session-assistant",
        "temporal": "temporal-reasoning",
        "aggregation": "multi-session",
        "state_evolution": "knowledge-update",
        "preference": "single-session-preference",
    }

    correct_dominant = 0
    total = len(dev_items)

    confusion = defaultdict(Counter)
    by_true_type = defaultdict(list)
    budgets = []
    triggers = Counter()

    print("=" * 80)
    print(f"  P2.2: MEMORY-CONDITIONED COGNITIVE EXECUTION PLANNER (DEV 250)")
    print(f"  Pipeline: Fast Lexical Retrieval (Top-25) -> Candidate IR -> Planner")
    print("=" * 80)

    t0 = time.perf_counter()

    for idx, item in enumerate(dev_items):
        qid = item["question_id"]
        true_type = item["question_type"]
        q = item["question"]
        q_date = item.get("question_date", "")

        top_turns = fast_retrieve_candidate_turns(
            q,
            item.get("haystack_sessions", []),
            item.get("haystack_dates", []),
            item.get("haystack_session_ids", []),
            top_k=25,
        )

        candidate_records: list[StructuredIR] = []
        for sid, s_date, role, content in top_turns:
            recs = extractor.extract(content, default_source=role)
            for r in recs:
                r.raw_content = f"[{sid} on {s_date}] {role}: {content}" if s_date else f"[{sid}] {role}: {content}"
                r.time_scope = s_date
                candidate_records.append(r)

        plan = planner.plan(q, candidate_records, reference_date=q_date)

        pred_cat = mode_to_category.get(plan.dominant_mode, "unknown")
        is_match = (pred_cat == true_type)

        if is_match:
            correct_dominant += 1

        confusion[true_type][pred_cat] += 1
        by_true_type[true_type].append(is_match)
        budgets.append(plan.recommended_budget)

        if plan.requires_chronos_diff:
            triggers["chronos"] += 1
        if plan.requires_event_aggregation:
            triggers["aggregation"] += 1
        if plan.requires_state_timeline:
            triggers["timeline"] += 1
        if plan.requires_profile_grounding:
            triggers["profile"] += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  Processed [{idx+1:03d}/{total:03d}] items... (elapsed: {time.perf_counter()-t0:.1f}s)", flush=True)

    elapsed = time.perf_counter() - t0
    acc = correct_dominant / total * 100
    mean_budget = sum(budgets) / total

    print("\n" + "=" * 80)
    print("  P2.2 RESULTS: MEMORY-CONDITIONED ROUTING ACCURACY")
    print("=" * 80)
    print(f"Overall Dominant-Mode Alignment: {acc:.2f}% ({correct_dominant}/{total}) [vs R0 Query-Only: 44.0%]")
    print(f"Mean Adaptive Budget: {mean_budget:.1f} tokens / query")
    print(f"Elapsed Time: {elapsed:.2f}s ({elapsed/total*1000:.1f} ms/query)")
    print(f"Co-processor Triggers: Chronos={triggers['chronos']}, Agg={triggers['aggregation']}, Timeline={triggers['timeline']}, Profile={triggers['profile']}")

    print("\n--- Alignment by Benchmark Category ---")
    print(f"{'Category':<28} | {'Total':<6} | {'Matched':<8} | {'Rate':<8}")
    print("-" * 55)
    for qt, matches in sorted(by_true_type.items()):
        c = sum(matches)
        tot = len(matches)
        print(f"{qt:<28} | {tot:<6} | {c:<8} | {c/tot*100:6.1f}%")

    print("\n--- Confusion Matrix (Row=True, Col=Predicted) ---")
    types = sorted(by_true_type.keys())
    header = f"{'True / Pred':<26} " + "".join(f"{t[:8]:>10}" for t in types)
    print(header)
    print("-" * len(header))
    for t_row in types:
        row_str = f"{t_row:<26} "
        for t_col in types:
            cnt = confusion[t_row][t_col]
            row_str += f"{cnt:>10}"
        print(row_str)


if __name__ == "__main__":
    main()
