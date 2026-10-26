"""M15.1 Product Context Composition Audit (read-only, 0 LLM).

Purpose (V4 4.28): explain WHY the flag-on product builder converts far
fewer contradiction questions than the benchmark compile player
(M15 E2E: 1/5 product vs 4/5 benchmark at 50q; 4.24) *without* touching
any code.  The audit walks the actual data-flow of
``EnhancedContextBuilder`` for every contradiction question and records,
per stage, where the planted evidence pair survives or drops:

  store        both planted values exist in the paged active corpus
  recall<L>    the builder's own recall outputs (per level) contain a line
               carrying each value
  alloc        an allocated ContextPart (beyond the NOTE) carries each value
  final        the finally selected context carries each value
  note         the reserved NOTE part survived into the final selection

Stage leakage boundaries: gold value strings come from the scorer-side
question records and are read ONLY here, in the audit (the same boundary
as the probe/pricing scripts); no player or runtime code touches them.
The audit is read-only: it calls the player's public-ish internals
(``_build_context_parts`` / ``_select_within_budget``) and the recall
engine (deterministic; the builder's own internal recall calls hit the
same cache afterwards).

Output: per-question rows + per-stage summary JSON artifact.  The first
run is the Phase-2 baseline; later runs of a single change must compare
against it.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from artificial_memory.core.models import RecallLevel  # noqa: E402
from artificial_memory.research.benchmarks.arena import build_dataset  # noqa: E402
from am_current_player import AMRuntimeConflictPlayer  # noqa: E402

TRY_AUDIT_TOKENS = 2000
NOTE_SENTINEL = "NOTE: conflicting reports"

# Mirrors recall_configs inside EnhancedContextBuilder._build_context_parts.
LEVELS = (
    RecallLevel.CURRENT_ONLY,
    RecallLevel.LONG_TERM_SUMMARY,
    RecallLevel.EPISODE,
    RecallLevel.LIGHT_COMPRESSION,
)


def _without_note_lines(text: str) -> str:
    """Strip the NOTE line so values are measured on source/other lines."""
    return "\n".join(
        line for line in text.splitlines()
        if not line.startswith(NOTE_SENTINEL)
    )


def stage_flags(build, player, question) -> dict:
    rt = player._runtime
    topic = rt.conversation_manager.get_or_create_topic("Arena")
    current = player._all_active_current(topic.id)
    query = question.question

    values = [str(v) for v in question.ground_truth[0]]

    store_blob = "\n".join(m.content for m in current)
    flags = {
        "stage_store_both": all(v in store_blob for v in values),
    }
    for level in LEVELS:
        mems, _ = build.recall_engine.recall(
            query, topic.id, level, max_tokens=TRY_AUDIT_TOKENS // 2)
        blob = "\n".join(m.content for m in mems)
        if any(v in blob for v in values):
            flags[f"stage_recall_{level.name}_evidence"] = True

    # Life through allocation and final selection (the real builder call).
    parts, budget, _stats = build._build_context_parts(
        query, topic.id, TRY_AUDIT_TOKENS, current)
    # M15.1: the conflict part is note + source assertion lines; measure
    # the values on its LINES (non-note), not on the whole note chain.
    alloc_evidence = {v: False for v in values}
    alloc_note = False
    for p in parts:
        if NOTE_SENTINEL in p.content:
            alloc_note = True
            blob = _without_note_lines(p.content)
        else:
            blob = p.content
        for v in values:
            alloc_evidence[v] = alloc_evidence[v] or v in blob
    flags["stage_alloc_evidence"] = all(alloc_evidence.values())
    flags["stage_alloc_note"] = alloc_note

    selected, _, _ = build._select_within_budget(parts, TRY_AUDIT_TOKENS)
    final = "\n\n---\n\n".join(selected)
    evidence_final = [
        v for v in values
        if any(v in _without_note_lines(s) for s in selected)
    ]
    flags["final_evidence_values"] = evidence_final
    flags["stage_final_evidence_both"] = len(evidence_final) == len(values)
    flags["stage_final_note"] = NOTE_SENTINEL in final
    flags["final_source_lines"] = sum(
        1 for s in selected if NOTE_SENTINEL in s
        for line in s.splitlines() if line.startswith("- ")
    )
    flags["ctx_tokens_est"] = len(final) // 4
    flags["parts_allocated"] = len(parts)
    flags["parts_selected"] = len(selected)
    flags["tier_used"] = budget.get_usage_stats()["used"]
    return flags


def main() -> None:
    dataset = build_dataset()
    player = AMRuntimeConflictPlayer(
        {"lexical_recall": True, "conflict_surfacing": True}, answerer=None)
    player.ingest(list(dataset.scenarios))
    build = player._runtime.context_builder

    rows = []
    per_stage = defaultdict(lambda: {"hit": 0, "n": 0})
    for question in dataset.questions:
        if question.category.value != "contradiction_handling":
            continue
        flags = stage_flags(build, player, question)
        flags["question_id"] = question.question_id
        rows.append(flags)
        for key, val in flags.items():
            if key.startswith("stage_") and val:
                per_stage[key]["hit"] += 1
            if key.startswith("stage_"):
                per_stage[key]["n"] += 1

    n = len(rows)
    summary = {
        "n_questions": n,
        "audit_tokens": TRY_AUDIT_TOKENS,
        "stages": {
            key: {
                "survived": d["hit"],
                "of": d["n"],
                "rate": round(d["hit"] / max(d["n"], 1), 3),
            }
            for key, d in sorted(per_stage.items())
        },
        "co_presence_rate": round(
            sum(1 for r in rows
                if r["stage_final_evidence_both"] and r["stage_final_note"]) / n, 3),
        "mean_ctx_tokens_est": round(sum(r["ctx_tokens_est"] for r in rows) / n, 1),
        "rows": rows,
    }

    out = REPO / "benchmark_results" / "am_current" / "context_composition_audit.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"M15.1 composition audit -- {n} contradiction questions, 0 LLM")
    print(f"{'stage':<44}{'survived':>10}  rate")
    print("-" * 66)
    for key, d in sorted(per_stage.items()):
        print(f"{key:<44}{d['hit']:>4}/{d['n']:<5}  "
              f"{d['hit'] / max(d['n'], 1):.1%}")
    note_both = sum(1 for r in rows
                    if r["stage_final_evidence_both"] and r["stage_final_note"])
    print(f"\nNOTE + both-evidence co-presence in final context: {note_both}/{n}")
    print(f"mean final ctx tokens: {summary['mean_ctx_tokens_est']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
