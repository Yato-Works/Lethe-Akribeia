"""M15.3 fill-ablation pricing (read-only, 0 LLM).

Baseline probe recorded the candidate mechanism: on contradiction
questions the query-unrelated recency fill is ~92% of a ~1270-token
context and the reserved conflict part sits at ~93% position.  The ONE
change: when the annotation fires, the current-state fill is skipped
(``conflict_fill_ablation``), leaving the context as the conflict part
only — which already carries the NOTE and the pair's source lines.

This prices the change over the 200 real product contexts, comparing
the control arm (flag on, in-her-M15.1 composition) with the bare arm:

  contradiction questions: the conflict part must be byte-identical
  between arms (values, provenance, ordering) — only the fill gone;
  co-presence must stay 20/20.
  every other question: contexts BYTE-IDENTICAL between arms — the
  change cannot touch anything but fired contexts (0/180 firing).

Output: JSON artifact + stdout.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from artificial_memory.research.benchmarks.arena import build_dataset  # noqa: E402
from am_current_player import AMRuntimeConflictPlayer  # noqa: E402

SENTINEL = "NOTE: conflicting reports"


def collect(name: str, config: dict) -> dict:
    dataset = build_dataset()
    player = AMRuntimeConflictPlayer(config, answerer=None)
    player.ingest(list(dataset.scenarios))
    topic = player._runtime.conversation_manager.get_or_create_topic("Arena")
    current = player._all_active_current(topic.id)
    builder = player._runtime.context_builder
    per_cat = defaultdict(lambda: {"n": 0, "note": 0})
    ctxs = {}
    tokens = []
    for question in dataset.questions:
        cat = question.category.value
        ctx = builder.build_context(
            question.question, topic.id, player.context_budget,
            current_memories=current)
        ctxs[question.question_id] = ctx
        tokens.append(len(ctx) // 4)
        has_note = any(line.startswith(SENTINEL)
                       for line in ctx.splitlines())
        per_cat[cat]["n"] += 1
        if has_note:
            per_cat[cat]["note"] += 1
    result = {
        "per_cat": {k: dict(v) for k, v in per_cat.items()},
        "mean_ctx_tokens_est": int(sum(tokens) / len(tokens)),
        "max_ctx_tokens_est": max(tokens),
        "contradiction_tokens": [
            len(ctxs[f"contradiction_handling-{i:02d}"]) // 4
            for i in range(20)
        ],
        "ctxs": ctxs,
    }
    print(f"arm {name}: mean ~{result['mean_ctx_tokens_est']}, "
          f"max ~{result['max_ctx_tokens_est']}, "
          f"contradiction tokens {result['contradiction_tokens'][:5]}...")
    return result


if __name__ == "__main__":
    control = collect("conflict", {"conflict_surfacing": True})
    bare = collect(
        "bare", {"conflict_surfacing": True, "conflict_fill_ablation": True})

    # fired contexts: the conflict part must be byte-identical
    note_part_equal = 0
    copresence = 0
    values_by_q = {}
    for i in range(20):
        qid = f"contradiction_handling-{i:02d}"
        ctx_a, ctx_b = control["ctxs"][qid], bare["ctxs"][qid]
        note_a = next(line for line in ctx_a.splitlines()
                      if line.startswith(SENTINEL))
        note_b = next(line for line in ctx_b.splitlines()
                      if line.startswith(SENTINEL))
        if note_a == note_b:
            note_part_equal += 1
        part_b = "\n".join(line for line in ctx_b.splitlines()
                          if line.startswith(SENTINEL)
                          or line.startswith("- "))
        block_a = "\n\n---\n\n".join(
            part for part in ctx_a.split("\n\n---\n\n")
            if SENTINEL in part)
        if block_a == part_b:
            copresence += 1

    # identical spec: every non-fired context byte-identical
    identical_others = sum(
        1 for qid, ctx in control["ctxs"].items()
        if not qid.startswith("contradiction_handling")
        and bare["ctxs"][qid] == ctx)

    print(f"conflict part byte-equal: {note_part_equal}/20 | "
          f"reserved part identical (note+lines): {copresence}/20 | "
          f"non-fired contexts byte-identical: {identical_others}/180")

    for blob in (control, bare):
        blob.pop("ctxs")
    results = {"conflict": control, "bare": bare,
               "non_fired_byte_identical": identical_others}
    out = REPO / "benchmark_results" / "am_current" / "fill_ablation_pricing.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
