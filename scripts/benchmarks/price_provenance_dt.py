"""M15.2 provenance-distinguishability pricing (read-only, 0 LLM).

Phase-1 finding recorded: the product ingest collapses every planted
turn's real time point (``ArenaTurn.day`` -> ``turn.timestamp``, a fixed
epoch plus data-backed days) into one wall-clock date, so every
provenance label the builder prints -- inside the NOTE chain and on the
source lines -- reads the same "2026-10-25" for BOTH reports.  The
benchmark compile prints distinct labels ("day 45" vs "day 48").
``facade.remember`` now accepts ``occurred_at`` (default None keeps the
frozen behavior) and the ``provenance_days`` ingest knob passes the
scenario's own timestamps through.

This probe prices that ONE change on the real product path at 0 LLM,
comparing two flag-on arms over the same 200 contexts:

  conflict     flag on, same-day labels (M15.1 end-state)
  dt           flag on, distinct labels via provenance_days

Measured per arm: per-category NOTE firing, contradiction-context label
distinctness, NOTE + source-line co-presence, token profile.  The claim
under test: the dt change touches ONLY the label surface -- same firing
profile, same co-presence, same token budget -- with labels on a
data-backed timeline.

Output: JSON artifact + stdout table.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from artificial_memory.research.benchmarks.arena import build_dataset  # noqa: E402
from am_current_player import AMRuntimeConflictPlayer  # noqa: E402

NOTE_RE = re.compile(
    r"NOTE: conflicting reports for project (?P<entity>[^ ]+)"
    r" \((?P<prop>[^)]+)\):"
    r" (?P<v1>[^ (]+) \((?P<label1>[^)]+)\) vs (?P<v2>[^ (]+)"
    r" \((?P<label2>[^)]+)\)")
SENTINEL = "NOTE: conflicting reports"


def _note_line_of(ctx_text: str) -> str | None:
    for line in ctx_text.splitlines():
        if line.startswith(SENTINEL):
            return line
    return None


def run_arm(name: str, config: dict) -> dict:
    dataset = build_dataset()
    player = AMRuntimeConflictPlayer(config, answerer=None)
    player.ingest(list(dataset.scenarios))
    topic = player._runtime.conversation_manager.get_or_create_topic("Arena")
    current = player._all_active_current(topic.id)
    builder = player._runtime.context_builder
    per_cat = defaultdict(lambda: {"n": 0, "note": 0})
    tokens = []
    label_distinct = 0
    contradiction_n = 0
    copresence = 0
    for question in dataset.questions:
        ctx = builder.build_context(
            question.question, topic.id, player.context_budget,
            current_memories=current)
        cat = question.category.value
        per_cat[cat]["n"] += 1
        tokens.append(len(ctx) // 4)
        note = _note_line_of(ctx)
        if note:
            per_cat[cat]["note"] += 1
        if cat == "contradiction_handling":
            contradiction_n += 1
            m = NOTE_RE.search(note or "")
            outside = "\n".join(
                line for line in ctx.splitlines()
                if not line.startswith(SENTINEL))
            if m and m.group("label1") != m.group("label2"):
                label_distinct += 1
            if m and m.group("v1") in outside \
                    and m.group("v2") in outside:
                copresence += 1
    result = {
        "per_cat": {k: dict(v) for k, v in per_cat.items()},
        "contradiction_n": contradiction_n,
        "label_distinct_notes": label_distinct,
        "copresence": copresence,
        "mean_ctx_tokens_est": int(sum(tokens) / len(tokens)),
        "max_ctx_tokens_est": max(tokens),
    }
    print(f"arm {name}: contradiction notes "
          f"{result['per_cat']['contradiction_handling']['note']}/20, "
          f"label-distinct {label_distinct}/{contradiction_n}, "
          f"copresence {copresence}/{contradiction_n}, "
          f"mean ctx ~{result['mean_ctx_tokens_est']}, "
          f"max ~{result['max_ctx_tokens_est']}")
    return result


if __name__ == "__main__":
    results = {
        "conflict": run_arm("conflict", {"conflict_surfacing": True}),
        "dt": run_arm("dt", {"conflict_surfacing": True,
                             "provenance_days": True}),
    }
    for arm in ("conflict", "dt"):
        fires = sum(v["note"] for k, v in
                    results[arm]["per_cat"].items()
                    if k != "contradiction_handling")
        print(f"arm {arm}: non-contradiction note fires "
              f"{fires}/180 "
              f"(off-path collateral same as M15.1 pricing if 0)")

    out = REPO / "benchmark_results" / "am_current" / "provenance_dt_pricing.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
