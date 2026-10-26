"""M15.3 Fill-Mass baseline audit (read-only, 0 LLM).

Phase-1 instrumentation for the fill-mass ablation.  The prior audits
proved the planted evidence can be fully co-present with a distinct
provenance and the reader still refuses; the remaining explain-candidate
is the mass of query-unrelated recency fill around the annotation.  This
audit records, for every contradiction question's REAL product context:

  fill_tok     tokens in `current:semantic` parts (the recency fill)
  note_tok     tokens in the reserved conflict part (NOTE + source lines)
  note_pos     character offset of the NOTE as a fraction of the context
  note_line    which line index the NOTE lands on
  around       the two lines directly surrounding the NOTE

plus, as the comparability anchor for the coming ablation, the exact
NOTE-line text (its content must remain identical when the fill is
removed).

Output: JSON artifact + stdout summary.
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


def main() -> None:
    dataset = build_dataset()
    player = AMRuntimeConflictPlayer({"conflict_surfacing": True}, answerer=None)
    player.ingest(list(dataset.scenarios))
    topic = player._runtime.conversation_manager.get_or_create_topic("Arena")
    current = player._all_active_current(topic.id)
    builder = player._runtime.context_builder

    rows = []
    for question in dataset.questions:
        cat = question.category.value
        if cat != "contradiction_handling":
            continue
        ctx = builder.build_context(
            question.question, topic.id, player.context_budget,
            current_memories=current)
        lines = ctx.splitlines()
        note_idx = next((i for i, line in enumerate(lines)
                         if line.startswith(SENTINEL)), None)
        note_line = lines[note_idx] if note_idx is not None else None
        rows.append({
            "question_id": question.question_id,
            "ctx_tokens_est": len(ctx) // 4,
            "fill_tok": sum(len(line) for line in lines) // 4 - (len(note_line) // 4 if note_line else 0),
            "note_tok": len(note_line) // 4 if note_line else 0,
            "note_pos_frac": round(len("\n".join(lines[:note_idx])) / len(ctx), 3)
                if note_idx is not None else None,
            "note_line_idx": note_idx,
            "lines_total": len(lines),
            "around": [lines[note_idx - 1][:90], lines[note_idx + 1][:90]]
                if note_idx is not None and 0 < note_idx < len(lines) - 1 else None,
            "note_text": note_line,
        })
        del cat, lines, note_idx, note_line

    n = len(rows)
    total = sum(r["ctx_tokens_est"] for r in rows) / n
    fill = sum(r["fill_tok"] for r in rows) / n
    note = sum(r["note_tok"] for r in rows) / n
    pos = sum(r["note_pos_frac"] for r in rows) / n
    summary = {
        "n_questions": n,
        "mean_ctx_tokens_est": round(total, 1),
        "mean_fill_tokens_est": round(fill, 1),
        "mean_note_tokens_est": round(note, 1),
        "fill_share": round(fill / total, 3),
        "mean_note_position_frac": round(pos, 3),
        "note_reaches_end_in": sum(
            1 for r in rows
            if r["note_line_idx"] is not None
            and r["note_line_idx"] >= r["lines_total"] - 2),
        "rows": rows,
    }
    out = REPO / "benchmark_results" / "am_current" / "fill_mass_baseline.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("M15.3 fill-mass baseline -- 20 contradiction questions, 0 LLM")
    print(f"  mean ctx      ~{summary['mean_ctx_tokens_est']} tokens")
    print(f"  fill share    {summary['fill_share']:.0%} "
          f"(~{summary['mean_fill_tokens_est']} tokens of recency fill)")
    print(f"  note part     ~{summary['mean_note_tokens_est']} tokens")
    print(f"  note position ~{summary['mean_note_position_frac']:.0%} "
          f"into the context; at/near the very end in "
          f"{summary['note_reaches_end_in']}/{n} questions")
    sample = rows[0]["around"]
    print("  lines around the note (question 00):")
    for line in sample or []:
        print("    ", repr(line))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
