"""0-LLM pricing of the flag-on PRODUCT path (V4 4.27 / M14).

The M13 benchmark player converts contradiction questions through the R7
compile; the shipped runtime answers through its own
``EnhancedContextBuilder`` (the exact path ``LLMManager.chat`` takes).
Before spending reader runs, this prices whether the conflict-surfacing
annotation fires on the product path's candidate set at all: for each of
the 200 compiled product contexts, how often a conflict NOTE appears,
per question category, with both players stubbed (no reader calls).

Output: JSON artifact + stdout table.
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


def main() -> None:
    dataset = build_dataset()
    results = {}
    for name, flag in (("off", False), ("on", True)):
        player = AMRuntimeConflictPlayer(
            {"lexical_recall": True, "conflict_surfacing": flag},
            answerer=None)
        player.ingest(list(dataset.scenarios))
        per_cat = defaultdict(lambda: {"n": 0, "note": 0})
        token_rows = []
        for question in dataset.questions:
            cat = question.category.value
            answer = player._answer_with_budget(
                question, player.context_budget)
            notes = answer.metadata.get("conflict_notes", 0)
            per_cat[cat]["n"] += 1
            per_cat[cat]["note"] += int(notes > 0)
            token_rows.append(answer.context_tokens_used)
        results[name] = {
            "per_cat": dict(per_cat),
            "max_ctx_tokens_est": max(token_rows),
            "mean_ctx_tokens_est": int(sum(token_rows) / len(token_rows)),
        }
        del player

    print("product path (flag off / on), 0-LLM, 200 contexts:")
    header = f"{'category':<30}{'flag off':>12}{'flag on':>12}"
    print(header)
    print("-" * len(header))
    cats = sorted(set(results["on"]["per_cat"]) | set(results["off"]["per_cat"]))
    for cat in cats:
        off = results["off"]["per_cat"].get(cat, {"n": 0, "note": 0})
        on = results["on"]["per_cat"].get(cat, {"n": 0, "note": 0})
        if off["n"] == on["n"] and off["n"]:
            print(f"{cat:<30}{off['note']}/{off['n']:>7}"
                  f"{on['note']}/{on['n']:>8}")
    for name, r in results.items():
        print(f"flag {name}: mean ctx tokens ~{r['mean_ctx_tokens_est']}, "
              f"max ~{r['max_ctx_tokens_est']}")

    out = REPO / "benchmark_results" / "am_current" / "runtime_conflict_pricing.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
