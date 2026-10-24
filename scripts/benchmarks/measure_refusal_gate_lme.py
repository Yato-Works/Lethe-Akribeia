"""Cross-benchmark guard: does the answer-provenance gate touch LongMemEval?

The gate ships on LoCoMo (+4.33pt dev).  The program requires that it does not
regress the other benchmark.  This script prices BOTH modes on LongMemEval dev,
with the frozen context cache and the same deterministic scorer that produced the
published LME number:

  name-only (default)  the question must name a speaker
  roles                a first-person question is attributed to the user, which
                       makes the gate applicable to user/assistant corpora

Measured answer, which is why `allow_roles` defaults to False:

  name-only   parses 128/128 contexts, fires 0, accuracy 82.03% -> 82.03%
  roles       fires 2, accuracy 82.03% -> 80.47%  (-1.56pt)

The assistant legitimately restates the user's facts, so "the answer's wording
comes from the other role" is not evidence of misattribution there.  The guard is
therefore a REAL negative, not a vacuous pass.

Usage
    python scripts/lme_context_cache.py --limit 250 \
        --out benchmark_results/lme_context_cache.jsonl
    python scripts/benchmarks/measure_refusal_gate_lme.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory.research.benchmarks.external.longmemeval_adapter import (  # noqa: E402
    LongMemEvalAdapter,
)
from artificial_memory.skills.refusal_gate import (  # noqa: E402
    AnswerProvenanceGate,
    _dialogue_lines,
)

REFUSAL = "not mentioned"


def run_mode(gate, rows, run) -> dict:
    matched = parsed = fired = before = after = 0
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        record = run.get(row["qid"])
        if record is None:
            continue
        matched += 1
        if _dialogue_lines(row["context_text"]):
            parsed += 1

        refused = gate.decide(record["question"], row["context_text"],
                              record["predicted_answer"]).refuse
        fired += int(refused)
        prediction = REFUSAL if refused else record["predicted_answer"]
        ok_before = bool(LongMemEvalAdapter.score_answer(
            question_type=row["qtype"], gt=row["gt"],
            predicted_answer=record["predicted_answer"],
            is_abstention_gt=bool(row["is_abstention_gt"]),
            pcc_is_abstention=bool(row.get("is_abstention_flag", False)),
        ))
        ok_after = bool(LongMemEvalAdapter.score_answer(
            question_type=row["qtype"], gt=row["gt"],
            predicted_answer=prediction,
            is_abstention_gt=bool(row["is_abstention_gt"]),
            pcc_is_abstention=refused,
        ))
        before += int(ok_before)
        after += int(ok_after)
        by_type[row["qtype"]][0] += int(ok_before)
        by_type[row["qtype"]][1] += int(ok_after)
    return {"matched": matched, "parsed": parsed, "fired": fired,
            "before": before, "after": after, "by_type": dict(by_type)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="benchmark_results/lme_context_cache.jsonl")
    ap.add_argument("--run", default="benchmark_results/pipeline_v4/lme_dev250_v4_R7.json")
    ap.add_argument("--out", default="benchmark_results/refusal_gate/lme_guard.json")
    args = ap.parse_args()

    # Split on "\n" only: str.splitlines() also breaks on U+2028 / U+0085, which
    # appear inside LongMemEval messages and would cut a JSON row in half.
    rows = [json.loads(line) for line in
            (ROOT / args.cache).read_text(encoding="utf-8").split("\n") if line.strip()]
    run = {r["question_id"]: r
           for r in json.loads((ROOT / args.run).read_text(encoding="utf-8"))["results"]}

    modes = {
        "name-only": AnswerProvenanceGate(),
        "roles": AnswerProvenanceGate(allow_roles=True),
    }
    results = {}
    for name, gate in modes.items():
        stats = run_mode(gate, rows, run)
        results[name] = stats
        n = max(1, stats["matched"])
        print(f"{name:<10} parsed {stats['parsed']:3d}/{stats['matched']:<3d} "
              f"fired {stats['fired']:3d}  "
              f"accuracy {stats['before'] / n * 100:6.2f}% -> "
              f"{stats['after'] / n * 100:6.2f}%  "
              f"({(stats['after'] - stats['before']) / n * 100:+.2f}pt)")

    gate_default = results["name-only"]
    if gate_default["parsed"] == 0:
        verdict = "VACUOUS: the parser cannot read LME contexts at all."
    elif gate_default["fired"] == 0 and results["roles"]["fired"] == 0:
        verdict = "No regression, but the gate never fires on LME either way."
    elif gate_default["fired"] == 0:
        verdict = ("PASS in the shipped configuration (name-only: 0 fires). "
                   "Enabling role mode costs the delta shown above, which is why "
                   "allow_roles defaults to False.")
    else:
        verdict = "The shipped gate fires on LME; the delta above is its measured effect."
    print(f"\nVERDICT: {verdict}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"verdict": verdict, "modes": results},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
