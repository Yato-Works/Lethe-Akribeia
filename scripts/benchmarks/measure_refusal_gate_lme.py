"""Cross-benchmark guard: does the answer-provenance gate touch LongMemEval?

The gate ships on LoCoMo (+4.33pt dev).  Before it can be a default, the program
requires that it does not regress the other benchmark.  This script prices it on
LongMemEval dev, using the frozen context cache and the same deterministic scorer
that produced the published LME number, so the comparison is like for like.

It reports THREE things, and the third is the one that matters:

  1. how many LME contexts the gate's dialogue parser can read,
  2. how often the gate fires,
  3. the accuracy before/after.

If (1) is zero the guard is vacuous, not passed: LME renders dialogue with role
labels (`user` / `assistant`) rather than names, so a gate keyed on "the answer's
wording belongs to another NAMED speaker" has nothing to match.  That is a
limitation of the mechanism, not evidence that it is safe.

Usage
    python scripts/lme_context_cache.py --limit 250 \
        --out benchmark_results/lme_context_cache.jsonl
    python scripts/benchmarks/measure_refusal_gate_lme.py \
        --cache benchmark_results/lme_context_cache.jsonl \
        --run benchmark_results/pipeline_v4/lme_dev250_v4_R7.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="benchmark_results/lme_context_cache.jsonl")
    ap.add_argument("--run", default="benchmark_results/pipeline_v4/lme_dev250_v4_R7.json")
    args = ap.parse_args()

    # Split on "\n" only: str.splitlines() also breaks on U+2028 / U+0085, which
    # appear inside LongMemEval messages and would cut a JSON row in half.
    rows = [json.loads(line) for line in
            (ROOT / args.cache).read_text(encoding="utf-8").split("\n") if line.strip()]
    run = {r["question_id"]: r
           for r in json.loads((ROOT / args.run).read_text(encoding="utf-8"))["results"]}

    gate = AnswerProvenanceGate()
    matched = parsed = fired = 0
    before = after = 0
    by_type: dict[str, list[int]] = {}
    verdicts: Counter = Counter()

    for row in rows:
        qid = row["qid"]
        record = run.get(qid)
        if record is None:
            continue
        matched += 1
        if _dialogue_lines(row["context_text"]):
            parsed += 1

        refused = gate.decide(record["question"], row["context_text"],
                              record["predicted_answer"]).refuse
        if refused:
            fired += 1
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
        verdicts[(refused, ok_before, ok_after)] += 1
        by_type.setdefault(row["qtype"], [0, 0])
        by_type[row["qtype"]][0] += int(ok_before)
        by_type[row["qtype"]][1] += int(ok_after)

    if not matched:
        raise SystemExit("no LME question matched between cache and run")

    print(f"LME questions joined : {matched}")
    print(f"  contexts parsed by the gate : {parsed}")
    print(f"  gate fired                  : {fired}")
    print(f"  accuracy before             : {before / matched * 100:.2f}%")
    print(f"  accuracy after              : {after / matched * 100:.2f}%")
    print(f"  delta                       : {(after - before) / matched * 100:+.2f}pt")
    for qtype in sorted(by_type):
        b, a = by_type[qtype]
        print(f"    {qtype:<24} {b} -> {a}")
    if parsed == 0:
        print("\nVERDICT: vacuous guard. The parser reads no LME context, so the gate "
              "cannot fire and this run says nothing about safety. LME labels dialogue "
              "with roles (user/assistant), not names.")
    elif fired == 0:
        print("\nVERDICT: no regression, and the gate is live but never fires here.")
    else:
        print("\nVERDICT: the gate fires on LME; the delta above is the measured effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
