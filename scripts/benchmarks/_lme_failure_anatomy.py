"""Scratch analysis: anatomy of the LongMemEval 500 answer-conversion failures.

Read-only. Answers: are the misses caused by the reader refusing on questions whose
evidence WAS retrieved (abstention failure), by the verifier discarding a correct
answer (verification failure), or by genuine comprehension/arithmetic errors?
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPORT = Path("benchmark_results/longmemeval/grand_longmemeval_report_coder7b_apex.json")


def main() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = data["results"]
    diag = {d["question_id"]: d for d in data.get("failure_diagnoses", [])}

    wrong = [r for r in rows if not r["is_correct"]]
    print(f"total={len(rows)} correct={sum(1 for r in rows if r['is_correct'])} wrong={len(wrong)}")
    print(f"oracle recall on wrong answers: {sum(1 for r in wrong if r['oracle_recall'])/len(wrong)*100:.1f}%"
          "  (evidence present but answer wrong => reader-side loss)")

    refusals = [r for r in wrong if str(r["predicted_answer"]).strip().lower().startswith("i don't know")]
    print(f"\nwrong answers that are literal refusals ('I don't know.'): {len(refusals)}")

    cat_counts = Counter(d["category"] for d in diag.values())
    print("\nfailure taxonomy:", dict(cat_counts))

    for cat in ("ABSTENTION_FAILURE", "VERIFICATION_FAILURE", "TEMPORAL_FAILURE", "COMPOSITION_FAILURE"):
        sample = [d for d in diag.values() if d["category"] == cat][:3]
        print(f"\n--- {cat} (n={cat_counts.get(cat, 0)}) ---")
        for d in sample:
            row = next((r for r in rows if r["question_id"] == d["question_id"]), None)
            pred = str(row["predicted_answer"])[:80].replace("\n", " ") if row else ""
            gt = str(row["ground_truth"])[:80].replace("\n", " ") if row else ""
            print(f"  {d['question_id']} [{row['question_type'] if row else '?'}]")
            print(f"    pred: {pred}")
            print(f"    gt  : {gt}")
            print(f"    why : {d['explanation'][:120]}")

    # evidence for the truncation hypothesis: prompt token sizes
    toks = [r["tokens_used"] for r in rows]
    print(f"\nprompt tokens: min={min(toks)} mean={sum(toks)/len(toks):.0f} max={max(toks)}"
          f" | over 8192: {sum(1 for t in toks if t > 8192)}")
    print("=> if max <= 8192, no prompt in this run could have been truncated by an 8K window")


if __name__ == "__main__":
    main()
