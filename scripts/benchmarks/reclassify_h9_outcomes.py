"""Re-classify the H9 Reader outcomes by what the Reader ACTUALLY emitted.

The first pass bucketed every non-correct answer as INCORRECT.  That is wrong for
answers like "I don't know." -- those are abstentions wearing a sentence, and counting
them as errors both overstates the failure rate and hides the real signal.  The mission
explicitly requires appropriate abstentions to be counted separately.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SRC = Path("benchmark_results/evidence_audit/v4_reader_h9.json")

_DONT_KNOW = re.compile(
    r"^\s*(?:i\s+don'?t\s+know|unknown|n/?a|not\s+(?:stated|mentioned|specified|"
    r"provided|available|clear|given)|cannot\s+(?:be\s+)?(?:determined|tell|say)|"
    r"no\s+(?:information|mention|evidence|answer)|sorry|unable\s+to)\b", re.I)


def main() -> int:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    rows = d["rows"]

    abstained, wrong_value = [], []
    for r in rows:
        if r["mutually_exclusive"] == "CORRECT":
            continue
        raw = str(r.get("raw_output") or "").strip()
        (abstained if _DONT_KNOW.match(raw) else wrong_value).append(r)

    n = len(rows)
    n_correct = sum(1 for r in rows if r["mutually_exclusive"] == "CORRECT")

    print("=" * 74)
    print("H9 READER OUTCOMES, re-classified by what was actually emitted")
    print("=" * 74)
    print(f"  n                              : {n}")
    print(f"  CORRECT                        : {n_correct}  "
          f"({100 * n_correct / n:.1f}%)")
    print(f"  ABSTAINED ('I don't know' etc) : {len(abstained)}  "
          f"({100 * len(abstained) / n:.1f}%)")
    print(f"  WRONG VALUE (a real error)     : {len(wrong_value)}  "
          f"({100 * len(wrong_value) / n:.1f}%)")
    print()

    answered = n_correct + len(wrong_value)
    print(f"  Reader answered (non-abstaining): {answered}")
    print(f"  accuracy on what it ANSWERED   : {n_correct}/{answered} = "
          f"{100 * n_correct / answered:.1f}%")
    print()

    print("EVIDENCE SUFFICIENCY")
    print("  abstentions:")
    for k, v in Counter(r["evidence_in_context"] for r in abstained).most_common():
        print(f"    {k:<16} {v}")
    print("  wrong values:")
    for k, v in Counter(r["evidence_in_context"] for r in wrong_value).most_common():
        print(f"    {k:<16} {v}")
    print()

    # Is the abstention appropriate?  An abstention is APPROPRIATE when the frozen
    # context does not carry the evidence, and INCOMPLETE when it does.
    app = [r for r in abstained if r["evidence_in_context"] == "VERBATIM_ABSENT"]
    incomplete = [r for r in abstained if r["evidence_in_context"] != "VERBATIM_ABSENT"]
    print(f"  appropriate abstentions (no evidence in context) : {len(app)}")
    print(f"  abstentions WITH evidence present (missed chance): {len(incomplete)}")
    print()

    print("WRONG VALUES -- the real Reader errors, with what it said")
    for r in wrong_value[:12]:
        print(f"  {r['question_id']}  [{r['evidence_in_context']}]")
        print(f"     Q    : {r['question'][:62]}")
        print(f"     gold : {str(r['gold'])[:40]!r}")
        print(f"     pred : {str(r['prediction'])[:52]!r}")
    print()

    out = SRC.with_name("v4_reader_h9_reclassified.json")
    out.write_text(json.dumps({
        "label": "H9 Reader outcomes re-classified by actual output",
        "n": n, "correct": n_correct,
        "abstained": len(abstained), "wrong_value": len(wrong_value),
        "answered": answered,
        "accuracy_on_answered": round(100 * n_correct / answered, 1),
        "appropriate_abstentions": len(app),
        "abstentions_with_evidence": len(incomplete),
        "abstain_evidence": dict(Counter(r["evidence_in_context"] for r in abstained)),
        "wrong_value_evidence": dict(Counter(r["evidence_in_context"]
                                             for r in wrong_value)),
        "wrong_value_qids": [r["question_id"] for r in wrong_value],
        "abstain_qids": [r["question_id"] for r in abstained],
        "incomplete_abstain_qids": [r["question_id"] for r in incomplete],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
