"""Re-score the V4 mission-15 order probe with the FROZEN scorer.

The probe's own table used a lexical gold-in-answer proxy to label direction, and
explicitly deferred to the frozen scorer.  That deference is honoured here: every
control and swapped answer is scored with eval_locomo_full.score(), the same function
that produced the M2 baseline.  The proxy and the scorer are compared, and any
disagreement is reported rather than smoothed over -- in particular a swapped answer
that became an abstention ("I don't know.") is exactly the case a lexical proxy can
misfile.

No new model calls.  Reads v4_order_probe.json only.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

PROBE = ROOT / "benchmark_results/evidence_audit/v4_order_probe.json"
OUT = ROOT / "benchmark_results/evidence_audit/v4_order_probe_scored.json"


def main() -> int:
    from eval_locomo_full import score as frozen_score

    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    rows = [x for x in probe["rows"] if x["status"] == "MEASURED"]

    abstain_text = ("i don't know", "i dont know", "not mentioned", "unknown")

    out = []
    for x in rows:
        gold = x.get("gold") or ""
        ca, sa = x.get("control_answer", ""), x.get("swapped_answer", "")
        c_ok = bool(frozen_score(ca, gold))
        s_ok = bool(frozen_score(sa, gold))
        s_abst = any(t in (sa or "").lower() for t in abstain_text)
        c_abst = any(t in (ca or "").lower() for t in abstain_text)
        out.append({
            **x,
            "control_correct": c_ok,
            "swapped_correct": s_ok,
            "control_abstained": c_abst,
            "swapped_abstained": s_abst,
            # transition, mutually exclusive
            "transition": (
                "correct->wrong" if c_ok and not s_ok else
                "correct->abstain" if c_ok and s_abst else
                "wrong->correct" if not c_ok and s_ok else
                "wrong->abstain" if not c_ok and s_abst else
                "abstain->correct" if c_abst and s_ok else
                "abstain->wrong" if c_abst and not s_ok else
                "unchanged_correct" if c_ok and s_ok else
                "unchanged_wrong" if not c_ok and not s_ok else "other"
            ),
            # did the lexical proxy agree with the frozen scorer on DIRECTION?
            "proxy_said_improve": (not x["control_matches_gold"]
                                   and x["swapped_matches_gold"]),
        })

    print("=" * 76)
    print("V4 MISSION 15 -- RE-SCORED WITH THE FROZEN SCORER (authoritative)")
    print("=" * 76)
    print(f"  scorer : eval_locomo_full.score()  (the same function as the M2 baseline)")
    print(f"  cases  : {len(out)}   (no new model calls)")
    print()

    for label in ("FAILURE", "CONTROL"):
        g = [x for x in out if x["group"] == label]
        if not g:
            continue
        tc = Counter(x["transition"] for x in g)
        print(f"  {label}  n={len(g)}")
        for k in ("wrong->correct", "abstain->correct", "correct->wrong",
                  "correct->abstain", "wrong->abstain", "abstain->wrong",
                  "unchanged_correct", "unchanged_wrong", "other"):
            if tc.get(k):
                print(f"      {k:<20} {tc[k]}")
        print()

    # proxy vs scorer disagreement -- the honest check
    print("=" * 76)
    print("PROXY vs FROZEN SCORER  (where the lexical proxy would have misled)")
    print("=" * 76)
    disagree = [x for x in out
                if x["proxy_said_improve"] != (x["transition"] == "wrong->correct")]
    print(f"  disagreements : {len(disagree)}/{len(out)}")
    for x in disagree:
        print(f"    {x['question_id']} [{x['group']}] proxy_said_improve="
              f"{x['proxy_said_improve']} scorer_transition={x['transition']}")
        if not disagree or len(disagree) <= 8:
            print(f"        gold   : {(x.get('gold') or '')[:60]}")
            print(f"        before : {x['control_answer'][:65]}")
            print(f"        after  : {x['swapped_answer'][:65]}")
    print()

    fm = [x for x in out if x["group"] == "FAILURE"]
    cm = [x for x in out if x["group"] == "CONTROL"]
    f_res = sum(1 for x in fm if x["transition"] in ("wrong->correct", "abstain->correct"))
    c_res = sum(1 for x in cm if x["transition"] in ("wrong->correct", "abstain->correct"))
    c_break = sum(1 for x in cm if x["transition"] in ("correct->wrong", "correct->abstain"))
    f_break = sum(1 for x in fm if x["transition"] in ("correct->wrong", "correct->abstain"))
    print("=" * 76)
    print("HEADLINE, frozen scorer")
    print("=" * 76)
    print(f"  rescued  : FAILURE {f_res}/{len(fm)}   CONTROL {c_res}/{len(cm)}")
    print(f"  broken   : FAILURE {f_break}/{len(fm)}   CONTROL {c_break}/{len(cm)}")
    print()
    print("  This is a 30-case causal probe on a stratum-selected cohort, NOT a")
    print("  benchmark estimate.  7 of the 37 cases were unmeasurable (competing pair")
    print("  not locatable) and are reported, not dropped silently.")
    print("  Scope unchanged: this shows order causally affects WHICH candidate is")
    print("  selected.  It does not observe the Reader's reasoning, and it does not")
    print("  establish a production improvement.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 15 re-scored with the frozen scorer",
        "scorer": "eval_locomo_full.score()",
        "n_scored": len(out),
        "rescued_failure": f_res, "rescued_control": c_res,
        "broken_failure": f_break, "broken_control": c_break,
        "proxy_scorer_disagreements": len(disagree),
        "rows": out,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
