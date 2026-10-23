#!/usr/bin/env python3
"""M30-45 Phase 5: Detail the recovered/regressed cases + the 3 target cases."""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
d = json.loads((PROJECT_ROOT / "results/m30_45/ab_results.json").read_text(encoding="utf-8"))
bmap = {r["qid"]: r for r in d["baseline_rows"]}
tmap = {r["qid"]: r for r in d["treatment_rows"]}

ABSTAIN = re.compile(r"(no mention|does not contain|does not provide|no information|"
                     r"not specified|no specific|cannot|does not specify)", re.I)

print("=" * 78)
print("M30-45 Phase 5: Recovered / Regressed / Target cases")
print("=" * 78)
print()

print("--- CHANGED CASES ---")
for qid in bmap:
    b, t = bmap[qid], tmap[qid]
    if b["correct"] != t["correct"] or b["predicted"].strip() != t["predicted"].strip():
        tag = "RECOVERED" if (t["correct"] and not b["correct"]) else \
              "REGRESSED" if (b["correct"] and not t["correct"]) else "CHANGED (same score)"
        print(f"{qid}  [{tag}]")
        print(f"  Q:   {b['question']}")
        print(f"  gold: {b['gold']}")
        print(f"  BASE ({'OK' if b['correct'] else '--'}): {b['predicted'][:90]}")
        print(f"  TREAT({'OK' if t['correct'] else '--'}): {t['predicted'][:90]}")
        print()

print("--- TARGET CASES (M30-44) ---")
for qid in ["lc_conv-48_12", "lc_conv-47_65", "lc_conv-26_63"]:
    b, t = bmap[qid], tmap[qid]
    print(f"{qid}")
    print(f"  gold: {b['gold']}")
    print(f"  BASE ({'OK' if b['correct'] else '--'}): {b['predicted'][:100]}")
    print(f"  TREAT({'OK' if t['correct'] else '--'}): {t['predicted'][:100]}")
    print()

# Did treatment reduce abstentions?
b_abst = sum(1 for r in d["baseline_rows"] if ABSTAIN.search(r["predicted"]))
t_abst = sum(1 for r in d["treatment_rows"] if ABSTAIN.search(r["predicted"]))
print("=" * 78)
print("ABSTENTION COUNT")
print("=" * 78)
print(f"  Baseline abstained:  {b_abst}/28")
print(f"  Treatment abstained: {t_abst}/28")
print()

# Unfounded answers (non-abstain, non-correct)
b_unf = sum(1 for r in d["baseline_rows"] if not r["correct"] and not ABSTAIN.search(r["predicted"]))
t_unf = sum(1 for r in d["treatment_rows"] if not r["correct"] and not ABSTAIN.search(r["predicted"]))
print(f"  Baseline wrong-but-answered:  {b_unf}")
print(f"  Treatment wrong-but-answered: {t_unf}")
print()

out = PROJECT_ROOT / "results/m30_45"
(out / "phase5_detail.json").write_text(json.dumps({
    "baseline_abstained": b_abst, "treatment_abstained": t_abst,
    "baseline_wrong_answered": b_unf, "treatment_wrong_answered": t_unf,
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved: {out / 'phase5_detail.json'}")
