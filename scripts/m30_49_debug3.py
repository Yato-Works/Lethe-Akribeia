#!/usr/bin/env python3
"""M30-49 Phase 3: Diagnostic — how many mechanism A cases can post-processing touch?

The A/B showed 0 changes because the reader output for the mechanism A
failure cases doesn't itself contain a relative expression — the reader
DID NOT resolve it and DID NOT emit it.

Let's verify: of the 7 mechanism A failures, how many reader outputs contain
a resolvable relative expression, and how many emit "I don't know"?
"""
import json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
exec(open("scripts/m30_49_phase2_ab.py").read().split("def main")[0])

MECH_A = ["lc_conv-26_63", "lc_conv-42_57", "lc_conv-47_43", "lc_conv-47_65",
          "lc_conv-48_12", "lc_conv-50_52", "lc_conv-41_20"]

prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
baseline = {r["qid"]: r for r in prod["rows"]}

print("M30-49 Phase 3: Mechanism A cases — reader output analysis")
print("=" * 60)
for qid in MECH_A:
    if qid not in baseline:
        print(f"{qid}: NOT IN BASELINE")
        continue
    bp = baseline[qid]
    raw = bp.get("predicted", "")
    rels = REL_RE.findall(raw)
    is_abstain = any(w in raw.lower() for w in ["i don't know", "not mentioned", "not enough", 
                                                 "not specified", "unknown", "unclear", "cannot determine"])
    has_rel = bool(rels)
    can_fix = has_rel  # can only post-process if the rel expr is in the answer
    print(f"{qid}")
    print(f"  gold: {bp['gold'][:40]!r}")
    print(f"  raw:  {raw[:80]!r}")
    print(f"  rel_expr: {rels}")
    print(f"  abstain_shape: {is_abstain}")
    print(f"  post-processable: {can_fix}")
    print()
