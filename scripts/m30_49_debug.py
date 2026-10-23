#!/usr/bin/env python3
"""M30-49 debug: check if post_process ever fires."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import directly
exec(open("scripts/m30_49_phase2_ab.py").read().split("def main")[0])

prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
baseline = {r["qid"]: r for r in prod["rows"]}

for qid, bp in baseline.items():
    raw = bp.get("reader_output", bp.get("predicted", ""))
    rels = REL_RE.findall(raw)
    if rels:
        print(f"{qid}: rels={rels}  raw='{raw[:80]}'")
    if "I don't know" in raw:
        print(f"{qid}: ABSTAIN  raw='{raw[:60]}'")