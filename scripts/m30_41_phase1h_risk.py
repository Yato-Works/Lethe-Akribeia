#!/usr/bin/env python3
"""M30-41 Phase 1h: Classify the 35 hardcoded files by runtime risk.

For each file containing benchmark-specific tokens, determine:
  - does the token appear in a DECISION path (if/elif/score/boost)?
  - or only in a stopword/comment/docstring?
  - is the file reachable from the production locomo path?

Output: risk-tiered list (HIGH / MEDIUM / LOW).
Read-only. No production edit.
"""
import re, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "artificial_memory"

LOCOMO_NAMES = ["caroline", "melanie", "gina", "jon", "maria", "joanna", "nate",
                "james", "deborah", "jolene", "evan", "calvin", "audrey", "tim",
                "andrew", "kadence", "dave"]

# Files on the production locomo path (from Phase 1b/1d)
PROD_PATH = {
    "context/msc_compiler.py",
    "recall/state_reconstructor.py",
    "recall/evidence_scorer.py",
    "recall/query_planner.py",
    "recall/adaptive_search.py",
    "recall/temporal_resolver.py",
    "recall/state_supersession_engine.py",
    "recall/proposition_integrity_gate.py",
    "recall/answer_verifier.py",
    "recall/evidence_graph.py",
}

DECISION_PAT = re.compile(
    r"(if|elif|and|or|==|!=|\bin\b|\+=|-=|\*=|score|boost|penalt)", re.I)
STOPWORD_CTX = re.compile(r"(STOP|stopword|stop_word|_STOP|COMMON_STOP)", re.I)


def classify(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    hits = []
    for i, ln in enumerate(lines, 1):
        low = ln.lower()
        if not any(re.search(rf"\b{n}\b", low) for n in LOCOMO_NAMES):
            continue
        is_stop = bool(STOPWORD_CTX.search(ln))
        is_decision = bool(DECISION_PAT.search(ln))
        hits.append({
            "line": i,
            "stopword": is_stop,
            "decision": is_decision,
            "code": ln.strip()[:100],
        })
    return hits


def main():
    files = {}
    for py in SRC.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        rel = str(py.relative_to(SRC)).replace("\\", "/")
        hits = classify(py)
        if hits:
            files[rel] = hits

    tiers = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for rel, hits in files.items():
        decision_hits = [h for h in hits if h["decision"] and not h["stopword"]]
        stop_hits = [h for h in hits if h["stopword"]]
        on_prod = rel in PROD_PATH
        if decision_hits and on_prod:
            tiers["HIGH"].append((rel, len(decision_hits), len(hits)))
        elif decision_hits:
            tiers["MEDIUM"].append((rel, len(decision_hits), len(hits)))
        else:
            tiers["LOW"].append((rel, len(decision_hits), len(hits)))

    print("=" * 78)
    print("M30-41 Phase 1h: Hardcoded-Name Risk Tiers")
    print("=" * 78)
    print()
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        print(f"### {tier} ({len(tiers[tier])} files)")
        for rel, nd, nh in sorted(tiers[tier], key=lambda x: -x[1]):
            print(f"  {rel}  (decision lines={nd}, total={nh})")
        print()

    print("=" * 78)
    print("HIGH-tier detail (decision path + production-reachable)")
    print("=" * 78)
    print()
    for rel, _, _ in tiers["HIGH"]:
        print(f"--- {rel} ---")
        for h in files[rel]:
            if h["decision"] and not h["stopword"]:
                print(f"  L{h['line']}: {h['code']}")
        print()

    out = PROJECT_ROOT / "results/m30_41"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hardcode_risk_tiers.json").write_text(json.dumps({
        "HIGH": [r for r, _, _ in tiers["HIGH"]],
        "MEDIUM": [r for r, _, _ in tiers["MEDIUM"]],
        "LOW": [r for r, _, _ in tiers["LOW"]],
        "high_detail": {rel: [h for h in files[rel] if h["decision"] and not h["stopword"]]
                        for rel, _, _ in tiers["HIGH"]},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'hardcode_risk_tiers.json'}")


if __name__ == "__main__":
    main()
