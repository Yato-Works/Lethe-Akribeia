#!/usr/bin/env python3
"""M30-54 Phase 2: Failure Taxonomy.

Classifies each wrong case into the taxonomy defined in M30-54 spec.
"""
import sys
from pathlib import Path
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load M30-51 fallback as module
spec = importlib.util.spec_from_file_location("fallback", "scripts/m30_51_phase2_fallback.py")
fallback_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fallback_mod)

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
import json, re

adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
sids = [c.get("sample_id", "") for c in locomo]

TURN_RE = re.compile(r"(?:\[SPEAKER:[^\]]+\]\s*)?\[(D\d+:\d+)\s+on\s+([^\]]+)\](.*)")

TAXONOMY = {
    "ENGINE_MISS": "Evidence not in retrieval results (IR records)",
    "EVIDENCE_SELECTION_MISS": "Evidence in IR but excluded from compiled context",
    "EVIDENCE_DELIVERY_FAILURE": "Data serialization/truncation loss",
    "READER_TURN_BINDING": "Evidence present but wrong turn/event selected",
    "READER_ABSTENTION": "Evidence present but Reader unnecessarily abstained",
    "READER_FORMAT_FAILURE": "Correct content but wrong output format",
    "SCORER_OR_ADAPTER": "Answer and ground truth don't align",
    "ANNOTATION_UNCERTAIN": "Ambiguous gold label or evidence",
}


def classify_case(qid, audit_entry, baseline_pred, treatment_pred, gold, question):
    """Classify a wrong case into the taxonomy."""
    if audit_entry.get("error"):
        return "ANNOTATION_UNCERTAIN", "Error in audit entry"

    delivered = audit_entry.get("evidence_delivered", False)
    retrieved = audit_entry.get("evidence_retrieved", False)

    if not retrieved:
        return "ENGINE_MISS", f"Gold evidence turn {audit_entry.get('gold_evidence_turn')} not in IR records"

    if not delivered:
        return "EVIDENCE_SELECTION_MISS", f"Evidence in IR but excluded from compiled context"

    # Evidence was delivered — now check Reader behavior
    if "I don't know" in treatment_pred and "I don't know" in baseline_pred:
        return "READER_ABSTENTION", "Reader abstained despite evidence"
    if "does not contain" in treatment_pred.lower() or "does not contain" in baseline_pred.lower():
        return "READER_ABSTENTION", "Reader claims no info despite evidence present"
    if treatment_pred.strip() == baseline_pred.strip():
        return "READER_TURN_BINDING", "Same wrong answer — wrong turn binding"
    if treatment_pred != baseline_pred and treatment_pred != gold:
        return "READER_TURN_BINDING", f"Answer changed but still wrong: {treatment_pred[:50]}"

    return "UNKNOWN", "Unclassified"


def main():
    ab = json.loads((PROJECT_ROOT / "results/m30_51/ab_results_v2.json").read_text(encoding="utf-8"))
    audit = json.loads((PROJECT_ROOT / "results/m30_54/phase1_oronical_audit.json".replace("nal","al").replace("audit/audit","audit/audit")).read_text(encoding="utf-8")) if False else \
        json.loads((PROJECT_ROOT / "results/m30_54/phase1_oracle_audit.json").read_text(encoding="utf-8"))

    # Load production baseline
    prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))

    print("M30-54 Phase 2: FAILURE TAXONOMY REPORT")
    print("="*70)

    taxonomy_counts = {}
    wrong_cases = []
    for r in ab['rows']:
        qid = r['qid']
        if r['treatment_correct']:
            continue
        wrong_cases.append(r)

    print(f"\nTotal wrong (after M30-51): {len(wrong_cases)}/28")
    print(f"\n{'QID':<16} {'Category':<32} {'Evidence':<10} {'Gold':<25}")
    print("-"*83)

    for r in wrong_cases:
        qid = r['qid']
        a = audit['per_qid'].get(qid, {})
        delivered = "DELIVERED" if a.get('evidence_delivered') else "MISSING"
        retrieved = "YES" if a.get('evidence_retrieved') else "NO"
        cat, reason = classify_case(qid, a, r['baseline_raw'], r['treatment_raw'], r['gold'], r['gold'])
        taxonomy_counts[cat] = taxonomy_counts.get(cat, 0) + 1

        evidence_status = f"ret={retrieved},del={delivered[:4]}"
        print(f"{qid:<16} {cat:<32} {evidence_status:<10} {r['gold'][:25]}")
        if any(kw in cat.lower() for kw in ['reader', 'engine', 'evidence_selection']):
            print(f"   → {reason}")

    print(f"\n{'='*70}")
    print("TAXONOMY SUMMARY:")
    for cat, count in sorted(taxotomy_counts.items() if False else taxonomy_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<32} {count:>3}  — {TAXONOMY.get(cat, '')}")

    # Save report
    out_dir = PROJECT_ROOT / "results/m30_54"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "total_wrong": len(wrong_cases),
        "taxonomy_counts": taxonomy_counts,
        "per_case": [],
    }
    for r in wrong_cases:
        qid = r['qid']
        a = audit['per_qid'].get(qid, {})
        cat, reason = classify_case(qid, a, r['baseline_raw'], r['treatment_raw'], r['gold'], r['gold'])
        report["per_case"].append({
            "qid": qid,
            "taxonomy": cat,
            "reason": reason,
            "evidence_delivered": a.get("evidence_delivered", False),
            "evidence_retrieved": a.get("evidence_retrieved", False),
            "gold_turn": a.get("gold_evidence_turn", ""),
            "baseline_answer": r['baseline_raw'][:80],
            "treatment_answer": r['treatment_raw'][:80],
            "gold_answer": r['gold'],
        })
    (out_dir / "phase2_taxonomy.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'phase2_taxonomy.json'}")


if __name__ == "__main__":
    main()
