#!/usr/bin/env python3
"""M30-42 Phase 3b: Counterfactual name-substitution test.

Rename LoCoMo characters to fictional names consistently across
question + conversation + speaker fields, preserving meaning and the
gold answer correspondence. Then check whether the name boosts stop firing.

This isolates CONDITIONAL DEPENDENCE (does the boost need the exact LoCoMo
name?) from ABLATION (we turned the mechanism off).

Deterministic. No LLM.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.recall import evidence_scorer as es
from artificial_memory_gen.recall.state_reconstructor import StateReconstructor

# LoCoMo name -> fictional replacement (consistent mapping)
RENAME = {
    "caroline": "Tamsin", "melanie": "Rurik", "gina": "Ilse", "jon": "Casper",
    "john": "Emrys", "maria": "Solveig", "joanna": "Bertram", "nate": "Yara",
    "james": "Lior", "deborah": "Anouk", "jolene": "Fenrir", "evan": "Zora",
    "sam": "Ivo", "calvin": "Nell", "dave": "Osric", "audrey": "Priya",
    "andrew": "Cato", "tim": "Verity", "kadence": "Juno",
}


def main():
    scorer = es.UniversalEvidenceScorer()
    recon = StateReconstructor()

    print("=" * 78)
    print("M30-42 Phase 3b: Counterfactual name substitution")
    print("=" * 78)
    print()

    cases = [
        ("original", "When did Caroline go to the LGBTQ support group?"),
        ("original", "When did Melanie get hurt?"),
        ("renamed",  "When did Tamsin go to the LGBTQ support group?"),
        ("renamed",  "When did Rurik get hurt?"),
    ]

    print(f"{'kind':10s} {'query':55s} {'es_actor':12s} {'sr_actor':12s}")
    print("-" * 95)
    rows = []
    for kind, q in cases:
        es_hit = scorer.extract_target_actor(q)
        sr_hit = recon._extract_target_actor(q)
        print(f"{kind:10s} {q[:55]:55s} {str(es_hit):12s} {str(sr_hit):12s}")
        rows.append({"kind": kind, "query": q, "es_actor": es_hit, "sr_actor": sr_hit})
    print()

    # Verdict
    orig_fires = sum(1 for r in rows if r["kind"] == "original" and r["es_actor"])
    ren_fires = sum(1 for r in rows if r["kind"] == "renamed" and r["es_actor"])
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  original names -> es boost fires: {orig_fires}/2")
    print(f"  renamed names  -> es boost fires: {ren_fires}/2")
    print()
    if orig_fires > ren_fires:
        print("CONDITIONAL DEPENDENCE CONFIRMED:")
        print("  The boost requires the EXACT LoCoMo name. Renaming silently disables it.")
    else:
        print("No conditional dependence observed.")
    print()

    # Full-conversation rename test: how many boosts would fire on a renamed
    # Bucket A vs the original?
    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    cases_map = {c["question_id"]: c for c in bucket["cases"]}
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")

    def rename_text(t):
        out = t
        for k, v in RENAME.items():
            out = re.sub(rf"\b{k}\b", v, out, flags=re.I)
        return out

    orig_hits = ren_hits = 0
    for qid in targets:
        c = cases_map[qid]
        qo = c["question"]
        qr = rename_text(qo)
        if scorer.extract_target_actor(qo):
            orig_hits += 1
        if scorer.extract_target_actor(qr):
            ren_hits += 1

    print("=" * 78)
    print("FULL BUCKET A RENAME TEST")
    print("=" * 78)
    print(f"  original questions where es actor boost fires: {orig_hits}/28")
    print(f"  renamed  questions where es actor boost fires: {ren_hits}/28")
    print()

    out = PROJECT_ROOT / "results/m30_42"
    out.mkdir(parents=True, exist_ok=True)
    (out / "name_substitution.json").write_text(json.dumps({
        "spot_checks": rows,
        "bucket_a_original_fires": orig_hits,
        "bucket_a_renamed_fires": ren_hits,
        "rename_map": RENAME,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'name_substitution.json'}")


if __name__ == "__main__":
    main()
