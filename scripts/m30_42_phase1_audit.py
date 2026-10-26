#!/usr/bin/env python3
"""M30-42 Phase 1: Complete audit table of benchmark-specific boosts.

Classify each hardcoded mechanism:
  - is it benchmark-specific (LoCoMo names / corpus-specific tokens)?
  - is it on the production path?
  - has it been A/B tested?
  - is it a name boost, a domain anchor, or a generic heuristic?

Output: structured audit table.
"""
import re, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "artificial_memory"

# Mechanisms to classify (file, label, kind)
MECHANISMS = [
    ("recall/evidence_scorer.py", "KNOWN_ACTORS (+15/+12/-8)", "benchmark_name"),
    ("recall/state_reconstructor.py", "_extract_target_actor (+25/+15/+20/-10)", "benchmark_name"),
    ("recall/query_planner.py", "NAMED_ENTITIES (target_entities)", "benchmark_name"),
    ("recall/evidence_graph.py", "STOPWORDS contains melanie/caroline", "benchmark_name"),
    ("memory/persona_store.py", '["melanie","caroline"] loops', "benchmark_name"),
    ("compiler/closure/planner.py", "_COMMON_ENTITIES + coref by name", "benchmark_name"),
    ("protein/evidence_fuser.py", '"mel"->"melanie" aliases', "benchmark_name"),
    ("protein/state_synthesizer.py", 'entity or "Caroline" default', "benchmark_name"),
    ("recall/state_reconstructor.py", "aggregation +35 topic anchor", "domain_anchor"),
    ("recall/state_reconstructor.py", "DOMAIN_SYNONYMS (fish/aquarium/wedding/festival...)", "domain_anchor"),
    ("recall/evidence_scorer.py", "SEMANTIC_EXPANSIONS", "domain_anchor"),
    ("recall/state_reconstructor.py", "session/family spillover", "generic_heuristic"),
    ("recall/state_reconstructor.py", "intent weighting (+30 evidence/state)", "generic_heuristic"),
]

PROD_PATH = {
    "context/msc_compiler.py", "recall/state_reconstructor.py",
    "recall/evidence_scorer.py", "recall/query_planner.py",
    "recall/adaptive_search.py", "recall/temporal_resolver.py",
    "recall/state_supersession_engine.py", "recall/proposition_integrity_gate.py",
    "recall/answer_verifier.py", "recall/evidence_graph.py",
}

TESTED = {
    "KNOWN_ACTORS (+15/+12/-8)": "Phase 8a: top-1 16->16 (+0)",
    "_extract_target_actor (+25/+15/+20/-10)": "Phase 8b: top-1 16->16 (+0)",
    "NAMED_ENTITIES (target_entities)": "Phase 8c: entities 7->4 (downstream unmeasured)",
}


def main():
    rows = []
    for rel, label, kind in MECHANISMS:
        p = SRC / rel
        on_prod = rel in PROD_PATH
        tested = TESTED.get(label, "UNTESTED")
        rows.append({
            "file": rel,
            "mechanism": label,
            "kind": kind,
            "on_production_path": on_prod,
            "ab_tested": tested,
        })

    print("=" * 78)
    print("M30-42 Phase 1: Benchmark-specific mechanism audit")
    print("=" * 78)
    print()
    print(f"{'file':42s} {'mechanism':40s} {'kind':18s} {'prod':5s}")
    print("-" * 110)
    for r in rows:
        print(f"{r['file']:42s} {r['mechanism'][:40]:40s} {r['kind']:18s} {'YES' if r['on_production_path'] else 'no':5s}")
    print()

    # Summary by kind
    from collections import Counter
    kinds = Counter(r["kind"] for r in rows)
    print("By kind:")
    for k, v in kinds.items():
        print(f"  {k}: {v}")
    print()

    prod_bench = [r for r in rows if r["on_production_path"] and r["kind"] == "benchmark_name"]
    print(f"Benchmark-NAME mechanisms on production path: {len(prod_bench)}")
    for r in prod_bench:
        print(f"  {r['file']}: {r['mechanism']}")
        print(f"    A/B: {r['ab_tested']}")
    print()

    prod_domain = [r for r in rows if r["on_production_path"] and r["kind"] == "domain_anchor"]
    print(f"Domain-ANCHOR mechanisms on production path: {len(prod_domain)}")
    for r in prod_domain:
        print(f"  {r['file']}: {r['mechanism']}")
        print(f"    A/B: {r['ab_tested']}")
    print()

    out = PROJECT_ROOT / "results/m30_42"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mechanism_audit.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'mechanism_audit.json'}")


if __name__ == "__main__":
    main()
