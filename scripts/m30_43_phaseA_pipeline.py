#!/usr/bin/env python3
"""M30-43 Phase A: Pipeline identity audit — which path did each experiment use?

The core question: did the M30 experiments evaluate the PRODUCTION pipeline
(locomo_adapter.load_conversation -> UniversalIRExtractor -> compile) or a
hand-rolled approximation (make_records -> compile)?

Evidence: read each M30 script's record construction and compare to the
production path.

Read-only.
"""
import re, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(PROJECT_ROOT.glob("scripts/m30_*.py"))

# Marker patterns
PROD_MARKERS = ["load_conversation", "LoCoMoAdapter", "locomo_adapter", "UniversalIRExtractor"]
HANDROLL_MARKERS = ["def make_records", "StructuredIR(", "raw_content=f\"["]

print("=" * 78)
print("M30-43 Phase A: Which pipeline did each experiment use?")
print("=" * 78)
print()

rows = []
for s in SCRIPTS:
    txt = s.read_text(encoding="utf-8", errors="ignore")
    uses_prod = [m for m in PROD_MARKERS if m in txt]
    uses_hand = [m for m in HANDROLL_MARKERS if m in txt]
    # does it call compile()?
    calls_compile = "compiler.compile(" in txt or ".compile(" in txt
    # does it use a custom extract_evidence_turns?
    custom_extract = "def extract_evidence_turns" in txt

    if uses_prod:
        path = "PRODUCTION"
    elif calls_compile and uses_hand:
        path = "HANDROLL->compile"
    elif custom_extract:
        path = "CUSTOM extract"
    elif uses_hand:
        path = "HANDROLL (no compile)"
    else:
        path = "other"

    rows.append({
        "script": s.name,
        "path": path,
        "prod_markers": uses_prod,
        "handroll_markers": uses_hand,
        "calls_compile": calls_compile,
        "custom_extract": custom_extract,
    })

from collections import Counter
paths = Counter(r["path"] for r in rows)
print("Pipeline classification:")
for p, n in paths.most_common():
    print(f"  {p}: {n}")
print()

print("Scripts by path:")
for r in rows:
    print(f"  [{r['path']:20s}] {r['script']}")
print()

# The production entry
print("=" * 78)
print("PRODUCTION PATH (from locomo_adapter.load_conversation)")
print("=" * 78)
print("""
  locomo10.json
    -> LoCoMoAdapter.load_conversation()
       -> UniversalIRExtractor.extract(prev_ctx + text)   <- structured IR
       -> r.raw_content = "[D1:1 on <date>] (In reply to X: ...) Speaker: text"
       -> r.time_scope = session_date
    -> MinimumSufficientContextCompiler.compile(question, ir_records)
       -> QueryPlanner.plan()
       -> TemporalResolver.resolve()
       -> StateSupersessionEngine.resolve()
       -> StateReconstructor.reconstruct_world()
       -> AdaptiveEvidenceSearcher.search()
       -> PropositionIntegrityGate.check()
       -> checker.check()  -> ProofCarryingContext
    -> OllamaAnswerer.answer(question, context_text)
    -> OfficialLongMemEvalGrader.grade()
""")

out = PROJECT_ROOT / "results/m30_43"
out.mkdir(parents=True, exist_ok=True)
(out / "pipeline_identity.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved: {out / 'pipeline_identity.json'}")
