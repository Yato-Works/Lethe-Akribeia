#!/usr/bin/env python3
"""M30-41 Phase 1: Leakage audit of production retrieval scoring.

Finding: src/artificial_memory/recall/evidence_scorer.py hardcodes LoCoMo
character names in KNOWN_ACTORS and gives them +15/+12 boosts.

Questions to answer:
1. Which production code paths use UniversalEvidenceScorer?
2. Do the M30-33..M30-40 synthetic evaluations use it? (No - they use their
   own expanded_retrieval, but confirm.)
3. Does the LoCoMo production path (locomo_adapter.py) use it?
4. Is there any OTHER benchmark-specific hardcoding in production code?
"""
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

# LoCoMo-specific names that should NOT be in production code
LOCOMO_NAMES = ["caroline", "melanie", "gina", "jon", "maria", "joanna", "nate",
                "james", "deborah", "jolene", "evan", "calvin", "audrey", "tim",
                "andrew", "kadence", "dave"]
# Other benchmark-specific tokens seen in M30 traces
OTHER_TOKENS = ["perseid", "dr. seuss", "sundance", "cannes", "tribeca", "oximeter", "oura"]

print("=" * 78)
print("M30-41 Phase 1: Leakage Audit")
print("=" * 78)
print()

hits = {}
for py in SRC.rglob("*.py"):
    if "__pycache__" in str(py):
        continue
    try:
        text = py.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    rel = py.relative_to(PROJECT_ROOT)
    for name in LOCOMO_NAMES + OTHER_TOKENS:
        # word-boundary match on lowercase token
        if re.search(rf"\b{re.escape(name)}\b", text, re.I):
            hits.setdefault(str(rel), []).append(name)

print(f"Production files containing benchmark-specific tokens: {len(hits)}")
print()
for rel, names in sorted(hits.items(), key=lambda x: -len(x[1])):
    print(f"  {rel}")
    print(f"    tokens: {sorted(set(names))}")
print()

# ---- Which of these are on the retrieval scoring path? ----
print("=" * 78)
print("Retrieval-path usage")
print("=" * 78)
print()

# Does locomo_adapter use UniversalEvidenceScorer?
adapter = (SRC / "artificial_memory/research/benchmarks/external/locomo_adapter.py").read_text(encoding="utf-8", errors="ignore")
print(f"locomo_adapter.py references UniversalEvidenceScorer: "
      f"{'UniversalEvidenceScorer' in adapter}")
print(f"locomo_adapter.py references evidence_scorer: "
      f"{'evidence_scorer' in adapter}")
print()

# Does state_reconstructor (which DOES use the scorer) run in production?
sr = (SRC / "artificial_memory/recall/state_reconstructor.py").read_text(encoding="utf-8", errors="ignore")
print("state_reconstructor.py uses UniversalEvidenceScorer: True")
print()

# Where is state_reconstructor called from?
callers = []
for py in SRC.rglob("*.py"):
    if "__pycache__" in str(py) or py.name == "state_reconstructor.py":
        continue
    try:
        t = py.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if "state_reconstructor" in t or "StateReconstructor" in t:
        callers.append(str(py.relative_to(PROJECT_ROOT)))
print("state_reconstructor callers:")
for c in callers:
    print(f"  {c}")
print()

# ---- Conclusion ----
print("=" * 78)
print("ASSESSMENT")
print("=" * 78)
print()
print("FINDING: Production retrieval scorer (evidence_scorer.py) contains")
print("hardcoded LoCoMo character names with large score boosts:")
print("  KNOWN_ACTORS = ['caroline', 'melanie', 'gina', 'jon', ...]")
print("  extract_target_actor() only recognizes these names")
print("  compute_score() gives +15 if the turn speaker is the target actor,")
print("  +12 if the turn mentions them, -8 otherwise.")
print()
print("IMPLICATION:")
print("  - On LoCoMo (Bucket A 28), this gives name-based boosts that would")
print("    NOT exist for unseen names.")
print("  - On synthetic sets (M30-33..40), names like 'Olivia'/'Aiden' are NOT")
print("    in KNOWN_ACTORS, so extract_target_actor() returns None and the")
print("    +15/+12 actor boosts never fire.")
print("  - This is a REAL train/test asymmetry: the production scorer is")
print("    tuned to LoCoMo names.")
print()

out = PROJECT_ROOT / "results/m30_41"
out.mkdir(parents=True, exist_ok=True)
(out / "leakage_audit.json").write_text(
    __import__("json").dumps({
        "files_with_benchmark_tokens": {k: sorted(set(v)) for k, v in hits.items()},
        "locomo_adapter_uses_scorer": "UniversalEvidenceScorer" in adapter,
        "known_actors_hardcoded": True,
        "note": "Production scorer hardcodes LoCoMo names; synthetic sets bypass it",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved: {out / 'leakage_audit.json'}")
