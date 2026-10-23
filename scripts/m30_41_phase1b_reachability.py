#!/usr/bin/env python3
"""M30-41 Phase 1b: Trace whether LoCoMo evaluation uses the hardcoded scorer.

Chain under investigation:
  locomo_adapter.py
    -> context.msc_compiler.MinimumSufficientContextCompiler
      -> recall.state_reconstructor.StateReconstructor
        -> recall.evidence_scorer.UniversalEvidenceScorer (KNOWN_ACTORS hardcoded)

If this chain is live during LoCoMo evaluation, then the +15/+12 actor
boosts (which only fire for LoCoMo names) inflate LoCoMo retrieval scores.
"""
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "artificial_memory"


def read(p):
    return (SRC / p).read_text(encoding="utf-8", errors="ignore")


print("=" * 78)
print("M30-41 Phase 1b: Hardcoded-Scorer Reachability")
print("=" * 78)
print()

# 1. locomo_adapter imports
adapter = read("research/benchmarks/external/locomo_adapter.py")
imports = re.findall(r"^from ([\w\.]+) import .*$", adapter, re.M)
print("locomo_adapter.py imports:")
for i in sorted(set(imports)):
    print(f"  {i}")
print()

uses_msc = "MinimumSufficientContextCompiler" in adapter
print(f"locomo_adapter uses MinimumSufficientContextCompiler: {uses_msc}")
print()

# 2. msc_compiler -> state_reconstructor?
msc = read("context/msc_compiler.py")
print(f"msc_compiler references StateReconstructor: {'StateReconstructor' in msc}")
print(f"msc_compiler references state_reconstructor: {'state_reconstructor' in msc}")
print()

# 3. state_reconstructor -> evidence_scorer?
sr = read("recall/state_reconstructor.py")
print(f"state_reconstructor uses UniversalEvidenceScorer: {'UniversalEvidenceScorer' in sr}")
print(f"state_reconstructor uses extract_target_actor: {'extract_target_actor' in sr}")
print()

# 4. Is msc_compiler actually invoked in locomo_adapter's evaluate path?
#    Look for the call site
call_sites = []
for m in re.finditer(r"MinimumSufficientContextCompiler[^\n]*", adapter):
    call_sites.append(m.group(0)[:100])
print("msc_compiler call sites in locomo_adapter:")
for c in call_sites:
    print(f"  {c}")
print()

# 5. Which methods reference it?
for m in re.finditer(r"def (\w+)[^\n]*\n(?:.*\n){0,60}?.*MinimumSufficientContextCompiler", adapter):
    print(f"  enclosing method: {m.group(1)}")
print()

print("=" * 78)
print("VERDICT")
print("=" * 78)
print()
if uses_msc and "StateReconstructor" in msc and "UniversalEvidenceScorer" in sr:
    print("The chain IS statically connected:")
    print("  locomo_adapter -> msc_compiler -> state_reconstructor -> evidence_scorer")
    print()
    print("But static connection != runtime execution. The scorer's compute_score()")
    print("is called from state_reconstructor.score_unit()/reconstruct(), which must")
    print("be invoked by msc_compiler for the hardcoded actor boosts to matter.")
    print()
    print("NEXT: check whether msc_compiler's state_reconstruction is enabled by")
    print("default or behind a flag, and whether locomo_adapter sets that flag.")
else:
    print("The chain is NOT fully connected. Hardcoded names may be dormant.")
print()

# 6. Check for a feature flag / enable switch
print("Flags referencing state reconstruction in msc_compiler:")
for m in re.finditer(r"(enable|use|with|disable)\w*\s*[:=]\s*(True|False|bool\([^)]*\))", msc):
    print(f"  {m.group(0)}")
print()

# 7. Does state_reconstructor have a guard?
print("state_reconstructor constructor / guards:")
for m in re.finditer(r"def (__init__|reconstruct|score_unit)\b[^\n]*", sr):
    print(f"  {m.group(0)[:90]}")
