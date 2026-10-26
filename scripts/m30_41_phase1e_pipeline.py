#!/usr/bin/env python3
"""M30-41 Phase 1e: Confirm the production compile() pipeline is live.

Read msc_compiler.compile() and list the ordered stages actually invoked.
This determines whether Phase 2-6 must IMPLEMENT or VERIFY.
"""
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "artificial_memory"
msc = (SRC / "context/msc_compiler.py").read_text(encoding="utf-8", errors="ignore")

print("=" * 78)
print("M30-41 Phase 1e: compile() pipeline stages")
print("=" * 78)
print()

# Find the main compile method
lines = msc.splitlines()
in_compile = False
stage_lines = []
for i, ln in enumerate(lines, 1):
    if re.match(r"\s*def compile\b", ln):
        in_compile = True
        print(f"  compile() at line {i}")
        continue
    if in_compile:
        if re.match(r"\s*def \w+", ln) and "def compile" not in ln:
            in_compile = False
            continue
        # self.<component>.<method>(...)
        m = re.search(r"self\.(\w+)\.(\w+)\(", ln)
        if m:
            stage_lines.append((i, m.group(1), m.group(2)))

print()
print("Pipeline stages (component.method):")
for ln, comp, meth in stage_lines:
    print(f"  L{ln}: {comp}.{meth}()")
print()

# Which mission capabilities map to which stage
mapping = {
    "QueryPlanner": "Phase 3 (Query Compiler)",
    "TemporalResolver": "Phase 5 (Temporal Resolution)",
    "StateSupersessionEngine": "Phase 5 (state update)",
    "StateReconstructor": "Phase 4 (Evidence Selection/Join)",
    "AdaptiveEvidenceSearcher": "Phase 4 (Evidence Selection)",
    "PropositionIntegrityGate": "Phase 6 (Answer Verification)",
    "AnswerVerifier": "Phase 6 (Answer Verification)",
    "checker": "Phase 6 (Proof certificate)",
}

print("=" * 78)
print("MISSION-PHASE MAPPING")
print("=" * 78)
print()
present = set(comp for _, comp, _ in stage_lines)
for comp, phase in mapping.items():
    mark = "LIVE" if comp in present else "not in compile()"
    print(f"  {comp:28s} -> {phase:38s} [{mark}]")
print()

print("=" * 78)
print("CONCLUSION")
print("=" * 78)
print()
live = [c for c in mapping if c in present]
print(f"Mission capabilities ALREADY IMPLEMENTED AND LIVE: {len(live)}/{len(mapping)}")
print()
print("Per constraint 6 (do not reimplement existing mechanisms),")
print("Phase 2-6 must VERIFY and MEASURE these existing stages,")
print("not write new parallel implementations.")
print()
print("The REAL open questions from Phase 1:")
print("  1. The hardcoded KNOWN_ACTORS actor boost (Phase 1 leakage finding)")
print("  2. Whether these stages actually improve E2E accuracy on unseen data")
print("     (M30-33..40 synthetic sets bypassed msc_compiler entirely)")
