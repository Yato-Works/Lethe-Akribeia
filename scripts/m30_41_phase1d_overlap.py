#!/usr/bin/env python3
"""M30-41 Phase 1d: Architecture overlap survey.

The mission requires NOT reimplementing what exists. Survey the existing
modules that already provide proof-carrying / provenance / query-plan
capability, and check whether they are connected to the production path.

Deliverable: for each capability, EXISTS? / CONNECTED? / gap.
"""
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src" / "artificial_memory"

# Capabilities the mission asks for -> candidate existing modules
CAPABILITIES = {
    "provenance_evidence": [
        "recall/evidence_graph.py",
        "core/ir/proposition.py",
        "recall/proposition_graph.py",
    ],
    "query_plan": [
        "recall/query_planner.py",
        "compiler/query_analyzer.py",
        "compiler/execution_planner.py",
    ],
    "temporal_resolution": [
        "recall/temporal_resolver.py",
        "temporal/temporal_compiler.py",
        "recall/state_timeline.py",
    ],
    "answer_verification": [
        "recall/answer_verifier.py",
        "compiler/sufficiency_judge.py",
        "recall/evidence_sufficiency_gate.py",
        "recall/proposition_integrity_gate.py",
    ],
    "evidence_selection": [
        "recall/evidence_scorer.py",
        "recall/adaptive_search.py",
        "compiler/selectors/coverage_selector.py",
    ],
    "state_update": [
        "recall/state_supersession_engine.py",
        "recall/state_reconstructor.py",
    ],
}


def read(rel):
    p = SRC / rel
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="ignore")


def public_api(text):
    """Extract class/def names."""
    classes = re.findall(r"^class (\w+)", text, re.M)
    defs = re.findall(r"^    def (\w+)", text, re.M)
    return classes, defs


print("=" * 78)
print("M30-41 Phase 1d: Architecture Overlap Survey")
print("=" * 78)
print()

for cap, files in CAPABILITIES.items():
    print(f"### {cap}")
    for rel in files:
        text = read(rel)
        if text is None:
            print(f"  MISSING: {rel}")
            continue
        classes, defs = public_api(text)
        size = len(text.splitlines())
        print(f"  {rel}  ({size} lines)")
        if classes:
            print(f"    classes: {', '.join(classes[:6])}")
        # key public methods
        key = [d for d in defs if not d.startswith("_")]
        if key:
            print(f"    methods: {', '.join(key[:10])}")
    print()

# ---- Connection check: is each capability reachable from the locomo path? ----
print("=" * 78)
print("Connection to production path (locomo_adapter)")
print("=" * 78)
print()

adapter = read("research/benchmarks/external/locomo_adapter.py")
imported = set(re.findall(r"from artificial_memory\.([\w\.]+) import ([\w, ]+)", adapter))
flat_imports = []
for mod, names in imported:
    for n in names.split(","):
        flat_imports.append(n.strip())

print("locomo_adapter imports:")
for n in sorted(flat_imports):
    print(f"  {n}")
print()

# Which capability modules are in the import closure?
for cap, files in CAPABILITIES.items():
    connected = []
    for rel in files:
        modname = rel.replace("/", ".").replace(".py", "")
        # is it imported directly or transitively via msc_compiler?
        if modname.split(".")[-1] in adapter:
            connected.append(f"{rel} (direct)")
    # transitively via msc_compiler
    msc = read("context/msc_compiler.py") or ""
    for rel in files:
        base = rel.split("/")[-1].replace(".py", "")
        if base in msc:
            connected.append(f"{rel} (via msc_compiler)")
    status = connected if connected else ["NOT CONNECTED"]
    print(f"  {cap}: {status}")
print()

print("=" * 78)
print("GAP ANALYSIS")
print("=" * 78)
print()
print("Capabilities that EXIST but are NOT connected to the locomo path:")
print("  -> these are candidates for Phase 2-6 (connect, don't reimplement)")
print()
print("Capabilities already connected:")
print("  -> Phase 2-6 should verify their behaviour, not duplicate them")
