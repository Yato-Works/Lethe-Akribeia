#!/usr/bin/env python3
"""M30-41 Phase 1f: Prove the reconstruct_world actor boost is live and
LoCoMo-specific.

state_reconstructor._extract_target_actor() hardcodes the same LoCoMo names
and gives +25/+20 in reconstruct_world()'s scoring loop (line 285-295).

This is called from msc_compiler.compile() line 417 - i.e. the production path.
"""
import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory_gen.recall.state_reconstructor import StateReconstructor

sr = StateReconstructor()

probes = [
    ("LoCoMo", "When did Caroline go to the LGBTQ support group?"),
    ("LoCoMo", "When did Melanie get hurt?"),
    ("LoCoMo", "When did Joanna plan to go to Nate's?"),
    ("synthetic", "When did Olivia go to the art exhibition?"),
    ("synthetic", "When did Aiden attend the poetry reading?"),
    ("synthetic", "When did Silas go to the wine tasting?"),
    ("synthetic", "In which month and year did Victor attend the art auction?"),
]

print("=" * 78)
print("M30-41 Phase 1f: reconstruct_world actor boost (live path)")
print("=" * 78)
print()
print("StateReconstructor._extract_target_actor() results:")
print()

locomo_hits = 0
synth_hits = 0
for kind, q in probes:
    r = sr._extract_target_actor(q)
    mark = "BOOST" if r else "none"
    if kind == "LoCoMo" and r:
        locomo_hits += 1
    if kind == "synthetic" and r:
        synth_hits += 1
    print(f"  [{kind:9s}] {q[:52]:52s} -> {str(r):10s} [{mark}]")
print()

print(f"LoCoMo-name queries boosted:    {locomo_hits}/3")
print(f"Synthetic-name queries boosted: {synth_hits}/4")
print()

print("=" * 78)
print("SCORE IMPACT")
print("=" * 78)
print()
print("In reconstruct_world() scoring (line 285-295):")
print("  u_source == target_actor        -> +25.0 (+15 if first-person)  = up to +40")
print("  target_actor in content_lower   -> +20.0")
print()
print("In UniversalEvidenceScorer.compute_score() (line 305-316):")
print("  u_speaker == target_actor       -> +15.0 (+10 if first-person)  = up to +25")
print("  target_actor in content_lower   -> +12.0")
print("  other KNOWN_ACTOR speaking      -> -8.0")
print()
print("These boosts fire ONLY when the query contains a LoCoMo name.")
print("For synthetic names the boosts are silently skipped.")
print()

print("=" * 78)
print("VERDICT")
print("=" * 78)
print()
print("CONFIRMED: The production path (locomo_adapter -> msc_compiler.compile")
print("-> reconstruct_world -> _extract_target_actor) contains name-based")
print("boosts of up to +40 that fire only for LoCoMo character names.")
print()
print("This is a benchmark-specific optimisation embedded in production code.")
print("It inflates LoCoMo retrieval but not unseen-name retrieval.")
print()

out = PROJECT_ROOT / "results/m30_41"
out.mkdir(parents=True, exist_ok=True)
(out / "reconstruct_world_probe.json").write_text(json.dumps({
    "locomo_boosted": locomo_hits,
    "synthetic_boosted": synth_hits,
    "max_boost_reconstruct_world": 40.0,
    "max_boost_evidence_scorer": 25.0,
    "note": "name-based boosts fire only for hardcoded LoCoMo names",
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved: {out / 'reconstruct_world_probe.json'}")
