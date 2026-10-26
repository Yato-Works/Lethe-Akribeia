#!/usr/bin/env python3
"""M30-41 Phase 1g: Quantify the actor-boost leak (read-only, deterministic).

Measure how much the hardcoded KNOWN_ACTORS boost changes the SCORE and
RANK of candidate turns, for a LoCoMo-style name vs a synthetic name.

No production change. No LLM. Pure scorer arithmetic.

Method: build identical candidate units; run compute_score twice —
  (a) with the real query (name in KNOWN_ACTORS)
  (b) with the name replaced by a synthetic one
and compare the score gap attributable to the actor term.
"""
import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.core.ir.memory_types import ApexMemoryUnit
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory_gen.recall.evidence_scorer import UniversalEvidenceScorer


def make_unit(speaker, text):
    ir = StructuredIR(
        entity=speaker, property="statement", value=text,
        source=speaker, raw_content=f"{speaker}: {text}",
    )
    return ApexMemoryUnit(ir=ir)


scorer = UniversalEvidenceScorer()

# A LoCoMo-shaped scenario
LOCOMO = {
    "query": "When did Caroline go to the LGBTQ support group?",
    "target_name": "Caroline",
    "units": [
        ("Caroline", "I went to the LGBTQ support group yesterday."),      # the answer
        ("Melanie", "That sounds great, Caroline! I signed up for pottery."),  # about Caroline
        ("Melanie", "I went to my pottery class yesterday, it was fun."),   # distractor
        ("Caroline", "Thanks Melanie, I am glad you liked it."),            # filler by target
    ],
}

# The SAME scenario with a synthetic name (what the frozen sets use)
SYNTH = {
    "query": "When did Olivia go to the art exhibition?",
    "target_name": "Olivia",
    "units": [
        ("Olivia", "I went to the art exhibition yesterday."),
        ("Aiden", "That sounds great, Olivia! I signed up for a class."),
        ("Aiden", "I went to my class yesterday, it was fun."),
        ("Olivia", "Thanks Aiden, I am glad you liked it."),
    ],
}


def rank_units(scenario):
    scored = []
    for sp, txt in scenario["units"]:
        u = make_unit(sp, txt)
        b = scorer.compute_score(scenario["query"], u)
        scored.append((b.total if hasattr(b, "total") else sum(
            v for k, v in vars(b).items() if isinstance(v, (int, float))
        ), sp, txt, b))
    scored.sort(key=lambda x: -x[0])
    return scored


def breakdown_dict(b):
    return {k: round(v, 2) for k, v in vars(b).items() if isinstance(v, (int, float))}


print("=" * 78)
print("M30-41 Phase 1g: Actor-Boost Leak Quantification")
print("=" * 78)
print()

print("--- LoCoMo-style (name in KNOWN_ACTORS) ---")
print(f"Query: {LOCOMO['query']}")
locomo_ranked = rank_units(LOCOMO)
for i, (tot, sp, txt, b) in enumerate(locomo_ranked, 1):
    print(f"  #{i} [{tot:8.2f}] {sp:9s} | {txt[:52]}")
    print(f"        {breakdown_dict(b)}")
print()

print("--- Synthetic-style (name NOT in KNOWN_ACTORS) ---")
print(f"Query: {SYNTH['query']}")
synth_ranked = rank_units(SYNTH)
for i, (tot, sp, txt, b) in enumerate(synth_ranked, 1):
    print(f"  #{i} [{tot:8.2f}] {sp:9s} | {txt[:52]}")
    print(f"        {breakdown_dict(b)}")
print()

# Extract the actor component difference
def actor_component(b):
    for k, v in vars(b).items():
        if "actor" in k.lower():
            return v
    return 0.0

locomo_actor = [actor_component(b) for _, _, _, b in locomo_ranked]
synth_actor = [actor_component(b) for _, _, _, b in synth_ranked]

print("=" * 78)
print("RESULT")
print("=" * 78)
print()
print(f"LoCoMo actor components:    {[round(x,2) for x in locomo_actor]}")
print(f"Synthetic actor components: {[round(x,2) for x in synth_actor]}")
print()

# Did the answer turn rank #1 in each?
loco_top = locomo_ranked[0][1]
synth_top = synth_ranked[0][1]
print(f"LoCoMo top-ranked speaker:    {loco_top} (target={LOCOMO['target_name']})")
print(f"Synthetic top-ranked speaker: {synth_top} (target={SYNTH['target_name']})")
print()

# The key claim: does the boost change the ordering?
loco_target_rank = next(i for i, (_, sp, _, _) in enumerate(locomo_ranked, 1) if sp == LOCOMO["target_name"])
synth_target_rank = next(i for i, (_, sp, _, _) in enumerate(synth_ranked, 1) if sp == SYNTH["target_name"])
print(f"Target speaker rank — LoCoMo: #{loco_target_rank}, Synthetic: #{synth_target_rank}")
print()

out = PROJECT_ROOT / "results/m30_41"
out.mkdir(parents=True, exist_ok=True)
(out / "actor_boost_quantification.json").write_text(json.dumps({
    "locomo_actor_components": [round(x, 2) for x in locomo_actor],
    "synthetic_actor_components": [round(x, 2) for x in synth_actor],
    "locomo_target_rank": loco_target_rank,
    "synthetic_target_rank": synth_target_rank,
    "note": "deterministic scorer arithmetic, no LLM, no production change",
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved: {out / 'actor_boost_quantification.json'}")
