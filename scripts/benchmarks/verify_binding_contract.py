"""PHASE 1 validation -- the Evidence Binding contract against the real sentences.

Run before wiring anything into the engine.  Every case is a sentence that actually
occurs in the frozen LoCoMo contexts, with the verdict the contract must produce, so a
design flaw surfaces here rather than inside the production path.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory.skills.evidence_binding import (  # noqa: E402
    BindingCandidate,
    BindingRequest,
    BindingVerdicts,
    bind,
)

#: (question, entity, sentence, value, value_index_or_None, expected, why)
CASES: list[tuple] = [
    # ---- the two defects that motivated this ---------------------------------
    ("How long has John been surfing?", "john",
     "I started surfing five years ago (2018) and it's been great.",
     "five years", "MATCH",
     "the real ground: 'surfing' is in the value's own clause"),
    ("How long has John been surfing?", "john",
     "We were teammates for four years in high school, so we've played together "
     "for quite some time.",
     "four years", "NO_MATCH",
     "the measured defect: 'four years' attaches to 'teammates', not to surfing"),
    ("How long did it take for James to complete his Witcher-inspired book?", "james",
     "Thanks, I'm glad you can feel the atmosphere. I got the idea from a walk with "
     "my dogs two weeks ago.",
     "two weeks", "NO_MATCH",
     "'two weeks' is about a dog walk; the ask is about finishing a book"),

    # ---- clause scoping is what makes the above work -------------------------
    ("How long has John been surfing?", "john",
     "We played together for four years in high school, so we surfed often.",
     "four years", "NO_MATCH",
     "the surfing mention is in the NEXT clause, so it cannot license 'four years'"),

    # ---- the entity is necessary but must never be sufficient ----------------
    ("How long has John been playing the piano?", "john",
     "I've been working at the shop for 4 years now.",
     "4 years", "NO_MATCH",
     "same person, different activity -- exactly what must not be accepted"),

    # ---- cases that MUST match ----------------------------------------------
    ("How long has Caroline had her current group of friends?", "caroline",
     "I've had this group of friends for 4 years now.",
     "4 years", "MATCH", "'friends' is the asked event term"),
    ("How long did the trail hike that Audrey went on take?", "audrey",
     "The hike took us about two hours in total.",
     "two hours", "MATCH", "'hike' is in the value's clause"),
    ("How long does John usually hold the yoga pose?", "john",
     "I typically hold it for 30-60 seconds.",
     "30-60 seconds", "MATCH", "'hold' is the asked predicate"),

    # ---- insufficient evidence must not become a match ----------------------
    ("How long has Tim been playing the piano?", "tim",
     "It's been about four months.",
     "four months", "AMBIGUOUS",
     "no content word in the clause, so the event cannot be tied to the value"),
]


def main() -> int:
    print("=" * 74)
    print("PHASE 1 -- Evidence Binding contract, validated on real sentences")
    print("=" * 74)
    bad = 0
    for q, ent, sentence, value, want, why in CASES:
        idx = sentence.find(value)
        req = BindingRequest.for_question(q, entity=ent, value_type="DURATION")
        cand = BindingCandidate(text=sentence, turn_id="D0:0", value=value,
                               value_start=idx, value_end=idx + len(value))
        v = bind(req, cand)
        ok = v.verdict.value == want
        if not ok:
            bad += 1
        print(f"  {'PASS' if ok else 'FAIL'}  want={want:<10} got={v.verdict.value:<10} {why}")
        if not ok:
            print(f"        Q : {q}")
            print(f"        S : {sentence[:78]}")
            print(f"        clause: {v.clause[:70]!r}")
            print(f"        reason: {v.reason[:90]}")
        # The entity must never be the deciding factor.
        if v.verdict is BindingVerdicts.MATCH and not v.overlap:
            print("        VIOLATION: MATCH with no event overlap")
            bad += 1
    print()
    print(f"  {len(CASES) - bad}/{len(CASES)} cases correct")
    if bad:
        print("  THE CONTRACT IS NOT READY -- do not wire it into the engine")
        return 1
    print("  Every MATCH carries an event overlap, so entity agreement alone can never")
    print("  produce a commit.  That is the invariant the mission asked for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
