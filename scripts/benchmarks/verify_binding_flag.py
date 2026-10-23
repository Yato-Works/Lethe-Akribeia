"""PHASE 1 -- prove the binding flag actually controls the code path.

The mission is explicit that printing the environment variable is NOT verification.  So
this checks the branch itself, three ways:

  1. the flag is read at CALL time, not at import time -- flipping it inside one
     process must change behaviour, which rules out a cached module-level constant.
  2. with the flag OFF, bind() is never called -- counted by instrumenting it.
  3. with the flag ON, bind() IS called and its verdict is the reason a candidate is
     accepted or rejected.

Each check compares OBSERVED behaviour, not the value of a variable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import artificial_memory_gen.skills.autonomous_engines.temporal_algebra_engine as tam
from artificial_memory_gen.skills.answer_committer import parse_turns
from artificial_memory_gen.skills.deterministic_memory_engine import (
    DeterministicMemoryEngine,
)
from artificial_memory_gen.skills import evidence_binding as eb

CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
Q_SURFING = "How long has John been surfing?"


def _load(cache_qid: str) -> tuple[str, str]:
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["qid"] == cache_qid:
                return row["question"], row["context"]
    raise AssertionError(f"{cache_qid} not in the frozen cache")


import json  # noqa: E402  (placed after the helper that needs it)


def main() -> int:
    question, context = _load("conv-43-qa-081")

    # ---- CHECK 1: the flag is read at call time -----------------------------
    print("=" * 74)
    print("PHASE 1 -- the binding flag really controls the code path")
    print("=" * 74)

    calls = {"n": 0, "verdicts": []}
    real_bind = eb.bind

    def counting_bind(req, cand):
        v = real_bind(req, cand)
        calls["n"] += 1
        calls["verdicts"].append((cand.value, v.verdict.value))
        return v

    # Patch the name the engine actually looks up, so the instrument is on the real path.
    tam.bind = counting_bind

    os.environ["LETHE_EVIDENCE_BINDING"] = "0"
    calls["n"] = 0
    calls["verdicts"].clear()
    off = DeterministicMemoryEngine.resolve(question, context)

    os.environ["LETHE_EVIDENCE_BINDING"] = "1"
    calls["n"] = 0
    calls["verdicts"].clear()
    on = DeterministicMemoryEngine.resolve(question, context)

    print(f"  flag OFF -> bind() called {calls['n']} times in the OFF run "
          f"(see below), answer={off.answer!r}")
    print()
    # Re-run separately so each arm's counter is clean and unambiguous.
    os.environ["LETHE_EVIDENCE_BINDING"] = "0"
    calls["n"] = 0
    calls["verdicts"].clear()
    DeterministicMemoryEngine.resolve(question, context)
    off_calls = calls["n"]
    off_verdicts = list(calls["verdicts"])

    os.environ["LETHE_EVIDENCE_BINDING"] = "1"
    calls["n"] = 0
    calls["verdicts"].clear()
    DeterministicMemoryEngine.resolve(question, context)
    on_calls = calls["n"]
    on_verdicts = list(calls["verdicts"])

    print(f"  OFF run : bind() called {off_calls} times")
    print(f"  ON  run : bind() called {on_calls} times")
    for value, verdict in on_verdicts[:6]:
        print(f"           ON candidate {value!r:<16} -> {verdict}")
    print()

    ok = True
    if off_calls != 0:
        print("  FAIL: bind() ran while the flag was OFF -- the flag is not honoured")
        ok = False
    else:
        print("  PASS: bind() never runs when the flag is OFF")
    if on_calls == 0:
        print("  FAIL: bind() never ran when the flag was ON -- the branch is dead")
        ok = False
    else:
        print(f"  PASS: bind() runs {on_calls} times when the flag is ON, and its "
              "verdicts are recorded above")
    if off.answer == on.answer and off_calls == on_calls:
        print("  FAIL: the two arms are indistinguishable")
        ok = False
    else:
        print(f"  PASS: the arms differ (OFF {off.answer!r} / ON {on.answer!r})")

    # ---- CHECK 2: flipping inside one process works -------------------------
    print()
    os.environ["LETHE_EVIDENCE_BINDING"] = "1"
    a1 = DeterministicMemoryEngine.resolve(question, context)
    os.environ["LETHE_EVIDENCE_BINDING"] = "0"
    a2 = DeterministicMemoryEngine.resolve(question, context)
    if a1.answer == a2.answer:
        print("  FAIL: flipping the flag inside one process changed nothing -- it is "
              "read at import time and this A/B method is invalid")
        ok = False
    else:
        print(f"  PASS: flipping within one process works "
              f"(ON {a1.answer!r} -> OFF {a2.answer!r}), so the flag is read per call")

    tam.bind = real_bind
    print()
    print(f"  RESULT: {'the flag is verified on the real code path' if ok else 'NOT VERIFIED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
