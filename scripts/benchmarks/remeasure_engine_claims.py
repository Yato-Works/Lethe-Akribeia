"""Re-measure the engine FIXED/BROKE claims with the numeral guard active.

The numeral guard removed 32 spurious credits across the stored runs.  Any engine
'FIXED' that was earned by a spurious credit must now be reported as what it
really is: not a fix.  This re-runs the recorded cohorts and re-scores them with
the corrected scorer, reporting (FIXED, BROKE, SPURIOUS) -- never a bare count.

A pair is SPURIOUS when:
  * it was recorded as fixed by the run, AND
  * under the corrected scorer it is still marked correct, BUT
  * the numerals it carries conflict with gold -- i.e. the credit came from the
    defect the guard removed, not from the engine.
"""
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = mod
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass
score = mod.score

NUMWORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "twice": 2,
}


def numerals(s: str) -> set[int]:
    out: set[int] = set()
    for w in re.findall(r"\b[a-z]+\b", str(s).lower()):
        if w in NUMWORDS:
            out.add(NUMWORDS[w])
    for d in re.findall(r"\b\d+\b", str(s)):
        out.add(int(d))
    return out


COHORTS = [
    ("counting discovery", "benchmark_results/rnd_counting_discovery_after_20261013.json"),
    ("counting after-final", "benchmark_results/rnd_counting_discovery_after_final_20261013.json"),
    ("counting after-final2", "benchmark_results/rnd_counting_discovery_after_final2_20261013.json"),
    ("temporal binding sprint", "benchmark_results/temporal_event_binding_sprint_eval_20261013.json"),
]


def main() -> int:
    grand = {"fixed": 0, "broke": 0, "spurious": 0}

    for label, rel in COHORTS:
        path = ROOT / rel
        if not path.exists():
            print(f"-- {label}: file missing, skipped")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        cases = data.get("records") or data.get("cases") or []
        fixed = broke = spurious = 0
        detail: list[str] = []

        for row in cases:
            pred = row.get("answer")
            gold = row.get("gold") or row.get("ground_truth")
            status = row.get("status")
            if not pred or not gold:
                continue
            now = score(pred, gold)
            if now and mod._numeral_contradiction(pred, gold):
                # credited only through the defect -> spurious, not a fix
                spurious += 1
                detail.append(
                    f"    SPURIOUS {row.get('question_id')} pred={str(pred)[:34]!r} "
                    f"gold={str(gold)[:24]!r}")
                continue
            if status == "abstain" and now:
                fixed += 1
            elif status in ("correct", "wrong") and not now:
                broke += 1

        grand["fixed"] += fixed
        grand["broke"] += broke
        grand["spurious"] += spurious
        print(f"== {label}  (n={len(cases)})")
        print(f"   FIXED={fixed}  BROKE={broke}  SPURIOUS={spurious}")
        for line in detail:
            print(line)
        print()

    print("=== TOTAL across cohorts (corrected scorer) ===")
    print(f"   FIXED   = {grand['fixed']}")
    print(f"   BROKE   = {grand['broke']}")
    print(f"   SPURIOUS= {grand['spurious']}   <- credits the defect produced, not the engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
