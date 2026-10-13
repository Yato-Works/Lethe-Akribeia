"""Measure spurious credit on numeral disagreement across every stored run.

The scorer's 50%-either-direction token-overlap rule can credit an answer whose
NUMERAL disagrees with gold whenever the two sides share a content word
("two dogs" vs "three dogs" -> correct).  This script measures how often that
happens on real prediction/gold pairs.

Reported as (TOTAL, numeral-conflicting pairs, spurious credits) -- never a bare
correct count.
"""
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "ev", ROOT / "scripts/benchmarks/eval_locomo_full.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["ev"] = mod
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass
score = mod.score

NUMWORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def numerals(s: str) -> set[int]:
    """Every cardinal value named in the string, digits or words."""
    out: set[int] = set()
    for w in re.findall(r"\b[a-z]+\b", str(s).lower()):
        if w in NUMWORDS:
            out.add(NUMWORDS[w])
    for d in re.findall(r"\b\d+\b", str(s)):
        out.add(int(d))
    return out


def main() -> int:
    runs = sorted((ROOT / "benchmark_results").glob("locomo_full_*.json"))
    total = conflicting = spurious = 0
    examples: list[tuple] = []
    seen: set[tuple] = set()

    for path in runs:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("results") or data.get("predictions") or []
        for row in rows:
            pred = row.get("predicted_answer")
            gold = row.get("ground_truth") or row.get("gold")
            if not pred or not gold:
                continue
            total += 1
            pn, gn = numerals(pred), numerals(gold)
            if not (pn and gn):
                continue          # not a numeral-bearing pair
            if pn & gn:
                continue          # numerals agree -- not this defect
            conflicting += 1
            if score(pred, gold):
                spurious += 1
                key = (str(row.get("question_id")), str(pred)[:40], str(gold)[:40])
                if key not in seen:
                    seen.add(key)
                    if len(examples) < 15:
                        examples.append(key)

    print(f"runs scanned                     : {len(runs)}")
    print(f"total pred/gold pairs            : {total}")
    print(f"pairs with CONFLICTING numerals  : {conflicting}")
    print(f"of those, credited correct       : {spurious}   <-- spurious")
    if conflicting:
        print(f"spurious rate on numeral pairs   : {spurious / conflicting * 100:.1f}%")
    print("\nexamples of a conflicting numeral credited correct:")
    for qid, pred, gold in examples:
        print(f"  {qid:<24} pred={pred!r:<44} gold={gold!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
