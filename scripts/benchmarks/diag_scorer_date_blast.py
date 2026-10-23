"""Blast-radius measurement for the 60%-token-overlap date defect.

E.score credits a wrong DATE when the month and year match, because the day number
is 1-2 characters and gets dropped by the `len(w) > 2` filter:
    pred '10 May, 2023' vs gold '27 May, 2023'  ->  scored CORRECT
Measured over the category-2 temporal questions where the committer claims an
answer.  Reports how many CORRECT verdicts would flip to WRONG, and how many
currently-WRONG verdicts would be unaffected -- so the scorer change is priced
before it is made, not after.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from artificial_memory.skills.answer_committer import commit_answer  # noqa: E402

CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
DATA = ROOT / "datasets/external/locomo10.json"

_MONTHS = ("january february march april may june july august september october "
           "november december").split()


def looks_like_date(s: str) -> bool:
    low = s.lower()
    return any(m in low for m in _MONTHS) or bool(re.search(r"\b\d{4}\b", s))


def date_day(s: str) -> str | None:
    """The day-of-month a date string commits to, if it names one."""
    low = s.lower()
    for m in _MONTHS:
        mm = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{m}\b", low)
        if mm:
            return mm.group(1)
        mm = re.search(rf"\b{m}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", low)
        if mm:
            return mm.group(1)
    return None


def main() -> None:
    golds = {}
    for conv in json.load(open(DATA, encoding="utf-8")):
        for qa in conv.get("qa", []):
            if "question" in qa and "answer" in qa:
                golds[str(qa["question"]).lower().strip()] = str(qa["answer"])

    rows = [json.loads(l) for l in open(CACHE, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if str(r.get("category")) == "2"]

    n = scored_correct = flip_to_wrong = unaffected = 0
    flips = []

    for r in rows:
        gold = golds.get(r["question"].lower().strip())
        if gold is None:
            continue
        ans = commit_answer(r["question"], r["context"], 1)
        if not ans.used:
            continue
        pred = str(ans.answer)
        n += 1

        if not E.score(pred, gold):
            unaffected += 1
            continue
        scored_correct += 1

        # Only date-shaped answers can be affected by the day-number defect.
        if not (looks_like_date(pred) and looks_like_date(gold)):
            unaffected += 1
            continue
        pd_, gd_ = date_day(pred), date_day(gold)
        if pd_ and gd_ and pd_ != gd_:
            flip_to_wrong += 1
            flips.append((r["question"], pred, gold))
        else:
            unaffected += 1

    print("=== blast radius of the 60%-token-overlap date defect ===")
    print(f"  committer claims evaluated      : {n}")
    print(f"  currently scored CORRECT        : {scored_correct}")
    print(f"  would FLIP to wrong (bad day)   : {flip_to_wrong}")
    print(f"  unaffected                      : {unaffected}")
    print()
    if n:
        print(f"  share of correct that is spurious: "
              f"{flip_to_wrong/scored_correct*100:.1f}%")
    print()
    print("=== examples of wrong dates currently credited ===")
    for q, p, g in flips[:8]:
        print(f"  Q : {q[:54]}")
        print(f"  pred {p[:26]!r}   gold {g[:26]!r}")


if __name__ == "__main__":
    main()
