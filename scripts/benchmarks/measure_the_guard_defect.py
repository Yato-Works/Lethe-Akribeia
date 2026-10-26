"""The THE-guard defect: a bare article credits against any 'The X' gold.

Found while validating the phase-C probe results: the attribute probe returned
'the same way with my team' for gold 'The Alchemist' and the scorer called it CORRECT.
Reducing that case gives pred 'the' against gold 'The Alchemist' -> also CORRECT.

Mechanism.  The 50%-token-overlap rule keeps tokens with len(w) > 2, which ADMITS
'the' (3 chars) while dropping 1-2 digit days.  When gold's content words are all
either short or absent from the prediction, 'the' alone can carry 100% of the
surviving gold tokens:

    gold 'The Alchemist' -> gold tokens kept = ['the']   ('alchemist' unmatched)
    pred contains 'the'  -> hit = 1/1 = 100%            -> credited

So any gold of the form 'The <word>' is credited by any prediction containing the
word 'the'.  That is not a rare edge case -- LoCoMo titles, artworks and band names
are routinely gold, and 'the' is in most sentences.

This measures the blast radius across all stored runs before anything is changed:
  * how many pairs contain a 'The X' gold,
  * how many are credited with 'the' as the ONLY overlapping content token,
  * concrete examples, so the claim is checkable.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = mod
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass
score = mod.score

#: Tokens that carry no content.  'the' is the one that matters here.
_ARTICLES = {"the", "and", "for", "with", "that", "this", "from", "was", "were",
             "his", "her", "its", "their", "our", "your", "my", "him", "she",
             "they", "them", "not", "but", "all", "any", "out", "about"}


def content_only(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", str(s).lower())
            if len(w) > 2 and w not in _ARTICLES}


def main() -> int:
    runs = sorted((ROOT / "benchmark_results").glob("locomo_full_*.json"))
    the_gold = 0
    credited = 0
    only_article_overlap = 0
    examples: list[tuple] = []

    for path in runs:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                       # noqa: BLE001
            continue
        for r in (data.get("results") or []):
            gold, pred = r.get("ground_truth"), r.get("predicted_answer")
            if not gold or not pred:
                continue
            gstr = str(gold).strip()
            # Gold of the shape 'The X' (a title, artwork, band, book).
            if not re.match(r"^[\"']?the\s+\w+", gstr, re.I):
                continue
            the_gold += 1
            gold_content = content_only(gold)
            pred_tokens = set(re.findall(r"[a-z]+", str(pred).lower()))
            overlap = gold_content & pred_tokens
            if score(pred, gold):
                credited += 1
                if not overlap:
                    only_article_overlap += 1
                    if len(examples) < 15:
                        examples.append((
                            str(r.get("question_id")), str(pred)[:38],
                            gstr[:34]))
            del overlap

    print("=" * 74)
    print("THE-GUARD DEFECT -- blast radius across all stored runs")
    print("=" * 74)
    print(f"  runs scanned                              : {len(runs)}")
    print(f"  pairs whose gold is 'The X'               : {the_gold}")
    print(f"  of those, credited CORRECT                : {credited}")
    print(f"  credited with NO content-word overlap     : {only_article_overlap}")
    print()
    print("  'no content-word overlap' means the only shared surviving token was an")
    print("  article -- 'the' -- so the credit is an artifact of the len(w)>2 filter.")
    print()
    print("EXAMPLES (pred credited against a 'The X' gold with no content overlap):")
    for qid, pred, gold in examples:
        print(f"  {qid:<20} pred={pred!r:<40} gold={gold!r}")
    print()
    print("DIRECT REDUCTION, showing the mechanism in its purest form:")
    for p, g in [("the", "The Alchemist"), ("the", "The End"),
                 ("the book", "The Alchemist"), ("the end", "The End")]:
        gold_content = content_only(g)
        pred_tokens = set(re.findall(r"[a-z]+", p.lower()))
        print(f"  score({p!r:<12}, {g!r:<16}) = {score(p, g)!s:<6} "
              f"gold_content={sorted(gold_content)} "
              f"overlap={sorted(gold_content & pred_tokens)}")
    print()
    print("READ THIS AS: a defect of the same family as day (c449aa0), numeral")
    print("(e6a305f) and month (75b1f2f), but broader -- it does not need a wrong")
    print("value, only a shared article.  Fix belongs with its siblings: a guard that")
    print("requires at least one CONTENT token to overlap before the token-overlap")
    print("rules may credit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
