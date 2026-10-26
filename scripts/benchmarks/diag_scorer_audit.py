"""Audit the 83 scoring-suspect failures: too strict, or correctly rejected?

Eli's stage 1: fix the measurement before chasing score.  The failure taxonomy
found 83 failures where prediction and gold agree semantically but score() said
no.  Those split into two very different things:

  SCORER_TOO_STRICT   the answer is right and the scorer is wrong
  CORRECTLY_REJECTED  the scorer is right -- subset of an enumeration, dropped
                      date offset, different person, etc.  These are INTENTIONAL
                      rejections I added in ff0227f / 36bc56c to block inflation

Only the first group is a measurement bug.  Reporting the split matters because
"83 questions" sounds like 83 free points and it is not.

Prediction discipline, stated BEFORE running:
  predicted  10-25 genuine scorer bugs, ~0.5-1.3pt available
  mechanism  the suspect cases are dominated by enumeration partial matches,
             which the scorer rejects by design
  if it fails the scorer is already sound and the 83 are a policy question, not
             a bug -- which is itself the answer we need
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from artificial_memory.compiler.eval.audit_strict_grade import is_list_gold  # noqa: E402

DATA = ROOT / "datasets" / "external" / "locomo10.json"
RUN = ROOT / "benchmark_results" / "locomo_full_20261012_211601.json"

STOP = frozenset({
    "i", "my", "me", "we", "you", "the", "a", "an", "and", "or", "but", "to",
    "of", "in", "on", "at", "for", "with", "was", "were", "is", "are", "be",
    "been", "it", "its", "this", "that", "these", "those", "so", "very",
    "just", "she", "her", "he", "his", "they", "them", "their",
})


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def items(s: str) -> set[str]:
    return {norm(x) for x in re.split(r",|\band\b", str(s)) if norm(x)}


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    recorded = {r["question_id"]: r
                for r in json.loads(RUN.read_text(encoding="utf-8"))["results"]}

    verdicts = Counter()
    examples = {}

    for conv in data:
        turns, _ = E.conversation_turns(conv["conversation"])
        for q in E.load_questions(conv):
            r = recorded.get(q["question_id"])
            if r is None or r["is_correct"]:
                continue
            pred = str(r["predicted_answer"])
            gold = str(q["answer"])
            p, g = norm(pred), norm(gold)
            if not p or not g:
                continue
            pt, gt = set(p.split()), set(g.split())
            shorter = pt if len(pt) <= len(gt) else gt
            if len(pt & gt) / max(1, len(shorter)) < 0.6:
                continue                      # not a suspect case

            # --- classify the rejection -----------------------------------
            if is_list_gold(gold) or is_list_gold(pred):
                gi, pi = items(gold), items(pred)
                if gi and pi and gi != pi and (gi & pi):
                    key = "REJECTED_enumeration_subset_by_design"
                elif gi == pi:
                    key = "SCORER_BUG_same_items_scored_wrong"
                else:
                    key = "REJECTED_enumeration_different_items"
            else:
                # date offset: same digits, gold has an offset word
                gn = re.findall(r"\d+", g)
                pn = re.findall(r"\d+", p)
                offset = re.search(r"\b(before|after|of)\b", g)
                if gn and pn and gn == pn and offset:
                    key = "REJECTED_dropped_date_offset_by_design"
                elif p in g or g in p:
                    key = "SCORER_BUG_containment_missed"
                elif gn and pn and gn == pn:
                    key = "SCORER_BUG_same_digits_scored_wrong"
                else:
                    key = "REJECTED_genuinely_different"

            verdicts[key] += 1
            if key not in examples:
                examples[key] = (q["question"], gold, pred)

    total = sum(verdicts.values())
    print("=" * 78)
    print("SCORING-SUSPECT AUDIT -- too strict, or correctly rejected?")
    print("=" * 78)
    print(f"suspect cases examined: {total}")
    print()
    bugs = 0
    print(f"{'verdict':44s} {'n':>5s}  free points?")
    print("-" * 78)
    for k, c in sorted(verdicts.items(), key=lambda kv: -kv[1]):
        is_bug = k.startswith("SCORER_BUG")
        bugs += c if is_bug else 0
        print(f"{k:44s} {c:5d}  {'YES' if is_bug else 'no (by design)'}")
        qq, gt, pred = examples[k]
        print(f"    e.g. {qq[:54]}")
        print(f"         GT   {gt[:44]}")
        print(f"         PRED {pred[:44]}")
    print("-" * 78)
    print(f"genuine scorer bugs : {bugs} / {total}")
    print(f"point value if fixed: +{bugs/1986*100:.2f}pt")
    print()
    if bugs == 0:
        print("VERDICT: the scorer is SOUND.  All 83 suspect rejections are the")
        print("  intentional anti-inflation rules (enumeration subsets, dropped date")
        print("  offsets).  There are no free points here, and the measurement is")
        print("  trustworthy on this axis.")
    else:
        print(f"VERDICT: {bugs} cases are genuine scorer bugs worth "
              f"+{bugs/1986*100:.2f}pt.")
        print("  The rest are intentional rejections and must stay.")
    print()
    print("PREDICTION CHECK (stated before running: 10-25 bugs, 0.5-1.3pt):")
    print(f"  actual {bugs} bugs, +{bugs/1986*100:.2f}pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
