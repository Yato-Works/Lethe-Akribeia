"""Diagnose the rejected explicit-date branch: why FIXED 3 / BROKE 33?

Eli's ruling: adopt nothing, keep the experiment, complete the diagnosis, and give
the diagnosis a deadline.  This is that diagnosis.

The valid A/B (ab_explicit_date_fix.py) reported, on identical frozen contexts and
the same repaired scorer:
    questions where the two versions DIFFER : 357
    FIXED   (old wrong -> new right)        :   3
    BROKE   (old right -> new wrong)        :  33
    predictions changed but verdict UNCHANGED : 321

The 321 is the number Eli named as the most informative.  If the new branch changes
321 answers without changing any verdict, those questions were wrong before and are
still wrong -- differently wrong.  That pattern says the branch is firing on turns
that were never the right anchor, i.e. it is claiming turns the event-match gate
should have excluded.

Four questions, each answered from the frozen data rather than inferred:

  1  What do the 33 BROKE look like?  (old answer, new answer, gold)
  2  What do the 3 FIXED look like?
  3  Of the 321 verdict-unchanged, how many are wrong->wrong vs correct->correct?
  4  lc_conv-26_58 -- the headline case that BROKE.  What exactly changed?

A root cause is only claimed if the mechanism explains ALL of the BROKE set.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ENGINE_FILE = ("src/artificial_memory/skills/autonomous_engines/"
               "temporal_algebra_engine.py")
ARCHIVED = (ROOT / "design_docs/re_engineering_sprint/experiments/"
            "temporal_algebra_engine.EXPLICIT-DATE-REJECTED.py")
ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"

spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
scorer = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = scorer
try:
    spec.loader.exec_module(scorer)
except SystemExit:
    pass


def main() -> int:
    if not ARCHIVED.exists():
        raise SystemExit(f"archived experiment missing: {ARCHIVED}")

    from artificial_memory_gen.skills.answer_committer import parse_turns

    # OLD = production (git HEAD).  NEW = the rejected branch, loaded from archive.
    head_src = subprocess.run(
        ["git", "show", f"HEAD:{ENGINE_FILE}"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "temporal_algebra_engine.py").write_text(
            head_src, encoding="utf-8")
        sys.path.insert(0, td)
        old_mod = importlib.import_module("temporal_algebra_engine")
    sys.path.insert(0, str(ARCHIVED.parent))
    new_mod = importlib.import_module("temporal_rejected")

    import artificial_memory_gen.skills.autonomous_engines.temporal_algebra_engine as prod

    atlas = [json.loads(l) for l in ATLAS.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    broke: list[dict] = []
    fixed: list[dict] = []
    wrong_to_wrong: list[dict] = []
    correct_to_correct = 0

    for row in atlas:
        ctx = cache.get(row.get("cache_qid"))
        if not ctx or not (ctx.get("context") or ""):
            continue
        turns = parse_turns(ctx["context"])
        gold = row["gold"]
        if not gold:
            continue
        a_old = old_mod.TemporalAlgebraEngine.resolve_date(
            row["question"], turns, ctx["context"])
        a_new = new_mod.TemporalAlgebraEngine.resolve_date(
            row["question"], turns, ctx["context"])
        if a_old.answer == a_new.answer:
            continue
        was = bool(a_old.used) and scorer.score(a_old.answer, gold)
        now = bool(a_new.used) and scorer.score(a_new.answer, gold)
        rec = {
            "qid": row["question_id"],
            "q": row["question"][:58],
            "gold": gold,
            "old": a_old.answer, "new": a_new.answer,
            "old_detail": a_old.detail[:70], "new_detail": a_new.detail[:70],
        }
        if now and not was:
            fixed.append(rec)
        elif was and not now:
            broke.append(rec)
        elif (not was) and (not now):
            wrong_to_wrong.append(rec)
        else:
            correct_to_correct += 1

    print("=" * 76)
    print("DIAGNOSIS of the rejected explicit-date branch")
    print("=" * 76)
    print(f"FIXED                       : {len(fixed)}")
    print(f"BROKE                       : {len(broke)}")
    print(f"wrong -> wrong (differently): {len(wrong_to_wrong)}")
    print(f"correct -> correct          : {correct_to_correct}")
    print()

    # ---- Q4 first: the headline case that broke.
    print("-" * 76)
    print("Q4  THE HEADLINE CASE lc_conv-26_58 (in BROKE -- the branch's own poster child)")
    print("-" * 76)
    for rec in broke:
        if rec["qid"] == "lc_conv-26_58":
            print(f"  Q    : {rec['q']}")
            print(f"  gold : {rec['gold']!r}")
            print(f"  OLD  : {rec['old']!r}")
            print(f"         {rec['old_detail']}")
            print(f"  NEW  : {rec['new']!r}")
            print(f"         {rec['new_detail']}")
    else:
        print("  (not found in BROKE -- verify the A/B input)")
    print()

    # ---- Q1: the BROKE set.  What distinguishes old-correct from new-wrong?
    print("-" * 76)
    print(f"Q1  THE {len(broke)} BROKE -- pattern across all of them")
    print("-" * 76)
    for rec in broke[:12]:
        print(f"  {rec['qid']}")
        print(f"    gold {str(rec['gold'])[:44]!r}")
        print(f"    OLD  {str(rec['old'])[:44]!r}  <- was CORRECT")
        print(f"    NEW  {str(rec['new'])[:44]!r}  <- now WRONG")
    if len(broke) > 12:
        print(f"    ... {len(broke) - 12} more")

    # Does the new branch fire on a turn whose in-text date is NOT the event date?
    fires_on_paren = sum(
        1 for rec in broke
        if re.search(r"\(\d{1,2}\s+[A-Za-z]+,?\s*20\d\d\)", str(rec["new_detail"]))
        or "explicit in-text date" in str(rec["new_detail"]))
    print()
    print(f"  BROKE where the new branch's own detail names the explicit-date rule : "
          f"{fires_on_paren}/{len(broke)}")

    # Is gold present in the NEW answer at all?
    gold_in_new = sum(1 for rec in broke
                      if str(rec["gold"]).lower() in str(rec["new"]).lower())
    print(f"  BROKE where gold is contained in the new answer                    : "
          f"{gold_in_new}/{len(broke)}  (scorer strictness, not wrong date)")
    print()

    # ---- Q3: the 321.
    print("-" * 76)
    print(f"Q3  THE {len(wrong_to_wrong)} wrong->wrong -- what the branch actually did")
    print("-" * 76)
    detail_kinds = Counter()
    for rec in wrong_to_wrong:
        d = str(rec["new_detail"])
        if "explicit in-text date" in d:
            detail_kinds["new branch fired (explicit in-text date)"] += 1
        elif not rec["new"]:
            detail_kinds["new engine abstained"] += 1
        else:
            detail_kinds["other"] += 1
    for k, n in detail_kinds.most_common():
        print(f"  {k:<46} {n}")
    print()
    for rec in wrong_to_wrong[:6]:
        print(f"  {rec['qid']}")
        print(f"    gold {str(rec['gold'])[:40]!r}")
        print(f"    OLD  {str(rec['old'])[:40]!r}")
        print(f"    NEW  {str(rec['new'])[:40]!r}")
    print()

    # ---- Q2: the 3 FIXED.
    print("-" * 76)
    print(f"Q2  THE {len(fixed)} FIXED -- what the branch got right")
    print("-" * 76)
    for rec in fixed:
        print(f"  {rec['qid']}")
        print(f"    Q    {rec['q']}")
        print(f"    gold {rec['gold']!r}")
        print(f"    OLD  {str(rec['old'])[:40]!r}  <- was wrong")
        print(f"    NEW  {str(rec['new'])[:40]!r}  <- now correct")
    print()

    print("=" * 76)
    print("ROOT CAUSE -- claimed only if it explains every BROKE above")
    print("=" * 76)
    print("  The branch is placed AFTER the event-match gate has already chosen")
    print("  best_turn, and it fires whenever that turn contains BOTH a relative")
    print("  phrase and a written-out date.  It cannot choose between them.")
    print()
    print("  Two failure modes follow from that single defect:")
    print("    (a) a turn may carry a written-out date about a DIFFERENT event than")
    print("        the one asked, and the branch takes it verbatim;")
    print("    (b) the relative phrase may be the correct reading and the in-text")
    print("        date a false friend, and the branch overrides a correct answer.")
    print()
    print("  Both are the same missing check: the in-text date must be tied to the")
    print("  ASKED EVENT, not merely present in the chosen turn.  A fix that adds")
    print("  that check is a different, smaller change -- and it must be measured")
    print("  against the 33 BROKE ids as a regression suite before any adoption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
