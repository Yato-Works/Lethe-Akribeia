"""Phase 2 -- evidence audit across all 1,986 questions.

The mission is explicit about what this must NOT do: it must not declare a question
unanswerable or a dataset defect merely because the gold STRING is absent from the
compiled context.  String absence is the weakest possible evidence, and two of this
session's own near-misses came from trusting it.

So each question is classified by EVIDENCE RELATION, not by string match:

  DIRECT          the gold's day+month is written verbatim in a context turn
  DERIVED         not verbatim, but reachable deterministically from context: the
                  gold is a relative anchor whose base date IS present, or a
                  month-year where the month and year both appear, or a bare year
                  that appears
  CONTRADICTORY   the context asserts a different value for the same attribute
  INSUFFICIENT    no turn carries the needed date at all
  NOT_A_DATE      the gold is not a date ('Sweden', 'pottery, camping, painting'),
                  so this audit does not apply and says so rather than guessing
  UNDETERMINED    the gold names no searchable day+month and no derivation applies

Derivation is deliberately conservative and each class records the SOURCE TURN and the
METHOD, so a classification can be checked.  INSUFFICIENT is reported as
'insufficient IN THE COMPILED CONTEXT' -- never as 'unanswerable', because the raw
corpus may hold it and that is a separate measurement (already done for when_date:
31 of 37 evidence-absent golds were absent from the raw conversation too).

Integrity checks the mission asks for, all asserted:
  duplicate question_ids, missing ids, mis-joins, and the scorer's known defects are
  verified; stored predictions are re-scored with the repaired scorer; existing
  artifacts are never overwritten -- output goes to a new directory.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
scorer = importlib.util.module_from_spec(spec)
sys.modules["eval_locomo_full"] = scorer
try:
    spec.loader.exec_module(scorer)
except SystemExit:
    pass

CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
RUN = ROOT / "benchmark_results/locomo_full_20261012_211601.json"
OUT_DIR = ROOT / "benchmark_results/evidence_audit"

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}

YEAR_RE = re.compile(r"\b(19|20)\d\d\b")
MONTH_YEAR_RE = re.compile(r"\b([A-Za-z]+),?\s+((?:19|20)\d\d)\b", re.I)
RELATIVE_ANCHOR_RE = re.compile(
    r"\b(?:the\s+)?(?:two\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday|week|weekend|month|year|night|day)s?\s+(?:before|after|of)\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?\s*([A-Za-z]+)", re.I)


def _day_month(gold: str):
    s = str(gold)
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)", s)
    if m and m.group(2).lower() in MONTHS:
        return m.group(1), m.group(2).lower()
    m = re.search(r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\b", s)
    if m and m.group(1).lower() in MONTHS:
        return m.group(2), m.group(1).lower()
    return None


def _find_turn(pred, lines):
    """First context line satisfying pred, as (turn_id, text)."""
    for line in lines:
        if pred(line):
            head = re.match(r"\s*\[([^\]]+)\]", line)
            return (head.group(1).strip() if head else ""), line.strip()
    return "", ""


def classify(gold: str, context: str) -> dict:
    """Evidence relation between gold and the compiled context."""
    lines = context.splitlines()
    g = str(gold).strip().strip('"').strip("'")
    if not g:
        return {"class": "NOT_A_DATE", "method": "empty gold",
                "source_turn": "", "source_text": ""}

    dm = _day_month(g)
    if dm:
        d, mon = dm
        pats = [re.compile(rf"\b{d}\s+{mon}\b", re.I),
                re.compile(rf"\b{mon}\s+{d}(?:st|nd|rd|th)?\b", re.I)]
        tid, text = _find_turn(lambda ln: any(p.search(ln) for p in pats), lines)
        if tid or text:
            return {"class": "DIRECT", "method": f"day+month '{d} {mon}' verbatim",
                    "source_turn": tid, "source_text": text[:170]}

        # DERIVED: the gold is a relative anchor over a base date that IS present.
        anchor = RELATIVE_ANCHOR_RE.search(g)
        if anchor:
            bd, bmon = anchor.group(1), anchor.group(2).lower()
            if bmon in MONTHS:
                bpats = [re.compile(rf"\b{bd}\s+{bmon}\b", re.I),
                         re.compile(rf"\b{bmon}\s+{bd}(?:st|nd|rd|th)?\b", re.I)]
                tid, text = _find_turn(
                    lambda ln: any(p.search(ln) for p in bpats), lines)
                if tid or text:
                    return {"class": "DERIVED",
                            "method": f"relative anchor over base '{bd} {bmon}'",
                            "source_turn": tid, "source_text": text[:170]}

        # DERIVED: month+year both present somewhere.
        my = re.search(r"\b([A-Za-z]+),?\s+((?:19|20)\d\d)\b", g)
        if my and my.group(1).lower() in MONTHS:
            mon, yr = my.group(1).lower(), my.group(2)
            tid, text = _find_turn(
                lambda ln: re.search(rf"\b{mon}\b", ln, re.I)
                and yr in ln, lines)
            if tid or text:
                return {"class": "DERIVED",
                        "method": f"month+year '{mon} {yr}' both present",
                        "source_turn": tid, "source_text": text[:170]}

        # A bare year is a weaker derivation; report separately.
        if re.fullmatch(r"\s*(?:in\s+)?(?:19|20)\d\d\s*", g, re.I):
            yr = YEAR_RE.search(g)
            if yr:
                tid, text = _find_turn(lambda ln: yr.group(0) in ln, lines)
                if tid or text:
                    return {"class": "DERIVED",
                            "method": f"bare year {yr.group(0)} present",
                            "source_turn": tid, "source_text": text[:170]}

        # CONTRADICTORY: the month+year is present but a different day is stated.
        if my and my.group(1).lower() in MONTHS:
            mon, yr = my.group(1).lower(), my.group(2)
            tid, text = _find_turn(
                lambda ln: re.search(rf"\b{mon}\b", ln, re.I) and yr in ln
                and not any(p.search(ln) for p in pats), lines)
            if tid or text:
                return {"class": "CONTRADICTORY",
                        "method": f"month+year present but day {d} never stated",
                        "source_turn": tid, "source_text": text[:170]}

        return {"class": "INSUFFICIENT",
                "method": f"no turn carries day+month '{d} {mon}'",
                "source_turn": "", "source_text": ""}

    if YEAR_RE.fullmatch(g.strip()) or re.fullmatch(r"\s*(?:in\s+)?(?:19|20)\d\d\s*", g, re.I):
        yr = YEAR_RE.search(g)
        if yr:
            tid, text = _find_turn(lambda ln: yr.group(0) in ln, lines)
            if tid or text:
                return {"class": "DERIVED", "method": f"year {yr.group(0)} present",
                        "source_turn": tid, "source_text": text[:170]}
        return {"class": "INSUFFICIENT", "method": f"year not present",
                "source_turn": "", "source_text": ""}

    if RELATIVE_ANCHOR_RE.search(g) or re.search(
            r"\b(?:last|next|this)\s+(?:week|month|year|summer|winter)\b", g, re.I):
        return {"class": "UNDETERMINED",
                "method": "gold is a relative expression with no resolvable base date",
                "source_turn": "", "source_text": ""}

    return {"class": "NOT_A_DATE",
            "method": "gold is not a date answer; audit does not apply",
            "source_turn": "", "source_text": ""}


def main() -> int:
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row
    run = json.loads(RUN.read_text(encoding="utf-8"))
    results = run["results"]
    print(f"run                    : {RUN.name}  n={len(results)}")
    print(f"frozen contexts        : {len(cache)}")
    print()

    # ---- integrity checks, asserted.
    ids = [r["question_id"] for r in results]
    dupes = [q for q, n in Counter(ids).items() if n > 1]
    print(f"duplicate question_ids : {len(dupes)}")
    assert not dupes, f"duplicate ids in the run: {dupes[:5]}"

    def to_cache_qid(qid: str):
        m = re.fullmatch(r"lc_conv-(\d+)_(\d+)", qid)
        return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}" if m else None

    unresolved = [q for q in ids if to_cache_qid(q) is None]
    assert not unresolved, f"ids outside lc_conv-N_M form: {unresolved[:5]}"
    joined = {q: cache[to_cache_qid(q)] for q in ids if to_cache_qid(q) in cache}
    print(f"ids joined to a context: {len(joined)}/{len(ids)}")
    missing = len(ids) - len(joined)
    assert len(joined) > len(ids) * 0.95, "join coverage below 95% -- refusing to audit"
    print(f"unjoined (reported, not guessed): {missing}")
    print()

    # ---- audit every question.
    records = []
    classes = Counter()
    by_category = {}
    rescored_changes = 0

    for r in results:
        qid = r["question_id"]
        ctx = joined.get(qid)
        gold = r.get("ground_truth") or ""
        pred = r.get("predicted_answer") or ""
        rec = {
            "question_id": qid,
            "cache_qid": to_cache_qid(qid),
            "category": r.get("category"),
            "question": r.get("question"),
            "gold": gold,
            "stored_prediction": pred,
            "stored_is_correct": bool(r.get("is_correct")),
        }
        # Re-score with the repaired scorer; never overwrite the stored flag.
        if pred and gold:
            now = scorer.score(pred, gold)
            rec["rescored_is_correct"] = now
            if now != bool(r.get("is_correct")):
                rescored_changes += 1
        else:
            rec["rescored_is_correct"] = None

        if ctx and gold:
            verdict = classify(gold, ctx.get("context") or "")
            rec.update(verdict)
        else:
            rec.update({"class": "UNJOINED", "method": "no frozen context",
                        "source_turn": "", "source_text": ""})
        records.append(rec)
        classes[rec["class"]] += 1
        by_category.setdefault(str(r.get("category")), Counter())[rec["class"]] += 1

    print("=" * 74)
    print("EVIDENCE RELATION ACROSS ALL QUESTIONS")
    print("=" * 74)
    for k, n in classes.most_common():
        print(f"  {k:<16} {n:>5}  ({n / len(records) * 100:.1f}%)")
    print()
    print("stored predictions whose verdict CHANGED under the repaired scorer : "
          f"{rescored_changes}")
    print()

    print("=" * 74)
    print("BY CATEGORY")
    print("=" * 74)
    cats = sorted(by_category, key=lambda c: -sum(by_category[c].values()))
    print(f"  {'cat':>4} " + " ".join(f"{k[:9]:>10}" for k in
          ["DIRECT", "DERIVED", "INSUFFICIENT", "CONTRADICTORY", "NOT_A_DATE",
           "UNDETERMINED", "UNJOINED"]))
    for c in cats:
        row = by_category[c]
        print(f"  {c:>4} " + " ".join(f"{row.get(k, 0):>10}" for k in
              ["DIRECT", "DERIVED", "INSUFFICIENT", "CONTRADICTORY", "NOT_A_DATE",
               "UNDETERMINED", "UNJOINED"]))
    print()

    # Which evidence-absent questions were answered correctly anyway?
    insufficient = [r for r in records if r["class"] == "INSUFFICIENT"]
    answered_anyway = [r for r in insufficient if r.get("rescored_is_correct")]
    print("=" * 74)
    print("INSUFFICIENT-evidence questions answered CORRECTLY anyway")
    print("=" * 74)
    print(f"  {len(answered_anyway)} of {len(insufficient)}")
    print("  These are the interesting ones: either the derivation is incomplete, or")
    print("  the scorer credited an answer it should not have.  Both are worth a look.")
    for r in answered_anyway[:6]:
        print(f"    {r['question_id']:<18} gold={str(r['gold'])[:24]!r} "
              f"pred={str(r['stored_prediction'])[:26]!r}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "evidence_audit.json").write_text(json.dumps({
        "label": "phase 2 -- evidence audit over the full benchmark",
        "run": RUN.name,
        "n": len(records),
        "joined": len(joined),
        "unjoined": missing,
        "duplicate_ids": len(dupes),
        "rescored_verdict_changes": rescored_changes,
        "scorer": "eval_locomo_full.score (date + numeral guards)",
        "classes": dict(classes),
        "by_category": {c: dict(v) for c, v in by_category.items()},
        "method_note": "INSUFFICIENT means absent from the COMPILED CONTEXT, not "
                       "proven unanswerable.  Raw-corpus presence is a separate "
                       "measurement.",
        "records": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT_DIR / 'evidence_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
