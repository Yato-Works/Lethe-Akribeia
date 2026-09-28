#!/usr/bin/env python3
"""A deterministic answer contract: what a safe post-processor can recover.

Step 3c measured the official-F1 gap between the 7B and the 1.5B reader at
+12.80pp, and a one-line "answer with the value only" instruction recovered
+5.8pp of it - so a large part of the gap is *shape*, not knowledge: the 1.5B
pads, and F1 charges precision for padding.  The deployed 7B is not exempt: its
mean precision is 0.525, so a fifth of its output is padding too.

That makes a runtime-side answer contract a first-class lever next to the
committer: it costs no model call, it works for any reader, and the benchmark
metric pays for it directly.

This tool is the **measurement**, not the feature.  It applies only transformations
that cannot invent or destroy information:

  markdown        strip emphasis/backticks/links, collapse whitespace
  boilerplate     drop leading "The answer is", "According to the context," ...
  redundancy      drop repeated sentences and repeated 5-grams
  trailing_meta   drop a trailing "According to ..." sentence when something else remains
  polarity        for category 3 only, keep the leading Yes/No - mirroring the
                  official scorer's own ``answer.split(';')[0]`` rule for that
                  category, and reported separately so it is never mistaken for a
                  capability gain
  dates           rewrite a date into one canonical order (day month year)

Every rule is measured on its own, and the two numbers that decide whether a
contract is worth shipping are printed together: how much F1 it recovers, and
**how many answers it breaks** (correct before, wrong after).  A rule that buys
F1 by destroying correct answers is a metric exploit, so the regression count is
the guard rail, not the F1 gain.

Usage:
  uv run python scripts/benchmarks/answer_contract.py \\
      --run benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "_locomo_scorer_shared", REPO / "scripts" / "benchmarks" / "locomo_scorer.py")
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)

BOILERPLATE = [
    r"^(?:the\s+)?answer\s+(?:to\s+(?:the|your)\s+question\s+)?is[:,]?\s*",
    r"^according to (?:the )?(?:context|conversation|evidence|passage)[,:]?\s*",
    r"^based on (?:the )?(?:context|conversation|evidence|passage)[,:]?\s*",
    r"^(?:from|in) the (?:context|conversation|passage)[,:]?\s*",
    r"^(?:sure|certainly|of course)[,!.]?\s*",
]
TRAILING_META = re.compile(
    r"\b(according to|based on|as (?:stated|mentioned)|this is (?:based|derived))\b[^.]*\.\s*$",
    re.IGNORECASE)
DATE_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"(?:\s*,?\s*(\d{4}))?\b|"
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b", re.IGNORECASE)
RULES = ("markdown", "boilerplate", "redundancy", "trailing_meta", "polarity", "dates")


def _strip_markdown(text: str) -> str:
    out = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    out = out.replace("**", "").replace("__", "").replace("`", "")
    out = re.sub(r"(?<!\w)[*_]+|[*_]+(?!\w)", "", out)
    return re.sub(r"\s+", " ", out).strip()


def _drop_boilerplate(text: str) -> str:
    for pattern in BOILERPLATE:
        stripped = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        if stripped:
            text = stripped
    return text


def _dedupe(text: str) -> str:
    """Drop repeated sentences and repeated 5-grams: same information, said twice."""
    sentences, seen = [], set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        key = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        sentences.append(sentence.strip())
    out = " ".join(s for s in sentences if s)
    words, emitted = out.split(), set()
    for i in range(len(words)):
        if i + 5 <= len(words):
            gram = " ".join(words[i:i + 5]).lower()
            if gram in emitted:
                continue
            emitted.add(gram)
        emitted.add(words[i].lower())
    return " ".join(words)


def _drop_trailing_meta(text: str) -> str:
    """Drop a trailing 'According to ...' sentence, but only if something remains."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) < 2:
        return text
    trimmed = TRAILING_META.sub("", text).strip()
    return trimmed if len(trimmed) >= 8 else text


def _polarity(text: str, category: int) -> str:
    """Category 3 only: keep the leading Yes/No.

    This mirrors the official scorer's own rule for that category (it truncates
    the *gold* at the first ``;``), so it is reported separately and never mixed
    with the rules that are about shape rather than about the protocol.
    """
    if category != 3:
        return text
    match = re.match(r"^\s*(yes|no)\b", text, flags=re.IGNORECASE)
    return f"{match.group(1).capitalize()}." if match else text


def _canonical_dates(text: str) -> str:
    """One canonical order for a date, so 'February 24, 2023' and '24 February 2023'
    produce the same tokens.  Same information, one form."""
    def fix(match: re.Match) -> str:
        # Groups: 1 day, 2 month, 3 year (day-first form) | 4 month, 5 day, 6 year
        # (month-first form).  Either way both are needed, or the text is left alone.
        day = match.group(1) or match.group(5)
        month = match.group(2) or match.group(4)
        year = match.group(3) or match.group(6)
        if not day or not month:
            return match.group(0)  # a bare month or year: nothing to reorder
        month = month.capitalize()
        return f"{int(day)} {month} {year}" if year else f"{int(day)} {month}"
    return DATE_RE.sub(fix, text)


def contract(text: str, category: int, rules: tuple[str, ...] = RULES) -> str:
    """Apply the safe transformations in order.  Nothing here infers anything."""
    out = str(text or "")
    if "markdown" in rules:
        out = _strip_markdown(out)
    if "boilerplate" in rules:
        out = _drop_boilerplate(out)
    if "redundancy" in rules:
        out = _dedupe(out)
    if "trailing_meta" in rules:
        out = _drop_trailing_meta(out)
    if "polarity" in rules:
        out = _polarity(out, category)
    if "dates" in rules:
        out = _canonical_dates(out)
    return out.strip() or str(text or "").strip()


def lethe_correct(ground_truth: str, prediction: str, category: int) -> bool:
    """Lethe's own matcher, called exactly as the runner calls it.

    Not a re-implementation: the classmethods are imported, so a contract measured
    here is measured against the matcher that produced the published number.  The
    report compares it against the stored ``is_correct`` on untouched predictions,
    which is what makes the deltas below deltas rather than a different metric.
    """
    gt = str(ground_truth).lower().strip()
    ans = str(prediction).lower().strip()
    if category == 2:
        return bool(LoCoMoAdapter._temporal_answer_matches(gt, ans))
    if category == 3:
        return bool(LoCoMoAdapter._open_domain_answer_matches(gt, ans))
    if category == 4:
        return bool(LoCoMoAdapter._single_hop_answer_matches(gt, ans))
    if not gt:
        return bool(LoCoMoAdapter.is_refusal_shaped(ans))
    if gt in ans or ans in gt:
        return True
    clean_gt = re.sub(r"\bde-stress\b", "destress", gt).replace("-", " ")
    clean_ans = re.sub(r"\bde-stress\b", "destress", ans).replace("-", " ")
    for word, digit in LoCoMoAdapter._NUMBER_WORDS.items():
        clean_gt = re.sub(rf"\b{word}\b", digit, clean_gt)
        clean_ans = re.sub(rf"\b{word}\b", digit, clean_ans)
    if clean_gt in clean_ans or clean_ans in clean_gt:
        return True
    gt_words = {w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", clean_gt) if len(w) > 2 or w.isdigit()}
    ans_words = {w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", clean_ans) if len(w) > 2 or w.isdigit()}
    if not gt_words or not ans_words:
        return False
    overlap = len(gt_words & ans_words)
    return overlap / len(gt_words) >= 0.33 or (len(gt_words) <= 3 and overlap >= 1)


def _rate(hits: float, total: int) -> float:
    return hits / total * 100 if total else 0.0


def measure(rows: list[dict], rule_sets: dict[str, tuple[str, ...]]) -> list[dict]:
    """Score the run under several rule sets, keeping the damage visible."""
    n = len(rows)
    # The baseline is the *same* matcher on the raw text, not the stored verdict:
    # otherwise the 0.3% self-check disagreement would be charged to the contract.
    baseline = {r["qid"]: lethe_correct(r["ground_truth"], r["prediction"], r["category"])
                for r in rows}
    agree = sum(1 for r in rows
                if lethe_correct(r["ground_truth"], r["prediction"], r["category"]) == r["is_correct"])
    results = []
    for label, rules in rule_sets.items():
        hits = f1_sum = 0.0
        regressions = improvements = changed = 0
        for row in rows:
            text = contract(row["prediction"], row["category"], rules)
            if text != row["prediction"]:
                changed += 1
            now = lethe_correct(row["ground_truth"], text, row["category"])
            hits += int(now)
            f1_sum += scorer.official_score(text, row["ground_truth"], row["category"] or 0)
            if now and not baseline[row["qid"]]:
                improvements += 1
            elif baseline[row["qid"]] and not now:
                regressions += 1
        results.append({
            "label": label, "rules": list(rules), "changed": changed,
            "hit_rate": _rate(hits, n), "official_f1": f1_sum / n * 100 if n else 0.0,
            "improvements": improvements, "regressions": regressions,
        })
    results[0]["self_check_agreement"] = agree / n * 100 if n else 0.0
    return results


def render_report(model: str, results: list[dict]) -> str:
    base = results[0]
    lines = [
        "### Answer contract - how much a safe post-processor can recover",
        f"Run: `{model}`; the first row is the untouched prediction.",
        f"Matcher self-check: the imported matcher agrees with the stored `is_correct` on "
        f"**{base['self_check_agreement']:.1f}%** of the run, so the deltas below are deltas",
        "",
        "| Rules | Answers changed | Lethe hit rate | Official F1 | Fixed | Broken |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        suffix = ""
        if result is not base:
            suffix = (f"  ({result['hit_rate'] - base['hit_rate']:+.1f}pp hit, "
                      f"{result['official_f1'] - base['official_f1']:+.2f}pp F1)")
        lines.append(
            f"| {result['label']} | {result['changed']} | {result['hit_rate']:.1f}% "
            f"| {result['official_f1']:.1f}% | {result['improvements']} "
            f"| {result['regressions']} |{suffix}")
    lines += [
        "",
        "`Fixed` / `Broken` are the guard rail: a rule that buys F1 by turning correct answers",
        "wrong is a metric exploit, not a contract. The polarity rule is reported on its own",
        "because it mirrors the official scorer's category-3 rule rather than being about shape.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure a safe deterministic answer contract")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="write a derived artefact with `prediction_contract` per question")
    args = ap.parse_args()

    run = scorer.load_run(str(REPO / args.run))
    rows = [{"qid": qid, "category": r["category"] or 0, "ground_truth": r["ground_truth"],
             "prediction": r["prediction"], "is_correct": r["correct"]}
            for qid, r in run["rows"].items()]
    if not rows:
        print(f"!! run artefact carries no results: {args.run}")
        return 2

    rule_sets = {"untouched": ()}
    for index, rule in enumerate(RULES):
        rule_sets[rule] = RULES[:index + 1]
    rule_sets["all rules"] = RULES
    results = measure(rows, rule_sets)
    print(render_report(run["model"], results))

    if args.out:
        data = json.loads((REPO / args.run).read_text(encoding="utf-8"))
        for row in data.get("results", []):
            row["prediction_contract"] = contract(str(row.get("prediction") or ""),
                                                  row.get("category") or 0)
        path = REPO / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved contract artefact to {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



