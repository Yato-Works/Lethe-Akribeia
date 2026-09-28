#!/usr/bin/env python3
"""The LoCoMo scoring contract: three policies, side by side, never one silently.

Step 3a.5 found 137 answers in each arm scored correct without the gold string
being in the prediction at all.  Two very different things can be going on there,
and only one of them is a bug:

* the **official protocol** really does truncate an open-domain gold answer at the
  first semicolon (``answer.split(';')[0]``), so a bare ``"Yes."`` is the *correct*
  answer to ``"Yes; it's classical music"``;
* everything else is **Lethe's own matcher**, which accepts a substring in either
  direction, accepts one word when the gold has three or fewer, and accepts 33% word
  overlap.  Those rules are Lethe's, not LoCoMo's.

So the score is not one number but a function of the policy, and this tool prints
all three rather than picking one:

  ``lethe_current``  the binary verdict the runner already stored (``is_correct``) -
                     what the scorecard publishes today.
  ``lethe_strict``   a binary containment rule: the normalised gold must appear in
                     the normalised prediction.  No partial credit, no substring in
                     the answer's favour.
  ``official_f1``    the LoCoMo protocol (snap-research/locomo, ``task_eval/
                     evaluation.py``): normalise (lowercase, drop punctuation, drop
                     a/an/the/and), token-overlap F1, category 3's gold truncated at
                     ``;``, category 1 (multi-hop) scored as partial F1 over its
                     sub-answers, categories 2/3/4 as plain F1, category 5 as the
                     refusal check.  Reported as a mean F1, i.e. **not** a 0/1 rate.

The point of printing three is that the absolute score is a policy choice while the
*comparison* is not: paired over two readers, the conclusion must not depend on which
column you read.  This tool reports the paired delta under each policy so that can be
checked instead of assumed.

Usage:
  uv run python scripts/benchmarks/locomo_scorer.py \\
      --run-a benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json \\
      --run-b benchmark_results/locomo1540/temporal321_rules_commit_15b.json
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
ARTICLES = re.compile(r"\b(a|an|the|and)\b")
OFFICIAL_SOURCE = "snap-research/locomo task_eval/evaluation.py (eval_question_answering, f1_score)"
POLICIES = ("lethe_current", "lethe_strict", "official_f1")


def normalize_official(text: str) -> str:
    """LoCoMo's ``normalize_answer``: lowercase, drop punctuation and articles."""
    text = str(text or "").replace(",", "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def f1_official(prediction: str, ground_truth: str) -> float:
    """Token-overlap F1 over the normalised strings, as the official scorer does."""
    pred_tokens = normalize_official(prediction).split()
    gold_tokens = normalize_official(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common: dict[str, int] = {}
    for token in pred_tokens:
        if token in gold_tokens:
            common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def official_score(prediction: str, ground_truth, category: int) -> float:
    """One question under the official protocol.

    Category 3's gold is truncated at the first ``;`` (the official harness does this,
    which is why a bare ``"Yes."`` is a correct open-domain answer), category 1 is
    scored as partial F1 over its sub-answers, categories 2/3/4 as plain F1, and
    category 5 as the refusal check.
    """
    if category == 5:
        text = str(prediction).lower()
        return 1.0 if ("no information available" in text or "not mentioned" in text) else 0.0
    if category == 1 and isinstance(ground_truth, (list, tuple)):
        parts = [f1_official(prediction, str(part)) for part in ground_truth]
        return sum(parts) / len(parts) if parts else 0.0
    gold = ground_truth
    if category == 3:
        gold = str(ground_truth).split(";")[0].strip()
    return f1_official(prediction, str(gold))


def strict_score(prediction: str, ground_truth) -> float:
    """Containment, no partial credit and no substring in the answer's favour."""
    gold = str(ground_truth)
    if isinstance(ground_truth, (list, tuple)):
        gold = " ; ".join(str(part) for part in ground_truth)
    gold_norm = normalize_official(gold)
    if not gold_norm:
        return float("no information available" in str(prediction).lower())
    return float(gold_norm in normalize_official(prediction))


def load_run(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {}
    for r in data.get("results", []):
        qid = str(r.get("question_id") or "")
        if qid:
            rows[qid] = {
                "correct": bool(r.get("is_correct")),
                "category": r.get("category") if isinstance(r.get("category"), int) else None,
                "prediction": str(r.get("prediction") or r.get("predicted_answer") or ""),
                "ground_truth": r.get("ground_truth"),
            }
    return {"path": str(path), "model": str(data.get("model") or Path(path).stem), "rows": rows}


def score_run(run: dict) -> dict:
    """Counts only, per policy and per category.  Every ratio is derived later."""
    policies = POLICIES
    totals = {p: {"n": 0, "sum": 0.0, "hits": 0} for p in policies}
    per_category: dict[str, dict] = {}
    for row in run["rows"].values():
        category = row["category"]
        group = per_category.setdefault(
            CATEGORY_NAMES.get(category, "unknown"),
            {p: {"n": 0, "sum": 0.0, "hits": 0} for p in policies})
        values = {
            "lethe_current": float(row["correct"]),
            "lethe_strict": strict_score(row["prediction"], row["ground_truth"]),
            "official_f1": official_score(row["prediction"], row["ground_truth"], category or 0),
        }
        for policy, value in values.items():
            totals[policy]["n"] += 1
            totals[policy]["sum"] += value
            totals[policy]["hits"] += int(value >= 1.0)
            group[policy]["n"] += 1
            group[policy]["sum"] += value
            group[policy]["hits"] += int(value >= 1.0)
    return {"totals": totals, "per_category": per_category}


def _rate(counts: dict) -> float:
    return counts["sum"] / counts["n"] * 100 if counts["n"] else 0.0


def render_report(a: dict, b: dict | None, sa: dict, sb: dict | None) -> str:
    lines = [
        "### LoCoMo scoring contract - one run set, three policies",
        f"Official protocol: `{OFFICIAL_SOURCE}`",
        "",
        "| Policy | Metric | A = `" + a["model"] + "` |"
        + (f" B = `{b['model']}` |" if b else " |"),
        "|---|---|---:|" + ("---:|" if b else ""),
    ]
    for policy in POLICIES:
        metric = "mean F1" if policy == "official_f1" else "hit rate"
        a_rate = _rate(sa["totals"][policy])
        if b is not None and sb is not None:
            b_rate = _rate(sb["totals"][policy])
            lines.append(f"| {policy} | {metric} | {a_rate:.1f}% | {b_rate:.1f}% |")
        else:
            lines.append(f"| {policy} | {metric} | {a_rate:.1f}% | |")
    if b is not None and sb is not None:
        lines += ["", "Paired delta (A - B) per policy - the conclusion has to survive all of them:", ""]
        for policy in POLICIES:
            a_rate, b_rate = _rate(sa["totals"][policy]), _rate(sb["totals"][policy])
            lines.append(f"* **{policy}**: {a_rate - b_rate:+.2f}pp")
    lines += ["", "Per category (A" + (" / B" if b else "") + "):", "",
              "| Category | N | " + " | ".join(POLICIES) + " |",
              "|---|---:|" + "---:|" * len(POLICIES)]
    for name in sorted(sa["per_category"]):
        cells = []
        for policy in POLICIES:
            rate = _rate(sa["per_category"][name][policy])
            if b is not None and sb is not None:
                rate = f"{rate:.1f} / {_rate(sb['per_category'][name][policy]):.1f}"
            cells.append(rate)
        n = sa["per_category"][name]["lethe_current"]["n"]
        lines.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Reading it: the absolute score is a policy choice, the paired delta is the claim.",
        "`official_f1` is graded, so it is lower than any binary rate by construction -",
        "it rewards a complete answer over a partial one, which is the official protocol.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Score LoCoMo runs under three policies")
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", default=None, help="second reader run, for the paired delta")
    ap.add_argument("--out", default=None, metavar="PATH", help="write the counts as JSON")
    args = ap.parse_args()

    a = load_run(str(REPO / args.run_a))
    b = load_run(str(REPO / args.run_b)) if args.run_b else None
    if not a["rows"]:
        print(f"!! run artefact carries no results: {args.run_a}")
        return 2
    sa = score_run(a)
    sb = score_run(b) if b else None
    print(render_report(a, b, sa, sb))

    if args.out:
        payload = {
            "official_source": OFFICIAL_SOURCE,
            "policies": list(POLICIES),
            "a": {"run": args.run_a, "model": a["model"], "scored": sa},
            "b": {"run": args.run_b, "model": b["model"], "scored": sb} if b else None,
        }
        path = REPO / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved scorer artefact to {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


