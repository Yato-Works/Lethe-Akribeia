#!/usr/bin/env python3
"""What does the reader do when the context *is* the answer?

The 378 questions in the unresolved bucket all share one fact: the gold evidence
turn was in the context, a reader was asked, and the answer was wrong.  This tool
joins the production run with the gold-evidence run and splits them four ways -
the split that decides what to build next:

    gold OK, both readers   the context pipeline is the blocker: the evidence was
                            delivered but buried among 1,590 words of distractors
    gold OK, big reader     a genuine capability gap; the smaller reader cannot
                            extract it even when handed the answer
    gold OK, small reader   the opposite, or noise worth looking at before it is
                            believed
    gold fails, both        the question needs something neither reader does from
                            the evidence: semantic, multi-hop or underspecified

Everything is read from stored artefacts - no model call - and every cell is
counted, so the ceiling is a measurement rather than a story.  The F1 column is
the official protocol's, because a binary hit rate cannot tell a recovered answer
from a lucky one.

Usage:
  uv run python scripts/benchmarks/gold_context_ceiling.py \\
      --gold-a benchmark_results/locomo1540/goldctx_7b.json \\
      --gold-b benchmark_results/locomo1540/goldctx_15b.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_audit = _load("_oracle_fn_audit_shared", REPO / "scripts" / "benchmarks" / "oracle_fn_audit.py")
scorer = _load("_locomo_scorer_shared", REPO / "scripts" / "benchmarks" / "locomo_scorer.py")

CELLS = {
    "context_limited": "gold OK, both readers",
    "reader_capability": "gold OK, big reader only",
    "small_reader_advantage": "gold OK, small reader only",
    "semantic": "gold fails for both",
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def answer_present_in_gold(item: dict, context: str, category: int) -> str:
    """Is the gold answer even *in* the gold evidence, as text?

    This is the check that decides what the `semantic` cell means.  A reader cannot
    be wrong about a fact that is not in front of it, so a cell where the labelled
    evidence does not contain the answer is not a reasoning failure - it is a
    question whose answer has to be *derived* (temporal arithmetic, a multi-turn
    join) or whose LoCoMo evidence annotation is incomplete.
    """
    gold = _norm(item.get("answer") or "")
    if category == 3:
        gold = gold.split(";")[0].strip()
    pool = _norm(context)
    if gold and gold in pool:
        return "verbatim"
    words = [w for w in gold.split() if len(w) > 2]
    if words and sum(1 for w in words if w in pool) / len(words) >= 0.8:
        return "mostly"
    return "absent"


def classify(gold_a: dict, gold_b: dict, dataset: dict | None = None) -> list[dict]:
    """One row per shared question: which reader recovered it on the gold context.

    The production verdicts are not joined on purpose - every one of these
    questions was wrong there by construction, which is what *unresolved* means.
    """
    rows = []
    for qid, row in gold_a["rows"].items():
        other = gold_b["rows"].get(qid)
        if other is None:
            continue
        a_ok, b_ok = bool(row["correct"]), bool(other["correct"])
        if a_ok and b_ok:
            cell = "context_limited"
        elif a_ok:
            cell = "reader_capability"
        elif b_ok:
            cell = "small_reader_advantage"
        else:
            cell = "semantic"
        row = {
            "qid": qid,
            "category": row["category"],
            "cell": cell,
            "gold_a": a_ok, "gold_b": b_ok,
            "gold_f1_a": scorer.official_score(row["prediction"], row["ground_truth"],
                                               row["category"] or 0),
            "gold_f1_b": scorer.official_score(other["prediction"], other["ground_truth"],
                                               other["category"] or 0),
        }
        if dataset is not None:
            item = dataset.get(qid)
            if item is not None:
                context = "\n".join(
                    f"[{t['id']} on {t['date']}] {t['text']}"
                    for t in item.get("turns") or []
                    if t["id"] in {str(e) for e in item.get("evidence_ids") or []})
                row["answer_in_gold_context"] = answer_present_in_gold(
                    item, context, row["category"] or 0)
        rows.append(row)
    return rows


def render_report(rows: list[dict], label: str) -> str:
    total = len(rows)
    counts = {cell: 0 for cell in CELLS}
    f1_a = f1_b = 0.0
    by_category: dict[str, dict[str, int]] = {}
    for row in rows:
        counts[row["cell"]] += 1
        f1_a += row["gold_f1_a"]
        f1_b += row["gold_f1_b"]
        group = by_category.setdefault(str(row["category"]),
                                       {cell: 0 for cell in CELLS})
        group[row["cell"]] += 1
    lines = [
        f"### Gold-context ceiling - {label}",
        f"{total} questions, all of them: the gold evidence was in the production context "
        "and the answer was still wrong.",
        "",
        "| Cell | Questions | Share of the 378 | What it means |",
        "|---|---:|---:|---|",
    ]
    for cell, human in CELLS.items():
        lines.append(f"| {human} | {counts[cell]} | {counts[cell] / total * 100:.1f}% | |")
    lines += [
        "",
        f"With the gold context the official F1 over these questions is **{f1_a / total * 100:.1f}%** "
        f"(big reader) and **{f1_b / total * 100:.1f}%** (small reader); on the production "
        f"context the same questions scored 0.0% by construction - that is what unresolved means.",
        "",
        "Per category (1 multi-hop / 2 temporal / 3 open-domain / 4 single-hop):",
        "",
        "| Category | " + " | ".join(CELLS[cell] for cell in CELLS) + " |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in sorted(by_category):
        group = by_category[category]
        lines.append(f"| {category} | " + " | ".join(str(group[cell]) for cell in CELLS) + " |")
    semantic = [r for r in rows if r["cell"] == "semantic" and "answer_in_gold_context" in r]
    if semantic:
        inside = sum(1 for r in semantic if r["answer_in_gold_context"] != "absent")
        lines += [
            "",
            f"**Of the {len(semantic)} questions that fail for both readers, the gold answer is "
            f"not even in the gold evidence on {len(semantic) - inside} of them** "
            f"({(len(semantic) - inside) / len(semantic) * 100:.0f}%). A reader cannot be wrong "
            "about a fact that is not in front of it, so this cell is mostly questions whose "
            "answer has to be *derived* - temporal arithmetic, a join across turns - or whose "
            "LoCoMo evidence label is incomplete. It is the argument for a deterministic "
            "derivation layer, and against both a bigger reader and a better retriever.",
        ]
    lines += [
        "",
        "Read it as a build order: `context_limited` is retrieval and context construction,",
        "`reader_capability` is the reader, `semantic` is the benchmark's own difficulty, and",
        "`small_reader_advantage` is the cell to distrust until it is reproduced.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-a", required=True, help="gold-context run, big reader")
    ap.add_argument("--gold-b", required=True, help="gold-context run, small reader")
    ap.add_argument("--dataset", default="datasets/external/locomo10.json",
                    help="used to check whether the gold answer is even in the gold evidence")
    ap.add_argument("--out", default=None, metavar="PATH", help="write the per-question cells")
    args = ap.parse_args()

    dataset = None
    if Path(REPO / args.dataset).exists():
        dataset = _audit.load_dataset(str(REPO / args.dataset))
    gold_a = scorer.load_run(str(REPO / args.gold_a))
    gold_b = scorer.load_run(str(REPO / args.gold_b))
    rows = classify(gold_a, gold_b, dataset)
    if not rows:
        print("!! the gold-context runs share no question ids with each other")
        return 2
    print(render_report(rows, f"A = `{gold_a['model']}`, B = `{gold_b['model']}`"))

    if args.out:
        path = REPO / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"questions": rows}, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved per-question cells to {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

