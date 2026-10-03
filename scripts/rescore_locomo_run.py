"""Re-score a stored LoCoMo run with the single frozen binary matcher.

Why this exists
---------------
Every published ``is_correct`` flag is produced by exactly one implementation,
``LoCoMoAdapter.score_binary`` (matcher-v2 in v0.3.0: generic rules only,
deterministic stemming).  This script re-applies that same implementation to
stored predictions offline - no LLM calls, no protocol change - so:

* a run artefact created by an older matcher revision can be brought up to the
  current revision (``--write``), and
* a published number can always be re-derived from the raw artefact.

Usage
-----
Report only (no writes), accepts run directories and run artefact JSONs::

    python scripts/rescore_locomo_run.py benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json
    python scripts/rescore_locomo_run.py benchmark_results/_official_scoring/run_temporal321_rules_commit_7b_postfix

Rewrite an artefact in place with matcher-v2 ``is_correct`` flags and a
provenance block::

    python scripts/rescore_locomo_run.py --write benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]

CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
CAT_NAME = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop", 5: "Adversarial"}
HOLDOUT_CONVS = ("conv-42", "conv-48")


def score(category: int, ground_truth: str, predicted_answer: str) -> bool:
    """The one matcher: delegate to the adapter's shared implementation."""
    return LoCoMoAdapter.score_binary(category, ground_truth, predicted_answer)


def _normalised_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise run-dir / artefact rows onto one schema."""
    rows: list[dict[str, Any]] = []
    for r in raw_rows:
        pred = r.get("prediction")
        if pred is None:
            pred = r.get("predicted_answer")
        rows.append(
            {
                "question_id": r.get("question_id", ""),
                "category": int(r.get("category", 0)),
                "ground_truth": str(r.get("ground_truth") or ""),
                "predicted_answer": str(pred or ""),
                "stored": bool(r.get("is_correct")),
            }
        )
    return rows


def load_run_dir(root: Path) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for p in sorted(root.glob("conv_*_results.json")):
        raw.extend(json.loads(p.read_text(encoding="utf-8"))["results"])
    return _normalised_rows(raw)


def report(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    for r in rows:
        r["rescored"] = score(r["category"], r["ground_truth"], r["predicted_answer"])

    n = len(rows)
    stored = sum(1 for r in rows if r["stored"])
    new = sum(1 for r in rows if r["rescored"])
    print(f"\n{name}: n={n}")
    print(f"  stored is_correct     : {stored / n * 100:6.2f}% ({stored})")
    print(f"  re-scored (matcher-v2): {new / n * 100:6.2f}% ({new})")
    print(f"  {'category':<14}{'n':>6}{'stored':>9}{'rescored':>10}")
    for cat in sorted({r["category"] for r in rows}):
        cat_rows = [r for r in rows if r["category"] == cat]
        so = sum(1 for r in cat_rows if r["stored"])
        ro = sum(1 for r in cat_rows if r["rescored"])
        print(f"  {CAT.get(cat, str(cat)):<14}{len(cat_rows):>6}"
              f"{so / len(cat_rows) * 100:>8.1f}%{ro / len(cat_rows) * 100:>9.1f}%")

    for label, keep in (("holdout (conv-42/48)", True), ("dev (8 convs)", False)):
        sub = [r for r in rows if (any(c in r["question_id"] for c in HOLDOUT_CONVS)) == keep]
        if sub:
            so = sum(1 for r in sub if r["stored"])
            ro = sum(1 for r in sub if r["rescored"])
            print(f"  {label:<14}{len(sub):>6}{so / len(sub) * 100:>8.1f}%{ro / len(sub) * 100:>9.1f}%")

    flips = [r for r in rows if r["stored"] != r["rescored"]]
    if flips:
        print(f"  flips ({len(flips)}):")
        for r in flips[:40]:
            print(f"    {r['question_id']:<18} stored={r['stored']!s:<5} -> {r['rescored']!s:<5} "
                  f"GT={r['ground_truth'][:42]!r} PRED={r['predicted_answer'][:42]!r}")
    return {"n": n, "stored": stored, "rescored": new}


def apply_rescore(blob: dict[str, Any]) -> int:
    """Rewrite ``is_correct`` / ``overall_accuracy`` / ``categories`` in place."""
    rows = _normalised_rows(blob["results"])
    flipped = 0
    for original, r in zip(blob["results"], rows):
        new = score(r["category"], r["ground_truth"], r["predicted_answer"])
        r["rescored"] = new
        if bool(original.get("is_correct")) != new:
            flipped += 1
        original["is_correct"] = new

    # ``overall_accuracy`` is stored as a percentage (0-100), matching the
    # original artefact schema; ``categories[...]["acc"]`` stays a fraction.
    blob["overall_accuracy"] = sum(1 for r in rows if r["rescored"]) / len(rows) * 100.0
    cats = blob.get("categories")
    if isinstance(cats, dict):
        for cat, name in CAT_NAME.items():
            if name not in cats:
                continue
            sub = [r for r in rows if r["category"] == cat]
            if not sub:
                continue
            corr = sum(1 for r in sub if r["rescored"])
            cats[name] = {"acc": corr / len(sub), "corr": corr, "tot": len(sub)}
    blob["scorer_revision"] = "matcher-v2"
    if not isinstance(blob.get("rescoring"), dict):
        blob["rescoring"] = {
            "matcher": "LoCoMoAdapter.score_binary",
            "from_revision": "matcher-v1",
            "tool": "scripts/rescore_locomo_run.py",
            "note": (
                "matcher-v1 artefacts contained ground-truth keyword branches and "
                "wiring-dependent stemming; matcher-v2 is generic and deterministic "
                "(see LoCoMoAdapter.score_binary docstring)"
            ),
        }
    return flipped


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="+",
                    help="run directory (conv_*_results.json) or run artefact JSON")
    ap.add_argument("--write", action="store_true",
                    help="rewrite JSON artefacts in place with matcher-v2 is_correct flags")
    args = ap.parse_args()

    for raw in args.paths:
        p = Path(raw)
        if not p.is_absolute():
            p = REPO / p
        shown = p.relative_to(REPO) if p.is_relative_to(REPO) else p
        if p.is_dir():
            report(str(shown), load_run_dir(p))
        elif p.suffix == ".json":
            blob = json.loads(p.read_text(encoding="utf-8"))
            raw_rows = blob.get("results")
            if not isinstance(raw_rows, list):
                print(f"!! {shown}: no 'results' list; skipping")
                continue
            report(str(shown), _normalised_rows(raw_rows))
            if args.write:
                flipped = apply_rescore(blob)
                p.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  wrote {shown} ({flipped} rows flipped)")
        else:
            print(f"!! {shown}: unrecognised target")
    return 0


if __name__ == "__main__":
    sys.exit(main())

