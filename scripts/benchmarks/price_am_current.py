"""Price two arena runs as a paired comparison, question by question.

Usage:
    python scripts/benchmarks/price_am_current.py <dir_A> <dir_B> [--names A,B]

Joins the two runs' official scores by question_id and reports:
- paired deltas (both correct, A only, B only, both wrong) overall and per
  category;
- the accuracy of each run.

A and B must be arena run dirs containing scores.csv (each holding players
that were asked the same questions; any player is priced against any other).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def load_run(run_dir: Path, player: str | None = None) -> dict[str, dict]:
    """Official per-question rows of one run dir, keyed by question_id."""
    path = run_dir / "scores.csv"
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["is_official"] not in ("1", "True", "true"):
                continue
            if player and row["player"] != player:
                continue
            # keep the first official row per question
            out.setdefault(row["question_id"], row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="paired arena comparison")
    parser.add_argument("dir_a")
    parser.add_argument("dir_b")
    parser.add_argument("--names", default="A,B")
    parser.add_argument("--player-a", default=None)
    parser.add_argument("--player-b", default=None)
    args = parser.parse_args()
    name_a, name_b = (s.strip() for s in args.names.split(",", 1))

    a = load_run(Path(args.dir_a), args.player_a)
    b = load_run(Path(args.dir_b), args.player_b)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no common official questions between the two runs")

    cells = defaultdict(int)
    per_cat = defaultdict(lambda: defaultdict(int))
    for qid in common:
        ca = int(a[qid]["correct"] not in ("0", "False", "false", ""))
        cb = int(b[qid]["correct"] not in ("0", "False", "false", ""))
        key = ("both_pass" if ca and cb else
               "both_fail" if not ca and not cb else
               f"{name_a}_only" if ca else f"{name_b}_only")
        cells[key] += 1
        per_cat[a[qid]["category"]][key] += 1

    n = len(common)
    acc_a = sum(1 for qid in common
                if a[qid]["correct"] not in ("0", "False", "false", "")) / n * 100
    acc_b = sum(1 for qid in common
                if b[qid]["correct"] not in ("0", "False", "false", "")) / n * 100
    delta = (cells[f"{name_a}_only"] - cells[f"{name_b}_only"]) / n * 100

    print(f"\n paired comparison over {n} common official questions")
    print(f"  {name_a}: {acc_a:.1f}%   {name_b}: {acc_b:.1f}%   paired delta: {delta:+.1f}pt")
    for key in ("both_pass", f"{name_a}_only", f"{name_b}_only", "both_fail"):
        print(f"  {key:<14}: {cells[key]}")

    print(f"\n  {'category':<28} {'both':>5} {name_a+'_only':>14} {name_b+'_only':>14} {'none':>5}")
    for cat in sorted(per_cat):
        c = per_cat[cat]
        bp = c["both_pass"]
        a_only = c[f"{name_a}_only"]
        b_only = c[f"{name_b}_only"]
        bf = c["both_fail"]
        print(f"  {cat:<28} {bp:>5} {a_only:>14} {b_only:>14} {bf:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
