"""Paired A/B comparison of two LoCoMo runner artefacts (McNemar exact).

Usage:
  uv run python scripts/benchmarks/_ab_compare.py --a cap24_coder7b --b opt2_coder7b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
NAMES = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop"}


def mcnemar_exact(x: dict[str, bool], y: dict[str, bool]) -> float:
    b = sum(1 for k in x if x[k] and not y[k])
    c = sum(1 for k in x if not x[k] and y[k])
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    p = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n) * 2
    return min(1.0, p)


def load(tag: str) -> dict[str, dict]:
    path = REPO / "benchmark_results" / "locomo1540" / f"{tag}.json"
    return {r["question_id"]: r for r in json.load(open(path, encoding="utf-8"))["results"]}


def report(name: str, keys: list[str], A: dict, B: dict) -> None:
    xa = {k: A[k]["is_correct"] for k in keys}
    xb = {k: B[k]["is_correct"] for k in keys}
    ca, cb = sum(xa.values()), sum(xb.values())
    n = len(keys)
    gained = sum(1 for k in keys if not xa[k] and xb[k])
    lost = sum(1 for k in keys if xa[k] and not xb[k])
    print(f"== {name}: n={n}")
    print(f"   A {100 * ca / n:6.2f}% ({ca})   B {100 * cb / n:6.2f}% ({cb})"
          f"   delta {100 * (cb - ca) / n:+.2f}pp")
    print(f"   gained {gained} / lost {lost}   McNemar exact p={mcnemar_exact(xa, xb):.4f}")
    for c in sorted(NAMES):
        ks = [k for k in keys if A[k]["category"] == c]
        if not ks:
            continue
        a_ = {k: A[k]["is_correct"] for k in ks}
        b_ = {k: B[k]["is_correct"] for k in ks}
        g = sum(1 for k in ks if not a_[k] and b_[k])
        l = sum(1 for k in ks if a_[k] and not b_[k])
        print(f"   {NAMES[c]:<11} {100 * sum(a_.values()) / len(ks):5.1f}% -> "
              f"{100 * sum(b_.values()) / len(ks):5.1f}%  "
              f"({sum(b_.values()) - sum(a_.values()):+3d})  "
              f"gained {g:<3} lost {l:<3} p={mcnemar_exact(a_, b_):.3f}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="cap24_coder7b")
    ap.add_argument("--b", default="opt2_coder7b")
    ap.add_argument("--drift", default="benchmark_results/retrieval/_cache_drift_qids.json")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    assert set(A) == set(B), "question sets differ"
    drift = set(json.load(open(REPO / args.drift, encoding="utf-8")))
    official = set(A)
    report(f"FULL set ({args.a} vs {args.b})", sorted(official), A, B)
    clean = sorted(official - drift)
    report(f"CLEAN set (normalizer identical, drift={len(official & drift)} excluded)",
           clean, A, B)
    print(f"drift rows inside the official 1,540: {len(official & drift)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
