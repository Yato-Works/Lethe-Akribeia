#!/usr/bin/env python3
"""Three-layer evaluation report (Lethe Akribeia).

Separates the three evaluation layers defined by the system role:

  1. ORACLE RECALL   -- did Lethe put the required evidence in the context?
  2. READER          -- given that evidence, did the Reader answer correctly?
  3. END-TO-END      -- the product of both (query -> Lethe -> Reader -> answer)

Layer 1 is Lethe's metric.  Layer 2 is only measurable on the oracle-hit subset
(it is conditioned on Lethe having done its job).  Layer 3 is the headline number
and must never be quoted as a Lethe score on its own.

Usage:
  uv run python scripts/benchmarks/three_layer_report.py                     # all suites
  uv run python scripts/benchmarks/three_layer_report.py --suite locomo1540
  uv run python scripts/benchmarks/three_layer_report.py --latest
  uv run python scripts/benchmarks/three_layer_report.py --failures <file>   # isolation list
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def _rows(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("results", "details", "questions"):
            if isinstance(payload.get(key), list):
                return payload[key]
    if isinstance(payload, list):
        return payload
    return []


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(rows: list[dict]) -> dict:
    """Compute the three layers from a single run's rows."""
    n = len(rows)
    if not n:
        return {}
    correct = [bool(r.get("is_correct")) for r in rows]
    oracle = [bool(r.get("oracle_recall")) for r in rows]
    # Layer 2 = reader accuracy conditioned on the evidence being present.
    hit = [i for i in range(n) if oracle[i]]
    reader_n = len(hit)
    reader_acc = (100 * sum(correct[i] for i in hit) / reader_n) if reader_n else 0.0
    miss = [i for i in range(n) if not oracle[i]]
    # A miss that is still answered correctly is an unsupported guess that
    # happened to land -- it must never be counted as retrieval success.
    miss_correct = sum(1 for i in miss if correct[i])

    by_cat: dict[int, dict[str, int]] = defaultdict(lambda: {"n": 0, "hit": 0, "acc": 0, "hit_acc": 0})
    for i, r in enumerate(rows):
        c = int(r.get("category") or 0)
        slot = by_cat[c]
        slot["n"] += 1
        slot["hit"] += int(oracle[i])
        slot["acc"] += int(correct[i])
        slot["hit_acc"] += int(oracle[i] and correct[i])

    tokens = [r.get("tokens") for r in rows if isinstance(r.get("tokens"), (int, float))]
    lat = [r.get("latency_ms") for r in rows if isinstance(r.get("latency_ms"), (int, float))]

    return {
        "n": n,
        "e2e_accuracy": 100 * sum(correct) / n,
        "oracle_recall": 100 * sum(oracle) / n,
        "reader_accuracy_on_hits": reader_acc,
        "reader_n": reader_n,
        "oracle_misses": len(miss),
        "miss_still_correct": miss_correct,
        "miss_ids": [rows[i].get("question_id") or rows[i].get("qid") for i in miss],
        "by_cat": {c: dict(v) for c, v in sorted(by_cat.items())},
        "mean_tokens": statistics.mean(tokens) if tokens else None,
        "p95_latency": sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None,
    }


def print_summary(name: str, s: dict) -> None:
    if not s:
        return
    print(f"\n### {name}")
    print(
        f"  L1 oracle recall   : {s['oracle_recall']:6.2f}%  "
        f"(misses {s['oracle_misses']}, of which answered-right {s['miss_still_correct']})"
    )
    print(
        f"  L2 reader on hits  : {s['reader_accuracy_on_hits']:6.2f}%  "
        f"(n={s['reader_n']}; misses excluded -> reader is NOT blamed)"
    )
    print(f"  L3 end-to-end      : {s['e2e_accuracy']:6.2f}%  (n={s['n']})")
    if s["mean_tokens"]:
        print(f"  tokens/question    : {s['mean_tokens']:6.0f}")
    if s["p95_latency"]:
        print(f"  latency p95        : {s['p95_latency']:6.0f} ms")
    print("  per category (n | L1 | L2 | L3):")
    for c, v in s["by_cat"].items():
        name_ = CATEGORY_NAMES.get(c, str(c))
        l1 = 100 * v["hit"] / v["n"]
        l2 = 100 * v["hit_acc"] / v["hit"] if v["hit"] else 0.0
        l3 = 100 * v["acc"] / v["n"]
        print(f"    {name_:14s} {v['n']:5d} | {l1:6.2f} | {l2:6.2f} | {l3:6.2f}")


def find_latest(directory: Path, pattern: str) -> Path | None:
    hits = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Three-layer evaluation report")
    ap.add_argument("--suite", choices=["locomo1540", "locomo10", "longmemeval"], default=None)
    ap.add_argument("--latest", action="store_true", help="use the newest run file per suite")
    ap.add_argument("--file", default=None, help="explicit results JSON")
    ap.add_argument("--failures", default=None, help="write the oracle-miss id list to this path")
    args = ap.parse_args()

    targets: list[tuple[str, Path]] = []
    if args.file:
        targets.append(("explicit", REPO / args.file))
    else:
        suites = [args.suite] if args.suite else ["locomo1540", "longmemeval"]
        for suite in suites:
            if suite == "locomo1540":
                d = REPO / "benchmark_results" / "locomo1540"
                p = d / "temporal321_rules_commit_7b_postfix.json"
                if not p.exists():
                    p = d / "locomo_final.json"
                if args.latest:
                    cands = [x for x in d.glob("*.json")
                             if not x.name.startswith(("smoke", "test"))]
                    p = max(cands, key=lambda x: x.stat().st_mtime) if cands else p
            elif suite == "locomo10":
                p = REPO / "benchmark_results" / "locomo10" / "full_locomo10_report.json"
            else:
                d = REPO / "benchmark_results" / "longmemeval"
                p = d / "grand_longmemeval_report_7b_rescued_834.json"
                if not p.exists():
                    p = d / "grand_longmemeval_report_7b_apex_ctx8192_postfix.json"
                if not p.exists() or args.latest:
                    p = find_latest(d, "grand_longmemeval_report_*.json")
            if p and p.exists():
                targets.append((suite, p))

    all_miss: list[str] = []
    for name, path in targets:
        if not path.exists():
            print(f"!! missing {path}")
            continue
        rows = _rows(_load(path))
        if not rows or "is_correct" not in rows[0]:
            print(f"!! {path.name}: no is_correct field, skipping")
            continue
        s = summarize(rows)
        print_summary(f"{name} :: {path.relative_to(REPO)}", s)
        all_miss.extend(str(x) for x in s.get("miss_ids", []))

    if args.failures and all_miss:
        out = REPO / args.failures
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(all_miss) + "\n", encoding="utf-8")
        print(f"\nwrote {len(all_miss)} oracle-miss ids -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
