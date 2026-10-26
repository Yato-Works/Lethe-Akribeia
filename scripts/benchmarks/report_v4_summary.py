"""Type-wise accuracy/token/latency report for V4 pipeline runs.

Usage:
  python scripts/benchmarks/report_v4_summary.py RUN                 # single run
  python scripts/benchmarks/report_v4_summary.py BASELINE RUN        # paired compare
  python scripts/benchmarks/report_v4_summary.py BASELINE RUN --out report.md

Single-run output is the usual per-question-type list (accuracy, tokens
mean/median/p90/max, latency).  With two runs it adds a paired comparison
per type plus an exact McNemar test on the shared question ids -- one
variable changed per experiment, judged on paired outcomes only.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
REPO = Path(__file__).resolve().parents[2]


def load(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def type_rows(data: dict) -> list[dict]:
    res = data["results"]
    by = defaultdict(list)
    for r in res:
        by[r["question_type"]].append(r)
    rows = []
    for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        toks = [r["prompt_tokens"] for r in rs]
        lat = [r["latency_ms"] for r in rs]
        ordered = sorted(toks)
        rows.append({
            "type": t,
            "n": len(rs),
            "acc": sum(r["is_correct"] for r in rs) / len(rs),
            "tok_mean": st.mean(toks),
            "tok_med": st.median(toks),
            "tok_p90": ordered[max(0, int(len(ordered) * 0.9) - 1)],
            "tok_max": max(toks),
            "lat_mean": st.mean(lat),
        })
    return rows


def render_single(path: str, data: dict) -> str:
    cond = data.get("condition", "R0")
    out = [
        f"## {Path(path).name}  (condition={cond}, model={data.get('model', '?')})",
        "",
        f"Accuracy: **{data['correct_answers']}/{data['total_questions']} "
        f"({data['accuracy']:.1%})** | mean tokens {data['mean_prompt_tokens']:.0f} "
        f"| elapsed {data.get('elapsed_seconds', 0):.0f}s",
        "",
        f"{'question_type':<28} {'n':>4} {'acc':>7} {'tok_mean':>8} {'tok_med':>7} "
        f"{'tok_p90':>7} {'tok_max':>7} {'lat_ms':>7}",
        "-" * 86,
    ]
    for r in type_rows(data):
        out.append(
            f"{r['type']:<28} {r['n']:>4} {r['acc']*100:>6.1f}% {r['tok_mean']:>8.0f} "
            f"{r['tok_med']:>7.0f} {r['tok_p90']:>7} {r['tok_max']:>7} {r['lat_mean']:>7.0f}"
        )
    toks = [r["prompt_tokens"] for r in data["results"]]
    lat = [r["latency_ms"] for r in data["results"]]
    out.append("-" * 86)
    out.append(
        f"{'ALL':<28} {len(data['results']):>4} {data['accuracy']*100:>6.1f}% "
        f"{st.mean(toks):>8.0f} {st.median(toks):>7.0f} "
        f"{sorted(toks)[max(0, int(len(toks)*0.9)-1)]:>7} {max(toks):>7} {st.mean(lat):>7.0f}"
    )
    return "\n".join(out)


def render_compare(a_path: str, a: dict, b_path: str, b: dict) -> str:
    ra = {r["question_id"]: r for r in a["results"]}
    rb = {r["question_id"]: r for r in b["results"]}
    shared = sorted(set(ra) & set(rb))
    a_only = sum(1 for q in ra if q not in rb)
    b_only = sum(1 for q in rb if q not in ra)
    both_correct = sum(1 for q in shared if ra[q]["is_correct"] and rb[q]["is_correct"])
    a_hit = sum(1 for q in shared if ra[q]["is_correct"] and not rb[q]["is_correct"])
    b_hit = sum(1 for q in shared if rb[q]["is_correct"] and not ra[q]["is_correct"])
    p = mcnemar_exact(a_hit, b_hit)

    acc_a = sum(ra[q]["is_correct"] for q in shared) / len(shared)
    acc_b = sum(rb[q]["is_correct"] for q in shared) / len(shared)
    tok_a = st.mean([ra[q]["prompt_tokens"] for q in shared])
    tok_b = st.mean([rb[q]["prompt_tokens"] for q in shared])
    lat_a = st.mean([ra[q]["latency_ms"] for q in shared])
    lat_b = st.mean([rb[q]["latency_ms"] for q in shared])

    out = [
        f"## Paired comparison: {Path(a_path).name} (A) vs {Path(b_path).name} (B)",
        "",
        f"paired on {len(shared)} shared questions (A-only {a_only}, B-only {b_only})",
        f"- Accuracy: A {acc_a:.1%} vs B {acc_b:.1%} (delta {(acc_b-acc_a)*100:+.1f} pp)",
        f"- Mean tokens: A {tok_a:.0f} vs B {tok_b:.0f} (delta {tok_b-tok_a:+.0f})",
        f"- Mean latency: A {lat_a:.0f}ms vs B {lat_b:.0f}ms (delta {lat_b-lat_a:+.0f}ms)",
        f"- McNemar: both-correct {both_correct}, A-only {a_hit}, "
        f"B-only {b_hit}, exact two-sided p = {p:.4f}",
        "",
        f"{'question_type':<28} {'n':>4} {'acc_A':>7} {'acc_B':>7} {'d_pp':>7} "
        f"{'tok_A':>7} {'tok_B':>7} {'d_tok':>7}",
        "-" * 84,
    ]
    by_type = defaultdict(list)
    for q in shared:
        by_type[ra[q]["question_type"]].append(q)
    for t, qs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        a_acc = sum(ra[q]["is_correct"] for q in qs) / len(qs)
        b_acc = sum(rb[q]["is_correct"] for q in qs) / len(qs)
        a_t = st.mean([ra[q]["prompt_tokens"] for q in qs])
        b_t = st.mean([rb[q]["prompt_tokens"] for q in qs])
        out.append(
            f"{t:<28} {len(qs):>4} {a_acc*100:>6.1f}% {b_acc*100:>6.1f}% "
            f"{(b_acc-a_acc)*100:>+6.1f} {a_t:>7.0f} {b_t:>7.0f} {b_t-a_t:>+7.0f}"
        )
    out.append("")
    flips = [q for q in shared if ra[q]["is_correct"] != rb[q]["is_correct"]]
    if flips:
        out.append("### Flipped questions")
        for q in flips:
            mark = "B-only" if rb[q]["is_correct"] else "A-only"
            out.append(
                f"- {q} [{mark}] {ra[q]['question_type']}: "
                f"A='{ra[q]['predicted_answer'][:60]}' B='{rb[q]['predicted_answer'][:60]}'"
            )
    else:
        out.append("### No flipped questions")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="one run json (summary) or two (paired compare)")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args()
    if len(args.runs) > 2:
        ap.error("give one run for a summary, or two for a paired comparison")

    if len(args.runs) == 1:
        path = args.runs[0]
        report = render_single(path, load(path))
    else:
        pa, pb = args.runs
        report = render_compare(pa, load(pa), pb, load(pb))

    print(report)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
