"""Hypothesis classification for reader-side failures (Phase 0-4).

For every failed question whose ground truth IS in the frozen reader input
(cache/v4_contexts/<cond>), assign a *hypothesis* label for WHY the reader
still missed it.  Labels are hypotheses for manual audit, never final
verdicts (the 36-case caveat: verbatim GT presence does not mean the answer
was easy to derive).

Categories:
  output_format  - normalized pred == normalized GT (format-only mismatch)
  aggregation    - count/sum/difference question, numeric GT
  temporal       - ordering / before-after / latest question
  entity         - GT and pred are both named entities but differ
  conflict       - the wrong value (pred) ALSO occurs in the context
  extraction     - GT in context, reader refused or answered off-string
  unclassified   - none of the rules fired (manual audit queue)

Usage:
  python scripts/benchmarks/classify_reader_fails.py \\
      --run benchmark_results/pipeline_v4/lme_dev250_v4_R0rerun.json \\
      [--cond R0] [--out benchmark_results/pipeline_v4/reader_fail_hypotheses.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

_WORD_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}


def normalize(s: str) -> str:
    s = s.lower().strip().rstrip(".!")
    s = re.sub(r"[,$]", "", s)
    words = [_WORD_NUM.get(w, w) for w in re.findall(r"[a-z0-9]+", s)]
    return " ".join(words)


def gt_variants(gt: str) -> list[str]:
    out = {normalize(gt)}
    for m in re.findall(r"\d[\d,]*", gt):
        out.add(m.replace(",", ""))
    for w, d in _WORD_NUM.items():
        if re.search(rf"\b{w}\b", gt.lower()):
            out.add(d)
    return [v for v in out if v]


def has_number(s: str) -> bool:
    return bool(re.search(r"\d", s)) or any(w in s.lower() for w in _WORD_NUM)


def looks_entity(s: str) -> bool:
    return bool(re.search(r"\b[A-Z][a-z]{2,}(\s+[A-Z][a-z]{2,})*", s.strip()))


def classify(question: str, qtype: str, gt: str, pred: str, ctx: str) -> tuple[str, str]:
    """Return (hypothesis, evidence) -- first matching rule wins."""
    q = question.lower()
    ctx_l = ctx.lower()
    n_gt, n_pred = normalize(gt), normalize(pred)

    if n_gt and n_pred and (n_gt == n_pred or n_gt in n_pred or n_pred in n_gt):
        return "output_format", f"normalized pred '{n_pred}' matches GT '{n_gt}'"

    refusal = any(r in pred.lower() for r in
                  ("i don't know", "not enough", "not mentioned", "no information"))
    if refusal:
        return "extraction", "reader refused although GT is in the context"

    agg_q = re.search(
        r"how many|total|sum|count|how much more|how much less|difference|"
        r"combined|altogether|in total", q)
    if agg_q and has_number(gt):
        return "aggregation", f"aggregation question ({agg_q.group(0)!r}) with numeric GT"

    temporal_q = re.search(
        r"\bbefore\b|\bafter\b|\bfirst\b|\blast\b|\blatest\b|\bmost recent\b|"
        r"\border\b|\bwhen\b|\bwhich .{0,30} (earlier|newer|older)\b|"
        r"how (many|long|much) (days|weeks|months|years)", q)
    if temporal_q:
        return "temporal", f"temporal question ({temporal_q.group(0)!r})"

    if looks_entity(gt) and looks_entity(pred) and normalize(gt) != normalize(pred):
        return "entity", f"GT entity '{gt[:40]}' vs pred entity '{pred[:40]}'"

    if n_pred and len(n_pred) >= 3 and n_pred in ctx_l:
        return "conflict", f"pred value '{n_pred[:40]}' also occurs in the context"

    if qtype == "knowledge-update":
        return "conflict", "knowledge-update: reader likely adopted the superseded value"

    return "unclassified", "no rule fired -- manual audit"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="benchmark_results/pipeline_v4/lme_dev250_v4_R0rerun.json")
    ap.add_argument("--cond", default="R0", help="context cache condition dir")
    ap.add_argument("--out", default="", help="optional json output path")
    args = ap.parse_args()

    run_path = Path(args.run)
    if not run_path.is_absolute():
        run_path = REPO / run_path
    data = json.loads(run_path.read_text(encoding="utf-8"))
    cache_dir = REPO / "cache" / "v4_contexts" / args.cond

    rows = []
    reader_side, no_cache, not_in_ctx = 0, 0, 0
    for r in data["results"]:
        if r["is_correct"]:
            continue
        qid = r["question_id"]
        cpath = cache_dir / f"{qid}.json"
        if not cpath.exists():
            no_cache += 1
            continue
        cached = json.loads(cpath.read_text(encoding="utf-8"))
        ctx = cached.get("prompt_context") or cached.get("compiled_context") or ""
        gt = str(r.get("ground_truth", ""))
        in_ctx = any(v in ctx.lower() for v in gt_variants(gt))
        if not in_ctx:
            not_in_ctx += 1
            continue
        reader_side += 1
        hyp, ev = classify(r["question"], r["question_type"], gt,
                           r["predicted_answer"], ctx)
        rows.append({
            "question_id": qid,
            "question_type": r["question_type"],
            "hypothesis": hyp,
            "evidence": ev,
            "ground_truth": gt,
            "predicted_answer": r["predicted_answer"],
        })

    total_fail = sum(1 for r in data["results"] if not r["is_correct"])
    print(f"=== Reader-failure hypothesis classification ===")
    print(f"run: {run_path.name}  failures: {total_fail}  "
          f"reader-side (GT in cached ctx): {reader_side}  "
          f"GT not in ctx: {not_in_ctx}  no cache: {no_cache}\n")
    hdr = f"{'question_id':<22} {'type':<24} {'hypothesis':<16} {'GT':<30} {'PRED':<30}"
    print(hdr)
    print("-" * len(hdr))
    for row in sorted(rows, key=lambda x: x["hypothesis"]):
        print(f"{row['question_id']:<22} {row['question_type']:<24} "
              f"{row['hypothesis']:<16} {row['ground_truth'][:28]:<30} "
              f"{row['predicted_answer'][:28]:<30}")
    print("\n=== hypothesis counts (HYPOTHESES, not verdicts) ===")
    for k, v in Counter(r["hypothesis"] for r in rows).most_common():
        print(f"  {k:<16} {v:>3}  ({v/max(1, len(rows))*100:.0f}%)")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "run": str(run_path),
            "condition": args.cond,
            "total_failures": total_fail,
            "reader_side": reader_side,
            "rows": rows,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
