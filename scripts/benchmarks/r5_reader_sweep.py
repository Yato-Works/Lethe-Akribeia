"""R5: Reader sweep over FROZEN context caches -- identical inputs, one model changed.

Every condition of the Phase-1 ladder compiled its reader inputs into
``cache/v4_contexts/<cond>/<qid>.json``.  This tool replays those exact
prompts against several reader models, so the ONLY variable is the reader.
That answers two questions at once:

  1. capability gap  - how much of the score is the reader (7b vs 1.5b vs 4b vs 8b)
  2. generalization  - does a context-design condition's effect survive a
                       different reader? (run again with --cond R2 and compare)

Deterministic-override questions (prompt_context is None) are copied from
the reference run for EVERY model: they are 0-token pipeline decisions that
do not depend on the reader, so the paired set stays complete (n=250).

Usage:
  python scripts/benchmarks/r5_reader_sweep.py --cond R0
  python scripts/benchmarks/r5_reader_sweep.py --cond R0 --models qwen2.5:1.5b
  python scripts/benchmarks/r5_reader_sweep.py --compare   # table of saved runs
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from artificial_memory.research.benchmarks.external.longmemeval_adapter import (
    LongMemEvalAdapter,
)
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

DEFAULT_MODELS = [
    "qwen2.5:7b-instruct",       # sanity: must reproduce the R0 run
    "qwen2.5:1.5b",
    "qwen3:4b",
    "llama3.1:8b-instruct-q4_K_M",
]
SWEEP_DIR = REPO / "benchmark_results" / "pipeline_v4" / "r5_reader_sweep"
REFERENCE = REPO / "benchmark_results" / "pipeline_v4" / "lme_dev250_v4_R0rerun.json"


def load_entries(cond: str) -> tuple[list[dict], dict[str, dict]]:
    """Frozen prompts for the condition + reference outcomes (gt/overrides)."""
    cache = REPO / "cache" / "v4_contexts" / cond
    if not cache.is_dir():
        raise SystemExit(f"no context cache for cond={cond}: {cache}")
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    ref_by_id = {r["question_id"]: r for r in ref["results"]}
    entries = []
    for path in sorted(cache.glob("*.json")):
        e = json.loads(path.read_text(encoding="utf-8"))
        qid = e["question_id"]
        if qid not in ref_by_id:
            continue
        r = ref_by_id[qid]
        entries.append({
            "question_id": qid,
            "question_type": e["question_type"],
            "question": e["question"],
            "prompt_context": e["prompt_context"],   # None => deterministic override
            "ground_truth": r["ground_truth"],
            "deterministic_override": bool(e.get("deterministic_override")),
            # fixed, reader-independent outcome copied for every model:
            "override_answer": r["predicted_answer"] if e.get("deterministic_override") else None,
            "override_correct": r["is_correct"] if e.get("deterministic_override") else None,
        })
    return entries, ref_by_id


def abstention_reference(qtype: str, qid: str, gt: str) -> bool:
    return (
        qtype == "abstention"
        or qid.endswith("_abs")
        or "information provided is not enough" in gt.lower()
    )


def grade(e: dict, pred: str) -> bool:
    return bool(LongMemEvalAdapter.score_answer(
        question_type=e["question_type"],
        gt=e["ground_truth"],
        predicted_answer=pred,
        is_abstention_gt=abstention_reference(e["question_type"], e["question_id"], e["ground_truth"]),
        pcc_is_abstention=abstention_reference(e["question_type"], e["question_id"], e["ground_truth"]),
    ))


def run_model(model: str, cond: str, entries: list[dict], num_ctx: int, limit: int) -> dict:
    out_path = SWEEP_DIR / f"{model.replace(':', '_')}_{cond}.json"
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if prev.get("n") == len(entries):
            print(f"  [skip] {model}: {out_path.name} already complete (n={prev['n']})")
            return prev

    todo = [e for e in entries if not e["deterministic_override"]]
    if limit:
        keep = {e["question_id"] for e in entries[:limit]}
        todo = [e for e in todo if e["question_id"] in keep]
    kept_ids = {e["question_id"] for e in todo}

    print(f"  [model] {model}: {len(todo)} reader calls")
    # qwen3-family defaults to server-side thinking; under the frozen
    # num_predict=256 the reasoning consumes the whole budget and the model
    # returns an EMPTY content string (adapter config, not a prompt change).
    think = False if model.startswith("qwen3") else None
    answerer = OllamaAnswerer(model=model, num_ctx=num_ctx, think=think)
    t0 = time.perf_counter()
    results = []
    try:
        for i, e in enumerate(entries, 1):
            if e["question_id"] not in kept_ids:
                if e["deterministic_override"] and not limit:
                    results.append({
                        "question_id": e["question_id"],
                        "question_type": e["question_type"],
                        "predicted_answer": e["override_answer"],
                        "is_correct": bool(e["override_correct"]),
                        "prompt_tokens": 0,
                        "latency_ms": 0.0,
                        "deterministic_override": True,
                    })
                continue
            resp = answerer.answer(e["question"], e["prompt_context"])
            results.append({
                "question_id": e["question_id"],
                "question_type": e["question_type"],
                "predicted_answer": resp.text,
                "is_correct": grade(e, resp.text),
                "prompt_tokens": resp.prompt_tokens,
                "latency_ms": resp.latency_ms,
                "deterministic_override": False,
            })
            if i % 25 == 0 or i == len(entries):
                acc = sum(r["is_correct"] for r in results) / len(results)
                print(f"    [{i:03d}/{len(entries)}] acc={acc:.1%}")
    except Exception as exc:  # model load / OOM must not kill the sweep
        print(f"  [model] {model} FAILED after {len(results)} questions: {exc}")
        if not results:
            return {"model": model, "cond": cond, "n": 0, "error": str(exc)}

    total = len(results)
    correct = sum(r["is_correct"] for r in results)
    toks = [r["prompt_tokens"] for r in results] or [0]
    lat = [r["latency_ms"] for r in results] or [0.0]
    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        by_type[r["question_type"]][1] += 1
        by_type[r["question_type"]][0] += r["is_correct"]
    out = {
        "model": model,
        "cond": cond,
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "mean_prompt_tokens": st.mean(toks),
        "mean_latency_ms": st.mean(lat),
        "elapsed_seconds": time.perf_counter() - t0,
        "per_type": {t: {"correct": c, "total": n, "accuracy": c / n if n else 0.0}
                     for t, (c, n) in sorted(by_type.items())},
        "results": results,
    }
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [model] {model}: acc={out['accuracy']:.1%} "
          f"tok={out['mean_prompt_tokens']:.0f} -> {out_path.name}")
    return out


def render_compare(cond: str) -> str:
    paths = sorted(SWEEP_DIR.glob(f"*_{cond}.json"))
    if not paths:
        return f"no sweep artefacts for cond={cond} in {SWEEP_DIR}"
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    types = sorted({t for r in rows for t in r.get("per_type", {})})
    out = [f"## R5 reader sweep (cond={cond}, identical frozen inputs)", ""]
    hdr = f"{'model':<34} {'n':>4} {'acc':>7} {'tok':>6} {'lat_ms':>7}"
    out += [hdr, "-" * len(hdr)]
    for r in sorted(rows, key=lambda x: -x.get("accuracy", 0.0)):
        out.append(f"{r['model']:<34} {r['n']:>4} {r.get('accuracy', 0.0)*100:>6.1f}% "
                   f"{r.get('mean_prompt_tokens', 0):>6.0f} {r.get('mean_latency_ms', 0):>7.0f}")
    out += ["", f"{'question_type':<28}" + "".join(f"{m['model'][:12]:>14}" for m in rows)]
    for t in types:
        line = f"{t:<28}"
        for m in rows:
            pt = m.get("per_type", {}).get(t)
            line += f"{(str(round(pt['accuracy'] * 100, 1)) + '%') if pt else '-':>14}"
        out.append(line)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cond", default="R0")
    ap.add_argument("--models", default="", help="comma list (default: full sweep)")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compare", action="store_true", help="only print the saved-runs table")
    args = ap.parse_args()

    if args.compare:
        print(render_compare(args.cond))
        return 0

    entries, _ = load_entries(args.cond)
    models = [m.strip() for m in args.models.split(",") if m.strip()] or DEFAULT_MODELS
    print(f"R5 reader sweep: cond={args.cond} n={len(entries)} models={models}")
    for model in models:
        run_model(model, args.cond, entries, args.num_ctx, args.limit)
    print()
    print(render_compare(args.cond))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
