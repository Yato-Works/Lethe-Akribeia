"""BEAM 100K-10M Runner for Qwen2.5-Coder:7B (AM Apex Maximum Power).

Usage:
  uv run python scripts/run_coder7b_beam.py --scale 100K --limit-chats 1 --limit-questions 10
  uv run python scripts/run_coder7b_beam.py --scale 100K --full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artificial_memory.research.benchmarks.external.beam_adapter import BeamAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer


def main() -> None:
    parser = argparse.ArgumentParser(description="BEAM Runner with Qwen2.5-Coder:7B")
    parser.add_argument("--scale", default="100K", choices=["100K", "500K", "1M", "10M"])
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--limit-chats", type=int, default=1, help="Number of chats to evaluate")
    parser.add_argument("--limit-questions", type=int, default=10, help="Questions per chat")
    parser.add_argument("--tag", default="beam_coder7b_apex")
    args = parser.parse_args()

    print("=" * 80)
    print(f"      BEAM {args.scale} APEX EVALUATION with {args.model}")
    print("=" * 80)

    adapter = BeamAdapter()
    answerer = OllamaAnswerer(model=args.model, num_ctx=args.num_ctx)

    chats = adapter.list_chats(args.scale)
    if not chats:
        print(f"No chats found for scale {args.scale} in {adapter.base_dir}")
        return

    target_chats = chats[: args.limit_chats]
    print(f"Total chats available: {len(chats)} | Target chats: {len(target_chats)}")

    total_results = []
    cat_stats = defaultdict(lambda: [0, 0])
    t0_all = time.perf_counter()

    for chat_id in target_chats:
        print(f"\n--- Loading Chat: {chat_id} ---")
        turns = adapter.load_chat(args.scale, chat_id)
        if not turns:
            print(f"Skipping chat {chat_id} (no turns found).")
            continue

        print(f"Loaded {len(turns)} turns. Ingesting into Memory IR...")
        t0_ingest = time.perf_counter()
        records = adapter.ingest_turns(turns)
        print(f"Ingested {len(records)} records in {time.perf_counter() - t0_ingest:.2f}s.")

        questions = adapter.load_probing_questions(args.scale, chat_id)
        if not questions:
            print(f"No questions found for chat {chat_id}.")
            continue

        target_qs = questions[: args.limit_questions]
        print(f"Evaluating {len(target_qs)} questions...")

        for i, q in enumerate(target_qs, 1):
            res = adapter.evaluate_question(q, records, answerer)
            total_results.append(res)
            cat_stats[res.category][1] += 1
            cat_stats[res.category][0] += int(res.is_correct)

            mark = "PASS" if res.is_correct else "FAIL"
            print(f"  [{i:02d}/{len(target_qs)}] {res.category:<25} | {mark:<4} | Q: {q.question[:40]}...", flush=True)

    elapsed = time.perf_counter() - t0_all
    total_q = len(total_results)
    total_corr = sum(1 for r in total_results if r.is_correct)
    acc = (total_corr / total_q * 100) if total_q else 0.0

    print("\n" + "=" * 80)
    print(f"          BEAM {args.scale} EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Questions Evaluated: {total_q}")
    print(f"Overall Accuracy:          {acc:.1f}% ({total_corr}/{total_q})")
    print(f"Elapsed Time:              {elapsed:.1f}s ({elapsed/max(1, total_q):.2f}s/Q)")
    print("-" * 80)
    print(f"{'Category':<30} | {'Score':<10} | Correct / Total")
    print("-" * 80)
    for cat, (c, t) in sorted(cat_stats.items()):
        cat_acc = c / t * 100 if t else 0.0
        print(f"{cat:<30} | {cat_acc:5.1f}%    | {c}/{t}")
    print("=" * 80)

    # Save results
    out_dir = Path("benchmark_results/beam")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.tag}_{args.scale}.json"

    serializable = [
        {
            "question_id": r.question_id,
            "category": r.category,
            "question": r.question,
            "ground_truth": r.ground_truth,
            "predicted_answer": r.predicted_answer,
            "is_correct": r.is_correct,
            "tokens_used": r.tokens_used,
            "latency_ms": r.latency_ms,
        }
        for r in total_results
    ]

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scale": args.scale,
                "model": args.model,
                "num_ctx": args.num_ctx,
                "total_questions": total_q,
                "overall_accuracy": acc,
                "categories": {cat: {"correct": c, "total": t, "acc": c / t} for cat, (c, t) in cat_stats.items()},
                "results": serializable,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
