"""
E1 Experiment: Gold Context Ceiling for LongMemEval (Dev 250).

Measures the theoretical upper-bound accuracy when the memory system delivers
ONLY the ground-truth evidence turns (has_answer=True) with exact dates.
Zero distractor turns. Evaluates on dev 250 to keep holdout 250 sealed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"
SPLIT_PATH = REPO / "benchmark_results" / "splits" / "lme_dev_qids.json"
OUT_PATH = REPO / "benchmark_results" / "gold_ceiling" / "lme_dev250_gold_ceiling_7b.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import FROZEN_MODEL, OllamaAnswerer


def build_gold_context(item: dict) -> tuple[str, int]:
    """Extract only turns marked with has_answer=True across all haystack sessions."""
    gold_turns = []
    
    dates = item.get("haystack_dates", [])
    sids = item.get("haystack_session_ids", [])
    
    for s_idx, session in enumerate(item.get("haystack_sessions", [])):
        s_date = dates[s_idx] if s_idx < len(dates) else ""
        sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
        
        for turn in session:
            if turn.get("has_answer"):
                role = turn.get("role", "user")
                content = turn.get("content", "")
                header = f"[{sid} on {s_date}]" if s_date else f"[{sid}]"
                gold_turns.append(f"{header} {role.capitalize()}: {content}")
    
    if not gold_turns:
        # Abstention questions have no answer turns
        return "[No prior conversation mentions this topic.]", 0
    
    context_text = "\n\n".join(gold_turns)
    return context_text, len(gold_turns)


def main():
    parser = argparse.ArgumentParser(description="Run Gold Context Ceiling on LongMemEval Dev 250")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0, help="Test limit (0=all 250 dev items)")
    args = parser.parse_args()

    # Load dev QIDs
    with open(SPLIT_PATH, encoding="utf-8") as f:
        dev_qids = set(json.load(f))

    # Load dataset
    with open(DATASET_PATH, encoding="utf-8") as f:
        all_items = json.load(f)

    dev_items = [it for it in all_items if it["question_id"] in dev_qids]
    if args.limit > 0:
        dev_items = dev_items[:args.limit]

    print("=" * 80)
    print(f"  E1: GOLD CONTEXT CEILING ON LONGMEMEVAL DEV 250 (n={len(dev_items)})")
    print(f"  Reader: {FROZEN_MODEL} (num_ctx={args.num_ctx})")
    print("=" * 80)

    answerer = OllamaAnswerer(model=FROZEN_MODEL, num_ctx=args.num_ctx)
    results = []
    
    t0 = time.perf_counter()

    for idx, item in enumerate(dev_items):
        qid = item["question_id"]
        qtype = item["question_type"]
        q = item["question"]
        gt = str(item.get("answer", ""))
        is_abs = qtype == "abstention" or qid.endswith("_abs") or "information provided is not enough" in gt.lower()

        gold_ctx, num_turns = build_gold_context(item)

        # Build clean direct prompt for gold ceiling
        prompt_context = (
            f"[VERIFIED CONVERSATION EVIDENCE]\n"
            f"{gold_ctx}\n\n"
            f"[INSTRUCTION]\n"
            f"Question date: {item.get('question_date', 'N/A')}\n"
            f"Answer the question directly and concisely using ONLY the verified evidence above.\n"
            f"If the evidence does not contain the answer, reply: The information provided is not enough."
        )

        ans_resp = answerer.answer(q, prompt_context)
        pred = ans_resp.text

        is_ok = LongMemEvalAdapter.score_answer(
            question_type=qtype,
            gt=gt,
            predicted_answer=pred,
            is_abstention_gt=is_abs,
            pcc_is_abstention=is_abs,
        )

        results.append({
            "question_id": qid,
            "question_type": qtype,
            "question": q,
            "ground_truth": gt,
            "predicted_answer": pred,
            "is_correct": is_ok,
            "num_gold_turns": num_turns,
            "prompt_tokens": ans_resp.prompt_tokens,
            "completion_tokens": ans_resp.completion_tokens,
            "latency_ms": ans_resp.latency_ms,
        })

        status = "PASS" if is_ok else "FAIL"
        curr_acc = sum(1 for r in results if r["is_correct"]) / len(results) * 100
        mean_tok = sum(r["prompt_tokens"] for r in results) / len(results)
        
        if (idx + 1) % 5 == 0 or (idx + 1) == len(dev_items):
            print(f"  [{idx+1:03d}/{len(dev_items):03d}] [{status}] Acc: {curr_acc:5.1f}% | Tok: {mean_tok:4.0f} | Type: {qtype[:16]:<16} | Q: {q[:30]}", flush=True)

        # Checkpoint every 25 items
        if (idx + 1) % 25 == 0 or (idx + 1) == len(dev_items):
            ckpt_path = OUT_PATH.parent / "lme_dev250_gold_ceiling_checkpoint.json"
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - t0
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    acc = correct / total * 100 if total else 0

    print("\n" + "=" * 80)
    print("  E1 FINAL REPORT: GOLD CONTEXT CEILING")
    print("=" * 80)
    print(f"Total Evaluated: {total}")
    print(f"Overall Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Mean Input Tokens: {sum(r['prompt_tokens'] for r in results)/total:.1f}")
    print(f"Total Elapsed: {elapsed:.1f}s ({elapsed/total:.2f}s/q)")

    # Breakdown by question_type
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    print("\n--- Breakdown by Question Type ---")
    for qt, rlist in sorted(by_type.items()):
        c = sum(1 for r in rlist if r["is_correct"])
        print(f"  {qt:26}: {c:3d}/{len(rlist):3d} ({c/len(rlist)*100:5.1f}%) | Mean Tok: {sum(r['prompt_tokens'] for r in rlist)/len(rlist):4.0f}")

    # Save artifact
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "longmemeval_dev250_gold_ceiling",
            "model": FROZEN_MODEL,
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in results) / total,
            "elapsed_seconds": elapsed,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nArtifact saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
