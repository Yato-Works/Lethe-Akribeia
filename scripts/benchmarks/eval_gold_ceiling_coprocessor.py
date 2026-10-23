"""
E1 + CoProcessor Experiment: Gold Context + Deterministic Certificates on Dev 250.

Evaluates how much deterministic certificates (Chronos diffs, Event Ledger, Profile State)
lift the 7B Reader above the raw 63.6% gold-context ceiling.
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
OUT_PATH = REPO / "benchmark_results" / "gold_ceiling" / "lme_dev250_coprocessor_ceiling_7b.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "src"))

from artificial_memory_gen.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory_gen.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import FROZEN_MODEL, OllamaAnswerer


def build_gold_context_and_records(item: dict, extractor: UniversalIRExtractor) -> tuple[str, list[StructuredIR], int]:
    """Extract gold turns and convert them into StructuredIR records."""
    gold_turns = []
    records = []
    
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
                line = f"{header} {role.capitalize()}: {content}"
                gold_turns.append(line)

                recs = extractor.extract(content, default_source=role)
                for r in recs:
                    r.raw_content = line
                    r.time_scope = s_date
                    records.append(r)
    
    if not gold_turns:
        return "[No prior conversation mentions this topic.]", [], 0
    
    context_text = "\n\n".join(gold_turns)
    return context_text, records, len(gold_turns)


def main():
    parser = argparse.ArgumentParser(description="Run Gold Context + CoProcessors on Dev 250")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(SPLIT_PATH, encoding="utf-8") as f:
        dev_qids = set(json.load(f))

    with open(DATASET_PATH, encoding="utf-8") as f:
        all_items = json.load(f)

    dev_items = [it for it in all_items if it["question_id"] in dev_qids]
    if args.limit > 0:
        dev_items = dev_items[:args.limit]

    print("=" * 80)
    print(f"  E1 + CO-PROCESSORS: GOLD EVIDENCE + DETERMINISTIC CERTIFICATES (n={len(dev_items)})")
    print(f"  Reader: {FROZEN_MODEL} (num_ctx={args.num_ctx})")
    print("=" * 80)

    extractor = UniversalIRExtractor()
    planner = CognitiveExecutionPlanner()
    coprocessor = CoProcessorEngine()
    answerer = OllamaAnswerer(model=FROZEN_MODEL, num_ctx=args.num_ctx)

    results = []
    t0 = time.perf_counter()
    overridden_count = 0

    for idx, item in enumerate(dev_items):
        qid = item["question_id"]
        qtype = item["question_type"]
        q = item["question"]
        q_date = item.get("question_date", "")
        gt = str(item.get("answer", ""))
        is_abs = qtype == "abstention" or qid.endswith("_abs") or "information provided is not enough" in gt.lower()

        gold_ctx, records, num_turns = build_gold_context_and_records(item, extractor)

        # 1. Cognitive Execution Planning
        plan = planner.plan(q, records, reference_date=q_date)

        # 2. Co-processor Reasoning & Certificates
        bundle = coprocessor.process(q, records, plan, reference_date=q_date)

        # Check for 0-token deterministic override
        if bundle.deterministic_answer_override:
            pred = bundle.deterministic_answer_override
            tok_used = 0
            lat_ms = 0.0
            overridden_count += 1
        else:
            # Build prompt with certificates injected
            prompt_parts = []
            if bundle.assembled_certificate_block:
                prompt_parts.append(bundle.assembled_certificate_block)
            
            prompt_parts.append(f"[VERIFIED CONVERSATION EVIDENCE]\n{gold_ctx}")

            instruction_lines = [
                f"[INSTRUCTION]",
                f"Question date: {q_date or 'N/A'}",
                f"Answer the question directly and concisely based on the verified evidence and certificates above.",
            ]
            if plan.preference_demand >= 0.40:
                instruction_lines.append(
                    "The question asks for recommendations or thoughts based on the user's personal situation.\n"
                    "Explicitly incorporate their background, gear, experiences, and preferences from the evidence.\n"
                    "Mention the relevant domain or gear category (e.g. photography accessories, evening activities, cocktail mixology) directly.\n"
                    "Do NOT say 'I don't know'. Give concrete, personalized suggestions."
                )
            elif plan.state_evolution_demand >= 0.50:
                instruction_lines.append(
                    "If the question asks for current/now/latest state, use the state from the latest date session,\n"
                    "including newly added items or records. If asking for earlier/previous, use the state before the update."
                )
            else:
                instruction_lines.append(
                    "If the answer is a duration or count, state the exact number directly."
                )

            prompt_parts.append("\n".join(instruction_lines))
            prompt_context = "\n\n".join(prompt_parts)

            ans_resp = answerer.answer(q, prompt_context)
            pred = ans_resp.text
            tok_used = ans_resp.prompt_tokens
            lat_ms = ans_resp.latency_ms

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
            "deterministic_override": bool(bundle.deterministic_answer_override),
            "prompt_tokens": tok_used,
            "latency_ms": lat_ms,
        })

        status = "PASS" if is_ok else "FAIL"
        curr_acc = sum(1 for r in results if r["is_correct"]) / len(results) * 100
        mean_tok = sum(r["prompt_tokens"] for r in results) / len(results)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(dev_items):
            print(f"  [{idx+1:03d}/{len(dev_items):03d}] [{status}] Acc: {curr_acc:5.1f}% | Tok: {mean_tok:4.0f} | Type: {qtype[:16]:<16} | Q: {q[:28]}", flush=True)

        # Checkpoint every 25 items
        if (idx + 1) % 25 == 0 or (idx + 1) == len(dev_items):
            ckpt = OUT_PATH.parent / "lme_dev250_coprocessor_checkpoint.json"
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - t0
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    acc = correct / total * 100 if total else 0

    print("\n" + "=" * 80)
    print("  E1 + CO-PROCESSORS FINAL REPORT")
    print("=" * 80)
    print(f"Total Evaluated: {total}")
    print(f"Overall Accuracy: {acc:.2f}% ({correct}/{total}) [vs Raw Gold: 63.60% | vs V3: 83.40%]")
    print(f"Deterministic Overrides (0 tokens): {overridden_count} queries ({overridden_count/total*100:.1f}%)")
    print(f"Mean Input Tokens: {sum(r['prompt_tokens'] for r in results)/total:.1f}")
    print(f"Total Elapsed: {elapsed:.1f}s")

    # Breakdown by question_type
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    print("\n--- Breakdown by Question Type ---")
    for qt, rlist in sorted(by_type.items()):
        c = sum(1 for r in rlist if r["is_correct"])
        print(f"  {qt:26}: {c:3d}/{len(rlist):3d} ({c/len(rlist)*100:5.1f}%) | Mean Tok: {sum(r['prompt_tokens'] for r in rlist)/len(rlist):4.0f}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "longmemeval_dev250_coprocessor_ceiling",
            "model": FROZEN_MODEL,
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "deterministic_overrides": overridden_count,
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in results) / total,
            "elapsed_seconds": elapsed,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nArtifact saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
