"""LoCoMo 1,540 Standard QA Benchmark Runner (Mem0 Protocol).

Evaluates the official 1,540 non-adversarial QA questions of LoCoMo-10:
  - Category 1: Multi-Hop (282 questions)
  - Category 2: Temporal (321 questions)
  - Category 3: Open-Domain (96 questions)
  - Category 4: Single-Hop (841 questions)
  Total: 1,540 questions (excludes 446 Adversarial questions).

Usage:
  uv run python scripts/run_locomo_1540.py --limit 50
  uv run python scripts/run_locomo_1540.py --full
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The src/ path insert above has to precede these imports, so they carry the
# project's usual E402 waiver (same convention as scripts/benchmarks/*.py).
from artificial_memory.recall.answer_shape import shape_directive  # noqa: E402
from artificial_memory.research.benchmarks.external.locomo_adapter import (  # noqa: E402
    LoCoMoAdapter,
    LoCoMoQuestion,
)
from artificial_memory.research.benchmarks.llm import (  # noqa: E402
    FROZEN_MODEL,
    OFFICIAL_ABSTENTION_TEXT,
    OllamaAnswerer,
)
from artificial_memory.skills.answer_committer import extract_temporal_answer  # noqa: E402


class ConciseAnswerer:
    """Asks the reader for the value and nothing else, changing nothing else.

    Step 3c measured a **12.8pp official-F1 gap** between the 7B and the 1.5B reader
    that the binary matcher does not see, and found the 1.5B's answers run 11.7
    words against the 7B's 5.4 with half of its hits scoring below 0.5 F1.  That
    leaves the question the artefacts cannot answer: is the small reader *less
    capable*, or merely *more verbose*?  This wrapper is the cheap experiment that
    separates the two - same model, same context, same questions, one extra
    instruction - and it is the control for every later claim that the reader is
    the bottleneck.

    Wrapping the answerer rather than editing the eight prompt sites in
    ``locomo_adapter`` keeps the frozen path byte-identical when the flag is off.
    """

    INSTRUCTION = (
        "[ANSWER FORMAT] Reply with the answer only: the exact value (a name, a date, "
        "a number or a short phrase). No preamble, no explanation, no restatement of "
        "the question, no full sentence. If the question asks for several items, list "
        "only those items."
    )

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        # model, num_ctx and answer_adapter must read through to the real answerer.
        return getattr(self._inner, name)

    def answer(self, question_text: str, context: str):
        return self._inner.answer(question_text, f"{context}\n\n{self.INSTRUCTION}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo 1,540 Standard Benchmark (Mem0 Protocol)")
    parser.add_argument("--model", default=FROZEN_MODEL, help="Reader model (default: qwen2.5-coder:7b)")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--tag", default="locomo_1540_coder7b")
    parser.add_argument("--use-cache", action="store_true", default=True, help="Use precomputed MSC context cache")
    parser.add_argument("--no-cache", action="store_true", default=False, help="Disable precomputed MSC context cache")
    parser.add_argument("--cache-file", default="benchmark_results/locomo_context_cache.jsonl",
                        help="Context cache to replay (lets a variant cache be scored with the frozen prompt)")
    parser.add_argument("--qids", default=None,
                        help="Optional file of question ids (one per line); only those questions are evaluated")
    parser.add_argument("--commit-temporal", action="store_true", default=False,
                        help="AM decides temporal answers deterministically (skip the reader LLM "
                             "for questions the committer can prove); default off = frozen path")
    parser.add_argument("--answer-shape-gate", action="store_true", default=False,
                        help="runtime classifies the open-domain answer shape (polarity / choice / "
                             "attribute) and instructs that shape; default off = the hand-written "
                             "ANSWER SHAPE FIRST block measured in AM_APEX_STATUS.md 0f")
    parser.add_argument("--concise-answer", action="store_true", default=False,
                        help="ask the reader for the exact value only (A/B for step 3c: is the "
                             "official-F1 gap capability or answer shape?); default off = frozen path")
    args = parser.parse_args()

    print("=" * 80)
    print("      LoCoMo 1,540 STANDARD BENCHMARK RUNNER (Mem0 Protocol)")
    print(f"      Model: {args.model} | num_ctx: {args.num_ctx}")
    print("=" * 80)

    adapter = LoCoMoAdapter()
    answerer = OllamaAnswerer(model=args.model, num_ctx=args.num_ctx)
    if args.concise_answer:
        answerer = ConciseAnswerer(answerer)
        print("READER: concise-answer variant ON (value only, no explanation)")

    cache_file = Path(args.cache_file)
    has_cache = cache_file.exists() and args.use_cache and not args.no_cache

    # Load ground truth map from locomo10.json
    gt_map = {}
    with open(adapter.dataset_path, encoding="utf-8") as f:
        raw_convs = json.load(f)
    for c in raw_convs:
        sid = c.get("sample_id", "")
        for i, qa in enumerate(c.get("qa", [])):
            qid = f"{sid}-qa-{i:03d}"
            gt_map[qid] = {
                "answer": str(qa.get("answer", "")),
                "evidence": qa.get("evidence", []),
                "category": qa.get("category", 1),
            }

    # 1. Load questions (either from cache or locomo10.json)
    target_records = []
    if has_cache:
        print(f"Loading context cache from {cache_file}...")
        with open(cache_file, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("category") != 5:  # Exclude adversarial -> exactly 1,540 questions
                    qid = r.get("qid") or r.get("question_id")
                    meta = gt_map.get(qid, {})
                    r["question_id"] = qid
                    r["ground_truth"] = meta.get("answer", "")
                    r["evidence_ids"] = meta.get("evidence", [])
                    r["context_text"] = r.get("context") or r.get("context_text")
                    target_records.append(r)
        print(f"Loaded {len(target_records)} non-adversarial questions from cache (Target: 1,540).")
    else:
        print("Loading from raw datasets/external/locomo10.json...")
        for c_idx in range(len(raw_convs)):
            turns, questions, ir_records = adapter.load_conversation(c_idx)
            for q in questions:
                if q.category != 5:
                    target_records.append({"question_obj": q, "turns": turns, "ir_records": ir_records})
        print(f"Loaded {len(target_records)} non-adversarial questions (Target: 1,540).")

    if args.limit:
        target_records = target_records[: args.limit]
        print(f"Evaluating subset: {len(target_records)} questions.")

    if args.qids:
        wanted = {
            line.strip()
            for line in Path(args.qids).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        target_records = [
            r for r in target_records
            if (r.get("question_id") or r.get("qid") or r["question_obj"].question_id) in wanted
        ]
        print(f"Evaluating --qids subset: {len(target_records)} questions.")

    # 2. Evaluation loop
    cat_names = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop"}
    cat_stats = collections.defaultdict(lambda: [0, 0])
    results = []
    t0_all = time.perf_counter()

    for i, item in enumerate(target_records, 1):
        if has_cache:
            qid = item["question_id"]
            cat = item["category"]
            question_text = item["question"]
            gt = item["ground_truth"]
            ctx = item["context_text"]

            q_obj = LoCoMoQuestion(
                question_id=qid,
                conv_id=qid.split("-qa-")[0] if "-qa-" in qid else "conv-0",
                question=question_text,
                ground_truth=gt,
                evidence_ids=item.get("evidence_ids", []),
                category=cat,
            )

            # Build category-specific prompt
            if cat == 3:
                # Answer-shape gating, two implementations:
                #   * frozen path (default): the hand-written block below, measured
                #     in AM_APEX_STATUS.md 0f at +5 questions (+5.2pp, p=0.125);
                #   * --answer-shape-gate: the runtime classifies the shape
                #     deterministically (recall/answer_shape.py) and the prompt
                #     states that shape instead of *asking the model to decide*.
                if args.answer_shape_gate:
                    shape_block = shape_directive(question_text)
                else:
                    shape_block = (
                        "- ANSWER SHAPE FIRST: decide what kind of answer the question wants.\n"
                        "  If it can be affirmed or denied ('Would X ...?', 'Is X ...?', 'Did X ...?'),\n"
                        "  answer 'Yes', 'Likely no' or 'No'.\n"
                        "  If it starts with what/which/who/whose, answer with the concrete noun,\n"
                        "  trait or stance - NEVER a yes/no. Example: 'What would X's political\n"
                        "  leaning likely be?' -> 'Liberal', NOT 'Likely no'.\n"
                    )
                prompt = (
                    f"[INSTRUCTION: COMMONSENSE & OPEN-DOMAIN MEMORY REASONING]\n"
                    f"Answer using the dialogue context below.\n"
                    f"{shape_block}"
                    f"- For polar questions, use the character's known behaviors and traits\n"
                    f"  to make a reasoned yes/no/likely-no prediction.\n"
                    f"- Check BOTH supporting AND contradicting evidence: if the evidence\n"
                    f"  only shows X supporting something, but the question asks IF X is THAT\n"
                    f"  thing (e.g., 'ally' vs 'member'), respond 'Likely no' — being supportive\n"
                    f"  of a community does NOT make someone a member of it.\n"
                    f"- Check for negative qualifiers: 'not', 'doesn't identify as', 'wouldn't want'\n"
                    f"  in the evidence — if present, lean 'Likely no'.\n"
                    f"- Check for explicit refusals: 'no', 'not interested', 'wouldn't enjoy'\n"
                    f"  in the evidence — if present, lean 'Likely no'.\n"
                    f"- If the question refers to an event the context describes negatively\n"
                    f"  (accident, scary, bad, upset, went wrong), then 'would X do/again/enjoy'\n"
                    f"  resolves to 'Likely no'.\n"
                    f"- CRITICAL: You are FORBIDDEN from saying 'I don't know', 'Unsure', 'Not enough information',\n"
                    f"  'I cannot determine', 'Unknown', or any refusal. The evidence was retrieved FOR this question.\n"
                    f"  If you cannot find the exact answer, make your BEST direct deduction from the evidence.\n"
                    f"- Do not include polite conversation, reasoning preambles, or explanations.\n"
                    f"- Return ONLY the concise target answer.\n\n"
                    f"=== DIALOGUE CONTEXT ===\n{ctx}"
                )
            elif cat == 1:
                prompt = (
                    f"[INSTRUCTION: MULTI-HOP EVIDENCE SYNTHESIS]\n"
                    f"Answer the question using ONLY the dialogue context below.\n"
                    f"- Name every item/person/event/location the question asks about; never answer with\n"
                    f"  only one part of a multi-part question. Explicitly list all distinct entities.\n"
                    f"- Filter BEFORE you enumerate: list every matching entity in the context, then\n"
                    f"  discard any that fails the question's qualifier (e.g. 'to help children').\n"
                    f"  Answer with only the survivors, comma-separated.\n"
                    f"- Answer with the entities the evidence line itself lists - never with the\n"
                    f"  conversation partner, unless the question names them.\n"
                    f"- Quote names, dates, and facts exactly as they appear in the context.\n"
                    f"- Do not include polite conversation, reasoning preambles, or explanations.\n"
                    f"- Return ONLY the concise target answer/entity/date.\n"
                    f"- CRITICAL: You are FORBIDDEN from saying 'I don't know', 'Unsure', 'Not enough information',\n"
                    f"  'I cannot determine', 'Unknown', or any refusal. The evidence was retrieved FOR this question.\n"
                    f"  If you cannot find the exact answer, make your BEST direct deduction from the evidence.\n"
                    f"- Abstention is a LAST RESORT: check every line of the context first; if any\n"
                    f"  line names the value, answer with it directly.\n"
                    f"- Only if the value appears nowhere in the context, reply exactly: "
                    f"{OFFICIAL_ABSTENTION_TEXT}\n\n"
                    f"{ctx}"
                )
            elif cat == 2:
                prompt = (
                    f"[INSTRUCTION: TEMPORAL REASONING]\n"
                    f"Answer the temporal question using ONLY the dated dialogue context below.\n"
                    f"1. Find the turn that states when the event happened.\n"
                    f"2. Copy the date form the context uses (e.g. 'The week before 9 June 2023',\n"
                    f"   '7 May 2023', '2022'); do not convert 'the week before X' into a plain\n"
                    f"   calendar date. But if the context only gives a bare abbreviation such as\n"
                    f"   'Last Fri', expand it using that unit's own header date first, so the answer\n"
                    f"   reads 'The Friday before 15 July 2023'. Never answer a bare abbreviation.\n"
                    f"3. Only do calendar arithmetic when the question asks for a duration or a gap\n"
                    f"(e.g. 'how many days between ...', 'how long after ...'), and then answer with\n"
                    f"   just the number and unit (e.g. '3 weeks', '7 days').\n"
                    f"4. Never answer with the session date you started from, and never add any\n"
                    f"   preamble, explanation or quotes.\n"
                    f"5. CRITICAL: You are FORBIDDEN from saying 'I don't know', 'Unsure', 'Not enough information',\n"
                    f"   'I cannot determine', 'Unknown', or any refusal. The evidence was retrieved FOR this question.\n"
                    f"   If you cannot find the exact answer, make your BEST direct deduction from the evidence.\n"
                    f"=== DIALOGUE CONTEXT ===\n{ctx}\n\n"
                )
            else:
                prompt = (
                    f"[INSTRUCTION: EVIDENCE DIRECTOR - FACT EXTRACTION]\n"
                    f"Answer the question directly based on the dialogue context below.\n"
                    f"- First identify the EXACT subject (who did it) and the action/object asked.\n"
                    f"- Extract the exact facts, names, numbers, or reasons concisely.\n"
                    f"- For 'what', 'when', 'how many', 'why' questions: answer with the\n"
                    f"  exact value from the context. Do NOT add extra information.\n"
                    f"- For 'how' questions: state the reason/purpose in your own words\n"
                    f"  ONLY if the context gives a clear reason.\n"
                    f"- Do not include polite conversation, reasoning preambles, or explanations.\n"
                    f"- Return ONLY the concise target answer/entity/date/number.\n"
                    f"- CRITICAL: You are FORBIDDEN from saying 'I don't know', 'Unsure', 'Not enough information',\n"
                    f"  'I cannot determine', 'Unknown', or any refusal. The evidence was retrieved FOR this question.\n"
                    f"  If you cannot find the exact answer, make your BEST direct deduction from the evidence.\n"
                    f"- Abstention is a LAST RESORT: check every line of the context first; if any\n"
                    f"  line names the value, answer with it directly.\n"
                    f"- Only if the value appears nowhere in the context, reply: {OFFICIAL_ABSTENTION_TEXT}\n\n"
                    f"{ctx}"
                )

            # AM-decides path (default off): when the deterministic committer can
            # prove the temporal answer from the compiled context, the reader LLM
            # is not called at all - the runtime already holds the answer, and the
            # model's only remaining jobs are the retrieval query and rendering.
            committed = (
                extract_temporal_answer(question_text, ctx)
                if (args.commit_temporal and cat == 2)
                else None
            )
            commit_source = "reader"
            if committed is not None and committed.used:
                pred = committed.answer
                lat_ms = 0.0
                commit_source = f"committed: {committed.detail}"
            else:
                t0_q = time.perf_counter()
                ans = answerer.answer(question_text, prompt)
                lat_ms = (time.perf_counter() - t0_q) * 1000
                pred = ans.text

            # Score answer using LoCoMo standard match
            gt_lower = str(gt).lower().strip()
            ans_lower = pred.lower().strip()
            if cat == 2:
                is_ok = LoCoMoAdapter._temporal_answer_matches(gt_lower, ans_lower)
            elif cat == 3:
                is_ok = LoCoMoAdapter._open_domain_answer_matches(gt_lower, ans_lower)
            elif cat == 4:
                is_ok = LoCoMoAdapter._single_hop_answer_matches(gt_lower, ans_lower)
            elif gt_lower in ans_lower or ans_lower in gt_lower:
                is_ok = True
            else:
                gt_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", gt_lower) if len(w) > 2 or w.isdigit())
                ans_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", ans_lower) if len(w) > 2 or w.isdigit())
                if gt_words and ans_words:
                    overlap = len(gt_words & ans_words)
                    is_ok = (overlap / len(gt_words) >= 0.33) or (len(gt_words) <= 3 and overlap >= 1)
                else:
                    is_ok = False

            ora = any(ev in ctx for ev in q_obj.evidence_ids) if q_obj.evidence_ids else True
            tok = item.get("token_cost", len(ctx.split()))
            # The cached path has no adapter result to ask, and the cache predates
            # the content-based oracle, so there is no label-test flag to record.
            ora_by_id = item.get("oracle_recall_by_id")
        else:
            q_obj = item["question_obj"]
            res = adapter.evaluate_question(q_obj, item["turns"], item["ir_records"], answerer)
            cat = q_obj.category
            is_ok = res.is_correct
            pred = res.predicted_answer
            gt = q_obj.ground_truth
            ora = res.oracle_recall
            tok = res.tokens_used
            lat_ms = res.latency_ms
            ora_by_id = res.oracle_recall_by_id

        cat_stats[cat][1] += 1
        cat_stats[cat][0] += int(is_ok)
        results.append({
            "question_id": q_obj.question_id,
            "category": cat,
            "question": q_obj.question,
            "ground_truth": gt,
            "prediction": pred,
            "is_correct": bool(is_ok),
            "oracle_recall": bool(ora),
            "oracle_recall_by_id": (bool(ora_by_id) if ora_by_id is not None else None),
            "tokens": tok,
            "latency_ms": lat_ms,
            "answer_source": commit_source,
        })

        if i % 10 == 0 or i == len(target_records):
            curr_corr = sum(c[0] for c in cat_stats.values())
            curr_tot = sum(c[1] for c in cat_stats.values())
            acc = curr_corr / curr_tot * 100
            print(f"[{i:04d}/{len(target_records):04d}] Acc: {acc:5.1f}% ({curr_corr}/{curr_tot}) | Cat: {cat_names.get(cat, 'Cat')} | Q: {q_obj.question[:35]}")

    elapsed = time.perf_counter() - t0_all
    tot_corr = sum(c[0] for c in cat_stats.values())
    tot_cnt = sum(c[1] for c in cat_stats.values())
    overall_acc = tot_corr / tot_cnt * 100 if tot_cnt else 0.0

    print("\n" + "=" * 80)
    print(f"          LoCoMo 1,540 EVALUATION SUMMARY ({args.tag})")
    print("=" * 80)
    print(f"Total Questions Evaluated: {tot_cnt}")
    print(f"Overall Accuracy:          {overall_acc:.2f}% ({tot_corr}/{tot_cnt})")
    print(f"Elapsed Time:              {elapsed:.1f}s ({elapsed/max(1, tot_cnt):.2f}s/Q)")
    print("-" * 80)
    print(f"{'Category':<25} | {'Score':<10} | Correct / Total")
    print("-" * 80)
    for c_id in sorted(cat_stats.keys()):
        c_name = cat_names.get(c_id, f"Cat {c_id}")
        corr, tot = cat_stats[c_id]
        c_acc = corr / tot * 100 if tot else 0.0
        print(f"{c_name:<25} | {c_acc:5.1f}%    | {corr}/{tot}")
    print("=" * 80)

    # Save output
    out_dir = Path("benchmark_results/locomo1540")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "tag": args.tag,
            "model": args.model,
            "n_questions": tot_cnt,
            "overall_accuracy": overall_acc,
            "categories": {cat_names.get(k, str(k)): {"acc": v[0]/v[1], "corr": v[0], "tot": v[1]} for k, v in cat_stats.items()},
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
