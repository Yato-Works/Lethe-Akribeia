#!/usr/bin/env python3
"""Run frontier reader evaluation across LoCoMo 1,540 with round-robin Gemini API keys.

Usage:
    uv run python scripts/benchmarks/run_frontier_eval.py [--model gemini-3.6-flash] [--concurrency 6]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
import locomo_scorer as ls

from google import genai
from google.genai import types
from google.genai.errors import APIError

DEFAULT_CACHE = REPO / "benchmark_results" / "locomo_context_cache_rules.jsonl"
DEFAULT_OUTPUT = REPO / "benchmark_results" / "gemini_answer.json"
DEFAULT_BASELINE = REPO / "benchmark_results" / "locomo1540" / "temporal321_rules_commit_7b_postfix.json"
OFFICIAL_ABSTENTION_TEXT = "No information available (not mentioned in the conversation)."


def get_api_keys() -> list[str]:
    env_keys = os.environ.get("GEMINI_API_KEYS", "")
    if env_keys:
        keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        if keys:
            return keys
    # Fallback to single GEMINI_API_KEY
    single_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if single_key:
        return [single_key]
    return []

def format_locomo_prompt(category: int, question: str, context: str) -> str:
    """Exact prompt construction matching src/artificial_memory/research/benchmarks/llm.py & scripts/run_locomo_1540.py."""
    if category == 3:
        shape_block = (
            "- ANSWER SHAPE FIRST: decide what kind of answer the question wants.\n"
            "  If it can be affirmed or denied ('Would X ...?', 'Is X ...?', 'Did X ...?'),\n"
            "  answer 'Yes', 'Likely no' or 'No'.\n"
            "  If it starts with what/which/who/whose, answer with the concrete noun,\n"
            "  trait or stance - NEVER a yes/no. Example: 'What would X's political\n"
            "  leaning likely be?' -> 'Liberal', NOT 'Likely no'.\n"
        )
        ctx_prompt = (
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
            f"=== DIALOGUE CONTEXT ===\n{context}"
        )
    elif category == 1:
        ctx_prompt = (
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
            f"{context}"
        )

    elif category == 2:
        ctx_prompt = (
            f"[INSTRUCTION: TEMPORAL REASONING]\n"
            f"Answer the temporal question using ONLY the dated dialogue context below.\n"
            f"1. Find the turn that states when the event happened.\n"
            f"2. Copy the date form the context uses (e.g. 'The week before 9 June 2023',\n"
            f"   '7 May 2023', '2022'); do not convert 'the week before X' into a plain\n"
            f"   calendar date. But if the context only gives a bare abbreviation such as\n"
            f"   'Last Fri', expand it using that unit's own header date first, so the answer\n"
            f"   reads 'The Friday before 15 July 2023'. Never answer a bare abbreviation.\n"
            f"3. Only do calendar arithmetic when the question asks for a duration or a gap\n"
            f"   (e.g. 'how many days between ...', 'how long after ...'), and then answer with\n"
            f"   just the number and unit (e.g. '3 weeks', '7 days').\n"
            f"4. Never answer with the session date you started from, and never add any\n"
            f"   preamble, explanation or quotes.\n"
            f"5. CRITICAL: You are FORBIDDEN from saying 'I don't know', 'Unsure', 'Not enough information',\n"
            f"   'I cannot determine', 'Unknown', or any refusal. The evidence was retrieved FOR this question.\n"
            f"   If you cannot find the exact answer, make your BEST direct deduction from the evidence.\n"
            f"=== DIALOGUE CONTEXT ===\n{context}\n\n"
        )
    else:
        ctx_prompt = (
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
            f"{context}"
        )
    return f"Conversation context:\n{ctx_prompt}\n\nQuestion: {question}\nAnswer:\n"

async def query_gemini_single(
    client: genai.Client,
    model: str,
    prompt: str,
    max_retries: int = 8,
) -> str:
    """Query Gemini with exponential backoff on 429/503 and transient errors."""
    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=64,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    backoff = 1.0
    for attempt in range(max_retries):
        try:
            resp = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            raw = resp.text if resp.text else ""
            return raw.strip()
        except APIError as e:
            if attempt == max_retries - 1:
                print(f"\n[ERROR] APIError after {max_retries} attempts: {e}")
                return ""
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n[ERROR] Exception after {max_retries} attempts: {e}")
                return ""
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)
    return ""

async def run_batch_inference(
    items: list[dict],
    output_path: Path,
    model: str = "gemini-3.6-flash",
    concurrency: int = 6,
) -> dict[str, str]:
    existing_answers: dict[str, str] = {}
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            existing_answers = json.loads(output_path.read_text(encoding="utf-8"))
            print(f"[Resume] Loaded {len(existing_answers)} existing answers from {output_path.name}")
        except Exception as e:
            print(f"[Warning] Could not parse existing {output_path.name}: {e}. Starting fresh.")
            existing_answers = {}

    keys = get_api_keys()
    if not keys:
        print("[Error] No Gemini API keys found. Please set GEMINI_API_KEYS (or GEMINI_API_KEY) environment variable.")
        sys.exit(1)
    print(f"[Config] Using {len(keys)} Gemini API keys round-robin with model={model}, concurrency={concurrency}")
    clients = [genai.Client(api_key=k) for k in keys]


    pending_items = [item for item in items if item["qid"] not in existing_answers]
    print(f"[Progress] {len(existing_answers)} already done, {len(pending_items)} pending out of {len(items)}")

    if not pending_items:
        return existing_answers

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    pbar = tqdm(total=len(items), initial=len(existing_answers), desc=f"Evaluating {model}")

    async def worker(index: int, item: dict):
        qid = item["qid"]
        category = item["category"]
        question = item["question"]
        context = item.get("context", "")

        prompt = format_locomo_prompt(category, question, context)
        client = clients[index % len(clients)]

        async with sem:
            answer = await query_gemini_single(client, model, prompt)

        async with write_lock:
            existing_answers[qid] = answer
            pbar.update(1)
            if len(existing_answers) % 10 == 0 or len(existing_answers) == len(items):
                output_path.write_text(json.dumps(existing_answers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    tasks = [worker(i, item) for i, item in enumerate(pending_items)]
    await asyncio.gather(*tasks)
    pbar.close()

    output_path.write_text(json.dumps(existing_answers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[Complete] Successfully saved all {len(existing_answers)} answers to {output_path.name}")
    return existing_answers

def evaluate_and_compare(answers: dict[str, str], baseline_path: Path):
    """Run standard LoCoMo evaluation and compare against 7B deployed baseline."""
    ref_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    ref_results = ref_data.get("results", [])
    ref_map = {r["question_id"]: r for r in ref_results}

    rows = {}
    for qid, pred in answers.items():
        if qid not in ref_map:
            continue
        r = ref_map[qid]
        cat = r["category"]
        gt = r["ground_truth"]

        gt_lower = str(gt).lower().strip()
        ans_lower = str(pred).lower().strip()

        is_ok = False
        if cat == 2:
            is_ok = LoCoMoAdapter._temporal_answer_matches(gt_lower, ans_lower)
        elif cat == 3:
            is_ok = LoCoMoAdapter._open_domain_answer_matches(gt_lower, ans_lower)
        elif cat == 4:
            is_ok = LoCoMoAdapter._single_hop_answer_matches(gt_lower, ans_lower)
        elif not gt_lower:
            is_ok = LoCoMoAdapter.is_refusal_shaped(ans_lower)
        elif gt_lower in ans_lower or ans_lower in gt_lower:
            is_ok = True
        else:
            clean_gt = re.sub(r"\bde-stress\b", "destress", gt_lower).replace("-", " ")
            clean_ans = re.sub(r"\bde-stress\b", "destress", ans_lower).replace("-", " ")
            for w, n in LoCoMoAdapter._NUMBER_WORDS.items():
                clean_gt = re.sub(rf"\b{w}\b", n, clean_gt)
                clean_ans = re.sub(rf"\b{w}\b", n, clean_ans)
            if clean_gt in clean_ans or clean_ans in clean_gt:
                is_ok = True
            else:
                gt_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", clean_gt) if len(w) > 2 or w.isdigit())
                ans_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", clean_ans) if len(w) > 2 or w.isdigit())
                if gt_words and ans_words:
                    overlap = len(gt_words & ans_words)
                    if overlap / len(gt_words) >= 0.33:
                        is_ok = True
                    elif len(gt_words) <= 3 and overlap >= 1:
                        is_ok = True

        rows[qid] = {
            "correct": is_ok,
            "category": cat,
            "prediction": str(pred),
            "ground_truth": gt,
        }

    scored = ls.score_run({"model": "gemini", "rows": rows})
    n = len(rows)
    hit_rate = sum(r["correct"] for r in rows.values()) / n * 100 if n else 0.0
    official_f1 = scored["totals"]["official_f1"]["sum"] / n * 100 if n else 0.0

    b_rows = {
        r["question_id"]: {
            "correct": r["is_correct"],
            "category": r["category"],
            "prediction": r["prediction"],
            "ground_truth": r["ground_truth"],
        }
        for r in ref_results
    }
    b_scored = ls.score_run({"model": ref_data.get("model", "7B"), "rows": b_rows})
    b_n = len(b_rows)
    b_hit_rate = sum(r["correct"] for r in b_rows.values()) / b_n * 100 if b_n else 0.0
    b_official_f1 = b_scored["totals"]["official_f1"]["sum"] / b_n * 100 if b_n else 0.0

    print("\n" + "=" * 70)
    print("                     EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Gemini':>12} | {'7B Deployed':>12} | {'Delta (pp)':>12}")
    print("-" * 70)
    delta_hit = hit_rate - b_hit_rate
    delta_f1 = official_f1 - b_official_f1
    print(f"{'Binary Hit Rate':<25} | {hit_rate:>11.2f}% | {b_hit_rate:>11.2f}% | {delta_hit:>+11.2f}pp")
    print(f"{'Official F1':<25} | {official_f1:>11.2f}% | {b_official_f1:>11.2f}% | {delta_f1:>+11.2f}pp")
    print("=" * 70)

    print("\nPer-Category Breakdown (Gemini vs 7B Deployed):")
    print(f"{'Category':<15} | {'N':>5} | {'Gemini Hit':>11} | {'7B Hit':>11} | {'Gemini F1':>11} | {'7B F1':>11}")
    print("-" * 75)
    for cat_name in sorted(scored["per_category"]):
        g_cat = scored["per_category"][cat_name]
        b_cat = b_scored["per_category"].get(cat_name, {})
        cat_n = g_cat["lethe_current"]["n"]
        b_cat_n = b_cat.get("lethe_current", {}).get("n", 0)
        g_hit = g_cat["lethe_current"]["sum"] / cat_n * 100 if cat_n else 0.0
        b_hit = b_cat.get("lethe_current", {}).get("sum", 0.0) / b_cat_n * 100 if b_cat_n else 0.0
        g_f1 = g_cat["official_f1"]["sum"] / cat_n * 100 if cat_n else 0.0
        b_f1 = b_cat.get("official_f1", {}).get("sum", 0.0) / b_cat_n * 100 if b_cat_n else 0.0
        print(f"{cat_name:<15} | {cat_n:>5} | {g_hit:>10.1f}% | {b_hit:>10.1f}% | {g_f1:>10.1f}% | {b_f1:>10.1f}%")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-3.6-flash", help="Gemini model to use (default: gemini-3.6-flash)")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="Context cache path")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Output answers JSON path")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="7B baseline run for comparison")
    parser.add_argument("--concurrency", type=int, default=6, help="Number of concurrent requests")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"Error: cache file not found at {cache_path}")
        sys.exit(1)

    print(f"Loading questions from {cache_path.name}...")
    items = []
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("category") != 5:  # Exclude category 5 adversarial
                items.append({
                    "qid": r.get("qid") or r.get("question_id"),
                    "category": r.get("category"),
                    "question": r.get("question"),
                    "context": r.get("context", ""),
                })

    print(f"Loaded {len(items)} questions (expected 1540).")
    output_path = Path(args.out)

    answers = asyncio.run(
        run_batch_inference(
            items=items,
            output_path=output_path,
            model=args.model,
            concurrency=args.concurrency,
        )
    )

    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        evaluate_and_compare(answers, baseline_path)
    else:
        print(f"[Warning] Baseline {baseline_path} not found. Skipping comparison.")


if __name__ == "__main__":
    main()

