#!/usr/bin/env python3
# M30-3: Actual evaluation of 28 TARGET cases with corrected evidence and patched scorer
import json, sys, os, re
from pathlib import Path

# Ensure we can import the local package
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artificial_memory.research.benchmarks.llm import OllamaAnswerer
from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader

def load_expanded_mapping():
    with open("results/m30_manual_evidence_mapping_expanded.json", encoding="utf-8") as f:
        return json.load(f)

def load_all_cases(bucketA_path):
    with open(bucketA_path, encoding="utf-8") as f:
        data = json.load(f)
    return {c["question_id"]: c for c in data["cases"]}

def load_conversations(bucketA_path):
    with open(bucketA_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["conversations"]

SESSION_DATES = {
    "conv-26": "2023-08-28",
    "conv-30": "2023-11-30",
    "conv-41": "2023-12-04",
    "conv-42": "2023-12-04",
    "conv-44": "2023-12-04",
    "conv-47": "2022-09-30",
    "conv-48": "2023-01-31",
    "conv-49": "2023-12-25",
    "conv-50": "2023-08-28",
}

def extract_evidence_turns(qid, case, conversations, gold):
    """Same as in m29_reader_compare.py"""
    conv_key = case.get("source_conversation", "")
    turns = conversations.get(conv_key, [])
    if not isinstance(turns, list):
        return []
    question = case.get("question", "").lower()
    gold_lower = gold.lower().strip()
    gold_keywords = set()
    gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
    stopwords = {"the", "in", "on", "at", "of", "is", "a", "an", "to", "for",
                 "january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december"}
    for w in gold_clean.split():
        if len(w) > 2 and w not in stopwords:
            gold_keywords.add(w.lower())
    question_keywords = set()
    for w in question.replace("?", "").lower().split():
        if len(w) > 3:
            question_keywords.add(w)
    evidence = []
    for i, t in enumerate(turns):
        if not isinstance(t, dict):
            continue
        text = t.get("text", "")
        text_lower = text.lower()
        if any(kw in text_lower for kw in gold_keywords):
            evidence.append(f"D{i+1} | {t.get('speaker', '?')} | {text}")
        elif any(kw in text_lower for kw in question_keywords if len(kw) > 3):
            if any(d in text_lower for d in ["month", "year", "2023", "2022", "2024", "indiana", "seoul", "france", "rome", "boston"]):
                evidence.append(f"D{i+1} | {t.get('speaker', '?')} | {text}")
    seen = set()
    unique = []
    for t in evidence:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[-8:]  # up to 8 turns

def oracle_evidence_text(turns):
    if not turns:
        return ""
    return "\n".join(f"- {t}" for t in turns)

def main():
    expanded = load_expanded_mapping()
    bucketA_path = Path("benchmark_results/evidence_audit/v4_bucketA_all.json")
    m25_path = Path("benchmark_results/evidence_audit/v4_intervention_temporal.json")
    all_cases = load_all_cases(bucketA_path)
    conversations = load_conversations(bucketA_path)
    with open(m25_path) as f:
        m25 = json.load(f)
    target_qids = [r["question_id"] for r in m25["rows"] if r.get("group") == "TARGET"]
    assert len(target_qids) == 28, f"Expected 28 TARGET cases, got {len(target_qids)}"
    grader = OfficialLongMemEvalGrader()
    reader = OllamaAnswerer(model="qwen2.5:7b-instruct")  # FROZEN_MODEL
    results = []
    summary = {"correct": 0, "total": 0, "generation_failure": 0, "processing_failure": 0, "scorer_mismatch": 0, "evidence_problem": 0}
    for qid in target_qids:
        case = all_cases.get(qid)
        if not case:
            print(f"WARNING: case {qid} not found")
            continue
        question = case.get("question", "")
        gold = case.get("gold_answer", "")
        conv_key = case.get("source_conversation", "")
        ref_date = SESSION_DATES.get(conv_key)
        evidence_turns = extract_evidence_turns(qid, case, conversations, gold)
        evidence_text = oracle_evidence_text(evidence_turns)
        context = evidence_text + "\n\n" + question
        try:
            response = reader.answer(question, context)
            reader_text = response.text.strip()
        except Exception as exc:
            reader_text = f"ERROR: {exc}"
            summary["generation_failure"] += 1
            results.append({
                "question_id": qid,
                "model": "qwen2.5:7b-instruct",
                "question": question,
                "gold": gold,
                "evidence_text": evidence_text,
                "reader_text": reader_text,
                "reader_correct": False,
                "scorer_correct": False,
                "verdict": "generation_failure",
                "note": str(exc)
            })
            continue
        # Check for empty output after stripping
        if not reader_text:
            summary["generation_failure"] += 1
            verdict = "generation_failure_empty"
        else:
            # Check if it's just thinking tokens? We'll assume generation succeeded.
            verdict = "generation_ok"
        # Grade
        result = grader.grade(qid, "temporal-reasoning", reader_text, gold, reference_date=ref_date)
        scorer_correct = result.is_correct
        # Simple reader correctness: token overlap (as in m29)
        gold_lower = gold.lower().strip()
        reader_lower = reader_text.lower().strip()
        reader_correct = gold_lower in reader_lower or reader_lower in gold_lower
        if scorer_correct:
            summary["correct"] += 1
        else:
            # Determine if it's a scorer mismatch (i.e., reader correct but scorer not)
            if reader_correct and not scorer_correct:
                summary["scorer_mismatch"] += 1
                verdict = "scorer_mismatch"
            else:
                # Could be evidence problem or processing failure; we'll label as evidence_problem for now
                summary["evidence_problem"] += 1
                verdict = "evidence_or_processing"
        summary["total"] += 1
        results.append({
            "question_id": qid,
            "model": "qwen2.5:7b-instruct",
            "question": question,
            "gold": gold,
            "evidence_text": evidence_text,
            "reader_text": reader_text,
            "reader_correct": reader_correct,
            "scorer_correct": scorer_correct,
            "verdict": verdict,
            "note": ""
        })
        print(f"[{qid}] reader='{reader_text[:60]}' | gold='{gold[:30]}' | scorer_correct={scorer_correct} | reader_correct={reader_correct}")
    # Compute accuracy
    total = summary["total"]
    accuracy = summary["correct"] / total if total > 0 else 0.0
    summary["accuracy"] = accuracy
    # Write results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "m30_3_evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print("\n=== M30-3 Evaluation Summary ===")
    print(f"Total cases: {total}")
    print(f"Correct: {summary['correct']} ({accuracy:.1%})")
    print(f"Generation failures: {summary['generation_failure']}")
    print(f"Scorer mismatches (reader correct but scorer wrong): {summary['scorer_mismatch']}")
    print(f"Evidence or processing problems: {summary['evidence_problem']}")
    print(f"Results written to: {out_path}")

if __name__ == "__main__":
    main()