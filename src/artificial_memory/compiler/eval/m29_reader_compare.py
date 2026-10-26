"""
M29 — Reader Comparison Experiment

Purpose
-------
Compare qwen2.5:7b vs qwen3:4b on the same Oracle evidence + question,
with the same scorer, to determine whether the 0/28 result is a model-side
limitation or a deeper evidence/scoring problem.

Usage:
    .venv/Scripts/python.exe src/artificial_memory/compiler/eval/m29_reader_compare.py

Output:
    results/m29_reader_compare.jsonl
    results/m29_reader_compare_summary.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader  # noqa: E402
from artificial_memory.research.benchmarks.llm import OllamaAnswerer  # noqa: E402


@dataclass
class M29Row:
    question_id: str
    model: str
    question: str
    gold: str
    source_conversation: str
    evidence_text: str
    reader_text: str
    reader_correct: bool
    scorer_correct: bool
    verdict: str
    note: str = ""


MODELS = ["qwen2.5:7b-instruct", "qwen3:4b"]

# Session reference dates (from M23 validation)
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


def load_all_cases(bucketA_path: Path) -> dict[str, dict]:
    """Load all cases from v4_bucketA_all.json into a qid->case map."""
    with open(bucketA_path, encoding="utf-8") as f:
        data = json.load(f)
    return {c["question_id"]: c for c in data["cases"]}


def load_conversations(bucketA_path: Path) -> dict[str, list[dict]]:
    """Load conversations from v4_bucketA_all.json."""
    with open(bucketA_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["conversations"]


def extract_evidence_turns(qid: str, case: dict, conversations: dict, gold: str) -> list[str]:
    """
    Automatically extract the turns that support the gold answer.

    Strategy:
    1. Find turns mentioning gold keywords / dates / entities.
    2. Include the source conversation's date anchor turn if present.
    """
    conv_key = case.get("source_conversation", "")
    turns = conversations.get(conv_key, [])
    if not isinstance(turns, list):
        return []

    question = case.get("question", "").lower()
    gold_lower = gold.lower().strip()

    # Gold keywords (strip stop words and punctuation)
    gold_keywords = set()
    gold_clean = re.sub(r"[^\w\s]", " ", gold_lower)
    stopwords = {"the", "in", "on", "at", "of", "is", "a", "an", "to", "for",
                 "january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december"}
    for w in gold_clean.split():
        if len(w) > 2 and w not in stopwords:
            gold_keywords.add(w.lower())

    # Question keywords
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

        # Match gold keywords
        if any(kw in text_lower for kw in gold_keywords):
            evidence.append(f"D{i+1} | {t.get('speaker', '?')} | {text}")
        # Match question keywords (for entity-related questions)
        elif any(kw in text_lower for kw in question_keywords if len(kw) > 3):
            # Include if the turn also has date or location context
            if any(d in text_lower for d in ["month", "year", "2023", "2022", "2024", "indiana", "seoul", "france", "rome", "boston"]):
                evidence.append(f"D{i+1} | {t.get('speaker', '?')} | {text}")

    # Deduplicate
    seen = set()
    unique = []
    for t in evidence:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique[-8:]  # up to 8 turns


def oracle_evidence_text(turns: list[str]) -> str:
    """Assemble evidence turns as a single text block."""
    if not turns:
        return ""
    return "\n".join(f"- {t}" for t in turns)


def run_m29():
    bucketA_path = Path("benchmark_results/evidence_audit/v4_bucketA_all.json")
    m25_path = Path("benchmark_results/evidence_audit/v4_intervention_temporal.json")

    all_cases = load_all_cases(bucketA_path)
    conversations = load_conversations(bucketA_path)

    # The 28 TARGET qids from M25
    with open(m25_path) as f:
        m25 = json.load(f)
    target_qids = [r["question_id"] for r in m25["rows"] if r.get("group") == "TARGET"]
    assert len(target_qids) == 28, f"Expected 28 TARGET cases, got {len(target_qids)}"

    grader = OfficialLongMemEvalGrader()

    rows = []
    summary = {m: {"correct": 0, "total": 0, "dont_know": 0, "errors": 0, "accuracy": 0.0} for m in MODELS}

    for qid in target_qids:
        case = all_cases.get(qid)
        if not case:
            print(f"WARNING: case {qid} not found in bucketA cases registry")
            continue

        question = case.get("question", "")
        gold = case.get("gold_answer", "")
        conv_key = case.get("source_conversation", "")
        ref_date = SESSION_DATES.get(conv_key)

        evidence_turns = extract_evidence_turns(qid, case, conversations, gold)
        evidence_text = oracle_evidence_text(evidence_turns)

        for model_name in MODELS:
            reader = OllamaAnswerer(model=model_name)
            context = evidence_text + "\n\n" + question

            try:
                response = reader.answer(question, context)
                reader_text = response.text.strip()
            except Exception as exc:
                reader_text = f"ERROR: {exc}"
                summary[model_name]["errors"] += 1

            # Use frozen official grader with relative-date normalization
            result = grader.grade(qid, "temporal-reasoning", reader_text, gold, reference_date=ref_date)
            scorer_correct = result.is_correct

            # Reader correctness: simple token overlap
            gold_lower = gold.lower().strip()
            reader_lower = reader_text.lower().strip()
            reader_correct = gold_lower in reader_lower or reader_lower in gold_lower

            if "don't know" in reader_text.lower():
                summary[model_name]["dont_know"] += 1

            row = M29Row(
                question_id=qid,
                model=model_name,
                question=question,
                gold=gold,
                source_conversation=conv_key,
                evidence_text=context,
                reader_text=reader_text,
                reader_correct=reader_correct,
                scorer_correct=scorer_correct,
                verdict="correct" if scorer_correct else "wrong",
            )
            rows.append(row)

            summary[model_name]["total"] += 1
            if scorer_correct:
                summary[model_name]["correct"] += 1

            print(f"[{model_name}] {qid}: correct={scorer_correct} | reader='{reader_text[:60]}' | gold='{gold[:30]}'")

    # Compute accuracy
    for m in summary:
        s = summary[m]
        total = s["total"]
        s["accuracy"] = s["correct"] / total if total > 0 else 0.0

    # Write results
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "m29_reader_compare.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            row_dict = {
                "question_id": r.question_id,
                "model": r.model,
                "question": r.question,
                "gold": r.gold,
                "source_conversation": r.source_conversation,
                "evidence_text": r.evidence_text,
                "reader_text": r.reader_text,
                "reader_correct": r.reader_correct,
                "scorer_correct": r.scorer_correct,
                "verdict": r.verdict,
            }
            f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")

    summary_path = output_dir / "m29_reader_compare_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=== M29 Reader Comparison Summary ===")
    for model_name, stats in summary.items():
        print(f"{model_name}: {stats['correct']}/{stats['total']} (accuracy={stats['accuracy']:.1%}), 'I don't know': {stats['dont_know']}, errors={stats['errors']}")

    print()
    print("Results written to:")
    print(f"  {jsonl_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    run_m29()
