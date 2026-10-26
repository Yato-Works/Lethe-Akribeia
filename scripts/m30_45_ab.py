#!/usr/bin/env python3
"""M30-45: Production Reader Temporal-Grounding A/B.

ONE change: the Reader's SYSTEM prompt gains temporal-interpretation rules.
Nothing else: retrieval, extraction, compile(), commit_answer, scorer unchanged.

Baseline : FROZEN_ANSWER_SYSTEM_PROMPT (production)
Treatment: same + temporal rules (no QID/name/gold-specific content)

Both arms run through the REAL LoCoMoAdapter path (load_conversation ->
UniversalIRExtractor -> compile -> commit_answer/answerer -> grader).

Scope: Bucket A 28 (dev data). 553 holdout untouched. No production edit.
"""
import sys, json, re, hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks import llm as llm_mod
from artificial_memory.research.benchmarks.llm import (
    OllamaAnswerer, FROZEN_ANSWER_SYSTEM_PROMPT, FROZEN_ANSWER_USER_TEMPLATE,
)

TEMPORAL_RULES = """
Temporal interpretation rules:
- Read each dated line in the context as carrying its own turn-level time.
- To interpret a relative expression (yesterday, tomorrow, last month, next month,
  this month, last year, one year ago), use the header timestamp of the
  question-relevant line that contains it as the reference date.
- yesterday = the day before the reference date; tomorrow = the day after;
  last month = the previous calendar month; next month = the next calendar month.
  Handle year boundaries correctly.
- For 'one year ago', preserve the month-year granularity derivable from the
  reference date. If the question asks for a month and year, do not reduce it to a year.
- Do not mix the timestamps of different lines. Do not treat a conversation-wide
  date as the reference for every line.
- A header timestamp is a turn's time; do not automatically equate it with the
  event's own date unless the evidence supports that.
- If a relative expression does not refer to the fact the question asks about,
  do not use it as the basis for the answer.
- If the needed date is missing, the relevant line cannot be identified, or more
  than one reading remains, do not guess; follow the existing abstention policy.
- If the evidence does answer the question, do the needed calculation and answer
  directly in the requested form; do not declare that there is no answer.
- Do not alter the evidence or add events/dates that are not present.
"""

TREATMENT_SYSTEM_PROMPT = FROZEN_ANSWER_SYSTEM_PROMPT + TEMPORAL_RULES


class TreatmentAnswerer(OllamaAnswerer):
    """Same model/config; only the system prompt differs."""
    def build_answer_messages(self, question_text: str, context: str):
        user_content = FROZEN_ANSWER_USER_TEMPLATE.format(
            context=context, question_text=question_text)
        return [
            {"role": "system", "content": TREATMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]


def to_conv_qid(qid):
    m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
    return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}" if m else qid


def main():
    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")
    tmap = {to_conv_qid(q): q for q in targets}

    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    convs_needed = sorted(set(q.split("-qa-")[0] for q in tmap))

    print("=" * 78)
    print("M30-45: Production Reader Temporal-Grounding A/B")
    print("=" * 78)
    print(f"Targets: {len(targets)}")
    print(f"Baseline system prompt hash:  {hashlib.sha256(FROZEN_ANSWER_SYSTEM_PROMPT.encode()).hexdigest()[:16]}")
    print(f"Treatment system prompt hash: {hashlib.sha256(TREATMENT_SYSTEM_PROMPT.encode()).hexdigest()[:16]}")
    print()

    def run(answerer, label):
        adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
        rows = []
        for idx, sid in enumerate(sample_ids):
            if sid not in convs_needed:
                continue
            turns, questions, ir_records = adapter.load_conversation(idx)
            for q in questions:
                if q.question_id not in tmap:
                    continue
                try:
                    res = adapter.evaluate_question(q, turns, ir_records, answerer)
                    pred, ok = res.predicted_answer, res.is_correct
                except Exception as e:
                    pred, ok = f"ERROR: {e}", False
                rows.append({"qid": tmap[q.question_id], "conv_qid": q.question_id,
                             "question": q.question, "gold": q.ground_truth,
                             "predicted": pred, "correct": bool(ok)})
        n = sum(1 for r in rows if r["correct"])
        print(f"{label}: {n}/{len(rows)}")
        return rows, n

    base_rows, base_n = run(OllamaAnswerer(model="qwen2.5:7b-instruct"), "Baseline")
    print()
    treat_rows, treat_n = run(TreatmentAnswerer(model="qwen2.5:7b-instruct"), "Treatment")
    print()

    # pair
    bmap = {r["qid"]: r for r in base_rows}
    rec = reg = unch = 0
    for r in treat_rows:
        b = bmap[r["qid"]]
        if r["correct"] and not b["correct"]:
            rec += 1
        elif b["correct"] and not r["correct"]:
            reg += 1
        else:
            unch += 1
    print("=" * 78)
    print("PAIRED RESULT")
    print("=" * 78)
    print(f"  Baseline:  {base_n}/28")
    print(f"  Treatment: {treat_n}/28")
    print(f"  recovered={rec} regressed={reg} unchanged={unch} (sum={rec+reg+unch})")
    print()

    out = PROJECT_ROOT / "results/m30_45"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ab_results.json").write_text(json.dumps({
        "baseline_correct": base_n, "treatment_correct": treat_n,
        "recovered": rec, "regressed": reg, "unchanged": unch,
        "baseline_prompt_hash": hashlib.sha256(FROZEN_ANSWER_SYSTEM_PROMPT.encode()).hexdigest()[:16],
        "treatment_prompt_hash": hashlib.sha256(TREATMENT_SYSTEM_PROMPT.encode()).hexdigest()[:16],
        "baseline_rows": base_rows, "treatment_rows": treat_rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "prompt_diff.txt").write_text(
        "=== BASELINE SYSTEM PROMPT ===\n" + FROZEN_ANSWER_SYSTEM_PROMPT +
        "\n\n=== TREATMENT SYSTEM PROMPT ===\n" + TREATMENT_SYSTEM_PROMPT +
        "\n\n=== ADDED BLOCK (treatment only) ===\n" + TEMPORAL_RULES, encoding="utf-8")
    print(f"Saved: {out / 'ab_results.json'}")
    print(f"Saved: {out / 'prompt_diff.txt'}")


if __name__ == "__main__":
    main()
