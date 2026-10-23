#!/usr/bin/env python3
"""M30-45 Phase 3: Did the Treatment prompt actually reach the reader?

Identical outputs across arms suggest the answerer may be bypassed.
Instrument evaluate_question to record, per question:
  - did commit_answer fire?
  - was the answerer called?
  - what system prompt did it receive?

This decides whether the A/B is valid (Phase 3 requirement).
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import (
    OllamaAnswerer, FROZEN_ANSWER_SYSTEM_PROMPT, FROZEN_ANSWER_USER_TEMPLATE,
)

TARGETS = ["lc_conv-48_12", "lc_conv-47_65", "lc_conv-26_63"]


def to_conv_qid(qid):
    m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
    return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}"


CALLS = {"answerer_calls": 0, "prompts": []}


class ProbeAnswerer(OllamaAnswerer):
    def build_answer_messages(self, question_text, context):
        CALLS["answerer_calls"] += 1
        msgs = super().build_answer_messages(question_text, context)
        CALLS["prompts"].append({"q": question_text[:50], "system": msgs[0]["content"][:80]})
        return msgs


def main():
    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")
    tmap = {to_conv_qid(q): q for q in targets}

    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    convs_needed = sorted(set(q.split("-qa-")[0] for q in tmap))

    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
    answerer = ProbeAnswerer(model="qwen2.5:7b-instruct")

    print("=" * 78)
    print("M30-45 Phase 3: Did the reader get the prompt?")
    print("=" * 78)
    print()

    from artificial_memory_gen.skills.answer_committer import commit_answer

    for idx, sid in enumerate(sample_ids):
        if sid not in convs_needed:
            continue
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            if q.question_id not in tmap:
                continue
            qid = tmap[q.question_id]
            pcc = adapter.compiler.compile(q.question, ir_records)
            committed = commit_answer(q.question, pcc.context_text, category=q.category)
            mark = " <<< TARGET" if qid in TARGETS else ""
            print(f"  {qid:16s} commit_answer.used={committed.used}{mark}")
            if committed.used:
                print(f"      committed answer: {str(committed.answer)[:70]}")

    print()
    print(f"Answerer build_answer_messages calls so far: {CALLS['answerer_calls']}")
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print()
    print("If commit_answer.used=True for a target, the LLM prompt never matters there.")
    print("If used=False and the answerer was called, the prompt DID reach the reader.")


if __name__ == "__main__":
    main()
