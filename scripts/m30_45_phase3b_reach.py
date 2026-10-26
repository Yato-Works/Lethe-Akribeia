#!/usr/bin/env python3
"""M30-45 Phase 3b: Run evaluate_question for the 3 targets and capture the
ACTUAL system prompt the reader received + the raw output.

This settles whether the treatment prompt reached the reader.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import (
    OllamaAnswerer, FROZEN_ANSWER_SYSTEM_PROMPT, FROZEN_ANSWER_USER_TEMPLATE,
)

TARGETS = {"conv-48-qa-012": "lc_conv-48_12",
           "conv-47-qa-065": "lc_conv-47_65",
           "conv-26-qa-063": "lc_conv-26_63"}

TREAT_EXTRA = "\nTemporal interpretation rules: (10 rules appended)"


class Probe(OllamaAnswerer):
    last_system = None
    calls = 0
    def build_answer_messages(self, q, c):
        msgs = super().build_answer_messages(q, c)
        Probe.last_system = msgs[0]["content"]
        Probe.calls += 1
        return msgs


class ProbeTreat(OllamaAnswerer):
    last_system = None
    calls = 0
    def build_answer_messages(self, q, c):
        user = FROZEN_ANSWER_USER_TEMPLATE.format(context=c, question_text=q)
        sysp = FROZEN_ANSWER_SYSTEM_PROMPT + TREAT_EXTRA
        ProbeTreat.last_system = sysp
        ProbeTreat.calls += 1
        return [{"role": "system", "content": sysp}, {"role": "user", "content": user}]


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]

    print("=" * 78)
    print("M30-45 Phase 3b: Does the treatment prompt reach the reader?")
    print("=" * 78)
    print()

    for label, ans_cls in [("BASELINE", Probe), ("TREATMENT", ProbeTreat)]:
        ans_cls.calls = 0
        ans_cls.last_system = None
        adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
        answerer = ans_cls(model="qwen2.5:7b-instruct")
        print(f"--- {label} ---")
        for idx, sid in enumerate(sample_ids):
            turns, questions, ir_records = adapter.load_conversation(idx)
            for q in questions:
                if q.question_id not in TARGETS:
                    continue
                res = adapter.evaluate_question(q, turns, ir_records, answerer)
                print(f"  {TARGETS[q.question_id]}: pred={str(res.predicted_answer)[:60]!r}")
        print(f"  answerer build calls: {ans_cls.calls}")
        print(f"  last system prompt head: {str(ans_cls.last_system)[:70]!r}")
        print(f"  system contains 'Temporal interpretation': "
              f"{'Temporal interpretation' in str(ans_cls.last_system)}")
        print()

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print()
    print(f"Baseline reader calls:  {Probe.calls}")
    print(f"Treatment reader calls: {ProbeTreat.calls}")
    if ProbeTreat.calls == 0:
        print("Treatment prompt NEVER reached the reader -> experiment INVALID.")
    else:
        print("Treatment prompt DID reach the reader.")
    print()

    out = PROJECT_ROOT / "results/m30_45"
    (out / "prompt_reach.json").write_text(json.dumps({
        "baseline_reader_calls": Probe.calls,
        "treatment_reader_calls": ProbeTreat.calls,
        "treatment_system_has_rules": "Temporal interpretation" in str(ProbeTreat.last_system),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'prompt_reach.json'}")


if __name__ == "__main__":
    main()
