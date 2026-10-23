#!/usr/bin/env python3
"""M30-46 Phase 3-0d: Direct determinism test on ONE question.

lc_conv-44_13 produced different answers with the SAME context in two runs.
Run the reader 5x on that exact context and count distinct outputs.

This decides whether temperature=0.1 is effectively deterministic here, which
sets the noise floor for every A/B in this project.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer


def to_conv(q):
    m = re.match(r"lc_conv-(\d+)_(\d+)", q)
    return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}"


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]

    target_qid = "conv-44-qa-013"
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
    answerer = OllamaAnswerer(model="qwen2.5:7b-instruct")

    q = None
    ctx = None
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        for cand in questions:
            if cand.question_id == target_qid:
                q = cand
                pcc = adapter.compiler.compile(cand.question, ir_records)
                ctx = pcc.context_text
                break
        if q:
            break

    print("=" * 78)
    print("M30-46 Phase 3-0d: determinism test (lc_conv-44_13, 5 runs)")
    print("=" * 78)
    print(f"Question: {q.question}")
    print(f"Gold:     {q.ground_truth}")
    print()

    outs = []
    for i in range(5):
        out = answerer.answer(q.question, ctx).text.strip()
        outs.append(out)
        print(f"  run {i+1}: {out[:80]!r}")
    print()
    uniq = set(outs)
    print(f"distinct outputs: {len(uniq)} / 5")
    print()

    out = PROJECT_ROOT / "results/m30_46"
    out.mkdir(parents=True, exist_ok=True)
    (out / "determinism_test.json").write_text(json.dumps({
        "qid": "lc_conv-44_13", "runs": outs, "distinct": len(uniq),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'determinism_test.json'}")
    print()
    if len(uniq) > 1:
        print("NONDETERMINISTIC: the same (question, context) yields different answers.")
        print("=> Every A/B in this project has a noise floor; small deltas are unreliable.")
    else:
        print("Deterministic on this input.")


if __name__ == "__main__":
    main()
