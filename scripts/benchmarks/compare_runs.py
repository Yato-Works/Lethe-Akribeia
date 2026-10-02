import argparse
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = REPO / "third_party" / "benchmarks" / "locomo"
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

def load_official_module():
    if importlib.util.find_spec("bert_score") is None:
        stub = types.ModuleType("bert_score")
        stub.score = lambda *a, **k: (None, None, [0.0])
        sys.modules["bert_score"] = stub
    path = OFFICIAL_DIR / "task_eval" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("official_locomo_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(OFFICIAL_DIR))
    spec.loader.exec_module(module)
    return module

def load_run(run_dir: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(run_dir.glob("conv_*_results.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for r in data.get("results", []):
                out[r["question_id"]] = r
    return out

def main():
    v7_dir = REPO / "benchmark_results" / "locomo10_runs" / "e2e_smoke_20261002_v7"
    v8_dir = REPO / "benchmark_results" / "locomo10_runs" / "e2e_smoke_20261002_v8"
    dataset_path = REPO / "benchmark_results" / "_official_scoring" / "locomo10_normalized.json"

    dataset = json.load(open(dataset_path, encoding="utf-8"))
    official = load_official_module()

    v7_run = load_run(v7_dir)
    v8_run = load_run(v8_dir)

    diffs = []
    cat_scores_v7 = defaultdict(list)
    cat_scores_v8 = defaultdict(list)

    for sample in dataset:
        sid = sample["sample_id"]
        qas7, qas8 = [], []
        for i, qa in enumerate(sample["qa"]):
            qid = f"{sid}-qa-{i:03d}"
            pred7 = v7_run.get(qid)
            pred8 = v8_run.get(qid)
            if pred7 is None or pred8 is None:
                continue
            item7 = {k: v for k, v in qa.items()}
            item7["pred"] = pred7["predicted_answer"]
            qas7.append((qid, item7))

            item8 = {k: v for k, v in qa.items()}
            item8["pred"] = pred8["predicted_answer"]
            qas8.append((qid, item8))

        if not qas7:
            continue

        ems7, _, _ = official.eval_question_answering([q for _, q in qas7], "pred", metric="f1")
        ems8, _, _ = official.eval_question_answering([q for _, q in qas8], "pred", metric="f1")

        for (qid, item7), (qid8, item8), em7, em8 in zip(qas7, qas8, ems7, ems8):
            cat = item7["category"]
            cat_scores_v7[cat].append(float(em7))
            cat_scores_v8[cat].append(float(em8))
            diffs.append((
                cat, qid, item7.get("question"), item7.get("answer"),
                item7["pred"], item8["pred"], float(em7), float(em8), float(em8) - float(em7)
            ))

    print(f"{'Category':<15} | {'v7 F1':<10} | {'v8 F1':<10} | {'Diff':<10}")
    print("-" * 52)
    all_s7, all_s8 = [], []
    for cat_id in [1, 2, 3, 4, 5]:
        v7_vals = cat_scores_v7[cat_id]
        v8_vals = cat_scores_v8[cat_id]
        m7 = (sum(v7_vals) / len(v7_vals)) * 100 if v7_vals else 0.0
        m8 = (sum(v8_vals) / len(v8_vals)) * 100 if v8_vals else 0.0
        all_s7.extend(v7_vals)
        all_s8.extend(v8_vals)
        print(f"{CAT[cat_id]:<15} | {m7:>9.2f}% | {m8:>9.2f}% | {m8 - m7:>+9.2f}%")
    print("-" * 52)
    tot7 = (sum(all_s7) / len(all_s7)) * 100
    tot8 = (sum(all_s8) / len(all_s8)) * 100
    print(f"{'Overall':<15} | {tot7:>9.2f}% | {tot8:>9.2f}% | {tot8 - tot7:>+9.2f}%")
    print()

    print("=== Questions where v8 scored LOWER than v7 (Diff < 0) ===")
    for cat, qid, quest, gold, p7, p8, s7, s8, d in diffs:
        if d < -0.05:
            print(f"[{qid}] (Cat {cat}: {CAT[cat]}) F1: {s7*100:.1f}% -> {s8*100:.1f}% (Diff: {d*100:+.1f}%)")
            print(f"   Q:    {quest}")
            print(f"   Gold: {gold}")
            print(f"   v7:   {p7}")
            print(f"   v8:   {p8}")
            print()

    print("=== Questions where v8 scored HIGHER than v7 (Diff > 0) ===")
    for cat, qid, quest, gold, p7, p8, s7, s8, d in diffs:
        if d > 0.05:
            print(f"[{qid}] (Cat {cat}: {CAT[cat]}) F1: {s7*100:.1f}% -> {s8*100:.1f}% (Diff: {d*100:+.1f}%)")
            print(f"   Q:    {quest}")
            print(f"   Gold: {gold}")
            print(f"   v7:   {p7}")
            print(f"   v8:   {p8}")
            print()

if __name__ == "__main__":
    main()
