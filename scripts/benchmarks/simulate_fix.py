import json
from pathlib import Path
import importlib.util
import types

REPO = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = REPO / "third_party" / "benchmarks" / "locomo"

def load_official():
    if importlib.util.find_spec("bert_score") is None:
        stub = types.ModuleType("bert_score")
        stub.score = lambda *a, **k: (None, None, [0.0])
        import sys
        sys.modules["bert_score"] = stub
    import sys
    sys.path.insert(0, str(OFFICIAL_DIR))
    path = OFFICIAL_DIR / "task_eval" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("official_locomo_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

from compare_runs import load_run

def main():
    module = load_official()
    dataset = json.load(open(REPO / "benchmark_results" / "_official_scoring" / "locomo10_normalized.json", encoding="utf-8"))
    v7 = load_run(REPO / "benchmark_results" / "locomo10_runs" / "e2e_smoke_20261002_v7")
    v8 = load_run(REPO / "benchmark_results" / "locomo10_runs" / "e2e_smoke_20261002_v8")

    # What if the 4 corrupted temporal answers were NOT corrupted?
    sim_v8 = {k: dict(v) for k, v in v8.items()}
    for qid in ["conv-26-qa-000", "conv-26-qa-035", "conv-26-qa-058", "conv-44-qa-025"]:
        sim_v8[qid]["predicted_answer"] = v7[qid]["predicted_answer"]

    qas = []
    for sample in dataset:
        for i, qa in enumerate(sample["qa"]):
            qid = f"{sample['sample_id']}-qa-{i:03d}"
            if qid in sim_v8:
                item = dict(qa)
                item["pred"] = sim_v8[qid]["predicted_answer"]
                qas.append(item)

    ems, _, _ = module.eval_question_answering(qas, "pred", metric="f1")
    cat_scores = {1: [], 2: [], 3: [], 4: [], 5: []}
    for q, em in zip(qas, ems):
        cat_scores[q["category"]].append(float(em))

    print("=== Simulated Scores (v8 without Temporal Corruption) ===")
    CAT = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop", 5: "Adversarial"}
    for cat in [1, 2, 3, 4, 5]:
        print(f"{CAT[cat]:<15}: {sum(cat_scores[cat])/len(cat_scores[cat])*100:.2f}%")
    print("-" * 35)
    print(f"{'Overall':<15}: {sum(ems)/len(ems)*100:.2f}%")

if __name__ == "__main__":
    main()
