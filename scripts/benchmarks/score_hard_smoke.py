import json
from pathlib import Path
import importlib.util
import types
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = REPO / "third_party" / "benchmarks" / "locomo"

def load_official_module():
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

def load_run(run_dir: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(run_dir.glob("conv_*_results.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            for r in data.get("results", []):
                out[r["question_id"]] = r
    return out

def main():
    official = load_official_module()
    ds_path = REPO / "datasets" / "external" / "locomo10.json"
    dataset = json.load(open(ds_path, encoding="utf-8"))
    run = load_run(REPO / "benchmark_results" / "locomo10_runs" / "hard_smoke_phase5")

    with open(REPO / "benchmark_results" / "hard_smoke_60q.json", encoding="utf-8") as f:
        target_qids = set(json.load(f))

    # Also load baseline v25 f1 if available
    v25_path = REPO / "benchmark_results" / "_official_scoring" / "f1_by_question_full_v25.json"
    v25_full = {x["qid"]: x for x in json.load(open(v25_path, encoding="utf-8"))} if v25_path.exists() else {}

    CAT_NAMES = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop', 5: 'adversarial'}
    cat_scores_new = defaultdict(list)
    cat_scores_old = defaultdict(list)

    diffs = []
    for sample in dataset:
        sid = sample["sample_id"]
        for i, qa in enumerate(sample["qa"]):
            qid = f"{sid}-qa-{i:03d}"
            if qid in target_qids:
                gold = qa.get("answer") or qa.get("adversarial_answer") or ""
                cat = qa["category"]
                
                # Old score
                s_old = v25_full.get(qid, {}).get("f1", 0.0)
                cat_scores_old[cat].append(s_old)

                # New score
                pred_obj = run.get(qid)
                pred_text = pred_obj.get("predicted_answer", "") if pred_obj else ""
                
                # Prepare QA item for official eval_question_answering
                qa_copy = dict(qa)
                qa_copy["pred"] = str(pred_text)
                qa_copy["answer"] = gold
                ems, _, _ = official.eval_question_answering([qa_copy], eval_key="pred", metric="f1")
                s_new = float(ems[0])
                cat_scores_new[cat].append(s_new)

                diffs.append((cat, qid, qa["question"], gold, v25_full.get(qid, {}).get("pred", ""), pred_text, s_old, s_new, s_new - s_old))

    print("=== HARD SMOKE (60 Worst Offenders) Official F1 Score ===")
    print(f"{'Category':<15} | {'Baseline':<10} | {'Phase 5':<10} | {'Gain':<10}")
    print("-" * 52)
    tot_old, tot_new = [], []
    for cat in [1, 2, 3, 4, 5]:
        m_old = sum(cat_scores_old[cat]) / len(cat_scores_old[cat]) * 100 if cat_scores_old[cat] else 0.0
        m_new = sum(cat_scores_new[cat]) / len(cat_scores_new[cat]) * 100 if cat_scores_new[cat] else 0.0
        tot_old.extend(cat_scores_old[cat])
        tot_new.extend(cat_scores_new[cat])
        print(f"{CAT_NAMES[cat]:<15} | {m_old:>9.2f}% | {m_new:>9.2f}% | {m_new - m_old:>+9.2f}pp")
    print("-" * 52)
    o_old = sum(tot_old) / len(tot_old) * 100
    o_new = sum(tot_new) / len(tot_new) * 100
    print(f"{'Overall':<15} | {o_old:>9.2f}% | {o_new:>9.2f}% | {o_new - o_old:>+9.2f}pp")
    print()

    print("=== Top Rescued Questions (Zeros turned into Wins!) ===")
    for cat, qid, q, gold, p_old, p_new, s_old, s_new, d in sorted(diffs, key=lambda x: x[8], reverse=True):
        if d > 0.2:
            print(f"[{qid}] (Cat {cat}: {CAT_NAMES[cat]}) F1: {s_old*100:.1f}% -> {s_new*100:.1f}% (+{d*100:.1f}pp)")
            print(f"   Q:    {q}")
            print(f"   GT:   {gold}")
            print(f"   Old:  {p_old}")
            print(f"   New:  {p_new}")
            print()

if __name__ == "__main__":
    main()
