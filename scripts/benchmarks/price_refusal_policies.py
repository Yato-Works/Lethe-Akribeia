"""Price refusal policies: deterministic gate, Reader verifier, and their unions.

Both signals answer "should this answer be replaced by a refusal?", and they cost
very different things:

  gate      deterministic, 0 LLM calls, +4.33pt dev (V4_PROGRAM.md 4.9)
  verifier  one extra 7B call per answered question, pilot projected +4.8pt on 40

They are not redundant: the gate only fires when the answer's wording is traceable
to another speaker, so it misses paraphrased answers; the verifier reads the
evidence and can catch those, at the price of friendlier errors on inference
questions.

This script prices every combination on the SAME frozen predictions with the
pinned official evaluator, and reports both directions (won / lost) per category,
so an apparent gain that is really a reshuffle stays visible.

Usage
    python scripts/benchmarks/price_refusal_policies.py \
        --contexts benchmark_results/frozen_ctx/locomo_r7_20261024/contexts.jsonl \
        --run benchmark_results/locomo_full_20261023_010055.json \
        --verdicts benchmark_results/support_verifier/dev_all.jsonl --scope dev
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory.skills.refusal_gate import AnswerProvenanceGate  # noqa: E402

OFFICIAL_DIR = ROOT / "third_party" / "benchmarks" / "locomo"
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
HOLDOUT = {"conv-42", "conv-48"}
REFUSAL = "not mentioned"
ABSTENTIONS = {"i don't know.", "", "not mentioned", "none"}


def load_official():
    if importlib.util.find_spec("bert_score") is None:
        stub = types.ModuleType("bert_score")
        stub.score = lambda *a, **k: (None, None, [0.0])
        sys.modules["bert_score"] = stub
    spec = importlib.util.spec_from_file_location(
        "official_locomo_evaluation", OFFICIAL_DIR / "task_eval" / "evaluation.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(OFFICIAL_DIR))
    spec.loader.exec_module(module)
    return module


def normalise(pred: str) -> str:
    low = str(pred).strip().lower()
    return REFUSAL if (low in ABSTENTIONS or "no information available" in low) else str(pred).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--dataset", default="datasets/external/locomo10.json")
    ap.add_argument("--scope", choices=["dev", "full"], default="dev")
    ap.add_argument("--out", default="", help="write the priced table as JSON")
    args = ap.parse_args()

    official = load_official()
    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    contexts = {}
    for line in (ROOT / args.contexts).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            contexts[row["question_id"]] = row
    run = {r["question_id"]: r
           for r in json.loads((ROOT / args.run).read_text(encoding="utf-8"))["results"]}
    verdicts = {}
    for line in (ROOT / args.verdicts).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            verdicts[row["question_id"]] = row["verdict"]

    # Per-question official F1 on the BASELINE predictions (refusals normalised).
    f1: dict[str, float] = {}
    cat_of: dict[str, int] = {}
    qid_pred: dict[str, str] = {}
    for sample in dataset:
        sid = sample["sample_id"]
        if args.scope == "dev" and sid in HOLDOUT:
            continue
        qas, ids = [], []
        for i, qa in enumerate(sample["qa"]):
            qid = f"lc_{sid}_{i}"
            row = run.get(qid)
            if row is None:
                continue
            item = dict(qa)
            if "answer" not in item:
                item["answer"] = item.get("adversarial_answer", "")
            pred = normalise(row.get("predicted_answer", ""))
            item["am_prediction"] = pred
            qas.append(item)
            ids.append(qid)
            qid_pred[qid] = pred
            cat_of[qid] = int(qa.get("category", 0))
        if not qas:
            continue
        ems, _, _ = official.eval_question_answering(qas, "am_prediction", metric="f1")
        for qid, em in zip(ids, ems):
            f1[qid] = float(em)

    gate = AnswerProvenanceGate()
    gate_fire = {}
    for qid, row in run.items():
        if qid not in contexts or qid not in f1:
            continue
        gate_fire[qid] = gate.decide(row["question"], contexts[qid]["context"],
                                     qid_pred[qid]).refuse

    policies = {
        "gate": lambda q: gate_fire.get(q, False),
        "verifier": lambda q: verdicts.get(q) == "NOT SUPPORTED",
        "or": lambda q: gate_fire.get(q, False) or verdicts.get(q) == "NOT SUPPORTED",
        "and": lambda q: gate_fire.get(q, False) and verdicts.get(q) == "NOT SUPPORTED",
        "verifier_beyond_gate": lambda q: (not gate_fire.get(q, False))
                                           and verdicts.get(q) == "NOT SUPPORTED",
    }

    keys = [q for q in f1 if q in qid_pred]
    base = sum(f1[q] for q in keys) / len(keys)
    print(f"questions {len(keys)}   baseline official F1 {base * 100:.2f}%   "
          f"scope={args.scope}")
    print(f"{'policy':<22}{'refused':>8}{'F1':>8}{'delta':>8}   per-category delta")
    report = {
        "questions": len(keys),
        "baseline_official_f1": round(base * 100, 3),
        "scope": args.scope,
        "policies": {},
    }
    for name, fn in policies.items():
        refused = [q for q in keys if fn(q)]
        total = 0.0
        by_cat: dict[int, float] = defaultdict(float)
        lost = won = 0
        for q in keys:
            cat = cat_of[q]
            if q in set(refused):
                new = 1.0 if cat == 5 else 0.0
                total += new
                by_cat[cat] += new - f1[q]
                if cat == 5:
                    won += int(f1[q] < 1.0)
                else:
                    lost += int(f1[q] > 0.0)
            else:
                total += f1[q]
        new_mean = total / len(keys)
        cats = " ".join(f"c{c}:{by_cat[c]:+.1f}" for c in sorted(by_cat))
        print(f"{name:<22}{len(refused):>8}{new_mean * 100:>7.2f}%"
              f"{(new_mean - base) * 100:>+8.2f}  {cats}  (won {won} / lost {lost})")
        report["policies"][name] = {
            "refused": len(refused),
            "official_f1": round(new_mean * 100, 3),
            "delta_pt": round((new_mean - base) * 100, 3),
            "won_cat5": won,
            "lost_cats1_4": lost,
            "per_category_delta": {CAT.get(c, str(c)): round(v, 2)
                                   for c, v in sorted(by_cat.items())},
        }
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
