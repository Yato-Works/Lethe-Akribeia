"""Per-question official F1 dump for a stored run.

Imports the PINNED official harness (task_eval/evaluation.py) and prints every
question sorted by ascending F1 so the loss surface is visible at question level.

Usage:
  python scripts/benchmarks/official_f1_by_question.py --run benchmark_results/locomo10_runs/e2e_smoke_20261002_v9
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.score_locomo_official import load_official_module, load_run, REPO, CAT  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="benchmark_results/locomo10_runs/e2e_smoke_20261002_v9")
    ap.add_argument("--dataset", default="datasets/external/locomo10.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    official = load_official_module()
    run = load_run(Path(args.run))
    dataset = json.load(open(REPO / args.dataset, encoding="utf-8"))

    rows = []
    for sample in dataset:
        sid = sample["sample_id"]
        for i, qa in enumerate(sample["qa"]):
            qid = f"{sid}-qa-{i:03d}"
            pred = run.get(qid)
            if pred is None:
                continue
            item = dict(qa)
            if "answer" not in item:
                item["answer"] = item.get("adversarial_answer", "")
            item["am_prediction"] = pred["predicted_answer"]
            ems, _, _ = official.eval_question_answering([item], "am_prediction", metric="f1")
            gt = item.get("answer") or item.get("adversarial_answer") or ""
            rows.append(
                {
                    "qid": qid,
                    "category": item["category"],
                    "cat_name": CAT.get(item["category"], str(item["category"])),
                    "f1": round(float(ems[0]), 4),
                    "oracle": bool(pred.get("oracle_recall")),
                    "run_correct": bool(pred.get("is_correct")),
                    "question": item["question"],
                    "gt": str(gt)[:160],
                    "pred": str(pred["predicted_answer"])[:160],
                }
            )

    rows.sort(key=lambda r: (r["f1"], r["qid"]))
    cats: dict[str, list[float]] = {}
    for r in rows:
        cats.setdefault(r["cat_name"], []).append(r["f1"])

    print(f"n={len(rows)}")
    for name, vals in sorted(cats.items()):
        print(f"{name:<14}{len(vals):>4}{sum(vals) / len(vals) * 100:>10.2f}%")
    allv = [r["f1"] for r in rows]
    print(f"{'ALL':<14}{len(allv):>4}{sum(allv) / len(allv) * 100:>10.2f}%")
    no5 = [r["f1"] for r in rows if r["category"] != 5]
    print(f"{'ALL(no cat5)':<14}{len(no5):>4}{sum(no5) / len(no5) * 100:>10.2f}%")

    hit = [r for r in rows if r["oracle"]]
    miss = [r for r in rows if not r["oracle"]]
    for label, grp in (("ORACLE-HIT", hit), ("ORACLE-MISS", miss)):
        v = [r["f1"] for r in grp]
        if v:
            print(f"{label:<14}{len(v):>4}{sum(v) / len(v) * 100:>10.2f}%")

    print("\n--- lowest F1 questions ---")
    print(f"{'qid':<24}{'cat':<13}{'F1':>7}{'orc':>5}{'run':>5}  pred | gt")
    for r in rows:
        print(
            f"{r['qid']:<24}{r['cat_name']:<13}{r['f1']:>7.2f}"
            f"{str(r['oracle'])[:1]:>5}{str(r['run_correct'])[:1]:>5}  "
            f"{r['pred']} | {r['gt']}"
        )

    if args.out:
        out = REPO / args.out
        out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
