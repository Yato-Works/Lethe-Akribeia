"""Paired official-F1 comparison between two stored runs.

`integration_gate.py` prices an engine change on the in-house binary status
(FIXED / BROKE / SPURIOUS).  It cannot price a change under the official metric,
because that metric is per-question token F1 and needs the pinned upstream
evaluator.  This is that counterpart.

It refuses to print a delta unless both runs cover the same question ids, and it
reports both directions (improved / worsened) plus the unchanged mass, so a
"gain" that is really a reshuffle is visible.

Usage
    python scripts/benchmarks/paired_official_delta.py \
        --a benchmark_results/locomo_full_20261023_010055.json \
        --b benchmark_results/locomo_full_<stamp>.json \
        --cats 1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = ROOT / "third_party" / "benchmarks" / "locomo"
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
HOLDOUT = {"conv-42", "conv-48"}


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


def qid_parts(qid: str) -> tuple[str, int]:
    rest = qid[3:] if qid.startswith("lc_") else qid
    sid, idx = rest.rsplit("_", 1) if "_" in rest and not rest.endswith("-qa") else (rest, "0")
    return sid, int(idx)


def load_predictions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", payload)
    return {r["question_id"]: str(r.get("predicted_answer", "")) for r in rows}


def per_question_f1(official, dataset, preds: dict[str, str], cats: set[int] | None):
    """{qid: f1} for the questions present in `preds`."""
    out: dict[str, float] = {}
    for sample in dataset:
        sid = sample["sample_id"]
        qas, ids = [], []
        for i, qa in enumerate(sample["qa"]):
            qid = f"lc_{sid}_{i}"
            if qid not in preds:
                continue
            if cats and int(qa.get("category", 0)) not in cats:
                continue
            item = dict(qa)
            if "answer" not in item:
                item["answer"] = item.get("adversarial_answer", "")
            item["am_prediction"] = preds[qid]
            qas.append(item)
            ids.append(qid)
        if not qas:
            continue
        ems, _, _ = official.eval_question_answering(qas, "am_prediction", metric="f1")
        for qid, em in zip(ids, ems):
            out[qid] = float(em)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline run json")
    ap.add_argument("--b", required=True, help="candidate run json")
    ap.add_argument("--dataset", default="datasets/external/locomo10.json")
    ap.add_argument("--cats", default="", help="comma-separated categories")
    ap.add_argument("--scope", choices=["full", "dev"], default="full")
    args = ap.parse_args()

    cats = {int(c) for c in args.cats.split(",")} if args.cats else None
    official = load_official()
    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    a = load_predictions(ROOT / args.a)
    b = load_predictions(ROOT / args.b)

    shared = set(a) & set(b)
    if not shared:
        raise SystemExit("no shared question ids; refusing to report")
    if args.scope == "dev":
        shared = {q for q in shared if qid_parts(q)[0] not in HOLDOUT}

    fa = per_question_f1(official, dataset, {q: a[q] for q in shared}, cats)
    fb = per_question_f1(official, dataset, {q: b[q] for q in shared}, cats)
    keys = sorted(set(fa) & set(fb))
    if not keys:
        raise SystemExit("no scored questions after filtering; refusing to report")

    # Assert the paired join: both sides scored exactly the same questions.
    assert set(fa) == set(fb), "question sets differ after scoring"

    delta = {q: fb[q] - fa[q] for q in keys}
    improved = sum(1 for d in delta.values() if d > 1e-9)
    worsened = sum(1 for d in delta.values() if d < -1e-9)
    unchanged = len(delta) - improved - worsened
    mean_a = sum(fa[q] for q in keys) / len(keys)
    mean_b = sum(fb[q] for q in keys) / len(keys)

    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print(f"questions: {len(keys)}  improved {improved}  worsened {worsened}  unchanged {unchanged}")
    print(f"official F1  A {mean_a * 100:6.2f}%   B {mean_b * 100:6.2f}%   "
          f"delta {(mean_b - mean_a) * 100:+.2f}pt")

    by_cat = defaultdict(list)
    qid_cat = {}
    for sample in dataset:
        sid = sample["sample_id"]
        for i, qa in enumerate(sample["qa"]):
            qid_cat[f"lc_{sid}_{i}"] = int(qa.get("category", 0))
    for q in keys:
        by_cat[qid_cat.get(q, 0)].append(delta[q])
    for cat in sorted(by_cat):
        vals = by_cat[cat]
        print(f"  cat{cat} {CAT.get(cat, '?'):<12} n={len(vals):4d}  "
              f"delta {sum(vals) / len(vals) * 100:+6.2f}pt")

    biggest = sorted(delta.items(), key=lambda kv: kv[1])
    print("\nworsened (top 5):", [f"{q}:{d * 100:+.0f}pt" for q, d in biggest[:5]])
    print("improved (top 5):", [f"{q}:{d * 100:+.0f}pt" for q, d in biggest[::-1][:5]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
