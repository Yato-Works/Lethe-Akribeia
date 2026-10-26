"""Does a Reader-side support verifier separate unsupported answers from supported ones?

The deterministic refusal gates failed (V4_PROGRAM.md 4.5): single-turn string
binding cannot tell "the fact is spread across turns" from "the fact is absent".
The remaining candidate for the adversarial category is a second, cheap model
call that audits the answer against the same evidence.

This script measures the verifier WITHOUT touching the runtime:

  - take a stored run's answered questions (refusals are skipped),
  - ask the frozen reader whether the evidence states the proposed answer ABOUT
    the asked person,
  - report the verdict rate per category, and the official-F1 delta that would
    follow if every NOT SUPPORTED answer were replaced by the canonical refusal.

The delta is priced with the pinned official evaluator on the same frozen
predictions, so it is comparable to every other V4 number.

Sampling is stratified and explicit, because running the verifier on 1,400
questions costs ~40 minutes and a bad prompt should be caught on 40.

Usage
    python scripts/benchmarks/measure_support_verifier.py \
        --contexts benchmark_results/frozen_ctx/locomo_r7_20261024/contexts.jsonl \
        --run benchmark_results/locomo_full_20261023_010055.json \
        --per-cat 20 --out benchmark_results/support_verifier/sample40.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CANONICAL_REFUSAL = "not mentioned"
ABSTENTIONS = {"i don't know.", "", "not mentioned", "none"}
HOLDOUT = {"conv-42", "conv-48"}

PROMPT = """You are auditing a memory system's answer against its evidence.

EVIDENCE:
{context}

QUESTION: {question}
PROPOSED ANSWER: {prediction}

Decide whether the EVIDENCE states the proposed answer ABOUT the person the
question asks about.

- If the evidence attributes that fact to a DIFFERENT person, or does not state
  it at all, answer NOT SUPPORTED.
- If the evidence does state it about the asked person, answer SUPPORTED.

Answer with exactly one of these two words and nothing else:
SUPPORTED
NOT SUPPORTED
"""


def is_refusal(text: str) -> bool:
    low = str(text).strip().lower()
    return low in ABSTENTIONS or "no information available" in low


def stratified_sample(rows: list[dict], per_cat: int, seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    by_cat: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[int(r["category"])].append(r)
    out = []
    for cat in sorted(by_cat):
        pool = sorted(by_cat[cat], key=lambda r: r["question_id"])
        rng.shuffle(pool)
        out.extend(pool[:per_cat])
    return out


def price_verdicts(records: list[dict], dataset: list[dict], run: dict[str, dict]) -> None:
    """Price the verdicts with the pinned official evaluator.

    A NOT SUPPORTED verdict becomes the canonical refusal.  For category 5 that
    is worth 1.0 (refusal is always correct upstream); for categories 1-4 it is
    worth 0.0, so every false NOT SUPPORTED on an answer that currently earns
    credit is a real loss.  Both sides are printed, never just the gain.
    """
    import importlib.util
    import types

    official_dir = ROOT / "third_party" / "benchmarks" / "locomo"
    if importlib.util.find_spec("bert_score") is None:
        stub = types.ModuleType("bert_score")
        stub.score = lambda *a, **k: (None, None, [0.0])
        sys.modules["bert_score"] = stub
    spec = importlib.util.spec_from_file_location(
        "official_locomo_evaluation", official_dir / "task_eval" / "evaluation.py")
    official = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(official_dir))
    spec.loader.exec_module(official)

    by_qid = {r["question_id"]: r for r in records}
    f1: dict[str, float] = {}
    cat_of: dict[str, int] = {}
    for sample in dataset:
        sid = sample["sample_id"]
        qas, ids = [], []
        for i, qa in enumerate(sample["qa"]):
            qid = f"lc_{sid}_{i}"
            if qid not in by_qid:
                continue
            item = dict(qa)
            if "answer" not in item:
                item["answer"] = item.get("adversarial_answer", "")
            row = run.get(qid)
            item["am_prediction"] = row.get("predicted_answer", "") if row else ""
            qas.append(item)
            ids.append(qid)
            cat_of[qid] = int(qa.get("category", 0))
        if not qas:
            continue
        ems, _, _ = official.eval_question_answering(qas, "am_prediction", metric="f1")
        for qid, em in zip(ids, ems):
            f1[qid] = float(em)

    delta_by_cat: dict[int, float] = defaultdict(float)
    counts: dict[int, Counter] = defaultdict(Counter)
    for qid, r in by_qid.items():
        if qid not in f1:
            continue
        cat = cat_of[qid]
        old = f1[qid]
        refused = r["verdict"] == "NOT SUPPORTED"
        new = (1.0 if cat == 5 else 0.0) if refused else old
        delta_by_cat[cat] += new - old
        counts[cat]["n"] += 1
        counts[cat]["refused"] += int(refused)
        counts[cat]["lost"] += int(refused and old > 0 and cat != 5)
        counts[cat]["won"] += int(refused and cat == 5)

    n = len(f1)
    total = sum(delta_by_cat.values())
    print(f"\npriced on {n} sampled questions (official F1, pinned evaluator)")
    for cat in sorted(counts):
        c = counts[cat]
        print(f"  cat{cat}: n={c['n']:3d} refused={c['refused']:3d} "
              f"won={c['won']:3d} lost={c['lost']:3d} "
              f"delta={delta_by_cat[cat]:+.2f} (sum of per-question F1)")
    print(f"  TOTAL delta on this sample: {total:+.2f} questions-worth "
          f"({total / max(1, n) * 100:+.1f}pt if the sample is representative)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--per-cat", type=int, default=20)
    ap.add_argument("--scope", choices=["dev", "full"], default="dev")
    ap.add_argument("--cats", default="", help="restrict to these categories")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--dataset", default="datasets/external/locomo10.json")
    ap.add_argument("--verdicts", default="",
                    help="price an existing verdicts jsonl instead of calling the model")
    args = ap.parse_args()

    from artificial_memory.research.benchmarks.llm import OllamaAnswerer

    run = {r["question_id"]: r
           for r in json.loads((ROOT / args.run).read_text(encoding="utf-8"))["results"]}
    contexts = {}
    for line in (ROOT / args.contexts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            contexts[row["question_id"]] = row

    wanted = {int(c) for c in args.cats.split(",")} if args.cats else None

    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    if args.verdicts:
        recs = [json.loads(line) for line in
                (ROOT / args.verdicts).read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"pricing {len(recs)} existing verdicts from {args.verdicts}")
        price_verdicts(recs, dataset, run)
        return 0

    rows = []
    for qid, r in run.items():
        if qid not in contexts:
            continue
        if wanted and int(r["category"]) not in wanted:
            continue
        sid = qid[3:].rsplit("_", 1)[0]
        if args.scope == "dev" and sid in HOLDOUT:
            continue
        if is_refusal(r.get("predicted_answer", "")):
            continue
        rows.append({
            "question_id": qid,
            "category": int(r["category"]),
            "question": r["question"],
            "prediction": r["predicted_answer"],
            "context": contexts[qid]["context"],
        })

    sample = stratified_sample(rows, args.per_cat)
    print(f"answered questions in scope: {len(rows)}   sampled: {len(sample)}")

    answerer = OllamaAnswerer(num_ctx=8192)
    verifier_system = (
        "You audit whether evidence supports a proposed answer. "
        "Reply with exactly one of: SUPPORTED, NOT SUPPORTED."
    )
    out_path = ROOT / args.out if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")

    verdicts: dict[int, Counter] = defaultdict(Counter)
    records = []
    for i, row in enumerate(sample, 1):
        prompt = PROMPT.format(context=row["context"], question=row["question"],
                               prediction=row["prediction"])
        try:
            # _chat keeps the frozen temperature/seed/num_ctx; only the messages
            # differ, so the verifier is the same model under the same protocol.
            text = answerer._chat(
                [{"role": "system", "content": verifier_system},
                 {"role": "user", "content": prompt}], 12).text.strip().upper()
        except Exception as exc:  # a single failure must not kill the sample
            text = f"<ERROR {type(exc).__name__}>"
        verdict = ("NOT SUPPORTED" if "NOT SUPPORTED" in text else
                   "SUPPORTED" if "SUPPORTED" in text else "UNPARSED")
        verdicts[row["category"]][verdict] += 1
        records.append({**{k: row[k] for k in ("question_id", "category", "question", "prediction")},
                        "raw": text[:80], "verdict": verdict})
        if out_path:
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
        if i % 10 == 0:
            print(f"  {i}/{len(sample)}")

    print("\nverdict by category (answered questions only):")
    for cat in sorted(verdicts):
        v = verdicts[cat]
        n = sum(v.values())
        print(f"  cat{cat}: n={n:3d}  supported={v['SUPPORTED']:3d}  "
              f"not_supported={v['NOT SUPPORTED']:3d}  unparsed={v['UNPARSED']}")
    price_verdicts(records, dataset, run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
