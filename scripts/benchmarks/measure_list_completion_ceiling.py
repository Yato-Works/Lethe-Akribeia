"""Price list-answer completion before building it: ceiling, then a cheap rule.

Two questions, in this order, because the second is pointless if the first is
small:

1. CEILING. If a perfect extractor appended every gold item that is present in the
   compiled context but missing from the prediction, how much would official F1
   gain?  This is an oracle (it reads gold), so it bounds any engine.
2. CHEAP RULE. Append the other members of a comma/"and" list in a turn that
   already contains a predicted item, attributed to the asked subject.  The
   hypothesis is that items enumerated together in one turn are usually both
   correct.

Both are priced with the pinned official evaluator on the same frozen predictions,
dev conversations only.

Usage
    python scripts/benchmarks/measure_list_completion_ceiling.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = ROOT / "third_party" / "benchmarks" / "locomo"
HOLDOUT = {"conv-42", "conv-48"}
_LINE = re.compile(r"^\[[^\]]+\]\s*([A-Za-z][A-Za-z0-9_ -]{0,24}?):\s*(.*)$")


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


def gold_items(gold: str) -> list[str]:
    return [x.strip().strip("\"'") for x in re.split(r",|;|\band\b", gold) if x.strip()]


def present(item: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(item.lower().rstrip('s'))}\w{{0,2}}\b",
                     text.lower()) is not None


def stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{4,}", text.lower())}


def segments(text: str) -> list[str]:
    out = []
    for part in re.split(r",|;|\band\b", text):
        part = part.strip(" .\"')(")
        if 2 <= len(part) <= 40:
            out.append(part)
    return out


def asked_subject(question: str, speakers: list[str]) -> str | None:
    low = question.lower()
    hits = [s for s in speakers if re.search(rf"\b{re.escape(s.lower())}\b", low)]
    return hits[0] if len(hits) == 1 else None


def sibling_completion(question: str, context: str, prediction: str) -> str | None:
    lines = []
    for raw in context.splitlines():
        m = _LINE.match(raw.strip())
        if m:
            lines.append((m.group(1), m.group(2)))
    speakers: list[str] = []
    for speaker, _ in lines:
        if speaker not in speakers:
            speakers.append(speaker)
    subject = asked_subject(question, speakers)
    if subject is None:
        return None
    predicted_stems = stems(prediction)
    extra: list[str] = []
    for speaker, text in lines:
        if speaker.lower() != subject.lower():
            continue
        parts = segments(text)
        if not parts or not any(predicted_stems & stems(p) for p in parts):
            continue
        for part in parts:
            if part.lower() in prediction.lower() or (stems(part) & predicted_stems):
                continue
            extra.append(part)
    if not extra:
        return None
    return prediction.rstrip(". ") + ", " + ", ".join(dict.fromkeys(extra))


def score(official, dataset, preds: dict[str, str], run: dict[str, dict],
          categories: set[int] | None = None, list_only: bool = False) -> dict[str, float]:
    out: dict[str, float] = {}
    for sample in dataset:
        sid = sample["sample_id"]
        if sid in HOLDOUT:
            continue
        qas, ids = [], []
        for i, qa in enumerate(sample["qa"]):
            qid = f"lc_{sid}_{i}"
            if qid not in preds or qid not in run:
                continue
            row = run[qid]
            if list_only and (not row.get("gold_is_list") or row["category"] == 5):
                continue
            if categories and int(qa.get("category", 0)) not in categories:
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
    ap.add_argument("--run", default="benchmark_results/ab/merged_no_committer.json")
    ap.add_argument("--contexts",
                    default="benchmark_results/frozen_ctx/locomo_r7_20261024/contexts.jsonl")
    ap.add_argument("--dataset", default="datasets/external/locomo10.json")
    ap.add_argument("--out", default="benchmark_results/list_completion/ceiling.json")
    args = ap.parse_args()

    official = load_official()
    run = {r["question_id"]: r
           for r in json.loads((ROOT / args.run).read_text(encoding="utf-8"))["results"]}
    contexts = {}
    for line in (ROOT / args.contexts).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            contexts[row["question_id"]] = row
    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))

    base = {q: r["predicted_answer"] for q, r in run.items()}
    oracle = dict(base)
    sibling = dict(base)
    appended = proposed = 0
    for sample in dataset:
        sid = sample["sample_id"]
        if sid in HOLDOUT:
            continue
        for i, qa in enumerate(sample["qa"]):
            qid = f"lc_{sid}_{i}"
            row = run.get(qid)
            if row is None or not row.get("gold_is_list") or row["category"] == 5:
                continue
            context = contexts[qid]["context"]
            prediction = str(row["predicted_answer"])
            missing = [it for it in gold_items(row["ground_truth"])
                       if present(it, context) and it.lower() not in prediction.lower()]
            if missing:
                oracle[qid] = prediction.rstrip(". ") + ", " + ", ".join(missing)
                appended += len(missing)
            out = sibling_completion(row["question"], context, prediction)
            if out:
                sibling[qid] = out
                proposed += 1

    a = score(official, dataset, base, run, list_only=True)
    b = score(official, dataset, oracle, run, list_only=True)
    c = score(official, dataset, sibling, run, list_only=True)
    keys = sorted(set(a) & set(b) & set(c))
    n = len(keys)
    mean = lambda d: sum(d[q] for q in keys) / max(1, n) * 100  # noqa: E731

    # Single-value ceiling on the same run (cats 3-4): how much would an extractor
    # that returns the exact gold gain, restricted to questions whose gold tokens
    # are already in the compiled context?
    sv_oracle = dict(base)
    sv_fixable = 0
    for qid, row in run.items():
        sid = qid[3:].rsplit("_", 1)[0]
        if sid in HOLDOUT or row["category"] not in (3, 4):
            continue
        context = contexts[qid]["context"]
        tokens = [w for w in re.findall(r"[a-z0-9]{3,}", str(row["ground_truth"]).lower())]
        if tokens and all(t in context.lower() for t in tokens):
            sv_oracle[qid] = row["ground_truth"]
            sv_fixable += 1
    sv_a = score(official, dataset, base, run, categories={3, 4})
    sv_b = score(official, dataset, sv_oracle, run, categories={3, 4})
    sv_keys = sorted(set(sv_a) & set(sv_b))
    sv_n = len(sv_keys)
    sv_mean = lambda d: sum(d[q] for q in sv_keys) / max(1, sv_n) * 100  # noqa: E731

    result = {
        "dev_list_questions": n,
        "oracle_items_appended": appended,
        "sibling_completions_proposed": proposed,
        "current_f1": round(mean(a), 3),
        "oracle_f1": round(mean(b), 3),
        "oracle_delta_pt": round(mean(b) - mean(a), 3),
        "sibling_f1": round(mean(c), 3),
        "sibling_delta_pt": round(mean(c) - mean(a), 3),
        "overall_ceiling_pt": round((mean(b) - mean(a)) * n / 1487, 3),
        "single_value_questions": sv_n,
        "single_value_oracle_fixable": sv_fixable,
        "single_value_current_f1": round(sv_mean(sv_a), 3),
        "single_value_oracle_f1": round(sv_mean(sv_b), 3),
        "single_value_ceiling_pt": round(sv_mean(sv_b) - sv_mean(sv_a), 3),
        "single_value_overall_pt": round((sv_mean(sv_b) - sv_mean(sv_a)) * sv_n / 1487, 3),
    }
    for k, v in result.items():
        print(f"{k}: {v}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
