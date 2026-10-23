"""Phase 1-4: build the failure atlas from stored predictions, re-scored.

Eli's rule: stored predictions get RE-SCORED with the fixed scorer (c449aa0).
The old is_correct flags came from the scorer that credited a wrong DATE, so they
are not used as-is.

Outputs (new files only -- nothing existing is modified):
  benchmark_results/failure_atlas/failure_atlas.jsonl
  benchmark_results/failure_atlas/failure_clusters.json
  benchmark_results/failure_atlas/evaluation_audit.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from artificial_memory_gen.skills.answer_committer import commit_answer, parse_turns  # noqa: E402

RUN = ROOT / "benchmark_results/locomo_full_20261010_115839.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/failure_atlas"
DATA = ROOT / "datasets/external/locomo10.json"

_ABSTENTION_WORDS = {"i don't know", "unknown", "cannot determine", "not enough",
                     "no information", "i cannot", "unanswerable", ""}


def nq(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def content_words(s: str) -> set:
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split() if len(w) > 2}


def gold_presence(gold: str, context: str) -> tuple[str, list[str]]:
    """full / partial / absent, plus the turn ids carrying gold tokens."""
    g = content_words(gold)
    if not g:
        return "unknown", []
    turns = parse_turns(context)
    hits_full = hits_any = 0
    found_ids: list[str] = []
    for t in turns:
        tw = content_words(t.text)
        shared = g & tw
        if shared:
            hits_any += 1
            found_ids.append(t.dia_id)
            if len(shared) >= max(1, len(g) // 2):
                hits_full += 1
    if hits_full:
        return "full", found_ids
    if hits_any:
        return "partial", found_ids
    return "absent", []


def qshape(q: str) -> str:
    ql = q.lower().strip()
    if re.search(r"\bwhen\b", ql):
        return "when_date" if re.search(
            r"\b(?:did|does|is|was|has|have|go|going|get|got|start|take|happen)\b", ql
        ) else "when_bare"
    if re.search(r"\bhow (?:many|much)\b", ql):
        return "what_count"
    if re.search(r"\bwho\b", ql):
        return "what_person"
    if re.search(r"\bwhere\b", ql):
        return "where_place"
    if re.search(r"\bwhy\b", ql):
        return "why_reason"
    if re.match(r"^(?:did|does|is|was|were|has|have|can|could|would|will)\b", ql):
        return "yesno"
    if re.search(r"\bwhat\b", ql):
        return "what_thing"
    return "other"


def committer_verdict(question: str, context: str):
    """What the current deterministic engines do, and why they don't answer."""
    ans = commit_answer(question, context, 1)
    if ans.used:
        return {"engine": ans.source, "detail": ans.detail,
                "engine_answer": str(ans.answer), "claims": True}
    return {"engine": None, "detail": ans.detail, "engine_answer": None,
            "claims": False}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    run = json.load(open(RUN, encoding="utf-8"))
    preds = run["results"]
    cache = {nq(json.loads(l)["question"]): json.loads(l)
             for l in open(CACHE, encoding="utf-8") if l.strip()}

    # --- dataset gold + evidence, for cross-checking the stored gold ---------
    ds_gold, ds_ev = {}, {}
    for conv in json.load(open(DATA, encoding="utf-8")):
        for qa in conv.get("qa", []):
            if "question" not in qa:
                continue
            k = nq(qa["question"])
            ds_gold[k] = str(qa.get("answer", qa.get("adversarial_answer", "")))
            ds_ev[k] = qa.get("evidence", [])

    records, clusters = [], defaultdict(list)
    audit = {
        "predictions_loaded": len(preds),
        "unique_question_texts": 0,
        "cache_rows": len(cache),
        "joined": 0,
        "duplicate_question_texts": 0,
        "conflicting_duplicates": 0,
        "gold_mismatch_vs_dataset": 0,
        "scorer": "eval_locomo_full.score (c449aa0, date-contradiction guard)",
        "scorer_note": "stored is_correct NOT used -- it came from the pre-fix "
                       "scorer that credited a wrong DATE when month/year agreed",
    }

    seen = Counter()
    for p in preds:
        k = nq(p["question"])
        seen[k] += 1
        if seen[k] > 1:
            audit["duplicate_question_texts"] += 1
            continue  # atlas keeps one record per unique question text

    audit["unique_question_texts"] = len(seen)
    audit["conflicting_duplicates"] = 0
    for k, c in seen.items():
        if c <= 1:
            continue
        grp = [p for p in preds if nq(p["question"]) == k]
        if len({(p["ground_truth"], p["predicted_answer"]) for p in grp}) > 1:
            audit["conflicting_duplicates"] += 1

    done = set()
    for p in preds:
        k = nq(p["question"])
        if k in done:
            continue
        done.add(k)
        audit["joined"] += 1

        gold = str(p["ground_truth"])
        if k in ds_gold and ds_gold[k] != gold:
            audit["gold_mismatch_vs_dataset"] += 1

        pred = str(p["predicted_answer"])
        pred_norm = pred.lower().strip()
        is_abstain = pred_norm in _ABSTENTION_WORDS or pred_norm.startswith("<error")

        # RE-SCORE with the fixed scorer.  This is the whole point: the stored
        # is_correct predates c449aa0.
        now_correct = E.score(pred, gold)
        was_correct = str(p["is_correct"]).lower() == "true"

        crow = cache.get(k)
        context = crow["context"] if crow else ""
        presence, ev_ids = gold_presence(gold, context) if context else ("unknown", [])

        rec = {
            "question_id": p["question_id"],
            "cache_qid": crow["qid"] if crow else None,
            "category": int(p["category"]),
            "question": p["question"],
            "gold": gold,
            "shape": qshape(p["question"]),
            "stored_is_correct": was_correct,
            "rescored_correct": bool(now_correct),
            "verdict": "correct" if now_correct else ("abstain" if is_abstain else "wrong"),
            "predicted_answer": pred[:200],
            "scorer_changed_verdict": bool(now_correct != was_correct),
            "gold_in_context": presence,
            "evidence_turns": ev_ids[:8],
            "dataset_evidence_ids": ds_ev.get(k, [])[:8],
            "engine": None,
            "engine_detail": None,
            "engine_claims": None,
        }

        # Current deterministic engines, on the frozen context.
        if context:
            cv = committer_verdict(p["question"], context)
            rec["engine"] = cv["engine"]
            rec["engine_detail"] = cv["detail"]
            rec["engine_claims"] = cv["claims"]

        # Only failures go into the atlas.
        if not now_correct:
            records.append(rec)
            key = _cluster_key(rec)
            clusters[key].append(rec)

    with open(OUT / "failure_atlas.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {}
    for key, recs in clusters.items():
        summary[key] = {
            "count": len(recs),
            "example_ids": [r["question_id"] for r in recs[:5]],
            "shapes": dict(Counter(r["shape"] for r in recs)),
            "gold_in_context": dict(Counter(r["gold_in_context"] for r in recs)),
            "categories": dict(Counter(r["category"] for r in recs)),
            "engine_claims_rate": round(
                sum(1 for r in recs if r["engine_claims"]) / len(recs), 3),
        }

    cluster_doc = {
        "total_joined": audit["joined"],
        "total_failures": len(records),
        "clusters": dict(sorted(summary.items(), key=lambda x: -x[1]["count"])),
    }
    with open(OUT / "failure_clusters.json", "w", encoding="utf-8") as f:
        json.dump(cluster_doc, f, indent=2, ensure_ascii=False)
    with open(OUT / "evaluation_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    # --- console ------------------------------------------------------------
    print(f"=== joined {audit['joined']} questions (unique texts) ===")
    print(f"  correct   : {audit['joined'] - len(records)}")
    print(f"  failures  : {len(records)}")
    print()
    verdicts = Counter(r["verdict"] for r in records)
    print(f"  abstain {verdicts['abstain']}  wrong {verdicts['wrong']}")
    print()
    flipped = sum(1 for r in records if r["scorer_changed_verdict"])
    print(f"scorer fix changed the verdict on {flipped} of the failures")
    print()
    print("=== clusters ===")
    for k, v in sorted(summary.items(), key=lambda x: -x[1]["count"])[:14]:
        print(f"  {v['count']:5d}  {k}")
        print(f"          gold_in_context={v['gold_in_context']}  "
              f"engine_claims={v['engine_claims_rate']}")
    print()
    print(f"written to {OUT}")


def _cluster_key(r: dict) -> str:
    """Failure pattern, decided from measured fields only."""
    if r["verdict"] == "abstain":
        return ("F1_abstain_gold_" + r["gold_in_context"]) if r["gold_in_context"] != "absent" \
            else "F1_abstain_gold_absent"
    if r["engine_claims"]:
        return "F4_engine_claims_but_wrong"
    if r["gold_in_context"] == "absent":
        return "F5_gold_not_in_context"
    if r["gold_in_context"] == "partial":
        return "F3_gold_partially_in_context"
    return "F2_evidence_present_but_unanswered"


if __name__ == "__main__":
    main()
