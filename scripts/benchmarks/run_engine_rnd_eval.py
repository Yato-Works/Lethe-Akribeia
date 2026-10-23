"""Paired, LLM-free evaluation of deterministic committer changes on frozen LoCoMo rows.

Example:
    python scripts/benchmarks/run_engine_rnd_eval.py --ids-file ... --out ...

The selected questions and gold answers come from the audited failure atlas;
contexts come from the immutable context cache.  This is a targeted engine
evaluation, not a reader/full-benchmark run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as scorer  # noqa: E402
from artificial_memory.skills.answer_committer import commit_answer  # noqa: E402
from artificial_memory.skills.answer_committer import CommittedAnswer  # noqa: E402
from artificial_memory.skills.autonomous_engines.event_action_resolver import EventActionResolver  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids-file", help="UTF-8 text file, one atlas question_id per line")
    ap.add_argument("--cohort", choices=("plural-list", "plural-list-gold", "plural-list-failures"),
                    help="Select a deterministic atlas cohort; IDs are saved in the result")
    ap.add_argument("--conversation-id", help="Restrict a cohort to one complete conversation")
    ap.add_argument("--paired-before", help="Compare selected IDs with a prior result JSON")
    ap.add_argument("--out", required=True, help="Write a new JSON result file")
    ap.add_argument("--label", default="")
    ap.add_argument("--run", default="benchmark_results/locomo_full_20261010_115839.json",
                    help="Ground-truth question table; stored verdicts are ignored")
    ap.add_argument("--disable-inspiration-resolver", action="store_true",
                    help="Paired baseline: do not invoke the new inspiration-query branch")
    ap.add_argument("--disable-implicit-cause", action="store_true",
                    help="Paired baseline: do not invoke the implicit why-purpose branch")
    args = ap.parse_args()

    if bool(args.ids_file) == bool(args.cohort):
        raise SystemExit("provide exactly one of --ids-file or --cohort")
    atlas_rows = read_jsonl(ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl")
    atlas = {row["question_id"]: row for row in atlas_rows}
    if args.ids_file:
        ids = [line.strip() for line in (ROOT / args.ids_file).read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    elif args.cohort == "plural-list-failures":
        # Atlas rows are failures after rescoring with the fixed evaluator.
        # This includes both abstentions and incorrect deterministic claims.
        ids = [row["question_id"] for row in atlas_rows
               if row["gold_in_context"] == "full"
               and not row["rescored_correct"]
               and row["shape"] in ("what_thing", "other")
               and re.match(r"\s*(?:what|which)\b", row["question"], re.I)
               and int(row["category"]) != 2
               and not re.search(r"\b(?:month|year|date|day|week|time|duration)\b", row["question"], re.I)
               and scorer.is_list_gold(str(row["gold"]))]
    else:
        # A conservative, reproducible discovery cohort: unresolved answers
        # whose complete multi-item gold is already in the frozen context.
        ids = [row["question_id"] for row in atlas_rows
               if row["gold_in_context"] == "full" and not row["engine_claims"]
               and row["shape"] in ("what_thing", "other")
               and re.match(r"\s*(?:what|which)\b", row["question"], re.I)
               and (("," in str(row["gold"]) or ";" in str(row["gold"])
                     or re.search(r"\band\b", str(row["gold"]), re.I))
                    if args.cohort == "plural-list"
                    else scorer.is_list_gold(str(row["gold"]))) ]
    if args.conversation_id:
        ids = [qid for qid in ids
               if (re.search(r"conv-(\d+)", qid) and
                   re.search(r"conv-(\d+)", qid).group(1) == args.conversation_id)]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate question IDs in ids-file")
    cache = {row["qid"]: row for row in read_jsonl(ROOT / "benchmark_results/locomo_context_cache_rules.jsonl")}
    run_rows = {row["question_id"]: row for row in json.loads((ROOT / args.run).read_text(encoding="utf-8"))["results"]}
    missing = [qid for qid in ids if qid not in run_rows]
    if missing:
        raise SystemExit(f"missing ground-truth rows: {missing}")

    records = []
    original_cause_resolver = EventActionResolver._resolve_explicit_cause
    if args.disable_inspiration_resolver:
        EventActionResolver._resolve_explicit_cause = classmethod(
            lambda cls, question, turns, person: (
                CommittedAnswer(used=False)
                if re.search(r"\b(?:inspir\w*|motivat\w*)\b", question, re.I)
                else original_cause_resolver(question, turns, person)
            )
        )
    if args.disable_implicit_cause:
        EventActionResolver._resolve_implicit_cause = classmethod(
            lambda cls, question, turns, person: CommittedAnswer(used=False)
        )
    started = time.perf_counter()
    for qid in ids:
        run_row = run_rows[qid]
        cache_qid = (atlas[qid]["cache_qid"] if qid in atlas else None)
        if cache_qid is None:
            match = re.fullmatch(r"lc_conv-(\d+)_(\d+)", qid)
            if match:
                cache_qid = f"conv-{match.group(1)}-qa-{int(match.group(2)):03d}"
        row = cache.get(cache_qid or "")
        if row is None:
            raise SystemExit(f"missing frozen context row for {qid} (derived key {cache_qid!r})")
        prediction = commit_answer(row["question"], row["context"], category=int(row["category"]))
        status = "abstain" if not prediction.used else (
            "correct" if scorer.score(str(prediction.answer), str(run_row["ground_truth"])) else "wrong"
        )
        records.append({
            "question_id": qid,
            "conversation": (re.search(r"conv-(\d+)", qid).group(1)
                             if re.search(r"conv-(\d+)", qid) else "unknown"),
            "category": int(row["category"]),
            "question": row["question"],
            "gold": run_row["ground_truth"],
            "status": status,
            "answer": str(prediction.answer) if prediction.used else None,
            "engine": prediction.source if prediction.used else None,
            "detail": prediction.detail,
            "evidence_turn": prediction.evidence_turn if prediction.used else "",
        })

    elapsed = time.perf_counter() - started
    counts = Counter(r["status"] for r in records)
    payload = {
        "label": args.label,
        "cohort": args.cohort,
        "question_ids": ids,
        "n": len(records),
        "correct": counts["correct"],
        "wrong": counts["wrong"],
        "abstain": counts["abstain"],
        "presented": counts["correct"] + counts["wrong"],
        "llm_calls": 0,
        "engine_evaluation_seconds": round(elapsed, 6),
        "scorer": "scripts/benchmarks/eval_locomo_full.py::score (unchanged, fixed scorer)",
        "context_source": "benchmark_results/locomo_context_cache_rules.jsonl",
        "gold_source": "benchmark_results/failure_atlas/failure_atlas.jsonl",
        "records": records,
        "invariants": {
            "n_eq_correct_plus_wrong_plus_abstain": len(records) == counts["correct"] + counts["wrong"] + counts["abstain"],
            "presented_eq_correct_plus_wrong": counts["correct"] + counts["wrong"] == sum(r["status"] != "abstain" for r in records),
        },
    }
    if args.cohort == "plural-list-failures":
        atlas_by_id = atlas
        baseline_status = {
            qid: ("wrong" if atlas_by_id[qid]["engine_claims"] else "abstain")
            for qid in ids
        }
        after_status = {r["question_id"]: r["status"] for r in records}
        counts_before = Counter(baseline_status.values())
        counts_after = Counter(after_status.values())
        payload["atlas_paired_comparison"] = {
            "baseline_source": "failure_atlas current-engine claims, fixed-scorer failures, same frozen cache IDs",
            "before": {k: counts_before[k] for k in ("correct", "wrong", "abstain")},
            "after": {k: counts_after[k] for k in ("correct", "wrong", "abstain")},
            "fixed_qids": [qid for qid in ids if baseline_status[qid] != "correct" and after_status[qid] == "correct"],
            "new_wrong_qids": [qid for qid in ids if baseline_status[qid] == "abstain" and after_status[qid] == "wrong"],
            "correctly_abstained_after_wrong_before_qids": [qid for qid in ids if baseline_status[qid] == "wrong" and after_status[qid] == "abstain"],
            "same_question_ids": True,
        }
    if args.paired_before:
        before_payload = json.loads((ROOT / args.paired_before).read_text(encoding="utf-8"))
        before_records = {r["question_id"]: r for r in before_payload["records"]}
        if any(qid not in before_records for qid in ids):
            raise SystemExit("paired-before artifact does not contain every selected question ID")
        fixed = [qid for qid in ids
                 if before_records[qid]["status"] != "correct"
                 and next(r for r in records if r["question_id"] == qid)["status"] == "correct"]
        broke = [qid for qid in ids
                 if before_records[qid]["status"] == "correct"
                 and next(r for r in records if r["question_id"] == qid)["status"] != "correct"]
        new_wrong = [qid for qid in ids
                     if before_records[qid]["status"] != "wrong"
                     and next(r for r in records if r["question_id"] == qid)["status"] == "wrong"]
        payload["paired_comparison"] = {
            "before_artifact": args.paired_before,
            "same_question_ids": True,
            "fixed_qids": fixed,
            "broke_qids": broke,
            "new_wrong_qids": new_wrong,
            "fixed": len(fixed),
            "broke": len(broke),
            "new_wrong": len(new_wrong),
        }
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing evaluation output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"n={len(records)} correct={counts['correct']} wrong={counts['wrong']} "
          f"abstain={counts['abstain']} presented={counts['correct'] + counts['wrong']}; saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
