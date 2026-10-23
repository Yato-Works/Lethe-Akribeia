#!/usr/bin/env python3
"""M30-44 Phase 2: For each production failure, did the gold evidence reach
the compiled context?

Runs the PRODUCTION path (LoCoMoAdapter) and, for each failing question,
checks whether the gold evidence turn's text appears in the compiled context.

  EVIDENCE_MISSING  -> retrieval/extraction dropped it (search failure)
  EVIDENCE_PRESENT  -> reader saw it and still failed (reader failure)

This is the decision that tells us WHERE to fix.

No production edit. No holdout.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

FAILS = json.loads((PROJECT_ROOT / "results/m30_44/production_failures.json").read_text(encoding="utf-8"))


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower())


def main():
    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucket["cases"]}

    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    gold_ev = {}
    for conv in locomo:
        sid = conv.get("sample_id", "")
        for qa in conv.get("qa", []):
            gold_ev[(sid, qa.get("question", "").strip().lower())] = qa.get("evidence", []) or []

    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    fail_qids = [r["qid"] for r in FAILS["rows"]]
    print("=" * 78)
    print("M30-44 Phase 2: Did gold evidence reach the compiled context?")
    print("=" * 78)
    print(f"Failing questions: {len(fail_qids)}")
    print()

    sample_ids = [c.get("sample_id", "") for c in locomo]
    results = []
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        turn_by_id = {t.dia_id: t for t in turns}
        for q in questions:
            # map conv qid back to lc qid
            m = re.match(r"conv-(\d+)-qa-(\d+)", q.question_id)
            if not m:
                continue
            lc_qid = f"lc_conv-{m.group(1)}_{int(m.group(2))}"
            if lc_qid not in fail_qids:
                continue

            # compile via the production path
            try:
                pcc = adapter.compiler.compile(q.question, ir_records)
                ctx = pcc.context_text
            except Exception as e:
                ctx = f"ERROR: {e}"

            gold_ids = gold_ev.get((sid, q.question.strip().lower()), [])
            pool = norm(ctx)
            # check each gold turn's text overlap with the context
            present = False
            details = []
            for gid in gold_ids:
                t = turn_by_id.get(gid)
                if not t:
                    details.append(f"{gid}:notfound")
                    continue
                words = [w for w in norm(t.text).split() if len(w) > 3]
                if not words:
                    details.append(f"{gid}:empty")
                    continue
                ov = sum(1 for w in words if w in pool) / len(words)
                details.append(f"{gid}:{ov:.0%}")
                if ov >= 0.8:
                    present = True

            results.append({
                "qid": lc_qid, "gold_evidence": gold_ids,
                "evidence_present": present, "overlap_details": details,
                "ctx_len": len(ctx),
            })
            print(f"  {lc_qid:16s} gold_ev={gold_ids} present={present}  [{', '.join(details)}]")

    print()
    n_present = sum(1 for r in results if r["evidence_present"])
    n_missing = len(results) - n_present
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  EVIDENCE_PRESENT (reader failure): {n_present}/{len(results)}")
    print(f"  EVIDENCE_MISSING (retrieval failure): {n_missing}/{len(results)}")
    print()

    out = PROJECT_ROOT / "results/m30_44"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence_reach.json").write_text(json.dumps({
        "present": n_present, "missing": n_missing, "rows": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'evidence_reach.json'}")


if __name__ == "__main__":
    main()
