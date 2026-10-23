#!/usr/bin/env python3
"""M30-41 Phase 8b: reconstruct_world A/B — the +25/+20 actor boost.

reconstruct_world() calls compute_score() AND adds its own actor term
(+25 speaker match, +15 first-person, +20 mention, -10 other). That is a
SECOND hardcoded-name boost, larger than the scorer's.

Method: runtime monkeypatch _extract_target_actor to return None (disable)
and compare the resulting top-k ranking on Bucket A 28.
No LLM. No production edit. No holdout.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory_gen.recall.state_reconstructor import StateReconstructor

LOCOMO = PROJECT_ROOT / "datasets/external/locomo10.json"


def load_gold():
    locomo = json.loads(LOCOMO.read_text(encoding="utf-8"))
    by_q = {}
    for conv in locomo:
        sid = conv.get("sample_id", "")
        for qa in conv.get("qa", []):
            by_q[(sid, qa.get("question", "").strip().lower())] = qa.get("evidence", []) or []
    return by_q


def make_records(conv_turns):
    recs = []
    for t in conv_turns:
        recs.append(StructuredIR(
            entity=t.get("speaker", ""), property="statement", value=t.get("text", ""),
            source=t.get("speaker", ""),
            raw_content=f"[{t.get('dia_id','')}] {t.get('speaker','')}: {t.get('text','')}",
        ))
    return recs


def main():
    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    cases = {c["question_id"]: c for c in bucket["cases"]}
    convs = bucket["conversations"]
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")
    gold_by_q = load_gold()

    recon = StateReconstructor()
    ORIG = StateReconstructor._extract_target_actor

    print("=" * 78)
    print("M30-41 Phase 8b: reconstruct_world actor-boost A/B")
    print("=" * 78)
    print(f"TARGET: {len(targets)}")
    print()

    def run(disable):
        if disable:
            StateReconstructor._extract_target_actor = lambda self, q: None
        else:
            StateReconstructor._extract_target_actor = ORIG

        rows = []
        for qid in targets:
            case = cases.get(qid)
            if not case:
                continue
            conv_key = case.get("source_conversation", "")
            turns = convs.get(conv_key, [])
            if not isinstance(turns, list):
                continue
            gold = set(g for g in gold_by_q.get((conv_key, case.get("question","").strip().lower()), []) if isinstance(g, str))
            recs = make_records(turns)
            try:
                intent, units = recon.reconstruct_world(case.get("question", ""), recs)
            except Exception as e:
                rows.append({"qid": qid, "rank": None, "err": str(e)[:60]})
                continue
            # units are ordered by score already? If not, we can't rank.
            # Find the position of the gold turn in the returned order.
            rank = None
            for i, u in enumerate(units, 1):
                # match by dia_id embedded in raw_content
                m = re.search(r"\[(D\d+:\d+)\]", u.ir.raw_content)
                if m and m.group(1) in gold:
                    rank = i
                    break
            rows.append({"qid": qid, "rank": rank, "n_units": len(units)})
        return rows

    base = run(False)
    off = run(True)

    def summ(rows, label):
        ok = [r for r in rows if "err" not in r]
        r1 = sum(1 for r in ok if r["rank"] == 1)
        r5 = sum(1 for r in ok if r["rank"] and r["rank"] <= 5)
        found = sum(1 for r in ok if r["rank"] is not None)
        print(f"{label}: n={len(rows)} err={len(rows)-len(ok)}")
        print(f"  gold@1={r1}  gold@5={r5}  found={found}")
        return {"n": len(rows), "top1": r1, "top5": r5, "found": found}

    b = summ(base, "Baseline (actor ON) ")
    o = summ(off,  "Treatment (actor OFF)")
    print()
    print(f"DELTA top-1: {o['top1']-b['top1']:+d}   top-5: {o['top5']-b['top5']:+d}")
    print()

    diffs = [(x["qid"], x["rank"], y["rank"]) for x, y in zip(base, off) if x.get("rank") != y.get("rank")]
    print(f"Rank changes: {len(diffs)}")
    for qid, br, orr in diffs[:15]:
        print(f"  {qid}: ON={br} -> OFF={orr}")
    print()

    out = PROJECT_ROOT / "results/m30_41"
    out.mkdir(parents=True, exist_ok=True)
    (out / "reconstruct_world_ab.json").write_text(json.dumps({
        "baseline_actor_on": b, "treatment_actor_off": o,
        "base_rows": base, "off_rows": off,
        "note": "runtime monkeypatch; no LLM; no production edit",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'reconstruct_world_ab.json'}")


if __name__ == "__main__":
    main()
