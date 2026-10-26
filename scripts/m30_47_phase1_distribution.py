#!/usr/bin/env python3
"""M30-47 Phase 1: Entity Grounding noise distribution (READ-ONLY).

Measures, for all 28 Bucket A questions:
  - presence / count / chars of "[Entity Grounding: ...]" lines
  - share of total context chars
  - position of the first such line
  - correlated with reader outcome (correct / abstain / wrong)

No deletion. No prompt change. No production code change.
Correlation only — this is NOT a causal claim.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

PROD = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
ABSTAIN = re.compile(r"(no mention|does not contain|does not provide|no information|"
                     r"not specified|no specific|cannot|does not specify)", re.I)


def to_conv(q):
    m = re.match(r"lc_conv-(\d+)_(\d+)", q)
    return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}"


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")
    tmap = {to_conv(q): q for q in targets}
    prod_map = {r["qid"]: r for r in PROD["rows"]}

    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    EG = re.compile(r"\[Entity Grounding:[^\]]*\]")
    rows = []
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            if q.question_id not in tmap:
                continue
            qid = tmap[q.question_id]
            pcc = adapter.compiler.compile(q.question, ir_records)
            ctx = pcc.context_text
            total = len(ctx)

            eg_lines = EG.findall(ctx)
            eg_chars = sum(len(x) for x in eg_lines)
            # share of lines that are EG
            lines = ctx.split("\n")
            eg_line_count = sum(1 for ln in lines if EG.search(ln))
            first_pos = ctx.find("[Entity Grounding:") if eg_lines else None

            # outcome
            r = prod_map[qid]
            if r["correct"]:
                outcome = "correct"
            elif ABSTAIN.search(r["predicted"]):
                outcome = "abstain"
            else:
                outcome = "wrong"

            rows.append({
                "qid": qid, "outcome": outcome,
                "ctx_chars": total, "eg_count": len(eg_lines),
                "eg_chars": eg_chars, "eg_char_share": round(eg_chars / max(total, 1), 4),
                "eg_line_count": eg_line_count, "total_lines": len(lines),
                "eg_line_share": round(eg_line_count / max(len(lines), 1), 4),
                "first_eg_pos": first_pos,
                "first_eg_rel": round(first_pos / max(total, 1), 4) if first_pos is not None else None,
            })

    print("=" * 78)
    print("M30-47 Phase 1: Entity Grounding noise distribution (read-only)")
    print("=" * 78)
    print()
    print(f"{'qid':16s} {'outcome':8s} {'eg_cnt':>6s} {'eg_share':>9s} {'line_share':>10s} {'first_rel':>9s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['qid']:16s} {r['outcome']:8s} {r['eg_count']:6d} "
              f"{r['eg_char_share']:9.4f} {r['eg_line_share']:10.4f} "
              f"{str(r['first_eg_rel']):>9s}")
    print()

    # Aggregates by outcome
    print("=" * 78)
    print("BY OUTCOME")
    print("=" * 78)
    for oc in ["correct", "abstain", "wrong"]:
        sub = [r for r in rows if r["outcome"] == oc]
        if not sub:
            continue
        with_eg = sum(1 for r in sub if r["eg_count"] > 0)
        avg_cnt = sum(r["eg_count"] for r in sub) / len(sub)
        avg_share = sum(r["eg_char_share"] for r in sub) / len(sub)
        print(f"  {oc:8s}: n={len(sub):2d}  with_EG={with_eg:2d}  avg_eg_count={avg_cnt:7.1f}  avg_char_share={avg_share:.4f}")
    print()

    # Distribution
    with_eg = [r for r in rows if r["eg_count"] > 0]
    print(f"Questions with ANY Entity Grounding line: {len(with_eg)}/28")
    print(f"Questions with EG count > 10: {sum(1 for r in rows if r['eg_count'] > 10)}")
    print(f"Questions with EG count > 50: {sum(1 for r in rows if r['eg_count'] > 50)}")
    print()
    top = sorted(rows, key=lambda r: -r["eg_count"])[:5]
    print("Top 5 by EG count:")
    for r in top:
        print(f"  {r['qid']:16s} count={r['eg_count']:4d} share={r['eg_char_share']:.4f} outcome={r['outcome']}")
    print()

    out = PROJECT_ROOT / "results/m30_47"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eg_distribution.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'eg_distribution.json'}")
    print()
    print("NOTE: correlation only. No causal claim. No context modified.")


if __name__ == "__main__":
    main()
