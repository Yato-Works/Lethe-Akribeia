#!/usr/bin/env python3
"""M30-46 Phase 1+2: Answer-route audit + evidence position in production context.

Phase 1: per-QID answer route (commit_answer vs fresh LLM), cache state.
Phase 2: where does the gold evidence sit inside the compiled context?
         (char position, relative position, context length)

Read-only. Uses the real LoCoMoAdapter path. No reader re-runs.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory_gen.skills.answer_committer import commit_answer

FAILS = json.loads((PROJECT_ROOT / "results/m30_44/production_failures.json").read_text(encoding="utf-8"))
EVID = json.loads((PROJECT_ROOT / "results/m30_44/evidence_reach.json").read_text(encoding="utf-8"))


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower())


def to_lc(qid):
    m = re.match(r"conv-(\d+)-qa-(\d+)", qid)
    return f"lc_conv-{m.group(1)}_{int(m.group(2))}" if m else qid


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    gold_ev = {}
    for conv in locomo:
        sid = conv.get("sample_id", "")
        for qa in conv.get("qa", []):
            gold_ev[(sid, qa.get("question", "").strip().lower())] = qa.get("evidence", []) or []

    fail_set = {r["qid"] for r in FAILS["rows"]}
    ev_map = {r["qid"]: r for r in EVID["rows"]}

    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    print("=" * 78)
    print("M30-46 Phase 1+2: Answer route + evidence position")
    print("=" * 78)
    print()

    rows = []
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        turn_by_id = {t.dia_id: t for t in turns}
        for q in questions:
            lc = to_lc(q.question_id)
            if lc not in fail_set:
                continue
            pcc = adapter.compiler.compile(q.question, ir_records)
            ctx = pcc.context_text

            # route
            committed = commit_answer(q.question, ctx, category=q.category)
            route = "DETERMINISTIC_ANSWER" if committed.used else "FRESH_LLM_GENERATION"

            # evidence position
            gold_ids = gold_ev.get((sid, q.question.strip().lower()), [])
            pos_info = []
            ctx_norm = norm(ctx)
            for gid in gold_ids:
                t = turn_by_id.get(gid)
                if not t:
                    continue
                # locate the header "[D.. on ...]" in the raw context
                m = re.search(rf"\[{re.escape(gid)}\s+on\s+[^\]]*\]", ctx)
                if m:
                    pos = m.start()
                    pos_info.append({"gid": gid, "char_pos": pos,
                                     "rel": round(pos / max(len(ctx), 1), 3)})
                else:
                    pos_info.append({"gid": gid, "char_pos": None, "rel": None})

            rows.append({
                "qid": lc, "route": route, "ctx_len": len(ctx),
                "evidence_pos": pos_info,
                "ev_present": ev_map.get(lc, {}).get("evidence_present"),
            })
            p = pos_info[0] if pos_info else {}
            print(f"  {lc:16s} route={route:22s} ctx={len(ctx):6d} "
                  f"ev_pos={p.get('char_pos')} rel={p.get('rel')} present={ev_map.get(lc,{}).get('evidence_present')}")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    from collections import Counter
    routes = Counter(r["route"] for r in rows)
    for k, v in routes.items():
        print(f"  {k}: {v}")
    print()
    rels = [p["rel"] for r in rows for p in r["evidence_pos"] if p.get("rel") is not None]
    if rels:
        print(f"  evidence relative position: min={min(rels):.3f} max={max(rels):.3f} "
              f"mean={sum(rels)/len(rels):.3f}")
        late = sum(1 for x in rels if x > 0.5)
        print(f"  evidence in the LATER half of the context: {late}/{len(rels)}")
    print()

    out = PROJECT_ROOT / "results/m30_46"
    out.mkdir(parents=True, exist_ok=True)
    (out / "route_and_position.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'route_and_position.json'}")


if __name__ == "__main__":
    main()
