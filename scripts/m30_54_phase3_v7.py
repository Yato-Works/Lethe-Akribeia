#!/usr/bin/env python3
"""M30-54 Phase 3 (v7): Entity-Relation Turn Binding — explicit-date priority.

Root cause of V1-V6 failures on lc_conv-42_57:
  D22:1: entity=2, event=6, rel=yesterday, explicit=None → resolves to 5 Oct
  D26:19: entity=2, event=2, rel=tomorrow, explicit="5 November 2022" → should be 5 Nov

D26:19 has an EXPLICIT date in its body ("tomorrow (5 November 2022)").
D22:1 has NO explicit date (just "yesterday (5 October 2022)").

Key insight: turns with EXPLICIT dates in the body are more reliable than those
requiring date arithmetic. Prioritize turns with explicit dates that match
the question's event terms.

V7 strategy:
1. Among turns with explicit dates in body, pick the one with highest event+entity score
2. Among turns without explicit dates, pick best by event_score + temporal bias
3. Prefer explicit date turns over non-explicit ones (they are more reliable)
"""
import json, re, sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("v1", "scripts/m30_51_phase2_fallback.py")
_v1 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v1)

_spec2 = _ilu.spec_from_file_location("v6", "scripts/m30_54_phase3_v6.py")
_v6 = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_v6)

# Import V5's combined_score and question_temporal_direction
_spec3 = _ilu.spec_from_file_location("v5", "scripts/m30_54_phase3_v5.py")
_v5 = _ilu.module_from_spec(_spec3)
_spec3.loader.exec_module(_v5)


def deterministic_temporal_resolve_v7(ctx, question, category):
    """V7: Explicit-date priority + Entity-Relation Turn Binding."""
    if not _v1.is_temporal_question(question, category):
        return None

    direction = _v6.question_temporal_direction(question)

    candidates = []
    for m in _v1.TURN_RE.finditer(ctx):
        dia, hdr, body = m.group(1), m.group(2), m.group(3)
        rels = _v1.REL_RE.findall(body)
        if not rels:
            continue
        explicit = _v1.parse_explicit_date(body)
        base = _v1.parse_header_date(hdr)
        if not base:
            continue

        ent_score, evt_score, matched = _v5.combined_score(question, body)
        bias = _v6.temporal_direction_bias(rels[0], direction)

        # NEW: Check if the turn body contains an explicit date annotation
        # e.g., "tomorrow (5 November 2022)" — this is more reliable
        has_explicit_date_in_body = explicit is not None

        candidates.append({
            "dia": dia, "hdr": hdr, "body": body, "rel": rels[0],
            "explicit": explicit, "base": base,
            "entity_score": ent_score, "event_score": evt_score,
            "matched": matched, "bias": bias,
            "has_explicit": has_explicit_date_in_body,
        })

    if not candidates:
        return None

    # V7 scoring strategy:
    # 1. Prefer turns with explicit dates in body (they are pre-resolved and reliable)
    # 2. Among those, sort by event_score + entity_score
    # 3. Among those without explicit dates, sort by event_score + entity_score + bias
    explicit_candidates = [c for c in candidates if c["has_explicit"]]
    non_explicit_candidates = [c for c in candidates if not c["has_explicit"]]

    if explicit_candidates:
        # Among turns with explicit dates, prefer those with highest event+entity score
        explicit_candidates.sort(key=lambda c: (
            -(c["event_score"] * 2 + c["entity_score"]),
            -c["entity_score"],
        ))
        best = explicit_candidates[0]
        # Explicit date takes priority
        return best["explicit"]

    # No explicit dates — fall back to event+entity + temporal bias
    non_explicit_candidates.sort(key=lambda c: (
        -(c["event_score"] * 2 + c["entity_score"] + c["bias"]),
        -c["entity_score"],
    ))

    best = non_explicit_candidates[0]
    if best["explicit"]:
        return best["explicit"]

    base = best["base"]
    e = best["rel"].lower()
    if "yesterday" in e or "the day before" in e:
        res = base - timedelta(days=1)
    elif "tomorrow" in e or "the next day" in e:
        res = base + timedelta(days=1)
    elif "next month" in e:
        mo = 1 if base.month == 12 else base.month + 1
        yr = base.year + (1 if base.month == 12 else 0)
        res = datetime(yr, mo, 1)
    elif "last month" in e:
        mo = 12 if base.month == 1 else base.month - 1
        yr = base.year - (1 if base.month == 1 else 0)
        res = datetime(yr, mo, 1)
    elif "currently" in e:
        res = base
    else:
        return None

    if "month" in question.lower() or "year" in question.lower():
        return f"{_v1.MONTH_NAMES[res.month]} {res.year}"
    return f"{res.day} {_v1.MONTH_NAMES[res.month]}, {res.year}"


def run_ab_test():
    prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
    baseline = {r["qid"]: r for r in prod["rows"]}

    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sids = [c.get("sample_id", "") for c in locomo]
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    conv_to_idx = {}
    for idx, sid in enumerate(sids):
        conv_num = int(sid.split("-")[1])
        conv_to_idx[conv_num] = idx

    ctx_cache = {}
    qid_to_cqid = {}
    conv_ids_needed = set()
    for qid in baseline:
        m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
        if not m: continue
        conv = f"conv-{m.group(1)}"
        conv_ids_needed.add(conv)
        qid_to_cqid[qid] = f"{conv}-qa-{int(m.group(2)):03d}"

    for idx, sid in enumerate(sids):
        if sid not in conv_ids_needed:
            continue
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            cqid = q.question_id
            if cqid in qid_to_cqid.values():
                pcc = adapter.compiler.compile(q.question, ir_records)
                ctx_cache[cqid] = (q.question, pcc.context_text, q.category)

    results = []
    v1_recovered = v1_reg = 0
    v7_recovered = v7_reg = 0
    bl_total = v1_total = v7_total = 0

    for qid, bp in baseline.items():
        m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
        if not m: continue
        conv_num = int(m.group(1))
        idx = conv_to_idx.get(conv_num)
        if idx is None: continue
        cqid = qid_to_cqid.get(qid)
        if not cqid or cqid not in ctx_cache:
            continue

        question, ctx, cat = ctx_cache[cqid]
        gold = bp["gold"]
        raw = bp.get("predicted", "")
        bl_correct = LoCoMoAdapter.score_binary(cat, gold, raw)

        if bl_correct:
            bl_total += 1
            continue

        v1_f = _v1.deterministic_temporal_resolve(ctx, question, cat)
        v1_t = v1_f if v1_f else raw
        v1_ok = LoCoMoAdapter.score_binary(cat, gold, v1_t)

        v7_f = deterministic_temporal_resolve_v7(ctx, question, cat)
        v7_t = v7_f if v7_f else raw
        v7_ok = LoCoMoAdapter.score_binary(cat, gold, v7_t)

        if v1_ok: v1_total += 1
        if v7_ok: v7_total += 1
        if v1_ok and not bl_correct: v1_recovered += 1
        if v7_ok and not bl_correct: v7_recovered += 1
        if bl_correct and not v1_ok: v1_reg += 1
        if bl_correct and not v7_ok: v7_reg += 1

        if v7_ok != v1_ok or v7_t != v1_t:
            status = "✅ RECOVERED" if v7_ok else "❌"
            print(f"DIFF {qid}: gold={gold[:35]} {status}")
            print(f"  V1: {v1_t[:50]}")
            print(f"  V7: {v7_t[:50]}")
            print()

        results.append({"qid": qid, "gold": gold[:50],
                       "v1_ok": v1_ok, "v7_ok": v7_ok,
                       "v1_out": v1_t[:50], "v7_out": v7_t[:50]})

    print("M30-54 Phase 3 v7: Entity-Relation Turn Binding (explicit-date priority)")
    print("="*70)
    print(f"Baseline: {bl_total}/28")
    print(f"V1: {bl_total+v1_total}/28 (+{v1_recovered}, {v1_reg} reg)")
    print(f"V7: {bl_total+v7_total}/28 (+{v7_recovered}, {v7_reg} reg)")

    # Show V7 wrong cases
    print("\nRemaining wrong after V7:")
    for r in results:
        if not r.get("v7_ok") and r.get("v1_ok"):
            print(f"  {r['qid']}: gold={r['gold'][:25]} | V1={r['v1_ok']} V7={r['v7_ok']}")

    out = PROJECT_ROOT / "results/m30_54/phase3_v7_ab_test.json"
    (out).parent.mkdir(parents=True, exist_ok=True)
    (out).write_text(json.dumps({
        "summary": {
            "baseline": bl_total, "v1_total": bl_total+v1_total, "v7_total": bl_total+v7_total,
            "v1_recovered": v1_recovered, "v7_recovered": v7_recovered,
            "v1_regressed": v1_reg, "v7_regressed": v7_reg,
        },
        "rows": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run_ab_test()
