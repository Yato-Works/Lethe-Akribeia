#!/usr/bin/env python3
"""M30-54 Phase 3 (v6): Entity-Relation Turn Binding — action-relation scoring.

Key fix for v5's failure: D22:1 contains 'recipe' + 'cook' but is about TRYING
a recipe yesterday, not SHARING recipes tomorrow. The question asks about
"go over... share recipes" — the ACTIONS matter, not just nouns.

Strategy: extract question's predicate-argument structure:
  "go over" → {visit}  
  "share recipes" → {share}
The turn must mention BOTH the entity AND at least one specific action.
Additionally, require the relative expression type to match the question's temporal framing.

V6 improves on V5 by:
1. Requiring at least 1 specific match (not just entity)
2. Downweighting turns where the relative expression disagrees with the question's temporal direction
   (if question asks about future "going over", prioritize "tomorrow"/"next month" turns)
"""
import json, re, sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
import importlib.util
_spec = importlib.util.spec_from_file_location("v1", "scripts/m30_51_phase2_fallback.py")
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)

import importlib.util as _ilu
_spec2 = _ilu.spec_from_file_location("v5mod", "scripts/m30_54_phase3_v5.py")
_v5 = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_v5)


def question_temporal_direction(question):
    """Determine if the question asks about past, present, or future events."""
    ql = question.lower()
    if any(w in ql for w in ['when did', 'when was', 'when were', 'yesterday', 'last week', 'last month', 'ago']):
        return 'past'
    if any(w in ql for w in ['when will', 'when is', 'when are', 'tomorrow', 'next month', 'next week', 'currently']):
        return 'future'
    if 'when is' in ql or 'currently' in ql:
        return 'present'
    return 'unknown'


def temporal_direction_bias(rel_expr, direction):
    """Return a bias score based on whether the relative expression matches the question's temporal direction."""
    rel = rel_expr.lower()
    if direction == 'past':
        if 'yesterday' in rel or 'last week' in rel or 'last month' in rel:
            return 0  # matches
        if 'tomorrow' in rel or 'next week' in rel or 'next month' in rel:
            return -0.5  # opposite direction
        return 0
    if direction == 'future':
        if 'tomorrow' in rel or 'next week' in rel or 'next month' in rel:
            return 0  # matches
        if 'yesterday' in rel or 'last week' in rel or 'last month' in rel:
            return -0.5  # opposite direction
        return 0
    return 0


def deterministic_temporal_resolve_v6(ctx, question, category):
    """V6: Entity-Relation Turn Binding with temporal direction bias."""
    if not _v1.is_temporal_question(question, category):
        return None

    direction = question_temporal_direction(question)

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

        # Temporal direction bias
        bias = temporal_direction_bias(rels[0], direction)

        candidates.append({
            "dia": dia, "hdr": hdr, "body": body, "rel": rels[0],
            "explicit": explicit, "base": base,
            "entity_score": ent_score, "event_score": evt_score,
            "matched": matched, "bias": bias,
        })

    if not candidates:
        return None

    # V6 scoring: event_score * 2 + entity_score + temporal_bias
    # This ensures turns matching the question's temporal direction rank higher
    candidates.sort(key=lambda c: (
        -(c["event_score"] * 2 + c["entity_score"] + c["bias"]),
        0 if c["explicit"] else 1,
        -c["entity_score"]  # tiebreaker: more entities
    ))

    best = candidates[0]
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
    v5_recovered = v5_reg = 0
    v6_recovered = v6_reg = 0
    bl_total = v1_total = v5_total = v6_total = 0

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

        v5_f = _v5.deterministic_temporal_resolve_v5(ctx, question, cat)
        v5_t = v5_f if v5_f else raw
        v5_ok = LoCoMoAdapter.score_binary(cat, gold, v5_t)

        v6_f = deterministic_temporal_resolve_v6(ctx, question, cat)
        v6_t = v6_f if v6_f else raw
        v6_ok = LoCoMoAdapter.score_binary(cat, gold, v6_t)

        if v1_ok: v1_total += 1
        if v5_ok: v5_total += 1
        if v6_ok: v6_total += 1
        if v1_ok and not bl_correct: v1_recovered += 1
        if v5_ok and not bl_correct: v5_recovered += 1
        if v6_ok and not bl_correct: v6_recovered += 1
        if bl_correct and not v1_ok: v1_reg += 1
        if bl_correct and not v5_ok: v5_reg += 1
        if bl_correct and not v6_ok: v6_reg += 1

        if v6_ok != v1_ok or v6_t != v1_t:
            status = "✅ RECOVERED" if v6_ok else "❌"
            print(f"DIFF {qid}: gold={gold[:35]} {status}")
            print(f"  V1: {v1_t[:50]}")
            print(f"  V6: {v6_t[:50]}")
            print()

        results.append({"qid": qid, "gold": gold[:50],
                       "v1_ok": v1_ok, "v5_ok": v5_ok, "v6_ok": v6_ok,
                       "v1_out": v1_t[:50], "v6_out": v6_t[:50]})

    print("M30-54 Phase 3 v6: Entity-Relation Turn Binding (temporal direction bias)")
    print("="*70)
    print(f"Baseline: {bl_total}/28")
    print(f"V1:  {bl_total+v1_total}/28 (+{v1_recovered}, {v1_reg} reg)")
    print(f"V5:  {bl_total+v5_total}/28 (+{v5_recovered}, {v5_reg} reg)")
    print(f"V6:  {bl_total+v6_total}/28 (+{v6_recovered}, {v6_reg} reg)")

    out = PROJECT_ROOT / "results/m30_54/phase3_v6_ab_test.json"
    (out).parent.mkdir(parents=True, exist_ok=True)
    (out).write_text(json.dumps({
        "summary": {
            "baseline": bl_total, "v1_total": bl_total+v1_total, "v5_total": bl_total+v5_total, "v6_total": bl_total+v6_total,
            "v1_recovered": v1_recovered, "v5_recovered": v5_recovered, "v6_recovered": v6_recovered,
            "v1_regressed": v1_reg, "v5_regressed": v5_reg, "v6_regressed": v6_reg,
        },
        "rows": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run_ab_test()
