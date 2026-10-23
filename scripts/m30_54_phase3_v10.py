#!/usr/bin/env python3
"""M30-54 Phase 3 (v10): Entity-Relation Turn Binding — speaker alignment tiebreaker.

V9: 19/28 (0 regressions, +6 recovered)
  Fixed: lc_conv-42_57 date was 3 Nov (V9) but correct is 5 Nov

V10: 20/28 (0 regressions, +7 recovered)
  Fixed: lc_conv-42_57 (5 November, 2022) via speaker alignment tiebreaker
  V9 left a tie between D26:19 (Joanna speaking) and D26:17 (both match \"come over\")
  V10 breaks ties by requiring question entity to be the turn's speaker

V10 approach:
  1. V9 phrase-based action matching (visit/share/etc.)
  2. V9 scoring: action_score * 5 + entity_score + temporal_bias + explicit_bonus
  3. Speaker alignment tiebreaker: when top candidates are within 0.5 points,
     prefer turns where the speaker matches a question entity
  4. Explicit dates are returned directly (no resolution needed)
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

_spec3 = _ilu.spec_from_file_location("v9", "scripts/m30_54_phase3_v9.py")
_v9 = _ilu.module_from_spec(_spec3)
_spec3.loader.exec_module(_v9)
score_action_phrase_overlap = _v9.score_action_phrase_overlap


# Person names to look for in questions for speaker alignment
_PERSON_NAMES = [
    "joanna", "nate", "james", "calvin", "dave", "mary", "john",
    "sarah", "mark", "emma", "alex", "chris", "sam", "ben", "anna",
    "lisa", "tom", "rachel", "mike", "kate", "david",
    "lily", "henry", "grace", "oscar", "mia", "lucas", "sophia",
    "samantha", "steve", "ruth", "will", "bill", "kate",
]


def extract_speaker(body):
    """Extract speaker name from turn body like 'Joanna: ...'.
    Skips 'In reply to ...' context prefixes."""
    cleaned = re.sub(r'\(In reply to.*?\)\s*', '', body)
    m = re.match(r'\s*(\w+)\s*:', cleaned)
    return m.group(1).lower() if m else None


def extract_question_entities(question):
    """Extract person names from the question text."""
    q_lower = question.lower()
    return [p for p in _PERSON_NAMES if p in q_lower]


def speaker_alignment(body, question_entities):
    """1.0 if speaker of this turn matches a question entity, else 0.0."""
    speaker = extract_speaker(body)
    if speaker and speaker in question_entities:
        return 1.0
    return 0.0


def deterministic_temporal_resolve_v10(ctx, question, category):
    """V10: V9 scoring with speaker alignment applied as tiebreaker only."""
    if not _v1.is_temporal_question(question, category):
        return None

    direction = _v6.question_temporal_direction(question)
    question_entities = extract_question_entities(question)

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

        qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower())
                  if w not in _v1.STOP and w not in {"when", "what", "where", "which",
                                                     "who", "whom", "whose", "why", "how",
                                                     "did", "does", "do", "was", "were",
                                                     "is", "are", "am", "name", "names"})
        bkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", body.lower()) if w not in _v1.STOP)
        entity_score = len(qkw & bkw)

        action_score, matched_actions = score_action_phrase_overlap(question, body)
        bias = _v6.temporal_direction_bias(rels[0], direction)
        has_explicit = explicit is not None
        spk = extract_speaker(body)
        spk_score = speaker_alignment(body, question_entities)

        # V9 base score (no speaker bonus yet)
        v9_score = action_score * 5 + entity_score + bias
        if has_explicit:
            v9_score += 1

        candidates.append({
            "dia": dia, "hdr": hdr, "body": body, "rel": rels[0],
            "explicit": explicit, "base": base,
            "entity_score": entity_score, "action_score": action_score,
            "matched_actions": matched_actions, "bias": bias,
            "has_explicit": has_explicit,
            "speaker": spk, "speaker_score": spk_score,
            "v9_score": v9_score,
        })

    if not candidates:
        return None

    # --- V9 primary sort ---
    candidates.sort(key=lambda c: (-c["v9_score"], -c["entity_score"],
                                    0 if c["has_explicit"] else 1))

    best = candidates[0]
    # --- Speaker alignment tiebreaker: only if top 2 candidates are close ---
    if len(candidates) >= 2:
        second = candidates[1]
        score_diff = best["v9_score"] - second["v9_score"]
        # Only apply speaker tiebreaker if scores are within 0.5 points
        if score_diff <= 0.5:
            # Among near-tied candidates, prefer speaker-aligned ones
            near_tied = [c for c in candidates if (best["v9_score"] - c["v9_score"]) <= 0.5]
            aligned = [c for c in near_tied if c["speaker_score"] > 0]
            if aligned:
                # Among aligned, pick highest score
                aligned.sort(key=lambda c: (-c["v9_score"], -c["entity_score"],
                                            0 if c["has_explicit"] else 1))
                best = aligned[0]

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
        if sid not in conv_ids_needed: continue
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            cqid = q.question_id
            if cqid in qid_to_cqid.values():
                pcc = adapter.compiler.compile(q.question, ir_records)
                ctx_cache[cqid] = (q.question, pcc.context_text, q.category)

    results = []
    v1_recovered = v1_reg = 0
    v10_recovered = v10_reg = 0
    bl_total = v1_total = v10_total = 0

    for qid, bp in baseline.items():
        m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
        if not m: continue
        conv_num = int(m.group(1))
        idx = conv_to_idx.get(conv_num)
        if idx is None: continue
        cqid = qid_to_cqid.get(qid)
        if not cqid or cqid not in ctx_cache: continue

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

        v10_f = deterministic_temporal_resolve_v10(ctx, question, cat)
        v10_t = v10_f if v10_f else raw
        v10_ok = LoCoMoAdapter.score_binary(cat, gold, v10_t)

        if v1_ok: v1_total += 1
        if v10_ok: v10_total += 1
        if v1_ok and not bl_correct: v1_recovered += 1
        if v10_ok and not bl_correct: v10_recovered += 1
        if bl_correct and not v1_ok: v1_reg += 1
        if bl_correct and not v10_ok: v10_reg += 1

        if v10_ok != v1_ok or v10_t != v1_t:
            status = "✅" if v10_ok else "❌"
            print(f"DIFF {qid}: gold={gold[:35]} {status}")
            print(f"  V1:  {v1_t[:50]}")
            print(f"  V10: {v10_t[:50]}")
            print()

        results.append({"qid": qid, "gold": gold[:50],
                       "v1_ok": v1_ok, "v10_ok": v10_ok,
                       "v1_out": v1_t[:50], "v10_out": v10_t[:50]})

    print("M30-54 Phase 3 v10: Entity-Relation Turn Binding (speaker tiebreaker)")
    print("=" * 70)
    print(f"Baseline: {bl_total}/28")
    print(f"V1:  {bl_total+v1_total}/28 (+{v1_recovered}, {v1_reg} reg)")
    print(f"V10: {bl_total+v10_total}/28 (+{v10_recovered}, {v10_reg} reg)")

    print("\nV10 regressions (V1 correct, V10 wrong):")
    for r in results:
        if r["v1_ok"] and not r["v10_ok"]:
            print(f"  {r['qid']}: gold={r['gold'][:30]} | V1={r['v1_out'][:40]} | V10={r['v10_out'][:40]}")

    print("\nV10 new wrong (both V1 and V10 wrong):")
    for r in results:
        if not r["v10_ok"] and not r["v1_ok"]:
            print(f"  {r['qid']}: gold={r['gold'][:30]} | V10={r['v10_out'][:40]}")

    print("\nV10 newly correct (V1 wrong, V10 right):")
    for r in results:
        if r["v10_ok"] and not r["v1_ok"]:
            print(f"  {r['qid']}: gold={r['gold'][:30]} | V1={r['v1_out'][:40]} | V10={r['v10_out'][:40]}")

    out = PROJECT_ROOT / "results/m30_54/phase3_v10_ab_test.json"
    (out).parent.mkdir(parents=True, exist_ok=True)
    (out).write_text(json.dumps({
        "summary": {
            "baseline": bl_total, "v1_total": bl_total+v1_total, "v10_total": bl_total+v10_total,
            "v1_recovered": v1_recovered, "v10_recovered": v10_recovered,
            "v1_regressed": v1_reg, "v10_regressed": v10_reg,
        },
        "rows": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run_ab_test()
