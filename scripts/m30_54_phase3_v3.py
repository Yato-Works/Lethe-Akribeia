#!/usr/bin/env python3
"""M30-54 Phase 3: Entity-Relation Turn Binding (v3).

Improves on M30-51 V1's noun-overlap-only scoring by adding:
  - Predicate/event overlap: match verbs/actions between question and candidate turn
  - This disambiguates when same entities (Joanna+Nate) appear across multiple turns
    about DIFFERENT events (recipe visit vs. game tournament).

No hardcoded names or QIDs. Generalizes via verb-stem matching.
"""
import json, re, sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory_gen.recall.evidence_scorer import UniversalEvidenceScorer

# Import noun-overlap version as base
import importlib.util
_spec = importlib.util.spec_from_file_location("fallback_v1", "scripts/m30_51_phase2_fallback.py")
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)

MONTHS = _v1.MONTHS
MONTH_NAMES = _v1.MONTH_NAMES
TURN_RE = _v1.TURN_RE

# Extended stop words for predicate extraction
PRED_STOP = _v1.STOP | {"got", "said", "says", "say", "says,", "got", "got", "get", "got", "would", "could", "should", "must"}

# Verb stems for key event categories
# These are derived from verb patterns, not specific entities
VERB_CATEGORIES = {
    "visit": ["visit", "come", "came", "go", "going", "went", "over"],  # "come over", "go over"
    "share/recipe": ["share", "recip", "bring", "made", "make", "cook", "cooked"],  # "share recipes"
    "plan": ["plan", "plann", "plann", "intend", "hope", "schedul"],
    "meet": ["meet", "met", "meet", "saw", "seen", "see", "visit"],
    "travel": ["travel", "trip", "road", "drive", "flew", "fly"],
    "game/tournament": ["game", "tournament", "compete", "won", "win", "play", "played", "score"],
    "adopt": ["adopt", "adopt", "care", "home", "foster"],
    "movie": ["movie", "film", "screen", "shoot", "filming", "direct"],
    "road_trip": ["road", "trip", "drive", "drove"],
}


def extract_predicates(text):
    """Extract verb stems from text for event matching."""
    words = re.findall(r"\b[a-z]+\b", text.lower())
    # Simple verb stem extraction: take first 4 chars of verbs
    # This is deterministic and doesn't require LLM
    predicates = set()
    for w in words:
        if w not in PRED_STOP and len(w) >= 4 and w not in {'this','that','with','from','have','they','them','theyre','theres','thats','would','could','should','there','their','about','after','before','during','through','between','which','while','where','when','what','when','whose','whom','which','that'}:
            # Match against known verb categories
            for cat, stems in VERB_CATEGORIES.items():
                if any(s in w for s in stems):
                    predicates.add(cat)
                    break
    return predicates


def entity_relation_score(question, turn_body):
    """Score a candidate turn using entity + predicate matching.

    Returns a score that prioritizes:
    1. Predicate/event overlap (NEW) — what is the turn about?
    2. Entity overlap (existing) — who is involved?

    A turn gets high score only if it's about the SAME EVENT as the question,
    not just mentioning the same people.
    """
    # Existing entity/noun overlap from V1
    qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower())
              if w not in _v1.STOP and w not in {"when", "what", "where", "which", "who", "whose", "why", "how"})
    bkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", turn_body.lower()) if w not in _v1.STOP)
    entity_score = len(qkw & bkw)

    # NEW: predicate/event overlap
    q_preds = extract_predicates(question)
    t_preds = extract_predicates(turn_body)
    pred_score = len(q_preds & t_preds) if q_preds else 0

    return entity_score, pred_score


def deterministic_temporal_resolve_v3(ctx, question, category):
    """Entity-Relation Turn Binding version.

    Same as V1 but uses predicate/event overlap as PRIMARY sort key,
    with noun overlap as tiebreaker.
    """
    if not _v1.is_temporal_question(question, category):
        return None

    candidates = []
    for m in TURN_RE.finditer(ctx):
        dia, hdr, body = m.group(1), m.group(2), m.group(3)
        rels = _v1.REL_RE.findall(body)
        if not rels:
            continue
        explicit = _v1.parse_explicit_date(body)
        base = _v1.parse_header_date(hdr)
        if not base:
            continue

        entity_score, pred_score = entity_relation_score(question, body)
        candidates.append({
            "dia": dia, "hdr": hdr, "body": body, "rel": rels[0],
            "explicit": explicit, "base": base,
            "entity_score": entity_score, "pred_score": pred_score,
        })

    if not candidates:
        return None

    # PRIMARY: predicate/event overlap (what happened)
    # SECONDARY: entity overlap (who is involved)
    # TERTIARY: turns with explicit dates
    candidates.sort(key=lambda c: (
        -c["pred_score"],      # prefer turns about the same event
        -c["entity_score"],    # then same entities
        0 if c["explicit"] else 1  # then explicit dates
    ))

    best = candidates[0]

    # Return explicit date if available, else resolve from base
    if best["explicit"]:
        return best["explicit"]

    from datetime import datetime, timedelta
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
        return f"{MONTH_NAMES[res.month]} {res.year}"
    return f"{res.day} {MONTH_NAMES[res.month]}, {res.year}"


def run_ab_test():
    """Run A/B test: V1 (noun-only) vs V3 (entity-relation) on all 28 cases."""
    prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
    baseline = {r["qid"]: r for r in prod["rows"]}

    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sids = [c.get("sample_id", "") for c in locomo]
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    conv_to_idx = {}
    for idx, sid in enumerate(sids):
        conv_num = int(sid.split("-")[1])
        conv_to_idx[conv_num] = idx

    # Cache contexts
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
    v1_recovered = v1_regressed = 0
    v3_recovered = v3_regressed = 0
    bl_total = tr_v1_total = tr_v3_total = 0

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
            results.append({"qid": qid, "gold": gold[:40], "baseline_correct": bl_correct,
                           "v1_correct": bl_correct, "v3_correct": bl_correct,
                           "v1_change": "same", "v3_change": "same"})
            continue

        # V1 treatment
        v1_fallback = _v1.deterministic_temporal_resolve(ctx, question, cat)
        v1_treated = v1_fallback if v1_fallback else raw
        v1_correct = LoCoMoAdapter.score_binary(cat, gold, v1_treated)

        # V3 treatment
        v3_fallback = deterministic_temporal_resolve_v3(ctx, question, cat)
        v3_treated = v3_fallback if v3_fallback else raw
        v3_correct = LoCoMoAdapter.score_binary(cat, gold, v3_treated)

        if v1_correct: tr_v1_total += 1
        if v3_correct: tr_v3_total += 1

        if v1_correct and not bl_correct: v1_recovered += 1
        if v1_correct != bl_correct and not v1_correct: v1_regressed += 1
        if v3_correct and not bl_correct: v3_recovered += 1
        if v3_correct != bl_correct and not v3_correct: v3_regressed += 1

        v1_change = "RECOVERED" if (v1_correct and not bl_correct) else ("REGRESSED" if (not v1_correct and bl_correct) else ("changed" if v1_treated != raw else "same"))
        v3_change = "RECOVERED" if (v3_correct and not bl_correct) else ("REGRESSED" if (not v3_correct and bl_correct) else ("changed" if v3_treated != raw else "same"))

        results.append({"qid": qid, "gold": gold[:40], "baseline_correct": bl_correct,
                       "v1_correct": v1_correct, "v3_correct": v3_correct,
                       "v1_change": v1_change, "v3_change": v3_change,
                       "v1_treated": v1_treated[:60], "v3_treated": v3_treated[:60],
                       "raw": raw[:60]})

    print("M30-54 Phase 3: Entity-Relation Turn Binding A/B Test")
    print("="*70)
    print(f"Baseline: {bl_total}/28")
    print(f"V1 (noun-only): {tr_v1_total}/28 (+{v1_recovered} recovered, {v1_regressed} regressed)")
    print(f"V3 (entity-relation): {tr_v3_total}/28 (+{v3_recovered} recovered, {v3_regressed} regressed)")
    print()

    # Detail: cases where V3 differs from V1
    print("Cases where V3 differs from V1:")
    for r in results:
        if r.get("v3_treated") != r.get("v1_treated") and r.get("v3_treated"):
            print(f"  {r['qid']}: gold={r['gold'][:25]}")
            print(f"    V1: {r.get('v1_treated','')[:50]}")
            print(f"    V3: {r.get('v3_treated','')[:50]}")

    # Wrong cases after V3
    print()
    print("Remaining wrong after V3:")
    for r in results:
        if not r.get("v3_correct"):
            print(f"  {r['qid']}: gold={r['gold'][:25]} | v1={'same' if r['v1_correct'] else 'wrong'} | v3={'same' if r['v3_correct'] else 'wrong'}")

    # Save results
    out_dir = PROJECT_ROOT / "results/m30_54"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "baseline_total": bl_total,
        "v1_total": tr_v1_total,
        "v3_total": tr_v3_total,
        "v1_recovered": v1_recovered,
        "v1_regressed": v1_regressed,
        "v3_recovered": v3_recovered,
        "v3_regressed": v3_regressed,
        "total": len(results),
    }
    (out_dir / "phase3_v3_ab_test.json").write_text(json.dumps({
        "summary": summary,
        "rows": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'phase3_v3_ab_test.json'}")


if __name__ == "__main__":
    run_ab_test()
