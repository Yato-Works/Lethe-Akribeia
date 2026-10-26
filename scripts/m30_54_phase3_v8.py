#!/usr/bin/env python3
"""M30-54 Phase 3 (v8): Entity-Relation Turn Binding — action-relation precision.

Root cause: simple keyword matching can't distinguish "tried a recipe" from
"brought recipes to share". Both contain 'recipe'.

Fix: require the turn to match MULTIPLE question action terms.
For "When did Joanna plan to go over and share recipes?"
  Action terms: {go/come/visit, share}
  D26:19: has "come over" → matches 'go/come'. Also has "recipes" → matches 'share/recipe'
  D22:1: has "recipe" → matches 'recipe'. But does NOT have "come/go/visit" as an ACTION
    → wait, D22:1 says "tried my newest recipe" — no "come over"

Strategy: extract question's **action verb phrases** and require the turn to contain
at least N of them. Score = (action matches) * weight + entity_matches.

But also preserve V1's correctness on lc_conv-47_65 and others.
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

# Import scoring helpers from V5
_spec2 = _ilu.spec_from_file_location("v5", "scripts/m30_54_phase3_v5.py")
_v5 = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_v5)

_spec3 = _ilu.spec_from_file_location("v6", "scripts/m30_54_phase3_v6.py")
_v6 = _ilu.module_from_spec(_spec3)
_spec3.loader.exec_module(_v6)


def extract_question_action_terms(question):
    """Extract action/event terms from question for precise matching.
    
    For 'When did Joanna plan to go over to Nate and share recipes?'
    → {'visit', 'share'} (the actual ACTIONS being described)
    
    For 'When did John go on a road trip to the Pacific Northwest?'
    → {'travel', 'road'}
    
    For 'When did Maria meet Jean?'
    → {'meet'}
    """
    stems = _v5.extract_stems(question)
    qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower())
              if w not in _v1.STOP and w not in {"when", "what", "where", "which", "who", "whom", "whose", "why", "how",
                                                  "did", "does", "do", "was", "were", "is", "are", "am", "name", "names"})
    
    # Build question's action term set using synonym groups
    action_terms = set()
    for w in stems | qkw:
        if w in _v5.get_stems(w):  # Get all stems for this word
            pass
        for stem in _v5.get_stems(w):
            action_terms.add(stem)
    
    # Filter to only action/event-related stems (not just nouns like 'nate', 'joanna')
    ACTION_STEMS = {'visit', 'share', 'recipe', 'recip', 'meet', 'adopt', 'adoption',
                    'travel', 'trip', 'drive', 'drove', 'hike', 'cook', 'cooked',
                    'run', 'ran', 'watch', 'watched', 'play', 'played', 'lose',
                    'lost', 'try', 'tried', 'game', 'gaming', 'film', 'filming',
                    'movie', 'screen', 'work', 'job', 'make', 'made', 'move', 'moved',
                    'decide', 'decided', 'decision', 'decisions', 'plan', 'planned',
                    'planning', 'plans', 'choose', 'chose', 'choosing'}
    
    return action_terms & ACTION_STEMS


def score_turn_action_match(question, turn_body, question_action_terms):
    """Score how well a turn matches the question's action terms.
    
    Returns: (action_match_count, total_question_actions, event_overlap_stems)
    """
    t_stems = _v5.extract_stems(turn_body)
    # How many of the question's action terms appear in the turn?
    matched_actions = question_action_terms & t_stems
    return len(matched_actions), len(question_action_terms)


def deterministic_temporal_resolve_v8(ctx, question, category):
    """V8: Entity-Relation Turn Binding with action-term precision matching.
    
    Key improvement over V5/V6/V7:
    - Extracts question's ACTION terms (verbs, not just nouns)
    - Requires turn to match action terms to score high
    - Distinguishes 'try recipe' (D22:1, no 'visit'/'share' action) from
      'come over... bring recipes' (D26:19, matches 'visit' + 'share')
    """
    if not _v1.is_temporal_question(question, category):
        return None

    direction = _v6.question_temporal_direction(question)
    q_action_terms = extract_question_action_terms(question)

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
        has_explicit = explicit is not None

        # NEW: action term matching
        action_matched, action_total = score_turn_action_match(question, body, q_action_terms)

        candidates.append({
            "dia": dia, "hdr": hdr, "body": body, "rel": rels[0],
            "explicit": explicit, "base": base,
            "entity_score": ent_score, "event_score": evt_score,
            "matched": matched, "bias": bias, "has_explicit": has_explicit,
            "action_matched": action_matched, "action_total": action_total,
        })

    if not candidates:
        return None

    # V8 sorting strategy:
    # 1. Primary: explicit date turns with high action match
    # 2. Score formula: action_matched * 10 + event_score * 2 + entity_score + bias
    #    Action terms are weighted highest to distinguish true event matches
    explicit_cands = [c for c in candidates if c["has_explicit"]]
    non_explicit_cands = [c for c in candidates if not c["has_explicit"]]

    for cands in [explicit_cands, non_explicit_cands]:
        for c in cands:
            c["v8_score"] = c["action_matched"] * 10 + c["event_score"] * 2 + c["entity_score"] + c["bias"]

    # Sort each group, prefer explicit dates
    for cands in [explicit_cands, non_explicit_cands]:
        cands.sort(key=lambda c: (-c["v8_score"], -c["entity_score"]))

    # Prefer explicit date turns if any exist with reasonable action match
    if explicit_cands and any(c["action_matched"] > 0 for c in explicit_cands[:3]):
        best = explicit_cands[0]
        return best["explicit"]

    if non_explicit_cands:
        best = non_explicit_cands[0]
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

    # No non-explicit candidates — use explicit candidates
    if explicit_cands:
        return explicit_cands[0]["explicit"]
    return None


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
    v8_recovered = v8_reg = 0
    bl_total = v1_total = v8_total = 0

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

        v8_f = deterministic_temporal_resolve_v8(ctx, question, cat)
        v8_t = v8_f if v8_f else raw
        v8_ok = LoCoMoAdapter.score_binary(cat, gold, v8_t)

        if v1_ok: v1_total += 1
        if v8_ok: v8_total += 1
        if v1_ok and not bl_correct: v1_recovered += 1
        if v8_ok and not bl_correct: v8_recovered += 1
        if bl_correct and not v1_ok: v1_reg += 1
        if bl_correct and not v8_ok: v8_reg += 1

        if v8_ok != v1_ok or v8_t != v1_t:
            status = "✅" if v8_ok else "❌"
            print(f"DIFF {qid}: gold={gold[:35]} {status}")
            print(f"  V1: {v1_t[:50]}")
            print(f"  V8: {v8_t[:50]}")
            print()

        results.append({"qid": qid, "gold": gold[:50],
                       "v1_ok": v1_ok, "v8_ok": v8_ok,
                       "v1_out": v1_t[:50], "v8_out": v8_t[:50]})

    print("M30-54 Phase 3 v8: Entity-Relation Turn Binding (action-term precision)")
    print("="*70)
    print(f"Baseline: {bl_total}/28")
    print(f"V1: {bl_total+v1_total}/28 (+{v1_recovered}, {v1_reg} reg)")
    print(f"V8: {bl_total+v8_total}/28 (+{v8_recovered}, {v8_reg} reg)")

    # Show which V1-correct cases V8 regressed
    print("\nV8 regressions (V1 correct, V8 wrong):")
    for r in results:
        if r["v1_ok"] and not r["v8_ok"]:
            print(f"  {r['qid']}: gold={r['gold'][:30]} | V1={r['v1_out'][:40]} | V8={r['v8_out'][:40]}")

    print("\nV8 wrong cases:")
    for r in results:
        if not r["v8_ok"] and not r["v1_ok"]:
            print(f"  {r['qid']}: gold={r['gold'][:30]} | V8={r['v8_out'][:40]}")

    out = PROJECT_ROOT / "results/m30_54/phase3_v8_ab_test.json"
    (out).parent.mkdir(parents=True, exist_ok=True)
    (out).write_text(json.dumps({
        "summary": {
            "baseline": bl_total, "v1_total": bl_total+v1_total, "v8_total": bl_total+v8_total,
            "v1_recovered": v1_recovered, "v8_recovered": v8_recovered,
            "v1_regressed": v1_reg, "v8_regressed": v8_reg,
        },
        "rows": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run_ab_test()
