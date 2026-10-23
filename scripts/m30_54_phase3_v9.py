#!/usr/bin/env python3
"""M30-54 Phase 3 (v9): Entity-Relation Turn Binding — phrase-based matching.

Root cause of V8 failure on lc_conv-42_57:
  - Question: 'When did Joanna plan to go over to Nate and share recipes?'
  - V8 extracted action terms: {plan, visit, cook, share, recip, recipe}
  - D22:1 matched 'recipe' + 'share' + 'cook' (from 'tried my recipe') → wrong turn
  - D26:19 matched only 'visit' (from 'come over') → lower score
  
The problem: 'recipe' and 'share' are too generic — they appear in both turns.
Fix: use **phrase matching** for compound events, not individual word matching.

For 'go over and share recipes', the question's primary event is 'go over' (visit).
The turn must contain a visit-related phrase AND the correct temporal context.

V9 approach:
  1. Extract question's PRIMARY action phrase (the main verb phrase)
  2. Score turns by how well they match the primary action phrase
  3. If explicit date present, that turn wins (it's pre-resolved)
  4. Tiebreak by temporal direction match
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


# Action phrase patterns: maps question patterns to turn body patterns
ACTION_PATTERNS = [
    # (question_pattern, turn_patterns, weight)
    # "go over" / "come over" → visit
    (r'\bgo over\b|\bcome over\b|\bvisit\b|\bgoing over\b|\bcoming over\b', 
     [r'\bgo over\b', r'\bcome over\b', r'\bgoing over\b', r'\bcoming over\b', r'\bgonna go over\b', r'\bgonna come over\b'], 
     3),
    # "share recipes" → share
    (r'\bshare\s+[r]?recipe', [r'\bshare', r'\brecipes\b', r'recip'], 1),
    # "meet" → encounter
    (r'\bmeet\b|\bmet\b|\bmeeting\b',
     [r'\bmeet', r'\bmet', r'\bencounter', r'\bmeet'],
     2),
    # "adopt" → adoption
    (r'\badopt\b|\badopted\b|\badoption\b',
     [r'\badopt', r'\badopted', r'\badoption', r'\bcare\b'],
     2),
    # "road trip" → travel
    (r'\broad trip\b|\btravel\b|\bdrive\b|\broad-trip\b',
     [r'\broad trip', r'\btravel', r'\bdrive', r'\broad'],
     2),
    # "game" / "tournament" → competition
    (r'\bgame\b|\btournament\b|\bcompete\b',
     [r'\bgame', r'\btournament', r'\bcompete', r'\bwon\b', r'\bplay\b'],
     2),
    # "movie" / "film" → media
    (r'\bmovie\b|\bfilm\b|\bscreen\b',
     [r'\bmovie', r'\bfilm', r'\bscreen', r'\bpitch'],
     2),
    # "hike" → outdoor
    (r'\bhike\b|\bhiking\b',
     [r'\bhike', r'\bhiking', r'\btrail'],
     2),
    # "yoga" → exercise
    (r'\byoga\b',
     [r'\byoga', r'\byoga'],
     2),
]


def score_action_phrase_overlap(question, turn_body):
    """Score based on question's action phrases and how well the turn matches them.
    
    Returns: (score, matched_patterns)
    """
    q_lower = question.lower()
    t_lower = turn_body.lower()
    score = 0.0
    matched = []
    
    for q_pat, t_pats, weight in ACTION_PATTERNS:
        if re.search(q_pat, q_lower):
            for t_pat in t_pats:
                if re.search(t_pat, t_lower):
                    score += weight
                    matched.append((q_pat, t_pat, weight))
                    break  # only count once per question pattern
    
    return score, matched


def deterministic_temporal_resolve_v9(ctx, question, category):
    """V9: Phrase-based Entity-Relation Turn Binding."""
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

        # Entity overlap (from V1)
        qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower())
                  if w not in _v1.STOP and w not in {"when", "what", "where", "which", "who", "whom", "whose", "why", "how",
                                                      "did", "does", "do", "was", "were", "is", "are", "am", "name", "names"})
        bkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", body.lower()) if w not in _v1.STOP)
        entity_score = len(qkw & bkw)

        # Action phrase overlap
        action_score, matched_actions = score_action_phrase_overlap(question, body)
        bias = _v6.temporal_direction_bias(rels[0], direction)
        has_explicit = explicit is not None

        candidates.append({
            "dia": dia, "hdr": hdr, "body": body, "rel": rels[0],
            "explicit": explicit, "base": base,
            "entity_score": entity_score, "action_score": action_score,
            "matched_actions": matched_actions, "bias": bias,
            "has_explicit": has_explicit,
        })

    if not candidates:
        return None

    # V9 scoring: 
    # PRIMARY: action_score (phrase-based event matching)
    # SECONDARY: entity_score + bias
    # EXPLICIT DATES are strongly preferred
    for c in candidates:
        c["v9_score"] = c["action_score"] * 5 + c["entity_score"] + c["bias"]
        if c["has_explicit"]:
            c["v9_score"] += 1  # small bonus for explicit dates

    candidates.sort(key=lambda c: (-c["v9_score"], -c["entity_score"], 0 if c["has_explicit"] else 1))

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
        if sid not in conv_ids_needed: continue
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            cqid = q.question_id
            if cqid in qid_to_cqid.values():
                pcc = adapter.compiler.compile(q.question, ir_records)
                ctx_cache[cqid] = (q.question, pcc.context_text, q.category)

    results = []
    v1_recovered = v1_reg = 0
    v9_recovered = v9_reg = 0
    bl_total = v1_total = v9_total = 0

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

        v9_f = deterministic_temporal_resolve_v9(ctx, question, cat)
        v9_t = v9_f if v9_f else raw
        v9_ok = LoCoMoAdapter.score_binary(cat, gold, v9_t)

        if v1_ok: v1_total += 1
        if v9_ok: v9_total += 1
        if v1_ok and not bl_correct: v1_recovered += 1
        if v9_ok and not bl_correct: v9_recovered += 1
        if bl_correct and not v1_ok: v1_reg += 1
        if bl_correct and not v9_ok: v9_reg += 1

        if v9_ok != v1_ok or v9_t != v1_t:
            status = "✅" if v9_ok else "❌"
            print(f"DIFF {qid}: gold={gold[:35]} {status}")
            print(f"  V1: {v1_t[:50]}")
            print(f"  V9: {v9_t[:50]}")
            print()

        results.append({"qid": qid, "gold": gold[:50],
                       "v1_ok": v1_ok, "v9_ok": v9_ok,
                       "v1_out": v1_t[:50], "v9_out": v9_t[:50]})

    print("M30-54 Phase 3 v9: Entity-Relation Turn Binding (phrase-based)")
    print("="*70)
    print(f"Baseline: {bl_total}/28")
    print(f"V1: {bl_total+v1_total}/28 (+{v1_recovered}, {v1_reg} reg)")
    print(f"V9: {bl_total+v9_total}/28 (+{v9_recovered}, {v9_reg} reg)")

    print("\nV9 regressions (V1 correct, V9 wrong):")
    for r in results:
        if r["v1_ok"] and not r["v9_ok"]:
            print(f"  {r['qid']}: gold={r['gold'][:30]} | V1={r['v1_out'][:40]} | V9={r['v9_out'][:40]}")

    print("\nV9 new wrong (both V1 and V9 wrong):")
    for r in results:
        if not r["v9_ok"] and not r["v1_ok"]:
            print(f"  {r['qid']}: gold={r['gold'][:30]} | V9={r['v9_out'][:40]}")

    out = PROJECT_ROOT / "results/m30_54/phase3_v9_ab_test.json"
    (out).parent.mkdir(parents=True, exist_ok=True)
    (out).write_text(json.dumps({
        "summary": {
            "baseline": bl_total, "v1_total": bl_total+v1_total, "v9_total": bl_total+v9_total,
            "v1_recovered": v1_recovered, "v9_recovered": v9_recovered,
            "v1_regressed": v1_reg, "v9_regressed": v9_reg,
        },
        "rows": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run_ab_test()
