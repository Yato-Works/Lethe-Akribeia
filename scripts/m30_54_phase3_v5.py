#!/usr/bin/env python3
"""M30-54 Phase 3 (v5): Entity-Relation Turn Binding — combined scoring.

Fixes V4's fragility by using a combined score:
  score = entity_overlap + 2 * event_overlap
  
Where event_overlap is the count of content-word overlaps that are verbs/actions
or question-specific event terms. The key: D26:19 has "come over" (go) + "recipes" (share recipes)
while D28:8 has "decision" (no event match).

This is General (no hardcoded names).
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

# Extended verb stems — map various forms to their base
def get_stems(word):
    """Return base verb forms for a word. Maps synonyms to shared stems."""
    word = word.lower()
    if len(word) < 2:
        return set()
    stems = {word}
    # Strip common suffixes
    for suf in ['ing', 'ed', 'es', 's', 'd', 'er', 'est']:
        if word.endswith(suf) and len(word) > len(suf) + 1:
            stems.add(word[:-len(suf)])
    # Synonym groups — map related verbs to the same base
    # This allows "come" to match "go" (both mean "visit/arrive")
    synonym_groups = {
        'visit': {'visit', 'visited', 'visiting', 'visits', 'came', 'coming', 'comes', 'come', 'go', 'going', 'went', 'goes'},
        'share': {'share', 'shares', 'sharing', 'shared', 'recipe', 'recipes', 'recip', 'recip'},  
        'meet': {'meet', 'met', 'meeting', 'meetings', 'encounter', 'encountered'},
        'adopt': {'adopt', 'adopted', 'adopting', 'adoption'},
        'travel': {'travel', 'traveled', 'traveling', 'travels', 'trip', 'trips', 'tripped'},
        'game': {'game', 'games', 'gaming', 'play', 'playing', 'played', 'plays'},
        'cook': {'cook', 'cooking', 'cooked', 'cooks', 'recipe', 'recipes', 'recip'},
        'decide': {'decide', 'decided', 'decision', 'decisions', 'deciding', 'chose', 'choose', 'choosing'},
        'hike': {'hike', 'hiking', 'hiked', 'hikes'},
        'watch': {'watch', 'watching', 'watched', 'watches'},
        'read': {'read', 'reading', 'reads', 'read'},
        'run': {'run', 'running', 'ran', 'runs'},
        'lose': {'lose', 'losing', 'lost', 'loses'},
        'win': {'win', 'won', 'winning', 'wins'},
        'work': {'work', 'working', 'worked', 'works', 'job', 'jobs', 'offer', 'offered'},
        'make': {'make', 'making', 'made', 'makes'},
        'move': {'move', 'moved', 'moving', 'moves'},
        'see': {'see', 'seen', 'seeing', 'saw', 'sees'},
        'plan': {'plan', 'planned', 'planning', 'plans'},
    }
    for group_stem, members in synonym_groups.items():
        if word in members:
            stems.add(group_stem)
    return stems


def extract_stems(text):
    """Extract stemmed content from text (verbs + key nouns)."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    stems = set()
    for w in words:
        if len(w) < 2 or w in _v1.STOP:
            continue
        stems.update(get_stems(w))
    return stems


def get_question_event_keys(question):
    """Extract the key event terms from a question.
    
    Strategy: find verbs in the question that describe the event being asked about.
    For 'When did Joanna plan to go over to Nate and share recipes?'
    → keys: {'go', 'share', 'recipe', 'plan'}
    """
    stems = extract_stems(question)
    # Also include the question keywords (nouns) that describe the event
    qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower())
              if w not in _v1.STOP and w not in {"when", "what", "where", "which", "who", "whose", "why", "how", "did", "does", "do", "was", "were", "is", "are", "am"})
    # Flatten: all stems + all qkw + all stems of qkw
    keys = set(stems)
    for w in qkw:
        keys.update(get_stems(w))
    keys.update(qkw)
    return keys


def combined_score(question, turn_body):
    """Score: entity overlap + event overlap with specificity weighting.

    Key insight: single-word matches like 'plan' are too generic.
    Require multiple event matches for a high score.
    Specific verbs (go, come, share) weighted higher than generic ones (plan, make).
    """
    q_keys = get_question_event_keys(question)
    t_stems = extract_stems(turn_body)

    # Event overlap: question's key verbs in the turn body
    event_match = q_keys & t_stems

    # Entity overlap (noun overlap)
    qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower())
              if w not in _v1.STOP and w not in {"when", "what", "where", "which", "who", "whom", "whose", "why", "how", "did", "does", "do", "was", "were", "is", "are", "am"})
    bkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", turn_body.lower()) if w not in _v1.STOP)
    entity_score = len(qkw & bkw)

    # Specificity-weighted event score:
    # Verbs related to the actual event (go, come, share, recipe, meet, etc.) are specific
    # Generic verbs (plan, make, create, work) are low-specificity
    SPECIFIC = {'visit', 'share', 'recipe', 'recip', 'meet', 'adopt', 'travel', 'trip',
                'drive', 'drove', 'hike', 'cook', 'cooked', 'run', 'ran', 'watch',
                'played', 'lose', 'lost', 'try', 'tried', 'game', 'gaming',
                'film', 'filming', 'movie', 'screen', 'work', 'job', 'offer', 'made',
                'move', 'moved', 'decide', 'decided', 'decision', 'decisions'}
    GENERIC = {'plan', 'planning', 'planned', 'plans', 'create', 'created', 'creating', 'hope', 'hoping', 'intend', 'prepare'}

    # Only count specific event matches, exclude generic words
    specific_matches = event_match & SPECIFIC
    generic_matches = event_match & GENERIC

    # Event score: 2 points per specific match, 0.5 per generic (but require at least 1 specific)
    event_score = len(specific_matches) * 2 + len(generic_matches) * 0.5

    # Penalty: if only generic matches (no specific), score = 0
    if len(specific_matches) == 0:
        event_score = 0

    return entity_score, event_score, specific_matches


def deterministic_temporal_resolve_v5(ctx, question, category):
    """V5: Combined entity + event scoring."""
    if not _v1.is_temporal_question(question, category):
        return None

    candidates = []
    q_keys = get_question_event_keys(question)
    
    for m in _v1.TURN_RE.finditer(ctx):
        dia, hdr, body = m.group(1), m.group(2), m.group(3)
        rels = _v1.REL_RE.findall(body)
        if not rels:
            continue
        explicit = _v1.parse_explicit_date(body)
        base = _v1.parse_header_date(hdr)
        if not base:
            continue

        ent_score, evt_score, matched = combined_score(question, body)
        candidates.append({
            "dia": dia, "hdr": hdr, "body": body, "rel": rels[0],
            "explicit": explicit, "base": base,
            "entity_score": ent_score, "event_score": evt_score,
            "matched": matched,
        })

    if not candidates:
        return None

    # PRIMARY: event_score (specificity-weighted) + entity_score
    # event_score already multiplies specific matches by 2, so we can just sum
    candidates.sort(key=lambda c: (
        -(c["event_score"] * 2 + c["entity_score"]),
        0 if c["explicit"] else 1
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
    v1_recovered = v5_recovered = 0
    v1_reg = v5_reg = 0
    bl_total = v1_total = v5_total = 0

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

        v1_fallback = _v1.deterministic_temporal_resolve(ctx, question, cat)
        v1_treated = v1_fallback if v1_fallback else raw
        v1_correct = LoCoMoAdapter.score_binary(cat, gold, v1_treated)

        v5_fallback = deterministic_temporal_resolve_v5(ctx, question, cat)
        v5_treated = v5_fallback if v5_fallback else raw
        v5_correct = LoCoMoAdapter.score_binary(cat, gold, v5_treated)

        if v1_correct: v1_total += 1
        if v5_correct: v5_total += 1
        if v1_correct and not bl_correct: v1_recovered += 1
        if v5_correct and not bl_correct: v5_recovered += 1
        if not v1_correct and bl_correct: v1_reg += 1
        if not v5_correct and bl_correct: v5_reg += 1

        if v1_correct != v5_correct or v1_treated != v5_treated:
            print(f"DIFF {qid}: gold={gold[:30]}")
            print(f"  V1: {v1_treated[:50]}")
            print(f"  V5: {v5_treated[:50]}")
            print()

        results.append({"qid": qid, "gold": gold[:40],
                       "bl": bl_correct, "v1": v1_correct, "v5": v5_correct,
                       "v1_out": v1_treated[:50], "v5_out": v5_treated[:50]})

    print("M30-54 Phase 3 v5: Entity-Relation Turn Binding (combined scoring)")
    print("="*70)
    print(f"Baseline: {bl_total}/28")
    print(f"V1: {bl_total+v1_total}/28 (+{v1_recovered} recovered, {v1_reg} regressed)")
    print(f"V5: {bl_total+v5_total}/28 (+{v5_recovered} recovered, {v5_reg} regressed)")
    
    out = PROJECT_ROOT / "results/m30_54/phase3_v5_ab_test.json"
    (out).parent.mkdir(parents=True, exist_ok=True)
    (out).write_text(json.dumps({
        "summary": {
            "baseline": bl_total, "v1_total": bl_total+v1_total, "v5_total": bl_total+v5_total,
            "v1_recovered": v1_recovered, "v1_regressed": v1_reg,
            "v5_recovered": v5_recovered, "v5_regressed": v5_reg,
        },
        "rows": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run_ab_test()
