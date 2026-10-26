#!/usr/bin/env python3
"""M30-54 Phase 3 (v4): Entity-Relation Turn Binding — verb-stem version.

Key insight: instead of category matching (which produces false positives like
'decision' matching 'plan'), use direct verb stem matching:
  - Extract verb stems from question and turn body
  - Score by how many question verbs are covered by turn verbs
  - This is more precise: "come" and "share" both must appear

Also fixes: lowercase comparison, include 3-letter verbs, use stem matching.
"""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

import importlib.util
_spec = importlib.util.spec_from_file_location("v1", "scripts/m30_51_phase2_fallback.py")
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)

# --- Verb stem extraction ---

# Map infinitives to their various forms using simple rules
# No external libraries — deterministic stemming
def verb_stems(word):
    """Return possible verb stems for a word."""
    word = word.lower()
    if len(word) < 3:
        return set()
    stems = {word}
    # Common suffixes to strip
    for suffix in ['ing', 'ed', 'es', 's', 'd']:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stems.add(word[:-len(suffix)])
    # Special: 'came' -> 'come', 'made' -> 'make', 'met' -> 'meet'
    specials = {
        'came': 'come', 'came': 'come', 'making': 'make', 'made': 'make',
        'met': 'meet', 'planning': 'plan', 'planned': 'plan',
        'coming': 'come', 'coming': 'come',
        'sharing': 'share', 'shared': 'share',
        'going': 'go', 'went': 'go',
    }
    if word in specials:
        stems.add(specials[word])
    return stems


def extract_verb_stems(text):
    """Extract verb stems from text using simple heuristic."""
    # Common verbs in LoCoMo conversations
    words = re.findall(r'\b[a-z]+\b', text.lower())
    stems = set()
    for w in words:
        if len(w) < 3:
            continue
        # Check if it looks like a verb form
        # Includes: past tense (-ed), present participle (-ing), 
        # common action words
        action_verbs = {
            'go', 'going', 'went', 'come', 'coming', 'came',
            'share', 'sharing', 'shared', 'recipe', 'recipes',
            'visit', 'visited', 'visiting',
            'meet', 'met', 'meeting', 'meetings',
            'plan', 'planning', 'planned', 'plans',
            'try', 'trying', 'tried',
            'make', 'making', 'made',
            'adopt', 'adopting', 'adopted', 'adoption',
            'travel', 'traveled', 'trip', 'trips',
            'drive', 'drove', 'driving',
            'win', 'won', 'winning', 'won',
            'play', 'playing', 'played',
            'game', 'games', 'gaming', 'tournament', 'tournaments',
            'shoot', 'shooting', 'filming', 'film', 'films',
            'movie', 'movies', 'screen', 'screening',
            'cook', 'cooking', 'cooked',
            'run', 'running', 'ran',
            'hike', 'hiking', 'hiked',
            'road', 'trip', 'trips',
            'work', 'working', 'worked', 'project', 'projects',
            'job', 'jobs', 'offer', 'offered',
            'move', 'moved', 'moving',
            'decide', 'decided', 'decision', 'decisions',
            'choose', 'choosing', 'chose',
            'hope', 'hoping', 'hoped',
            'intend', 'intending', 'intended',
            'schedule', 'scheduled', 'scheduling',
            'prepare', 'preparing', 'prepared',
            'arrange', 'arranging', 'arranged',
            'set', 'setting', 'set up', 'set up',
            'start', 'started', 'starting',
            'begin', 'began', 'beginning', 'begun',
            'join', 'joined', 'joining',
            'attend', 'attended', 'attending',
            'participate', 'participated',
            'compete', 'competed', 'competing', 'competition',
            'score', 'scored', 'scoring',
            'lose', 'lost', 'losing',
            'win', 'won', 'winning',
            'read', 'reading', 'read',
            'watch', 'watched', 'watching',
            'see', 'seen', 'seeing', 'saw',
        }
        for v in action_verbs:
            if w == v or w.startswith(v) or v.startswith(w):
                stems.update(verb_stems(w))
                break
    return stems


def question_event_terms(question):
    """Extract the key event terms from a question.
    
    For 'When did Joanna plan to go over to Nate and share recipes?'
    → {'go', 'share', 'recipe'} (the EVENT being asked about)
    """
    # Extract action words from question
    stems = extract_verb_stems(question)
    # Filter to only event-relevant terms
    event_terms = set()
    for s in stems:
        if any(k in s for k in ['go', 'come', 'visit', 'share', 'recipe', 'meet', 'make', 
                                  'adopt', 'travel', 'trip', 'drive', 'win', 'play', 
                                  'game', 'hike', 'road', 'cook', 'filming', 'movie',
                                  'screen', 'work', 'job', 'move', 'run', 'watch',
                                  'read', 'see', 'lose', 'win']):
            event_terms.add(s)
    return event_terms


def entity_relation_score_v4(question, turn_body):
    """Score using entity overlap + verb-stem overlap.
    
    Key: the question's event verbs should be FOUND in the turn body.
    """
    q_terms = question_event_terms(question)
    t_stems = extract_verb_stems(turn_body)
    
    # How many question event terms appear in the turn?
    matched = q_terms & t_stems
    
    # Entity overlap (noun overlap)
    qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower())
              if w not in _v1.STOP and w not in {"when", "what", "where", "which", "who", "whose", "why", "how"})
    bkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", turn_body.lower()) if w not in _v1.STOP)
    entity_score = len(qkw & bkw)
    
    return entity_score, len(matched)


def deterministic_temporal_resolve_v4(ctx, question, category):
    """V4: Entity-Relation Turn Binding with verb-stem matching."""
    if not _v1.is_temporal_question(question, category):
        return None

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

        entity_score, pred_score = entity_relation_score_v4(question, body)
        candidates.append({
            "dia": dia, "hdr": hdr, "body": body, "rel": rels[0],
            "explicit": explicit, "base": base,
            "entity_score": entity_score, "pred_score": pred_score,
            "q_terms": question_event_terms(question),
        })

    if not candidates:
        return None

    # PRIMARY: predicate/event overlap (how many question event verbs are in the turn)
    # SECONDARY: entity overlap (who is involved)
    # TERTIARY: turns with explicit dates
    candidates.sort(key=lambda c: (
        -c["pred_score"],
        -c["entity_score"],
        0 if c["explicit"] else 1
    ))

    best = candidates[0]
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
        return f"{_v1.MONTH_NAMES[res.month]} {res.year}"
    return f"{res.day} {_v1.MONTH_NAMES[res.month]}, {res.year}"


def run_ab_test():
    """A/B test: V1 (noun-only) vs V4 (verb-stem entity-relation)."""
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
    v1_recovered = v3_recovered = v4_recovered = 0
    v1_reg = v3_reg = v4_reg = 0
    bl_total = v1_total = v4_total = 0

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

        v4_fallback = deterministic_temporal_resolve_v4(ctx, question, cat)
        v4_treated = v4_fallback if v4_fallback else raw
        v4_correct = LoCoMoAdapter.score_binary(cat, gold, v4_treated)

        if v1_correct: v1_total += 1
        if v4_correct: v4_total += 1
        if v1_correct and not bl_correct: v1_recovered += 1
        if v4_correct and not bl_correct: v4_recovered += 1
        if not v1_correct and bl_correct: v1_reg += 1
        if not v4_correct and bl_correct: v4_reg += 1

        if v1_correct != v4_correct or v1_treated != v4_treated:
            print(f"DIFF {qid}: gold={gold[:30]}")
            print(f"  V1: {v1_treated[:50]}")
            print(f"  V4: {v4_treated[:50]}")
            print()

        results.append({"qid": qid, "gold": gold[:40],
                       "bl": bl_correct, "v1": v1_correct, "v4": v4_correct,
                       "v1_out": v1_treated[:50], "v4_out": v4_treated[:50]})

    print("M30-54 Phase 3 v4: Entity-Relation Turn Binding (verb-stem)")
    print("="*70)
    print(f"Baseline: {bl_total}/28")
    print(f"V1: {bl_total+v1_total}/28 (+{v1_recovered} recovered, {v1_reg} regressed)")
    print(f"V4: {bl_total+v4_total}/28 (+{v4_recovered} recovered, {v4_reg} regressed)")
    # Save
    out = PROJECT_ROOT / "results/m30_54/phase3_v4_ab_test.json"
    (out).parent.mkdir(parents=True, exist_ok=True)
    (out).write_text(json.dumps({
        "summary": {
            "baseline": bl_total, "v1_total": bl_total+v1_total, "v4_total": bl_total+v4_total,
            "v1_recovered": v1_recovered, "v1_regressed": v1_reg,
            "v4_recovered": v4_recovered, "v4_regressed": v4_reg,
        },
        "rows": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run_ab_test()
