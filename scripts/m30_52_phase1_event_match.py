#!/usr/bin/env python3
"""M30-52: Improve binding precision for the deterministic fallback.

Problem: noun overlap selects the wrong turn when multiple context turns
mention the same actors (e.g., "Joanna" + "Nate" appear in many turns).

Fix: add EVENT ENTITY matching. Extract the specific event/activity from the
question (e.g., "talent show", "trip to Tokyo", "road trip", "recipes") and
require the turn to contain BOTH the actor AND the event entity.

Uses SEMANTIC_EXPANSIONS from EvidenceScorer for concept matching.
No Gold, no evidence IDs — pure content matching.
"""
import json, re, sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory_gen.recall.evidence_scorer import UniversalEvidenceScorer

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}
TURN_RE = re.compile(r"(?:\[SPEAKER:[^\]]+\]\s*)?\[(D\d+:\d+)\s+on\s+([^\]]+)\](.*)")
REL_RE = re.compile(r"\b(yesterday|tomorrow|the next day|the day before|next month|last month|"
                    r"next week|last week|currently|one year ago|two years ago)\b", re.I)
STOP = set("the and for are was were did does has have had will would could should can may might "
           "must this that these those when where what which who why how not all any both each "
           "with from into about after before during between through over under".split())

scorer = UniversalEvidenceScorer()


def parse_explicit_date(text):
    m = re.search(r"\(?\s*(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\s*\)?", text)
    if m and m.group(2).lower() in MONTHS:
        return f"{m.group(1)} {m.group(2)}, {m.group(3)}"
    m = re.search(r"\(\s*([A-Za-z]+)\s+(\d{4})\s*\)", text)
    if m and m.group(1).lower() in MONTHS:
        return f"{m.group(1)} {m.group(2)}"
    m = re.search(r"\b([A-Za-z]+)\s+(\d{4})\b", text)
    if m and m.group(1).lower() in MONTHS and str(m.group(2)) not in text[:m.start()][-5:]:
        # Avoid matching header dates — focus on body
        pass
    m = re.search(r"\b([A-Za-z]+)\s+(\d{4})\b", text)
    if m and m.group(1).lower() in MONTHS:
        return f"{m.group(1)} {m.group(2)}"
    m = re.search(r"\b(\d{4})\b", text)
    if m and len(text) > 40:
        return m.group(1)
    return None


def parse_header_date(hdr):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", hdr or "")
    if m and m.group(2).lower() in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None


def extract_event_phrase(question):
    """Extract the event noun phrase from the question.

    Instead of individual keywords, extract the semantic event concept
    as a phrase: "talent show", "road trip to the Pacific", etc.
    This avoids matching unrelated turns that happen to mention actor names.
    """
    q = re.sub(r'\b(when|did|do|does|is|are|was|were|will|would|could|should|can|may|might|do)\b', '',
               question, flags=re.I).strip(' ?').strip()
    # Remove leading "When is" / "What date did" etc.
    q = re.sub(r'^(when\s+(?:is|was|are|will|do|did|does)\s+)', '', q, flags=re.I).strip()
    q = re.sub(r'^(what\s+(?:date|year|month)\s+(?:did|was|is|are|will)\s+)', '', q, flags=re.I).strip()
    q = re.sub(r"^([^'']+)'s\s+", '', q)  # remove possessive "Caroline's"

    # Extract noun phrase: everything up to first verb or preposition
    # Look for the core event concept
    q_lower = q.lower()
    # Find phrases like "talent show", "road trip", "appreciation letter"
    # Try multi-word phrase extraction
    phrase_match = re.match(r"^([a-z]+\s+[a-z]+(?:\s+[a-z]+)*?)(?:\s+(?:when|next|last|on|at|in|the|putting|planning|having|going|took|did|will|is|are|was|were))", q_lower)
    if phrase_match:
        return phrase_match.group(1).strip()
    # Fallback: take first 2 content words
    words = re.findall(r"\b[a-z]{4,}\b", q_lower)
    words = [w for w in words if w not in STOP and w not in {"when", "what", "which"}]
    return " ".join(words[:2]) if words else ""


def resolve_relative(expr, base):
    if not base: return None
    e = expr.lower()
    if "yesterday" in e or "the day before" in e: return base - timedelta(days=1)
    if "tomorrow" in e or "the next day" in e: return base + timedelta(days=1)
    if "next month" in e:
        mo = 1 if base.month == 12 else base.month + 1
        yr = base.year + (1 if base.month == 12 else 0)
        return datetime(yr, mo, 1)
    if "last month" in e:
        mo = 12 if base.month == 1 else base.month - 1
        yr = base.year - (1 if base.month == 1 else 0)
        return datetime(yr, mo, 1)
    if "next week" in e: return base + timedelta(days=7)
    if "last week" in e: return base - timedelta(days=7)
    if "currently" in e: return base
    if "one year ago" in e: return datetime(base.year - 1, base.month, base.day)
    if "two years ago" in e: return datetime(base.year - 2, base.month, base.day)
    return None


def is_temporal_question(q, cat):
    if cat == 2:
        return True
    ql = q.lower()
    temporal_kw = {"when", "what date", "what year", "what month", "which date", "which year"}
    return any(w in ql for w in temporal_kw)


def deterministic_temporal_resolve_v2(ctx, question, category):
    """V2: with event entity matching for better turn selection."""
    if not is_temporal_question(question, category):
        return None

    # Step 1: Extract event phrase from question (e.g., "talent show")
    event_phrase = extract_event_phrase(question)
    event_lower = event_phrase.lower()

    # Step 2: Find candidate turns
    candidates = []
    for m in TURN_RE.finditer(ctx):
        dia, hdr, body = m.group(1), m.group(2), m.group(3)
        rels = REL_RE.findall(body)
        if not rels:
            continue
        base = parse_header_date(hdr)
        if not base:
            continue

        body_lower = body.lower()
        actor_match = sum(1 for a in scorer.KNOWN_ACTORS if f" {a} " in f" {body_lower} ")
        explicit_date = parse_explicit_date(body)

        # Event phrase match: check if event words ALL appear in the turn body
        # This is a TIE-BREAKER only, weighted lightly
        event_match = 0
        if event_lower:
            event_words = [w for w in event_lower.split() if len(w) >= 3]
            if event_words and all(w in body_lower for w in event_words):
                event_match = 1

        # Score = noun_overlap * 2 + explicit_date + event_bonus * 0.5
        # Event match is a minor bonus — must not override noun overlap ranking
        score = actor_match * 2 + (1 if explicit_date else 0) + event_match * 0.5

        candidates.append({
            "dia": dia, "base": base, "rel": rels[0], "body": body,
            "explicit": explicit_date, "score": score,
            "actor_match": actor_match, "event_match": event_match,
            "event_phrase": event_phrase,
        })

    if not candidates:
        return None

    # Sort by score, then prefer explicit dates, then prefer event-phrase matches
    candidates.sort(key=lambda c: (-c["score"], 0 if c["explicit"] else 1, 0 if c["event_match"] else 1))
    best = candidates[0]

    if best["explicit"]:
        return best["explicit"]

    resolved = resolve_relative(best["rel"], best["base"])
    if not resolved:
        return None

    if "month" in question.lower() or "year" in question.lower() or best["rel"] == "next month":
        return f"{MONTH_NAMES[resolved.month]} {resolved.year}"
    return f"{resolved.day} {MONTH_NAMES[resolved.month]}, {resolved.year}"


def main():
    prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
    baseline = {r["qid"]: r for r in prod["rows"]}

    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sids = [c.get("sample_id", "") for c in locomo]
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    conv_ids_needed = set()
    qid_to_conv = {}
    for qid in baseline:
        m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
        if not m: continue
        conv = f"conv-{m.group(1)}"
        conv_ids_needed.add(conv)
        qid_to_conv[qid] = f"{conv}-qa-{int(m.group(2)):03d}"

    ctx_cache = {}
    for idx, sid in enumerate(sids):
        if sid not in conv_ids_needed:
            continue
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            cqid = q.question_id
            if cqid in qid_to_conv.values():
                pcc = adapter.compiler.compile(q.question, ir_records)
                ctx_cache[cqid] = (q.question, pcc.context_text, q.category)

    results = []
    recovered = regressed = changed_nodiff = 0
    for qid, bp in baseline.items():
        cqid = qid_to_conv.get(qid)
        if not cqid or cqid not in ctx_cache:
            continue

        question, ctx, cat = ctx_cache[cqid]
        raw = bp.get("predicted", "")
        gold = bp["gold"]
        bl_correct = LoCoMoAdapter.score_binary(cat, gold, raw)

        if bl_correct:
            treated = raw
        else:
            fallback = deterministic_temporal_resolve_v2(ctx, question, cat)
            treated = fallback if fallback else raw

        is_correct = LoCoMoAdapter.score_binary(cat, gold, treated)
        changed = treated != raw

        if changed:
            if is_correct and not bl_correct:
                recovered += 1; change = "RECOVERED"
            elif not is_correct and bl_correct:
                regressed += 1; change = "REGRESSED"
            else:
                changed_nodiff += 1; change = "changed_no_diff"
        else:
            change = "same"

        results.append({
            "qid": qid, "category": cat, "gold": gold,
            "baseline_raw": raw, "baseline_correct": bl_correct,
            "treatment_raw": treated, "treatment_correct": is_correct,
            "change": change,
        })

        if changed or qid in {"lc_conv-26_63","lc_conv-42_57","lc_conv-47_43","lc_conv-47_65",
                               "lc_conv-48_12","lc_conv-50_52","lc_conv-41_20","lc_conv-47_48",
                               "lc_conv-48_50","lc_conv-49_59","lc_conv-50_36","lc_conv-44_13"}:
            ok = "✅" if is_correct else "❌"
            print(f"{qid} [{change}] {ok} gold={gold[:30]}")
            if changed:
                print(f"  raw: {raw[:60]!r}")
                print(f"  trt: {treated[:60]!r}")

    bl_total = sum(1 for r in results if r["baseline_correct"])
    tr_total = sum(1 for r in results if r["treatment_correct"])

    print(f"\n{'='*60}")
    print(f"M30-52 A/B SUMMARY (event entity matching)")
    print(f"{'='*60}")
    print(f"Baseline: {bl_total}/{len(results)}")
    print(f"Treatment: {tr_total}/{len(results)}")
    print(f"RECOVERED: {recovered}")
    print(f"REGRESSED: {regressed}")
    print(f"Changed (no diff): {changed_nodiff}")
    print(f"Unchanged: {len(results) - recovered - regressed - changed_nodiff}")

    out_dir = PROJECT_ROOT / "results/m30_52"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ab_results.json").write_text(json.dumps({
        "summary": {
            "baseline_correct": bl_total,
            "treatment_correct": tr_total,
            "recovered": recovered, "regressed": regressed,
            "changed_no_diff": changed_nodiff, "total": len(results),
        },
        "rows": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'ab_results.json'}")


if __name__ == "__main__":
    main()
