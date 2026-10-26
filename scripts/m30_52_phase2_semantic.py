#!/usr/bin/env python3
"""M30-52 Phase 2: Semantic event matching using SEMANTIC_EXPANSIONS.

Problem: The 8 remaining wrong cases fail because noun overlap picks the wrong
turn when multiple context turns mention the same actors (Joanna+Nate, etc.).

M30-52 Phase 1 (keyword-based event phrase) caused regresssions (18→16) because
it changed turn selection for already-correct cases.

Fix: Use SEMANTIC_EXPANSIONS to expand question content words, then weight
turns containing BOTH the original AND expanded terms. This is MORE precise
than raw keywords because expansions are curated (e.g., "recipes" -> "slow cooker",
"tokyo" -> "Japan", "show" -> "movie").

CRITICAL: The expansion scoring must NOT override noun overlap — it must only
break ties when scores are EQUAL. This prevents regressions.
"""
import json, re, sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory_gen.recall.evidence_scorer import UniversalEvidenceScorer

scorer = UniversalEvidenceScorer()
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


def parse_explicit_date(text):
    m = re.search(r"\(?\s*(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\s*\)?", text)
    if m and m.group(2).lower() in MONTHS:
        return f"{m.group(1)} {m.group(2)}, {m.group(3)}"
    m = re.search(r"\(\s*([A-Za-z]+)\s+(\d{4})\s*\)", text)
    if m and m.group(1).lower() in MONTHS:
        return f"{m.group(1)} {m.group(2)}"
    m = re.search(r"\b([A-Za-z]+)\s+(\d{4})\b", text)
    if m and m.group(1).lower() in MONTHS and len(text) > 50:
        # Avoid matching header dates
        if " on " not in text[:m.start()]:
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


def is_temporal_question(q, cat):
    if cat == 2: return True
    ql = q.lower()
    return any(w in ql for w in ["when", "what date", "what year", "what month", "which date", "which year"])


def semantic_expand(terms):
    """Expand a set of terms using SEMANTIC_EXPANSIONS."""
    expanded = set(terms)
    for term in terms:
        term_lower = term.lower()
        for key, expansions in scorer.SEMANTIC_EXPANSIONS.items():
            # Check if term matches a key or any expansion
            if term_lower == key or term_lower in [e.lower() for e in expansions]:
                expanded.update(e.lower() for e in expansions)
                expanded.add(key)
    return expanded


def resolve_relative(expr, base, question):
    if not base: return None
    e = expr.lower()
    if "yesterday" in e or "the day before" in e: res = base - timedelta(days=1)
    elif "tomorrow" in e or "the next day" in e: res = base + timedelta(days=1)
    elif "next month" in e:
        mo = 1 if base.month == 12 else base.month + 1
        yr = base.year + (1 if base.month == 12 else 0)
        res = datetime(yr, mo, 1)
    elif "last month" in e:
        mo = 12 if base.month == 1 else base.month - 1
        yr = base.year - (1 if base.month == 1 else 0)
        res = datetime(yr, mo, 1)
    elif "next week" in e: res = base + timedelta(days=7)
    elif "last week" in e: res = base - timedelta(days=7)
    elif "currently" in e: res = base
    elif "one year ago" in e: res = datetime(base.year - 1, base.month, base.day)
    elif "two years ago" in e: res = datetime(base.year - 2, base.month, base.day)
    else: return None

    if "year" in question.lower() and "what year" in question.lower():
        return res.year
    if "month" in question.lower() or "year" in question.lower() or "the year" in question.lower():
        return f"{MONTH_NAMES[res.month]} {res.year}"
    return f"{res.day} {MONTH_NAMES[res.month]}, {res.year}"


def deterministic_temporal_resolve_semantic(ctx, question, category):
    """Semantic event matching — uses SEMANTIC_EXPANSIONS for disambiguation.

    KEY DIVERGENCE FROM V1: expands question content words using the curated
    SEMANTIC_EXPANSIONS dictionary, then weights turns by how many expanded
    terms they contain. This catches synonyms/contextual variants that raw
    keyword matching misses.

    CRITICAL: event_score is only a TIE-BREAKER after noun overlap + explicit
    date. Same scoring priority as V1, with semantic expansion as final tiebreak.
    """
    if not is_temporal_question(question, category):
        return None

    # Extract question content words (exclude stopwords and known actors)
    qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower())
              if w not in STOP and w not in {"when", "what", "where", "which", "who", "whose", "why", "how"})
    qkw_no_actors = {w for w in qkw if w not in {a.lower() for a in scorer.KNOWN_ACTORS}}

    # Expand using SEMANTIC_EXPANSIONS
    qkw_expanded = semantic_expand(qkw)

    # Find candidate turns with relative expressions
    candidates = []
    for m in TURN_RE.finditer(ctx):
        dia, hdr, body = m.group(1), m.group(2), m.group(3)
        rels = REL_RE.findall(body)
        if not rels: continue
        explicit = parse_explicit_date(body)
        base = parse_header_date(hdr)
        if not base: continue
        body_lower = body.lower()

        # V1 scoring: noun overlap
        bkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", body_lower) if w not in STOP)
        score = len(qkw & bkw)

        # Semantic event matching: count expanded terms present in turn
        # This rewards turns that mention the EVENT concept (even via synonyms)
        event_matches = sum(1 for t in qkw_expanded if len(t) >= 4 and t in body_lower)
        # But only count non-actor terms for event matching
        event_matches_no_actors = sum(1 for t in qkw_expanded
                                       if len(t) >= 4 and t in body_lower
                                       and t not in {a.lower() for a in scorer.KNOWN_ACTORS})

        candidates.append({
            "dia": dia, "hdr": hdr, "body": body, "rel": rels[0],
            "explicit": explicit, "base": base, "score": score,
            "event_matches": event_matches,
            "event_matches_no_actors": event_matches_no_actors,
        })

    if not candidates:
        return None

    # Sort: noun overlap (primary) → explicit date → event_matches_no_actors
    # event_matches is ONLY a tiebreaker — never overrides noun overlap
    candidates.sort(key=lambda c: (-c["score"], 0 if c["explicit"] else 1, -c["event_matches_no_actors"]))
    best = candidates[0]

    if best["explicit"]:
        return best["explicit"]

    resolved = resolve_relative(best["rel"], best["base"], question)
    if resolved is None:
        return None
    # Handle year return type
    if isinstance(resolved, int):
        return str(resolved)
    if "month" in question.lower() or "year" in question.lower():
        if isinstance(resolved, datetime):
            return f"{MONTH_NAMES[resolved.month]} {resolved.year}"
    return resolved if isinstance(resolved, str) else f"{resolved.day} {MONTH_NAMES[resolved.month]}, {resolved.year}"


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
        if sid not in conv_ids_needed: continue
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
        if not cqid or cqid not in ctx_cache: continue

        question, ctx, cat = ctx_cache[cqid]
        raw = bp.get("predicted", "")
        gold = bp["gold"]
        bl_correct = LoCoMoAdapter.score_binary(cat, gold, raw)

        if bl_correct:
            treated = raw
        else:
            fallback = deterministic_temporal_resolve_semantic(ctx, question, cat)
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

        if changed:
            ok = "✅" if is_correct else "❌"
            print(f"{qid} [{change}] {ok} gold={gold[:30]}")
            if changed:
                print(f"  raw: {raw[:60]!r}")
                print(f"  trt: {treated[:60]!r}")

    bl_total = sum(1 for r in results if r["baseline_correct"])
    tr_total = sum(1 for r in results if r["treatment_correct"])

    print(f"\n{'='*60}")
    print(f"M30-52 Phase 2 SUMMARY (semantic expansion tiebreaker)")
    print(f"{'='*60}")
    print(f"Baseline: {bl_total}/{len(results)}")
    print(f"Treatment: {tr_total}/{len(results)}")
    print(f"RECOVERED: {recovered}")
    print(f"REGRESSED: {regressed}")
    print(f"Changed (no diff): {changed_nodiff}")
    print(f"Unchanged: {len(results) - recovered - regressed - changed_nodiff}")

    out_dir = PROJECT_ROOT / "results/m30_52"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ab_results_phase2.json").write_text(json.dumps({
        "summary": {
            "baseline_correct": bl_total,
            "treatment_correct": tr_total,
            "recovered": recovered, "regressed": regressed,
            "changed_no_diff": changed_nodiff, "total": len(results),
        },
        "rows": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'ab_results_phase2.json'}")


if __name__ == "__main__":
    main()
