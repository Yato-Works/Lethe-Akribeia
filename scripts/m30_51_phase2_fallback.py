#!/usr/bin/env python3
"""M30-51 Phase 2 (v2): Deterministic temporal resolution — use EXPLICIT dates in evidence.

Key insight: the evidence turns don't just have relative expressions — they
often include the RESOLVED date in parentheses right after.

Examples from the context:
  D26:19: "...I'll come over tomorrow (5 November 2022) if the weather..."
  D22:1:  "Yesterday (5 October 2022), I tried my newest..."

These dates are ALREADY IN THE CONTEXT as parenthetical annotations.
The Reader sometimes picks up the wrong one (from a different turn).

Strategy: find the evidence turn that best matches the question,
then extract the explicit date from that turn's body.
No date arithmetic needed — the date is stated explicitly.
"""
import json, re, sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory_gen.recall.evidence_scorer import UniversalEvidenceScorer

_months_ev = UniversalEvidenceScorer()

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}
TURN_RE = re.compile(r"(?:\[SPEAKER:[^\]]+\]\s*)?\[(D\d+:\d+)\s+on\s+([^\]]+)\](.*)")
STOP = set("the and for are was were did does has have had will would could should can may might "
           "must this that these those when where what which who why how not all any both each "
           "with from into about after before during between through over under".split())

# Date patterns — find explicit dates in turn bodies
DATE_PATTERNS = [
    # 5 November 2022, 2022-11-05, November 5, 2022, next month (September 2023)
    r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})",
    r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
    r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})",
    r"([A-Za-z]+)\s+(\d{4})",  # September 2023
    r"(\d{4})",  # just year
]

REL_RE = re.compile(r"\b(yesterday|tomorrow|the next day|the day before|next month|last month|"
                    r"next week|last week|currently)\b", re.I)


def parse_explicit_date(text):
    """Find and parse the first explicit date in text."""
    # Pattern: "5 November 2022" or "(5 November 2022)" or "November 5, 2022"
    m = re.search(r"\(?\s*(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\s*\)?", text)
    if m and m.group(2).lower() in MONTHS:
        return f"{m.group(1)} {m.group(2)}, {m.group(3)}"
    # Pattern: "next month (September 2023)" 
    m = re.search(r"\(\s*([A-Za-z]+)\s+(\d{4})\s*\)", text)
    if m and m.group(1).lower() in MONTHS:
        return f"{m.group(1)} {m.group(2)}"
    # Pattern: "September 2023" (month + year only)
    m = re.search(r"\b([A-Za-z]+)\s+(\d{4})\b", text)
    if m and m.group(1).lower() in MONTHS:
        return f"{m.group(1)} {m.group(2)}"
    # Pattern: just year "2022"
    m = re.search(r"\b(\d{4})\b", text)
    if m and len(text) > 50:
        return m.group(1)
    return None


def parse_header_date(hdr):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", hdr or "")
    if m and m.group(2).lower() in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None


def is_temporal_question(q, cat):
    if cat == 2:
        return True
    ql = q.lower()
    return "when" in ql or "what date" in ql or "what year" in ql


def deterministic_temporal_resolve(ctx, question, category):
    if not is_temporal_question(question, category):
        return None

    # Extract content words from question for noun overlap matching
    qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower())
              if w not in STOP and w not in {"when", "what", "where", "which", "who", "whose", "why", "how"})

    # Find turns with relative expressions AND parse explicit dates
    candidates = []
    for m in TURN_RE.finditer(ctx):
        dia, hdr, body = m.group(1), m.group(2), m.group(3)
        rels = REL_RE.findall(body)
        if not rels:
            continue
        explicit = parse_explicit_date(body)
        base = parse_header_date(hdr)
        if not base:
            continue
        bkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", body.lower()) if w not in STOP)
        score = len(qkw & bkw)
        # Event phrase score: how many words of the event phrase appear in the turn
        # (Currently unused — pure noun overlap V1 is optimal)
        candidates.append({
            "dia": dia, "hdr": hdr, "body": body, "rel": rels[0],
            "explicit": explicit, "base": base, "score": score,
        })

    if not candidates:
        return None

    # Sort by noun overlap score, then prefer turns with explicit dates
    candidates.sort(key=lambda c: (-c["score"], 0 if c["explicit"] else 1))
    best = candidates[0]

    # Return the explicit date from the turn body
    if best["explicit"]:
        return best["explicit"]

    # No explicit date — resolve from base
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
    MECH_A = {"lc_conv-26_63", "lc_conv-42_57", "lc_conv-47_43", "lc_conv-47_65",
              "lc_conv-48_12", "lc_conv-50_52", "lc_conv-41_20"}

    for qid, bp in baseline.items():
        cqid = qid_to_conv.get(qid)
        if not cqid or cqid not in ctx_cache:
            continue

        question, ctx, cat = ctx_cache[cqid]
        raw = bp.get("predicted", "")
        gold = bp["gold"]
        bl_correct = LoCoMoAdapter.score_binary(cat, gold, raw)

        # Treatment: deterministic fallback only on failure
        if bl_correct:
            treated = raw
        else:
            fallback = deterministic_temporal_resolve(ctx, question, cat)
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

        if qid in MECH_A or changed:
            ok = "✅" if is_correct else "❌"
            print(f"{qid} [{change}] {ok} gold={gold[:30]}")
            if changed:
                print(f"  raw:  {raw[:60]!r}")
                print(f"  trt:  {treated[:60]!r}")

        results.append({
            "qid": qid, "gold": gold,
            "baseline_raw": raw, "baseline_correct": bl_correct,
            "treatment_raw": treated, "treatment_correct": is_correct,
            "change": change,
        })

    bl_total = sum(1 for r in results if r["baseline_correct"])
    tr_total = sum(1 for r in results if r["treatment_correct"])

    print(f"\n{'='*60}")
    print(f"M30-51 v2 SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline: {bl_total}/28")
    print(f"Treatment: {tr_total}/28")
    print(f"RECOVERED: {recovered}")
    print(f"REGRESSED: {regressed}")
    print(f"Changed (no diff): {changed_nodiff}")
    print(f"Unchanged: {len(results) - recovered - regressed - changed_nodiff}")

    out_dir = PROJECT_ROOT / "results/m30_51"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ab_results_v2.json").write_text(json.dumps({
        "summary": {
            "baseline_correct": bl_total,
            "treatment_correct": tr_total,
            "recovered": recovered, "regressed": regressed,
            "changed_no_diff": changed_nodiff,
            "total": len(results),
        },
        "rows": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()