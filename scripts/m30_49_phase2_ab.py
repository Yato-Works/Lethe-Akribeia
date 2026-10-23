#!/usr/bin/env python3
"""M30-49 Phase 2: Post-processing with deterministic temporal resolution.

Optimized: load only the conversations needed for Bucket A (precompute + cache),
no LLM calls — deterministic post-processing only.

Baseline: production reader outputs from M30-43 saved files
Treatment: same outputs + deterministic temporal resolution post-processor
"""
import json, re, sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader
# Production uses LoCoMoAdapter.score_binary, NOT OfficialLongMemEvalGrader
from functools import partial

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}
REL_RE = re.compile(r"\b(yesterday|tomorrow|the next day|the day before|next month|last month|"
                    r"next week|last week|currently|one year ago|two years ago|three years ago)\b", re.I)
LINE_RE = re.compile(r"^\[SPEAKER:[^\]]+\]\s*\[(D\d+:\d+) on ([^\]]+)\](.*)$")
STOP = set("the and for are was were did does has have had will would could should can may might "
           "must this that these those when where what which who why how not all any both each "
           "with from into about after before during between through over under".split())


def parse_header(sd: str):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", sd or "")
    if m and m.group(2).lower() in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None


def resolve_relative(expr: str, base: datetime):
    if not base: return None
    e = expr.lower()
    if "yesterday" in e or "the day before" in e: return base - timedelta(days=1)
    if "tomorrow" in e or "the next day" in e: return base + timedelta(days=1)
    if "next month" in e:
        y, mo = (base.year + (1 if base.month == 12 else 0), 1 if base.month == 12 else base.month + 1)
        return datetime(y, mo, 1)
    if "last month" in e:
        y, mo = (base.year - (1 if base.month == 1 else 0), 12 if base.month == 1 else base.month - 1)
        return datetime(y, mo, 1)
    if "next week" in e: return base + timedelta(days=7)
    if "last week" in e: return base - timedelta(days=7)
    if "currently" in e: return base
    return None


def find_relevant_turn(ctx: str, question: str):
    """Find the turn in ctx whose body has best noun-overlap with question."""
    qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower())
              if w not in {"when", "what", "where", "which", "who", "whose", "why", "how"})
    best = None
    best_score = -1
    for m in LINE_RE.finditer(ctx):
        dia, hdr, body = m.group(1), m.group(2), m.group(3)
        nouns = set(w for w in re.findall(r"\b[a-z]{4,}\b", body.lower()) if w not in STOP)
        overlap = len(qkw & nouns)
        base = parse_header(hdr)
        if base and overlap > best_score:
            best = (dia, base, body)
            best_score = overlap
    return best


def post_process_temporal(raw_answer: str, ctx: str, question: str) -> str:
    """Deterministic: resolve relative temporal expressions in the answer."""
    rels = REL_RE.findall(raw_answer)
    if not rels:
        return raw_answer
    turn = find_relevant_turn(ctx, question)
    if not turn:
        return raw_answer
    dia, base_date, body = turn
    resolved = resolve_relative(rels[0], base_date)
    if not resolved:
        return raw_answer
    # Granularity from question
    if "month" in question.lower() or "year" in question.lower():
        return f"{MONTH_NAMES[resolved.month]} {resolved.year}"
    return f"{resolved.day} {MONTH_NAMES[resolved.month]}, {resolved.year}"


def main():
    # Load production baseline
    prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
    baseline = {r["qid"]: r for r in prod["rows"]}

    # Determine which conversations we need
    conv_ids = set()
    qid_to_conv_qid = {}
    for qid in baseline:
        m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
        if not m: continue
        conv = f"conv-{m.group(1)}"
        conv_ids.add(conv)
        qid_to_conv_qid[qid] = f"{conv}-qa-{int(m.group(2)):03d}"

    print(f"Need conversations: {sorted(conv_ids)}")

    # Load only needed conversations
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sids = [c.get("sample_id", "") for c in locomo]
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    # Build a map: conv_qid -> (question, ir_records, compiled context)
    ctx_cache = {}
    for idx, sid in enumerate(sids):
        if sid not in conv_ids:
            continue
        print(f"Loading {sid}...")
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            cqid = q.question_id
            if cqid in qid_to_conv_qid.values():
                pcc = adapter.compiler.compile(q.question, ir_records)
                ctx_cache[cqid] = (q.question, pcc.context_text, q.category)

    print(f"Contexts cached: {len(ctx_cache)}")

    # Build qid -> conv_qid map
    conv_qid_map = {}
    for qid, cqid in qid_to_conv_qid.items():
        conv_qid_map[qid] = cqid

    grader = OfficialLongMemEvalGrader()
    results = []
    for qid, bp in baseline.items():
        cqid = conv_qid_map.get(qid)
        if not cqid or cqid not in ctx_cache:
            print(f"[skip] {qid}: no context")
            continue

        question, ctx, cat = ctx_cache[cqid]
        raw = bp.get("reader_output", bp.get("predicted", ""))
        gold = bp["gold"]
        baseline_correct = bp["correct"]

        # Treatment: post-process
        treated = post_process_temporal(raw, ctx, question)

        # Use the SAME scorer production uses: LoCoMoAdapter.score_binary
        # (NOT OfficialLongMemEvalGrader, which is a different, stricter scorer)
        is_correct = LoCoMoAdapter.score_binary(cat, gold, treated)
        bl_correct = LoCoMoAdapter.score_binary(cat, gold, raw)

        change = "same"
        if treated != raw:
            if is_correct and not baseline_correct:
                change = "RECOVERED"
            elif not is_correct and baseline_correct:
                change = "REGRESSED"
            else:
                change = "changed_no_score_diff"

        results.append({
            "qid": qid,
            "category": cat,
            "question": question,
            "gold": gold,
            "baseline_raw": raw,
            "baseline_correct": baseline_correct,
            "treatment_raw": treated,
            "treatment_correct": is_correct,
            "change": change,
        })

    baseline_correct = sum(1 for r in results if r["baseline_correct"])
    treatment_correct = sum(1 for r in results if r["treatment_correct"])
    recovered = sum(1 for r in results if r["change"] == "RECOVERED")
    regressed = sum(1 for r in results if r["change"] == "REGRESSED")
    changed_no_diff = sum(1 for r in results if r["change"] == "changed_no_score_diff")

    print(f"\n{'='*60}")
    print("M30-49 A/B SUMMARY")
    print("="*60)
    print(f"Total: {len(results)}")
    print(f"Baseline (production): {baseline_correct}/28")
    print(f"Treatment (post-processed): {treatment_correct}/28")
    print(f"RECOVERED:   {recovered}")
    print(f"REGRESSED:   {regressed}")
    print(f"Changed (no score diff): {changed_no_diff}")
    print(f"Unchanged:   {len(results) - recovered - regressed - changed_no_diff}")

    for r in results:
        if r["change"] != "same":
            print(f"\n  {r['qid']} {r['change']}: '{r['baseline_raw'][:50]}' -> '{r['treatment_raw'][:50]}'")

    out_dir = PROJECT_ROOT / "results/m30_49"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ab_results.json").write_text(json.dumps({
        "summary": {
            "baseline_correct": baseline_correct,
            "treatment_correct": treatment_correct,
            "recovered": recovered,
            "regressed": regressed,
            "changed_no_diff": changed_no_diff,
            "total": len(results),
        },
        "rows": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'ab_results.json'}")


if __name__ == "__main__":
    main()