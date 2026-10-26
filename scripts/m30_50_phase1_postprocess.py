#!/usr/bin/env python3
"""M30-50 Phase 1: Deterministic temporal resolution post-processing.

Fix the LINE_RE to handle both [SPEAKER:X] [D.. on ...] and [D.. on ...] formats.
Then test if post-processing can recover mechanism-A cases.
"""
import json, re, sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}

# Match [SPEAKER:X] [D.. on ...](...) or just [D.. on ...](...)
TURN_RE = re.compile(
    r"(?:\[SPEAKER:[^\]]+\]\s*)?\[(D\d+:\d+)\s+on\s+([^\]]+)\](.*)"
)


def parse_header(sd):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", sd or "")
    if m and m.group(2).lower() in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None


def resolve(expr, base):
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
    if "currently" in e: return base
    return None


def find_tomorrow_turns(ctx):
    """Find all context turns containing temporal references like tomorrow/next month/yesterday."""
    turns = []
    for line in ctx.split("\n"):
        m = TURN_RE.match(line.strip())
        if not m:
            continue
        dia, hdr, body = m.group(1), m.group(2), m.group(3)
        if re.search(r"\b(yesterday|tomorrow|next month|last month|next week|last week|currently)\b", body, re.I):
            base = parse_header(hdr)
            turns.append((dia, hdr, body, base))
    return turns


def post_process_temporal(raw_answer, ctx, question):
    """Resolve relative temporal expressions in reader output using context."""
    # Find relative expressions in the answer
    rels = re.findall(r"\b(yesterday|tomorrow|the next day|the day before|next month|last month|"
                      r"next week|last week|currently)\b", raw_answer, re.I)
    if not rels:
        return raw_answer

    # Find candidate turns in context with relative expressions
    turns = find_tomorrow_turns(ctx)
    if not turns:
        return raw_answer

    # Score turns by question-content overlap
    qkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower())
              if w not in {"when", "what", "where", "which", "who", "whose", "why", "how"})
    STOP = set("the and for are was were did does has have had will would could should can may might "
               "must this that these those when where what which who why how not all any both each "
               "with from into about after before during between through over under".split())

    best_turn = None
    best_score = -1
    for dia, hdr, body, base in turns:
        bkw = set(w for w in re.findall(r"\b[a-z]{4,}\b", body.lower()) if w not in STOP)
        score = len(qkw & bkw)
        if score > best_score:
            best_score = score
            best_turn = (dia, hdr, body, base)

    if not best_turn:
        return raw_answer

    dia, hdr, body, base = best_turn
    # Resolve the first relative expression
    res = resolve(rels[0], base)
    if not res:
        return raw_answer

    q_lower = question.lower()
    if "month" in q_lower or "year" in q_lower:
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
    recovered = regressed = changed_nodiff = unchanged = 0
    for qid, bp in baseline.items():
        cqid = qid_to_conv.get(qid)
        if not cqid or cqid not in ctx_cache:
            continue

        question, ctx, cat = ctx_cache[cqid]
        raw = bp.get("predicted", "")
        gold = bp["gold"]
        baseline_correct = bp["correct"]

        treated = post_process_temporal(raw, ctx, question)

        # Re-grade baseline and treatment
        bl_correct = LoCoMoAdapter.score_binary(cat, gold, raw)
        is_correct = LoCoMoAdapter.score_binary(cat, gold, treated)

        changed = treated != raw
        if changed:
            if is_correct and not bl_correct:
                recovered += 1
                change = "RECOVERED"
            elif not is_correct and bl_correct:
                regressed += 1
                change = "REGRESSED"
            else:
                changed_nodiff += 1
                change = "changed_no_diff"
        else:
            unchanged += 1
            change = "same"

        results.append({
            "qid": qid, "category": cat, "gold": gold,
            "baseline_raw": raw, "baseline_correct": bl_correct,
            "treatment_raw": treated, "treatment_correct": is_correct,
            "change": change,
        })

        if changed:
            print(f"{qid}: {change}")
            print(f"  raw: {raw[:80]!r}")
            print(f"  trt: {treated[:80]!r}")
            print()

    bl_total = sum(1 for r in results if r["baseline_correct"])
    tr_total = sum(1 for r in results if r["treatment_correct"])

    print(f"{'='*60}")
    print(f"M30-50 A/B SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline: {bl_total}/{len(results)}")
    print(f"Treatment: {tr_total}/{len(results)}")
    print(f"RECOVERED: {recovered}")
    print(f"REGRESSED: {regressed}")
    print(f"Changed (no diff): {changed_nodiff}")
    print(f"Unchanged: {unchanged}")

    out_dir = PROJECT_ROOT / "results/m30_50"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ab_results.json").write_text(json.dumps({
        "summary": {
            "baseline_correct": bl_total,
            "treatment_correct": tr_total,
            "recovered": recovered, "regressed": regressed,
            "changed_no_diff": changed_nodiff, "unchanged": unchanged,
            "total": len(results),
        },
        "rows": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'ab_results.json'}")


if __name__ == "__main__":
    main()
