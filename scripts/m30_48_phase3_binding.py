#!/usr/bin/env python3
"""M30-48 Phase 3: Can we pick the right turn WITHOUT gold?

For mechanism A: the reader emits a relative expression (or abstains).
The context contains many dated turns. To resolve correctly we must bind the
expression to the RIGHT turn's header date.

Non-oracle binding rule under test:
  - take the reader's relative expression (or the question's)
  - among context turns whose BODY contains that expression, pick the one
    whose body best matches the question's content words
  - resolve against that turn's header date

No gold, no evidence IDs. If this binds correctly on the A cases, it is a
candidate fix. If not, mechanism A is not solvable without retrieval changes.
"""
import json, re
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

M = json.loads((PROJECT_ROOT / "results/m30_48/mechanisms.json").read_text(encoding="utf-8"))
MONTHS = {m.lower(): i for i, m in enumerate([
    "january","february","march","april","may","june","july","august",
    "september","october","november","december"], start=1)}
LINE_RE = re.compile(r"^\[SPEAKER:[^\]]+\]\s*\[(D\d+:\d+) on ([^\]]+)\](.*)$")
REL_RE = re.compile(r"\b(yesterday|tomorrow|the next day|the day before|next month|last month|"
                    r"next week|last week|currently)\b", re.I)
STOP = set("the and for are was were did does has have had will would could should can may might "
           "must this that these those when where what which who why how not all any both each "
           "with from into about after before during between through over under".split())


def parse_header(sd):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", sd or "")
    if m and m.group(2).lower() in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None


def resolve(expr, base):
    e = expr.lower()
    if "yesterday" in e or "the day before" in e:
        return base - timedelta(days=1)
    if "tomorrow" in e or "the next day" in e:
        return base + timedelta(days=1)
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


def main():
    # We need the compiled context for each A case. Rebuild via the adapter.
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sids = [c.get("sample_id", "") for c in locomo]
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    A = set(M["mech"]["A_evidence_has_rel_reader_must_resolve"])
    rows = {r["qid"]: r for r in M["rows"]}

    print("=" * 78)
    print("M30-48 Phase 3: Non-oracle turn binding for mechanism A")
    print("=" * 78)
    print()

    bound_ok = 0
    for idx, sid in enumerate(sids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            m = re.match(r"conv-(\d+)-qa-(\d+)", q.question_id)
            if not m:
                continue
            qid = f"lc_conv-{m.group(1)}_{int(m.group(2))}"
            if qid not in A:
                continue
            pcc = adapter.compiler.compile(q.question, ir_records)
            ctx = pcc.context_text

            # candidate turns: body contains a relative expression
            cands = []
            for ln in ctx.split("\n"):
                lm = LINE_RE.match(ln)
                if not lm:
                    continue
                dia, hdr, body = lm.group(1), lm.group(2), lm.group(3)
                rels = REL_RE.findall(body)
                if rels:
                    cands.append((dia, hdr, body, rels))

            # score by question overlap
            qk = set(w for w in re.findall(r"\b[a-z]{4,}\b", q.question.lower()) if w not in STOP)
            scored = []
            for dia, hdr, body, rels in cands:
                bk = set(w for w in re.findall(r"\b[a-z]{4,}\b", body.lower()) if w not in STOP)
                scored.append((len(qk & bk), dia, hdr, rels))
            scored.sort(key=lambda x: -x[0])

            gold = rows[qid]["gold"]
            gold_ids = {t[0] for t in rows[qid]["ev_texts"]}
            if not scored:
                print(f"{qid}: NO candidate turns with a relative expression")
                continue
            top = scored[0]
            base = parse_header(top[2])
            res = resolve(top[3][0], base) if base else None
            hit = top[1] in gold_ids
            if hit:
                bound_ok += 1
            print(f"{qid}  gold={gold!r}")
            print(f"  bound turn={top[1]} (gold_ids={sorted(gold_ids)})  HIT={hit}")
            print(f"  base={base.date() if base else None} rel='{top[3][0]}' -> {res.date() if res else None}")
            print(f"  top-3 candidates: {[(s,d) for s,d,_,_ in scored[:3]]}")
            print()

    print("=" * 78)
    print(f"Non-oracle binding hit the gold turn: {bound_ok}/{len(A)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
