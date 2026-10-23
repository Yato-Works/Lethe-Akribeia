#!/usr/bin/env python3
"""M30-48 Phase 4b: Correct metric — does the RESOLVED DATE match gold?

Phase 3/4 used "bound turn == gold turn" which is too strict: a sibling turn
in the same session has the SAME header date, so resolving against it gives
the same answer. Measure the outcome that matters: resolved date vs gold.
"""
import json, re, sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

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


def matches_gold(gold, res):
    if not res: return False
    g = gold.lower()
    mon = res.strftime("%B").lower()
    if mon in g and str(res.year) in g:
        # if gold has a day, require it
        dm = re.search(r"\b(\d{1,2})\b", g.replace(str(res.year), ""))
        if dm and int(dm.group(1)) <= 31:
            return str(res.day) in g or dm.group(1) == str(res.day)
        return True
    if re.fullmatch(r"\d{4}", g.strip()):
        return str(res.year) in g
    return False


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sids = [c.get("sample_id", "") for c in locomo]
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
    A = set(M["mech"]["A_evidence_has_rel_reader_must_resolve"])
    rows = {r["qid"]: r for r in M["rows"]}

    print("=" * 78)
    print("M30-48 Phase 4b: does the RESOLVED DATE match gold?")
    print("=" * 78)
    print()

    noun_ok = act_ok = 0
    total = 0
    for idx, sid in enumerate(sids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            m = re.match(r"conv-(\d+)-qa-(\d+)", q.question_id)
            if not m: continue
            qid = f"lc_conv-{m.group(1)}_{int(m.group(2))}"
            if qid not in A: continue
            pcc = adapter.compiler.compile(q.question, ir_records)
            ctx = pcc.context_text
            gold = rows[qid]["gold"]

            ql = q.question.lower()
            qk = set(w for w in re.findall(r"\b[a-z]{4,}\b", ql) if w not in STOP)
            qent = set(w.lower() for w in re.findall(r"\b([A-Z][a-z]{2,})\b", q.question))

            cands = []
            for ln in ctx.split("\n"):
                lm = LINE_RE.match(ln)
                if not lm: continue
                dia, hdr, body = lm.group(1), lm.group(2), lm.group(3)
                rels = REL_RE.findall(body)
                if not rels: continue
                bl = body.lower()
                noun = len(qk & set(w for w in re.findall(r"\b[a-z]{4,}\b", bl) if w not in STOP))
                ent = len(qent & set(w.lower() for w in re.findall(r"\b([A-Z][a-z]{2,})\b", body)))
                cands.append((dia, hdr, rels, noun, ent))
            if not cands: continue
            total += 1

            # noun-best
            nb = max(cands, key=lambda c: (c[3], c[4]))
            nres = resolve(nb[2][0], parse_header(nb[1]))
            # entity-first
            eb = max(cands, key=lambda c: (c[4], c[3]))
            eres = resolve(eb[2][0], parse_header(eb[1]))

            nok = matches_gold(gold, nres)
            eok = matches_gold(gold, eres)
            noun_ok += nok
            act_ok += eok
            print(f"{qid} gold={gold!r}")
            print(f"  noun-best {nb[0]} @ {nb[1]} -> {nres.date() if nres else None}  MATCH={nok}")
            print(f"  ent-best  {eb[0]} @ {eb[1]} -> {eres.date() if eres else None}  MATCH={eok}")
            print()

    print("=" * 78)
    print(f"noun-best  resolved-date matches gold: {noun_ok}/{total}")
    print(f"entity-best resolved-date matches gold: {act_ok}/{total}")
    print("=" * 78)


if __name__ == "__main__":
    main()
