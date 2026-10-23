#!/usr/bin/env python3
"""M30-48 Phase 4: Does action/verb matching improve binding?

Phase 3 used noun-overlap and got 3/7. Hypothesis: the question's ACTION
(attend / visit / receive / start / move / meet) is the discriminating signal,
and current scoring ignores it.

Test: score candidate turns by ACTION match + entity match, compare binding.

No gold in the scoring. Gold only checks the outcome.
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

# action verb families
ACTIONS = {
    "attend": {"attend","attended","go","went","going","show","showed","show up"},
    "visit": {"visit","visited","visiting","went","come","came"},
    "receive": {"receive","received","got","get","letter","gift"},
    "start": {"start","started","starting","begin","began","took","take","joined","join"},
    "move": {"move","moved","moving","place","house","new place"},
    "meet": {"meet","met","meeting","met up"},
    "buy": {"buy","bought","purchase","purchased"},
    "plan": {"plan","planned","planning","organize","organizing"},
}


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


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sids = [c.get("sample_id", "") for c in locomo]
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    A = set(M["mech"]["A_evidence_has_rel_reader_must_resolve"])
    rows = {r["qid"]: r for r in M["rows"]}

    print("=" * 78)
    print("M30-48 Phase 4: action+entity binding vs noun-only binding")
    print("=" * 78)
    print()

    noun_hit = act_hit = 0
    for idx, sid in enumerate(sids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        for q in questions:
            m = re.match(r"conv-(\d+)-qa-(\d+)", q.question_id)
            if not m: continue
            qid = f"lc_conv-{m.group(1)}_{int(m.group(2))}"
            if qid not in A: continue
            pcc = adapter.compiler.compile(q.question, ir_records)
            ctx = pcc.context_text
            gold_ids = {t[0] for t in rows[qid]["ev_texts"]}

            # question action family
            ql = q.question.lower()
            q_actions = set()
            for fam, words in ACTIONS.items():
                if any(w in ql for w in words):
                    q_actions |= words
            qk = set(w for w in re.findall(r"\b[a-z]{4,}\b", ql) if w not in STOP)
            # question entity (capitalized)
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
                act = len(q_actions & set(re.findall(r"\b[a-z']+\b", bl)))
                ent = len(qent & set(w.lower() for w in re.findall(r"\b([A-Z][a-z]{2,})\b", body)))
                cands.append((dia, hdr, rels, noun, act, ent))

            if not cands:
                continue
            noun_best = max(cands, key=lambda c: (c[3], c[4], c[5]))[0]
            act_best = max(cands, key=lambda c: (c[4], c[5], c[3]))[0]
            noun_ok = noun_best in gold_ids
            act_ok = act_best in gold_ids
            noun_hit += noun_ok
            act_hit += act_ok
            print(f"{qid}  gold={sorted(gold_ids)}")
            print(f"  noun-only -> {noun_best}  HIT={noun_ok}")
            print(f"  action+ent-> {act_best}  HIT={act_ok}")
            print()

    print("=" * 78)
    print(f"noun-only binding:  {noun_hit}/7")
    print(f"action+entity:      {act_hit}/7")
    print("=" * 78)


if __name__ == "__main__":
    main()
