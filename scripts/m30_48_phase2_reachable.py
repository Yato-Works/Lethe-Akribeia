#!/usr/bin/env python3
"""M30-48 Phase 2: Can mechanism A be solved deterministically?

For the 7 mechanism-A failures: the evidence turn has a header date and a
relative expression. Test whether resolving the expression against the
header date yields the gold — WITHOUT using gold to pick the turn.

Two questions:
  1. Is the gold reachable by the rule? (necessary condition)
  2. Can we identify the right evidence turn WITHOUT gold? (sufficiency)

Q1 here; Q2 uses only the reader's own output + the context.
"""
import json, re
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

M = json.loads((PROJECT_ROOT / "results/m30_48/mechanisms.json").read_text(encoding="utf-8"))
MONTHS = {m.lower(): i for i, m in enumerate([
    "january","february","march","april","may","june","july","august",
    "september","october","november","december"], start=1)}


def parse_header(sd):
    """'3:19 pm on 28 August, 2023' -> date"""
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
    if "next week" in e:
        return base + timedelta(days=7)
    if "last week" in e:
        return base - timedelta(days=7)
    if "next sunday" in e or "next monday" in e or "next tuesday" in e or \
       "next wednesday" in e or "next thursday" in e or "next friday" in e or "next saturday" in e:
        wd = {"sunday":6,"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5}
        target = next(k for k in wd if k in e)
        ahead = (wd[target] - base.weekday()) % 7 or 7
        return base + timedelta(days=ahead)
    if "currently" in e or "just around the corner" in e:
        return base
    return None


def gold_reachable(gold, resolved):
    if not resolved:
        return None
    g = gold.lower()
    # month-year granularity
    if resolved.strftime("%B").lower() in g and str(resolved.year) in g:
        return True
    # full date
    if resolved.strftime("%B").lower() in g and str(resolved.day) in g and str(resolved.year) in g:
        return True
    # year only
    if re.fullmatch(r"\d{4}", g.strip()):
        return str(resolved.year) in g
    return False


def main():
    print("=" * 78)
    print("M30-48 Phase 2: Mechanism A — is the gold deterministically reachable?")
    print("=" * 78)
    print()
    A = [r for r in M["rows"] if r["qid"] in M["mech"]["A_evidence_has_rel_reader_must_resolve"]]
    ok = 0
    for r in A:
        gold = r["gold"]
        print(f"{r['qid']}  gold={gold!r}")
        for gid, txt, sd in r["ev_texts"]:
            base = parse_header(sd)
            rels = re.findall(r"(yesterday|tomorrow|the next day|the day before|next month|last month|"
                              r"next week|last week|next sunday|next monday|next tuesday|next wednesday|"
                              r"next thursday|next friday|next saturday|currently)", txt, re.I)
            if not rels:
                continue
            for rel in rels:
                res = resolve(rel, base) if base else None
                reach = gold_reachable(gold, res)
                print(f"    {gid} base={base.date() if base else None} rel='{rel}' "
                      f"-> {res.date() if res else None}  gold_reachable={reach}")
                if reach:
                    ok += 1
        print()

    print("=" * 78)
    print(f"Mechanism A cases where gold is reachable by the rule: {ok}/{len(A)}")
    print("=" * 78)
    print()
    print("CAVEAT: this used gold only to CHECK reachability (necessary condition).")
    print("It does NOT show we can pick the right turn without gold. That is Phase 3.")


if __name__ == "__main__":
    main()
