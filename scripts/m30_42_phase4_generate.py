#!/usr/bin/env python3
"""M30-42 Phase 4: Generate unseen generic evaluation set (no benchmark names).

Development 40 + Frozen 120, 8 categories x 15.
Uses fictional names/organizations/events/dates that do NOT appear in LoCoMo
or in the production hardcode lists.

Deterministic expected answers from structured data.
"""
import json, random
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
random.seed(90210)

MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
MONTH_IDX = {m: i+1 for i, m in enumerate(MONTHS)}

# Names chosen to avoid: caroline, melanie, gina, jon, maria, joanna, nate,
# james, deborah, jolene, evan, calvin, audrey, tim, andrew, kadence, dave,
# alice, bob, charlie, david, eve, frank, grace, oliver, luna, bailey, oscar
NAMES = ["Soren","Petra","Ingrid","Marek","Talia","Rowan","Nadia","Cyrus",
         "Elise","Dmitri","Farah","Gustav","Hana","Idris","Juno","Kiran",
         "Lena","Milo","Noor","Otto"]

ORGS = ["Verdant Labs","Harborlight Press","Northwind Studio","Quartzline",
        "Tidewater Coop","Emberfield Trust","Bluewick Guild","Cobalt Row"]

EVENTS = ["kiln firing","maple harvest","lantern parade","river regatta",
          "puppet workshop","cider pressing","weaving circle","star watch"]


def gen_conv(num_sessions=3, per=10):
    a, b = random.sample(NAMES, 2)
    org = random.choice(ORGS)
    ev = random.choice(EVENTS)
    base = datetime(2026, 1, 1) + timedelta(days=random.randint(0, 200))
    conv = {}
    turns_all = []
    for s in range(1, num_sessions + 1):
        skey = f"session_{s}"
        sdate = base + timedelta(days=s * 18 + random.randint(0, 6))
        dt = sdate.strftime(f"%I:%M %p on {sdate.day} %B, %Y")
        turns = []
        for t in range(1, per + 1):
            sp = a if t % 2 else b
            other = b if sp == a else a
            mn = MONTHS[sdate.month - 1]
            templates = [
                f"Hey {other}, I finally finished the {ev} for {org} on {mn} {sdate.day}, {sdate.year}.",
                f"The {ev} at {org} happened on {mn} {sdate.day}, {sdate.year}. Worth the wait.",
                f"On {mn} {sdate.day}, {sdate.year} I showed up at {org} for the {ev}.",
                f"{mn} {sdate.day}, {sdate.year} was when {org} ran the {ev}.",
                f"I recall the {ev} at {org} was on {mn} {sdate.day}, {sdate.year}.",
                f"We met at the {ev} in {org} on {mn} {sdate.day}, {sdate.year}.",
                f"Thanks for the note! The {ev} at {org} is on {mn} {sdate.day}, {sdate.year}.",
                f"I'm still thinking about the {ev} at {org} on {mn} {sdate.day}, {sdate.year}.",
                f"Can't wait for the next {ev}. Last one at {org} was {mn} {sdate.day}, {sdate.year}.",
                f"The {ev} at {org} on {mn} {sdate.day}, {sdate.year} exceeded expectations.",
            ]
            text = templates[t % len(templates)]
            turns.append({"dia_id": f"D{s}:{t}", "speaker": sp, "text": text})
            turns_all.append({"dia_id": f"D{s}:{t}", "speaker": sp, "text": text,
                              "date": sdate, "session": s})
        conv[skey] = turns
        conv[f"{skey}_date_time"] = dt
    return conv, turns_all, org, ev


def gen_q(cat, conv, turns_all, qid, org, ev):
    t = random.choice(turns_all)
    d = t["date"]; y, mo, day = d.year, d.month, d.day
    sp = t["speaker"]

    if cat == "direct_fact":
        q = f"On what date did {sp} attend the {ev}?"
        exp = f"{mo}/{day}/{y}"
    elif cat == "paraphrase":
        q = f"When did {sp} make the trip for the {ev}?"
        exp = f"{mo}/{day}/{y}"
    elif cat == "entity_disambiguation":
        other = random.choice([n for n in NAMES if n != sp])
        q = f"On what date did {sp} (not {other}) go to the {ev}?"
        exp = f"{mo}/{day}/{y}"
    elif cat == "multi_turn":
        t2 = random.choice(turns_all)
        delta = abs((d - t2["date"]).days)
        q = f"How many days apart are the two {ev}s {sp} mentioned?"
        exp = f"{delta} days"
    elif cat == "multi_session":
        sn = t["dia_id"].split(":")[0].replace("D", "")
        q = f"Which session has {sp}'s message about the {ev}?"
        exp = f"Session {sn}"
    elif cat == "relative_date":
        phrase, days = random.choice([("yesterday",1),("tomorrow",-1),
                                      ("last month",30),("next month",-30),("one year ago",365)])
        tgt = d - timedelta(days=days) if days > 0 else d + timedelta(days=-days)
        q = f"If today is {mo}/{day}/{y}, what is the date {phrase}?"
        exp = f"{tgt.month}/{tgt.day}/{tgt.year}"
    elif cat == "answerability":
        if random.random() < 0.5:
            q = f"On what date did {sp} attend the {ev}?"
            exp = f"{mo}/{day}/{y}"
        else:
            q = f"What is {sp}'s favorite season?"
            exp = "I don't know."
    elif cat == "granularity":
        if random.random() < 0.5:
            q = f"In which year did {sp} attend the {ev}?"
            exp = f"{y}"
        else:
            q = f"In which month and year did {sp} attend the {ev}?"
            exp = f"{mo}/{y}"
    return {"question_id": qid, "category": cat, "question": q,
            "expected_answer": exp, "answer_turn": t["dia_id"]}


def build(n, prefix, start):
    cats = ["direct_fact","paraphrase","entity_disambiguation","multi_turn",
            "multi_session","relative_date","answerability","granularity"]
    convs, qs = {}, []
    qid = start
    for cat in cats:
        for i in range(n):
            cid = f"{prefix}_{cat}_{i}"
            conv, turns_all, org, ev = gen_conv()
            convs[cid] = conv
            q = gen_q(cat, conv, turns_all, f"{prefix}_q{qid:04d}", org, ev)
            q["conversation_id"] = cid
            qs.append(q)
            qid += 1
    return convs, qs


def main():
    dev_c, dev_q = build(5, "dev", 0)
    frz_c, frz_q = build(15, "frz", 1000)

    out = PROJECT_ROOT / "results/m30_42"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dev_generic.json").write_text(json.dumps(
        {"conversations": dev_c, "questions": dev_q}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "frozen_generic.json").write_text(json.dumps(
        {"conversations": frz_c, "questions": frz_q}, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    print(f"Development: {len(dev_q)} questions")
    for k, v in sorted(Counter(q["category"] for q in dev_q).items()):
        print(f"  {k}: {v}")
    print(f"Frozen: {len(frz_q)} questions")
    for k, v in sorted(Counter(q["category"] for q in frz_q).items()):
        print(f"  {k}: {v}")
    print()
    print(f"Saved: {out / 'dev_generic.json'}")
    print(f"Saved: {out / 'frozen_generic.json'}")


if __name__ == "__main__":
    main()
