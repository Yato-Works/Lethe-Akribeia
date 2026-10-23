#!/usr/bin/env python3
"""M30-48: Failure mechanism analysis — read the traces, derive the cause.

Do NOT group by surface label (abstain vs wrong). Group by MECHANISM:
  - what the reader produced
  - what the gold requires
  - what the evidence turn actually says
  - whether the answer is derivable from the evidence as-is

Then the next experiment is decided by the mechanism distribution, not by
picking a favourite failure type.
"""
import sys, json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

PROD = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
REACH = {r["qid"]: r for r in json.loads(
    (PROJECT_ROOT / "results/m30_44/evidence_reach.json").read_text(encoding="utf-8"))["rows"]}


def to_conv(q):
    m = re.match(r"lc_conv-(\d+)_(\d+)", q)
    return f"conv-{m.group(1)}-qa-{int(m.group(2)):03d}"


DATE_RE = re.compile(r"(january|february|march|april|may|june|july|august|september|october|november|december)"
                     r"[\s,]+\d{1,2}[\s,]+\d{4}|\d{1,2}\s+(january|february|march|april|may|june|july|august|"
                     r"september|october|november|december)[\s,]+\d{4}", re.I)
REL_RE = re.compile(r"\b(yesterday|tomorrow|last month|next month|this month|last year|next year|"
                    r"one year ago|two years ago|\d+ years? ago|last week|next week|the day before|"
                    r"the next day|that day|currently|recently)\b", re.I)


def main():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    sample_ids = [c.get("sample_id", "") for c in locomo]
    adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

    # map conv qid -> turns  (build once, keep ALL conversations)
    conv_turns = {}
    for idx, sid in enumerate(sample_ids):
        turns, questions, ir_records = adapter.load_conversation(idx)
        conv_turns[sid] = {t.dia_id: t for t in turns}
    print(f"[debug] conversations loaded: {sorted(conv_turns)}")
    print(f"[debug] conv-26 turns: {len(conv_turns.get('conv-26', {}))}")

    rows = []
    for r in PROD["rows"]:
        if r["correct"]:
            continue
        qid = r["qid"]
        conv = "conv-" + qid.replace("lc_conv-", "").split("_")[0]
        ev = REACH.get(qid, {})
        gold_ids = ev.get("gold_evidence", [])
        # gather the evidence turn texts
        ev_texts = []
        for gid in gold_ids:
            t = conv_turns.get(conv, {}).get(gid)
            if t:
                ev_texts.append((gid, t.text, t.session_date))
        if not ev_texts and gold_ids:
            print(f"[warn] {qid}: conv={conv} gold_ids={gold_ids} "
                  f"conv_loaded={conv in conv_turns} "
                  f"has_dia={[gid for gid in gold_ids if gid in conv_turns.get(conv,{})]}")
        pred = str(r["predicted"])
        gold = str(r["gold"])
        rows.append({
            "qid": qid, "gold": gold, "pred": pred,
            "ev_present": ev.get("evidence_present"),
            "ev_texts": ev_texts,
            "pred_has_date": bool(DATE_RE.search(pred)),
            "gold_has_date": bool(DATE_RE.search(gold)),
            "ev_has_rel": any(REL_RE.search(t[1]) for t in ev_texts),
            "ev_has_explicit_date": any(DATE_RE.search(t[1]) for t in ev_texts),
        })

    print("=" * 78)
    print("M30-48: Failure mechanism table")
    print("=" * 78)
    print()
    for r in rows:
        print(f"{r['qid']}")
        print(f"  gold: {r['gold']!r}")
        print(f"  pred: {r['pred'][:85]!r}")
        print(f"  ev_present={r['ev_present']}  ev_rel={r['ev_has_rel']}  ev_explicit_date={r['ev_has_explicit_date']}")
        for gid, txt, sd in r["ev_texts"]:
            print(f"    {gid} [{sd}]: {txt[:100]}")
        print()

    # Mechanism classification
    print("=" * 78)
    print("MECHANISM CLASSIFICATION")
    print("=" * 78)
    mech = {"A_evidence_has_rel_reader_must_resolve": [],
            "B_evidence_has_explicit_date_reader_must_copy": [],
            "C_evidence_present_no_relative_no_explicit": [],
            "D_evidence_missing": []}
    for r in rows:
        if not r["ev_present"]:
            mech["D_evidence_missing"].append(r["qid"])
        elif r["ev_has_rel"] and not r["ev_has_explicit_date"]:
            mech["A_evidence_has_rel_reader_must_resolve"].append(r["qid"])
        elif r["ev_has_explicit_date"]:
            mech["B_evidence_has_explicit_date_reader_must_copy"].append(r["qid"])
        else:
            mech["C_evidence_present_no_relative_no_explicit"].append(r["qid"])
    for k, v in mech.items():
        print(f"  {k}: {len(v)}  {v}")
    print()

    out = PROJECT_ROOT / "results/m30_48"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mechanisms.json").write_text(json.dumps({"rows": rows, "mech": mech},
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'mechanisms.json'}")


if __name__ == "__main__":
    main()
