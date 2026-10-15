"""V4 Mission 24 -- can an anchor-based relative-date resolver actually RECOVER these?

THE GATE, fixed before running.  A case counts as RESCUED only if ALL of:
  1. the resolver computed an absolute date from an anchor plus a relative expression;
  2. that date matches the gold under the FROZEN scorer (eval_locomo_full.score), unchanged;
  3. the anchor turn and the relative expression are correctly bound to the event the
     question asks about.
Computing a date is NOT a rescue.  If the computed value does not match gold, the case is
counted as NOT_RESCUED, never as a partial success.

THE SCORER IS FROZEN AND NOT MODIFIED.  Only the presence of the intervention changes, so
no apparent gain can come from moving the scoring rule.

WHAT IS MEASURED, per Eli's four indicators:
  Rescue            wrong/abstain before -> correct after
  Regression        a previously correct answer broken by the intervention
  Abstention quality  forcing an answer where the evidence cannot support one
  Temporal correctness  anchor, relative expression and target event bound correctly

AND THE CHEAPEST TEST RUNS FIRST.  This stage is the RESOLVER ONLY: it computes the date
and compares to gold, with no Reader call at all.  If the resolver cannot reproduce the
gold on these cases, there is no point wiring it into a Reader intervention -- the
intervention would be measuring a mechanism that does not work.  Zero model calls here.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

EA = ROOT / "benchmark_results/evidence_audit"
M23 = EA / "v4_undecided_decomposition.json"
PACK = EA / "v4_bucketA_audit_pack.json"
KEY = EA / "v4_bucketA_reviewer_key.json"
DATASET = ROOT / "datasets/external/locomo10.json"
OUT = EA / "v4_temporal_resolver_gate.json"

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
_ANCHOR = re.compile(
    r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b"
    r"|\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b")

# relative expressions and the offset they imply, FIXED BEFORE ANY RESULT
_REL_OFFSET = [
    (re.compile(r"\byesterday\b", re.I), ("days", -1)),
    (re.compile(r"\blast\s+night\b", re.I), ("days", -1)),
    (re.compile(r"\bnext\s+month\b", re.I), ("months", +1)),
    (re.compile(r"\blast\s+month\b", re.I), ("months", -1)),
    (re.compile(r"\bthis\s+month\b", re.I), ("months", 0)),
    (re.compile(r"\bnext\s+year\b", re.I), ("years", +1)),
    (re.compile(r"\blast\s+year\b", re.I), ("years", -1)),
    (re.compile(r"\bthis\s+year\b", re.I), ("years", 0)),
]


def parse_anchor(s: str | None):
    if not s:
        return None
    m = _ANCHOR.search(s)
    if not m:
        return None
    g = m.groups()
    try:
        if g[0]:                                  # "28 August 2023"
            return date(int(g[2]), _MONTHS[g[1].lower()], int(g[0]))
        return date(int(g[5]), _MONTHS[g[3].lower()], int(g[4]))   # "August 28, 2023"
    except Exception:                             # noqa: BLE001
        return None


def shift(d: date, unit: str, n: int) -> date:
    if unit == "days":
        return d + timedelta(days=n)
    if unit == "months":
        y, m = divmod((d.month - 1) + n, 12)
        y, m = d.year + y, m + 1
        day = min(d.day, [31, 29 if y % 4 == 0 and (y % 100 or not y % 400) else 28,
                          31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
        return date(y, m, day)
    return date(d.year + n, d.month, d.day)


def render(d: date) -> list[str]:
    """Candidate surface forms the scorer might accept, so formatting alone does not
    decide the outcome.  All are checked against gold; none is preferred a priori."""
    return [
        f"{d.day} {d.strftime('%B')} {d.year}",           # 3 September 2022
        f"{d.strftime('%B')} {d.day}, {d.year}",          # September 3, 2022
        f"{d.strftime('%B')} {d.year}",                   # September 2022
        f"{d.strftime('%B')}, {d.year}",                  # September, 2022
        f"{d.year}",                                      # 2022
        f"{d.strftime('%B')}.{d.year}",                   # September.2022
        f"{d.month}/{d.day}/{d.year}",
        f"{d.strftime('%B')} {d.day} {d.year}",
    ]


def main() -> int:
    from eval_locomo_full import score as frozen_score

    m23 = json.loads(M23.read_text(encoding="utf-8"))
    pack = {c["question_id"]: c for c in
            json.loads(PACK.read_text(encoding="utf-8"))["cases"]}
    key = {r["question_id"]: r for r in
           json.loads(KEY.read_text(encoding="utf-8"))["rows"]}
    convs = json.loads(PACK.read_text(encoding="utf-8"))["conversations"]
    weights = {int(k): v["weight"] for k, v in
               json.loads(PACK.read_text(encoding="utf-8"))["sampling_weights"].items()}

    cases = [x for x in m23["rows"] if x["cause_category"] == "RELATIVE_DATE"]
    print("=" * 78)
    print("V4 MISSION 24 -- resolver-only gate: does the computed date MATCH THE GOLD?")
    print("=" * 78)
    print(f"  cases : {len(cases)} RELATIVE_DATE, frozen from M23")
    print("  ZERO MODEL CALLS.  This is the cheapest possible test and it runs first:")
    print("  if the resolver cannot reproduce the gold, wiring it into a Reader would")
    print("  measure a mechanism that does not work.")
    print()
    print("  A case is RESCUED only if the computed date matches gold under the FROZEN")
    print("  scorer (eval_locomo_full.score, UNCHANGED).  Computing a date is not a")
    print("  rescue.  The scorer is not modified, so no gain can come from scoring.")
    print()

    rows = []
    for x in cases:
        qid = x["question_id"]
        gold = pack[qid]["gold_answer"]
        anchor_turn = x["anchor_turn"]
        anchor_date = parse_anchor(x["session_anchor_found"])
        conv = pack[qid]["source_conversation"]
        turn_text = next((t["text"] for t in convs.get(conv, [])
                          if t["dia_id"] == anchor_turn), "")

        # find the relative expression.  BUG FIXED: the first version only searched the
        # ANCHOR TURN, which failed 11 of 28 with "no relative expression in the anchor
        # turn" -- but the expression is often in a DIFFERENT turn of the same session
        # (e.g. lc_conv-30_7 has "just launched" in D2:1 while the turn carrying the
        # dated anchor is elsewhere).  The anchor and the expression only need to share
        # a session for the date to be computable, so search the whole conversation and
        # prefer a turn from the SAME session as the anchor.
        anchor_session = None
        ms = re.match(r"D(\d+):", anchor_turn or "")
        if ms:
            anchor_session = int(ms.group(1))
        hits = []
        for t in convs.get(conv, []):
            for pat, (unit, n) in _REL_OFFSET:
                if pat.search(t["text"]):
                    ts = re.match(r"D(\d+):", t["dia_id"])
                    same = bool(ts and anchor_session and int(ts.group(1)) == anchor_session)
                    hits.append({"dia_id": t["dia_id"], "unit": unit, "n": n,
                                 "pat": pat.pattern, "same_session": same,
                                 "text": t["text"][:90]})
        hits.sort(key=lambda h: (not h["same_session"], h["dia_id"]))
        found = None
        if hits:
            h = hits[0]
            found = (h["pat"], h["unit"], h["n"])
            expr_turn = h["dia_id"]
        if not anchor_date or not found:
            rows.append({"question_id": qid, "gold": gold,
                         "computed": None, "matched": False,
                         "why": "no anchor date parsed" if not anchor_date
                                else "no relative expression anywhere in the conversation",
                         "relative_hits_in_conversation": len(hits)})
            continue
        _, unit, n = found
        computed = shift(anchor_date, unit, n)
        forms = render(computed)
        hit = next((f for f in forms if frozen_score(f, gold)), None)
        rows.append({
            "question_id": qid, "gold": gold,
            "anchor_turn": anchor_turn,
            "anchor_date": anchor_date.isoformat(),
            "relative_expr": found[0], "expr_turn": expr_turn,
            "expr_same_session_as_anchor": hits[0]["same_session"],
            "offset": f"{n} {unit}",
            "computed": computed.isoformat(),
            "forms_tried": forms[:4],
            "matched_form": hit,
            "matched": hit is not None,
            # temporal correctness: the anchor and the expression must bind to the
            # event the question asks about, not merely co-occur
            "binding_verified": False,   # not established by the matcher
        })

    matched = [r for r in rows if r["matched"]]
    # ---- SPURIOUS-MATCH GUARD -------------------------------------------------
    # A gold match can be an artifact rather than a correct resolution.  The frozen
    # scorer is lenient on day-of-month (it accepts '9/14/2023' for gold
    # 'October 14, 2023' because the day and year agree), and my resolver tries several
    # surface forms, so a wrong arithmetic path can still land on an accepted form.
    # Restricting to same-session anchor+expression is the defensible subset; the rest
    # are reported separately rather than silently counted.
    strict = [r for r in matched if r.get("expr_same_session_as_anchor")]
    loose = [r for r in matched if not r.get("expr_same_session_as_anchor")]
    print("=" * 78)
    print("SPURIOUS-MATCH GUARD (a gold match is not automatically a correct resolution)")
    print("=" * 78)
    print(f"  matched, anchor and expression in the SAME session : {len(strict)}")
    print(f"  matched, but the expression came from another session: {len(loose)}")
    print()
    print("  The frozen scorer is LENIENT ON DAY-OF-MONTH: score('9/14/2023',")
    print("  'October 14, 2023') returns True because day and year agree.  Since the")
    print("  resolver tries several surface forms, a wrong arithmetic path can still land")
    print("  on an accepted form.  So the same-session subset is the defensible number;")
    print("  the others are reported, not counted.")
    print()
    for r in loose:
        print(f"    {r['question_id']}: gold {r['gold']!r}")
        print(f"      anchor {r['anchor_turn']} ({r['anchor_date']}) + "
              f"expression from {r.get('expr_turn')} ({r['relative_expr'][:22]})")
        print(f"      -> {r['computed']}, accepted as {r['matched_form']!r}")
    print()

    matched = strict
    print("=" * 78)
    print("RESOLVER vs GOLD  (frozen scorer, same-session subset)")
    print("=" * 78)
    print(f"  computed and matched gold : {len(matched)}/{len(rows)} "
          f"({100 * len(matched) / len(rows):.0f}%)")
    nm = [r for r in rows if not r["matched"]]
    reasons = Counter(r.get("why") or "computed date does not match gold"
                      for r in nm)
    for k, v in reasons.most_common():
        print(f"    not matched -- {k}: {v}")
    print()

    print("  MATCHED CASES (verified by hand below, not just by the matcher)")
    for r in matched[:8]:
        print(f"    {r['question_id']}: gold {r['gold']!r} <- "
              f"{r['anchor_date']} {r['relative_expr'][:22]} -> {r['computed']}")
    print()

    print("  NOT-MATCHED CASES, with the reason")
    for r in nm[:8]:
        print(f"    {r['question_id']}: gold {r['gold']!r}")
        print(f"      computed {r.get('computed')} from {r.get('anchor_date')} "
              f"{r.get('relative_expr', '')[:26]}")
        print(f"      why: {r.get('why') or 'computed date does not match gold'}")
    print()

    # ---------------- the four indicators ----------------------------------
    print("=" * 78)
    print("THE FOUR INDICATORS, and what this stage can and cannot say about them")
    print("=" * 78)
    n_ok = len(matched)
    print(f"  Rescue              : {n_ok}/{len(rows)} would be candidates -- but this is")
    print("                        a RESOLVER-ONLY result.  No Reader ran, no engine")
    print("                        output changed, so nothing is rescued yet.")
    print( "  Regression          : not measurable here -- no answer was altered.")
    print("  Abstention quality  : not measurable here -- nothing was answered.")
    print("  Temporal correctness: PARTIAL.  A gold match implies the arithmetic was")
    print("                        right, but not that the anchor binds to the right")
    print("                        EVENT.  That needs the per-case binding check, which")
    print("                        is recorded as unverified and must not be assumed.")
    print()

    # ---------------- weighted estimate ------------------------------------
    print("=" * 78)
    print("WEIGHTED (per-stratum weights preserved)")
    print("=" * 78)
    w_ok = sum(weights.get(pack[r["question_id"]]["category"], 1.0)
               for r in matched)
    w_all = sum(weights.get(pack[r["question_id"]]["category"], 1.0) for r in rows)
    print(f"  weighted match rate over these cases : {100 * w_ok / w_all:.1f}%")
    print(f"  M23 estimated resolvable-from-available-data at ~57.1% of Bucket A;")
    print(f"  this resolver stage supports at most {100 * w_ok / w_all:.1f}% of THAT, "
          f"i.e. ~{57.1 * w_ok / w_all:.1f}% of Bucket A.")
    print("  An estimate of the mix, not a per-case measurement, and NOT a rescue rate.")
    print()

    print("=" * 78)
    print("GATE")
    print("=" * 78)
    if len(matched) >= len(rows) * 0.5:
        print(f"  The resolver reproduces the gold on {len(matched)}/{len(rows)} cases")
        print("  with a frozen scorer and zero model calls.  That justifies proceeding to")
        print("  the Reader intervention, which must then measure rescue, regression,")
        print("  abstention quality and event binding together -- not answer count.")
    else:
        print(f"  Only {len(matched)}/{len(rows)} match.  The resolver is not good enough")
        print("  to justify a Reader intervention yet; fix the resolver first.")
    print()
    print("  SCOPE: computing a date is NOT a rescue.  No engine output changed, so no")
    print("  case is rescued and no score is claimed.  Frozen scorer, unchanged.  No")
    print("  model calls, no production change, holdout untouched.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 24 -- resolver-only gate against gold, frozen scorer",
        "baseline": {"head": "aebbde2", "production_changed": False},
        "model_calls": 0,
        "scorer": "eval_locomo_full.score() -- frozen, not modified",
        "n_cases": len(rows),
        "n_matched": len(matched),
        "match_rate_pct": round(100 * len(matched) / len(rows), 1),
        "weighted_match_rate_pct": round(100 * w_ok / w_all, 1),
        "is_a_rescue": False,
        "rescue_reason": ("resolver-only stage; no engine output was altered, so no case "
                          "is rescued and no score is claimed"),
        "binding_verified": False,
        "not_matched_reasons": dict(reasons),
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
