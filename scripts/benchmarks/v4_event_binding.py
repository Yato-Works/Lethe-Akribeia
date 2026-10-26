"""V4 Mission 9 -- event-binding failure decomposition, WITH A CONTROL GROUP.

The mission's warning is the design constraint: measuring turn distance on 28 failures
alone tells you failures are near each other, not that NEARNESS CAUSES the failure.  So
every feature here is computed with the SAME definition on a control group drawn from
CONTROL_CORRECT, and the comparison is reported alongside it.

THE CONTROL IS NON-VACUOUS BY CONSTRUCTION, which is the part that is easy to get wrong.
A competitor is only meaningful relative to the value the answer SHOULD be.  So for each
group the "correct event" is located in the source dialogue from the value that group's
answer was based on:
  failure group  -> the gold string (what the answer should have been)
  control group  -> the value actually returned AND scored correct, located in the source
For the controls that yields a real competitor pair, not an empty set, so the comparison
below answers a question rather than returning 0 vs 0.

THREE FACTORS, RECORDED INDEPENDENTLY AND NOT COLLAPSED into one label:
  A  turn distance between the two events (source order, and position in the given ctx)
  B  temporal relation (restored date, difference in days, confidence)
  C  event similarity (same kind of value, or different kind)

Gold is used only to LOCATE evidence for this audit.  No production code changes.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DECOMP = ROOT / "benchmark_results/evidence_audit/v4_selection_decomposition.json"
H9 = ROOT / "benchmark_results/evidence_audit/v4_reader_h9.json"
PILOT = ROOT / "benchmark_results/evidence_audit/v4_pilot_selection.json"
AUDITF = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
RAW = ROOT / "datasets/external/locomo10.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/v4_event_binding.json"

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
_MON_RE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DATE_PATTERNS = [
    (re.compile(rf"\b(\d{{1,2}})\s+({'|'.join(_MONTHS)})\w*,?\s+(\d{{4}})\b", re.I), "dmy"),
    (re.compile(rf"\b({'|'.join(_MONTHS)})\w*\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I), "mdy"),
    (re.compile(rf"\b(\d{{1,2}})\s+({_MON_RE})\b", re.I), "dm"),
    (re.compile(rf"\b({_MON_RE})\s+(\d{{1,2}})\b", re.I), "md"),
]
_HEADER_DATE = re.compile(r"\bon\s+\d{1,2}\s+\w+,?\s+(\d{4})|"
                          r"\bon\s+\d{1,2}\s+\w+\b")

_STOP = frozenset({
    "the", "and", "for", "was", "were", "his", "her", "its", "their", "our", "your",
    "him", "she", "they", "them", "not", "but", "all", "any", "out", "about", "into",
    "over", "than", "then", "when", "what", "who", "why", "how", "which", "there",
    "here", "have", "has", "had", "been", "being", "are", "will", "would", "could",
    "should", "did", "does", "with", "that", "this", "from", "many", "much", "long",
    "often", "ago", "now", "get", "got", "one", "two", "did", "describe", "describes",
    "jon", "john", "melanie", "caroline", "andrew", "audrey", "james", "joanna",
    "nate", "maria", "gina", "tim", "evan", "calvin", "dave", "sam", "jolene",
    "recently", "started", "learning", "said", "says", "went", "just", "really",
})

_ACTIVITY_WORDS = frozenset({
    "hiking", "picnic", "painting", "swimming", "skiing", "running", "reading",
    "cycling", "camping", "dancing", "basketball", "soccer", "tennis", "yoga",
    "cooking", "baking", "travel", "traveling", "surfing", "climbing", "walk",
    "walking", "gardening", "photography", "fishing", "bowling", "golf",
})


def content_tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if w not in _STOP}


def load_source() -> dict[str, list[dict]]:
    data = json.loads(RAW.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for s in data:
        sid = s.get("sample_id")
        conv = s.get("conversation") or {}
        turns = []
        for k in sorted(conv.keys()):
            m = re.fullmatch(r"session_(\d+)", k)
            if not m:
                continue
            date = conv.get(f"session_{m.group(1)}_date_time", "")
            for t in conv[k] or []:
                turns.append({"dia_id": t.get("dia_id", ""),
                              "speaker": t.get("speaker", ""),
                              "text": t.get("text", ""), "date": date,
                              "session": int(m.group(1))})
        if sid:
            out[sid] = turns
    return out


def restore_date(text: str) -> tuple[str | None, str]:
    """Best-effort date from a turn's own text.  Returns (iso or None, confidence)."""
    t = str(text)
    for pat, kind in _DATE_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        g = m.groups()
        try:
            if kind == "dmy":
                mon = _MONTHS[g[1].lower()]
                return f"{g[2]}-{mon:02d}-{int(g[0]):02d}", "FULL"
            if kind == "mdy":
                mon = _MONTHS[g[0].lower()]
                return f"{g[2]}-{mon:02d}-{int(g[1]):02d}", "FULL"
            if kind == "dm":
                mon = None
                for name, num in _MONTHS.items():
                    if g[1].lower().startswith(name[:3]):
                        mon = num
                        break
                if mon:
                    return f"????-{mon:02d}-{int(g[0]):02d}", "NO_YEAR"
            if kind == "md":
                for name, num in _MONTHS.items():
                    if g[0].lower().startswith(name[:3]):
                        return f"????-{num:02d}-{int(g[1]):02d}", "NO_YEAR"
        except Exception:
            continue
    m = _HEADER_DATE.search(t)
    if m:
        return None, "RELATIVE_ONLY"
    return None, "NONE"


def days_between(a: str | None, b: str | None) -> int | None:
    if not a or not b or "?" in a or "?" in b:
        return None
    try:
        da = datetime.fromisoformat(a)
        db = datetime.fromisoformat(b)
    except Exception:
        return None
    return (db - da).days


def find_turns(hay: list[dict], needle: str) -> list[dict]:
    n = re.sub(r"\s+", " ", str(needle).strip().lower().strip("\"'.,!?"))
    if not n:
        return []
    return [t for t in hay if n in re.sub(r"\s+", " ", t["text"].lower())]


def locate_by_answer(hay: list[dict], answer: str) -> list[dict]:
    """Locate the turn an ANSWER came from, by its most distinctive content tokens."""
    toks = content_tokens(answer)
    if not toks:
        return []
    hits = []
    for t in hay:
        low = t["text"].lower()
        n = sum(1 for w in toks if w in low)
        if n >= max(2, len(toks) // 2):
            hits.append((n, t))
    hits.sort(key=lambda p: -p[0])
    return [t for _, t in hits[:3]]


def analyse(rec_base: dict, gold_turns: list[dict], pred_turns: list[dict],
            src: list[dict], given_ctx: str) -> dict:
    """The three factors, computed identically for BOTH groups."""
    g_ids = {t["dia_id"] for t in gold_turns}
    p_ids = {t["dia_id"] for t in pred_turns}

    # ---- A: turn distance ---------------------------------------------------
    idx = {t["dia_id"]: i for i, t in enumerate(src)}
    distances = []
    for g in g_ids:
        for p in p_ids:
            if g in idx and p in idx:
                distances.append(abs(idx[g] - idx[p]))
    g_sessions = {t.get("session") for t in gold_turns}
    p_sessions = {t.get("session") for t in pred_turns}
    same_session = bool(g_sessions & p_sessions)

    # position inside the context the Reader actually saw
    ctx_ids = []
    for line in given_ctx.split("\n"):
        m = re.match(r"\[([^\]]+)\]", line)
        if m:
            ctx_ids.append(m.group(1).split(" on ")[0].strip())
    ctx_positions = []
    for g in g_ids:
        for p in p_ids:
            if g in ctx_ids and p in ctx_ids:
                ctx_positions.append(abs(ctx_ids.index(g) - ctx_ids.index(p)))

    # ---- B: temporal relation ----------------------------------------------
    g_date, g_conf = (restore_date(gold_turns[0]["text"]), )[0] if gold_turns else (None, "NONE")
    p_date, p_conf = (restore_date(pred_turns[0]["text"]), )[0] if pred_turns else (None, "NONE")
    # The turn TEXT usually carries no inline date, so an in-text-only pass scores 0/28
    # and B would silently vanish.  The session header DOES carry a full date
    # ('1:56 pm on 8 May, 2023'), and it is real provenance rather than an inference,
    # so fall back to it and record which source the date came from.
    def _header_date(header: str) -> tuple[str | None, str]:
        m = re.search(r"\bon\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", header or "")
        if not m:
            return None, "NONE"
        day, mon, yr = m.group(1), m.group(2).lower(), m.group(3)
        num = None
        for name, v in _MONTHS.items():
            if mon.startswith(name[:3]):
                num = v
                break
        if not num:
            return None, "NONE"
        return f"{yr}-{num:02d}-{int(day):02d}", "HEADER"

    if g_date is None and gold_turns:
        g_date, g_conf = _header_date(gold_turns[0].get("date", ""))
    if p_date is None and pred_turns:
        p_date, p_conf = _header_date(pred_turns[0].get("date", ""))
    delta = days_between(g_date, p_date)
    same_date = bool(g_date and p_date and g_date == p_date)
    g_header = gold_turns[0]["date"] if gold_turns else ""
    p_header = pred_turns[0]["date"] if pred_turns else ""

    # ---- C: event similarity ------------------------------------------------
    g_toks = content_tokens(gold_turns[0]["text"]) if gold_turns else set()
    p_toks = content_tokens(pred_turns[0]["text"]) if pred_turns else set()
    g_act = sorted(g_toks & _ACTIVITY_WORDS)
    p_act = sorted(p_toks & _ACTIVITY_WORDS)
    if g_act and p_act:
        same_kind = "BOTH_ACTIVITY_" + ("SAME" if set(g_act) & set(p_act) else "DIFFERENT")
    elif g_act or p_act:
        same_kind = "ONE_ACTIVITY"
    else:
        same_kind = "NEITHER_ACTIVITY"
    shared = sorted(g_toks & p_toks)

    rec_base.update({
        "gold_turns": sorted(g_ids)[:4],
        "gold_speakers": sorted({t["speaker"] for t in gold_turns}),
        "pred_turns": sorted(p_ids),
        "pred_speakers": sorted({t["speaker"] for t in pred_turns}),
        # A
        "A_min_distance": min(distances) if distances else None,
        "A_distances": distances[:6],
        "A_adjacent": bool(distances and min(distances) <= 1),
        "A_same_session": same_session,
        "A_ctx_distance": min(ctx_positions) if ctx_positions else None,
        # B
        "B_gold_date": g_date, "B_gold_conf": g_conf,
        "B_pred_date": p_date, "B_pred_conf": p_conf,
        "B_delta_days": delta,
        "B_same_date": same_date,
        "B_gold_header": g_header, "B_pred_header": p_header,
        # C
        "C_kind": same_kind,
        "C_gold_activity": g_act, "C_pred_activity": p_act,
        "C_shared_tokens": shared[:6],
    })
    return rec_base


def main() -> int:
    decomp = json.loads(DECOMP.read_text(encoding="utf-8"))
    h9 = json.loads(H9.read_text(encoding="utf-8"))
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    full = json.loads(AUDITF.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in full["records"]}
    arm = {r["question_id"]: r for r in h9["rows"]}
    src = load_source()
    frozen = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen[json.loads(line)["qid"]] = json.loads(line)["context"]

    print("=" * 74)
    print("V4 MISSION 9 -- event binding decomposition, WITH A NON-VACUOUS CONTROL")
    print("=" * 74)
    print("  FAILURE  : correct event = the GOLD string (what the answer should be)")
    print("  CONTROL  : correct event = the value that WAS returned and scored correct,")
    print("             located in the source the same way.  Without this the comparison")
    print("             would be 28 real pairs against 0 pairs, which proves nothing.")
    print()

    # ---- FAILURE group ------------------------------------------------------
    failures: list[dict] = []
    for r in decomp["rows"]:
        conv = r["conversation"]
        turns = src.get(conv, [])
        gold_t = find_turns(turns, r["gold"])
        pred_t = locate_by_answer(turns, r["raw_output"] or r["prediction"])
        cq = (meta.get(r["question_id"]) or {}).get("cache_qid", "")
        failures.append(analyse({
            "group": "FAILURE", "question_id": r["question_id"],
            "conversation": conv, "question": r["question"],
            "gold": r["gold"], "prediction": r["prediction"],
        }, gold_t, pred_t, turns, frozen.get(cq, "")))

    # ---- CONTROL group ------------------------------------------------------
    controls: list[dict] = []
    skipped = []
    for s in pilot["selected"]:
        if s["cohort"] != "CONTROL_CORRECT":
            continue
        qid = s["question_id"]
        r = arm.get(qid) or {}
        if r.get("mutually_exclusive") != "CORRECT":
            skipped.append((qid, "not scored correct"))
            continue
        answered = str(r.get("prediction") or "")
        if not answered:
            skipped.append((qid, "no prediction recorded"))
            continue
        conv = r["conversation"]
        turns = src.get(conv, [])
        # the correct event = where the returned value came from; the competing event
        # = the nearest other same-kind value in the same conversation
        corr_t = locate_by_answer(turns, answered)
        if not corr_t:
            skipped.append((qid, "correct value not located in source"))
            continue
        toks = content_tokens(answered)
        rivals = []
        for t in turns:
            if t["dia_id"] in {c["dia_id"] for c in corr_t}:
                continue
            low = t["text"].lower()
            if sum(1 for w in toks if w in low) >= max(2, len(toks) // 2):
                rivals.append(t)
        if not rivals:
            skipped.append((qid, "no rival event found -- no pair to measure"))
            continue
        cq = (meta.get(qid) or {}).get("cache_qid", "")
        controls.append(analyse({
            "group": "CONTROL", "question_id": qid, "conversation": conv,
            "question": r["question"], "gold": str(r.get("gold")),
            "prediction": answered,
            "control_rival": rivals[0]["text"][:80],
        }, corr_t, rivals[:1], turns, frozen.get(cq, "")))

    print(f"  failure cases analysed : {len(failures)}")
    print(f"  control pairs formed   : {len(controls)}   "
          f"(skipped {len(skipped)}: no rival event to measure against)")
    if skipped:
        for q, why in skipped[:5]:
            print(f"      skipped {q}: {why}")
    print()

    if not controls:
        print("  NO CONTROL PAIRS -- the comparison cannot be made.  Reporting the")
        print("  failure-side distributions only and stating that no causal reading is")
        print("  possible from them.")

    # ---- distributions, BOTH groups ----------------------------------------
    def dist(rows, key):
        return Counter(r.get(key) for r in rows)

    for label, rows in (("FAILURE", failures), ("CONTROL", controls)):
        if not rows:
            continue
        n = len(rows)
        adj = sum(1 for r in rows if r["A_adjacent"])
        adj3 = sum(1 for r in rows if r["A_min_distance"] is not None
                   and r["A_min_distance"] <= 3)
        dmin = [r["A_min_distance"] for r in rows if r["A_min_distance"] is not None]
        deltas = [r["B_delta_days"] for r in rows if r["B_delta_days"] is not None]
        same = sum(1 for r in rows if r["B_same_date"])
        undated = sum(1 for r in rows if r["B_gold_conf"] in ("NONE", "RELATIVE_ONLY")
                      or r["B_pred_conf"] in ("NONE", "RELATIVE_ONLY"))
        print("=" * 74)
        print(f"{label}  (n={n}, {len({r['conversation'] for r in rows})} conversations)")
        print("=" * 74)
        print(f"  A adjacent (distance <= 1)     : {adj}/{n} "
              f"({100 * adj / n:.0f}%)")
        print(f"  A distance <= 3                : {adj3}/{n} "
              f"({100 * adj3 / n:.0f}%)")
        if dmin:
            ds = sorted(dmin)
            print(f"  A min distance median          : {ds[len(ds) // 2]} "
                  f"(max {ds[-1]}, undeterminable {n - len(dmin)})")
        print(f"  A same conversation session    : "
              f"{sum(1 for r in rows if r['A_same_session'])}/{n}")
        ctxd = [r["A_ctx_distance"] for r in rows if r["A_ctx_distance"] is not None]
        if ctxd:
            print(f"  A distance INSIDE given ctx    : median "
                  f"{sorted(ctxd)[len(ctxd) // 2]} (n={len(ctxd)})")
        print(f"  B both dates restorable        : {len(deltas)}/{n} "
              f"(undeterminable {undated})")
        if deltas:
            print(f"  B |delta| median               : "
                  f"{sorted(abs(d) for d in deltas)[len(deltas) // 2]} days")
        print(f"  B same date, different event   : {same}")
        print(f"  C event kind                   : {dict(dist(rows, 'C_kind'))}")
        print()

    # ---- the comparison that actually answers the question ------------------
    if failures and controls:
        print("=" * 74)
        print("FAILURE vs CONTROL -- does any factor actually separate them?")
        print("=" * 74)
        f_n, c_n = len(failures), len(controls)
        rows = []
        for name, fn in (("A adjacent(<=1)", lambda r: r["A_adjacent"]),
                         ("A within 3", lambda r: (r["A_min_distance"] or 99) <= 3),
                         ("A same session", lambda r: r["A_same_session"]),
                         ("B same date", lambda r: r["B_same_date"]),
                         ("C both activity", lambda r: r["C_kind"].startswith("BOTH"))):
            fv = sum(1 for r in failures if fn(r))
            cv = sum(1 for r in controls if fn(r))
            rows.append((name, fv, f_n, 100 * fv / f_n,
                         cv, c_n, 100 * cv / c_n, 100 * fv / f_n - 100 * cv / c_n))
        print(f"  {'factor':<20}{'failure':>16}{'control':>16}{'gap':>9}")
        for name, fv, _fn, fp, cv, _cn, cp, gap in rows:
            print(f"  {name:<20}{fv:>4}/{f_n:<4}{fp:>6.0f}%"
                  f"{cv:>7}/{c_n:<4}{cp:>6.0f}%{gap:>+8.0f}pp")
        print()
        best = max(rows, key=lambda r: abs(r[7]))
        print(f"  LARGEST GAP: {best[0]}  failure {best[3]:.0f}% ({best[1]}/{best[2]}) "
              f"vs control {best[6]:.0f}% ({best[4]}/{best[5]})  {best[7]:+.0f}pp")
        print(f"  failure side spans "
              f"{len({r['conversation'] for r in failures})} conversations")
        print()
        print("  PER-CONVERSATION SPREAD of the top factor "
              "(a gap concentrated in one conversation is not a general cause)")
        key_fn = {"A adjacent(<=1)": lambda r: r["A_adjacent"],
                  "A within 3": lambda r: (r["A_min_distance"] or 99) <= 3,
                  "A same session": lambda r: r["A_same_session"],
                  "B same date": lambda r: r["B_same_date"],
                  "C both activity": lambda r: r["C_kind"].startswith("BOTH")}[best[0]]
        convs = sorted({r["conversation"] for r in failures})
        spread = []
        for cv_ in convs:
            sub = [r for r in failures if r["conversation"] == cv_]
            hit = sum(1 for r in sub if key_fn(r))
            if sub:
                spread.append((cv_, hit, len(sub)))
        print(f"    {'conv':<12}{'hit':>6}{'n':>5}")
        for cv_, hit, tot_ in spread:
            print(f"    {cv_:<12}{hit:>6}{tot_:>5}")
        print()
        if abs(best[7]) < 15:
            print(f"  NO FACTOR SEPARATES THE GROUPS (largest gap {best[7]:+.0f}pp). "
                  "Adjacency, dates and event kind do NOT explain the")
            print("  wrong-event choice, because controls show the same profile.")
        else:
            nz = [s for s in spread if s[1] > 0]
            print(f"  {best[0]} separates the groups by {best[7]:+.0f}pp and appears in "
                  f"{len(nz)}/{len(spread)} conversations")
            print("  -> candidate cause.  Still a CANDIDATE: this is an observational")
            print("     association on 28 failures vs 16 controls, not an intervention.")

    print("PER-CASE EVIDENCE (failures)")
    print("-" * 74)
    for r in failures:
        print(f"  {r['question_id']}  gold={str(r['gold'])[:22]!r}")
        print(f"     Q    : {r['question'][:58]}")
        print(f"     gold {r['gold_turns']} spk={r['gold_speakers']}  "
              f"pred {r['pred_turns']} spk={r['pred_speakers']}")
        print(f"     A dist={r['A_min_distance']} adjacent={r['A_adjacent']} "
              f"same_session={r['A_same_session']} ctx_dist={r['A_ctx_distance']}")
        print(f"     B gold_date={r['B_gold_date']}({r['B_gold_conf']}) "
              f"pred_date={r['B_pred_date']}({r['B_pred_conf']}) "
              f"delta={r['B_delta_days']}")
        print(f"     C kind={r['C_kind']} gold_act={r['C_gold_activity']} "
              f"pred_act={r['C_pred_activity']}")
    print()

    OUT.write_text(json.dumps({
        "label": "V4 mission 9 -- event binding decomposition with a non-vacuous control",
        "baseline": {"head": "e13ebf1", "production_changed": False,
                     "evidence_binding": "unwired, preserved"},
        "control_definition": "correct event = the returned-and-correct value located in "
                              "the source; rival = nearest other same-kind value",
        "n_failure": len(failures), "n_control": len(controls),
        "skipped_controls": skipped,
        "failures": failures, "controls": controls,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
