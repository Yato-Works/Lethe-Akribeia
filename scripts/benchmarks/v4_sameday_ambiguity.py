"""V4 Mission 10 -- same-day event ambiguity across all 989 questions.

Labels the condition BLIND, before any outcome is joined, so "it failed, therefore it
must have been ambiguous" cannot happen.  The labelling pass sees only the question and
the Reader's context.  Outcomes are joined afterwards from the frozen mission-2
artefact.

THE CONDITION (mission's definition, kept strict):
  Same-Day Competing Events = inside the context the Reader was ACTUALLY given, two or
  more DISTINCT events fall on the SAME DATE, and at least one of them is a rival for
  the kind of value the question asks for.

Two things are deliberately separated, because conflating them was the M9 confound:
  * several events on the same day            -> 'co-occurring'
  * a rival for the ASKED kind of value       -> 'competing'
A question where the same day holds 'ate lunch' and 'saw a film' but the question is
about lunch has NO rival, so it is not the condition.

Dates come from the context header or the turn text, never inferred; the SOURCE of every
date is recorded, and an unrestorable date makes the case UNKNOWN rather than guessed.

FOUR CELLS are reported so M9's confound stays visible instead of being merged:
  same-day + adjacent | same-day + not adjacent | diff-day + adjacent | diff-day + not adjacent

Speakers are recorded as a SEPARATE field and never merged into the primary condition,
because M9 did not support the speaker-handoff hypothesis (1 of 7).

Outcome labels are joined ONLY after labelling is frozen.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

H9 = ROOT / "benchmark_results/evidence_audit/v4_reader_h9.json"
AUDITF = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
SPLIT = ROOT / "benchmark_results/evidence_audit/v4_holdout_split.json"
OUT = ROOT / "benchmark_results/evidence_audit/v4_sameday_ambiguity.json"

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
_MON_RE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"

_ACTIVITY = frozenset({
    "hiking", "hike", "picnic", "painting", "paint", "swimming", "swim", "skiing",
    "ski", "running", "run", "reading", "read", "cycling", "cycle", "camping",
    "camp", "dancing", "dance", "basketball", "soccer", "football", "tennis",
    "yoga", "cooking", "cook", "baking", "bake", "travel", "traveling", "surfing",
    "surf", "climbing", "climb", "walking", "walk", "gardening", "garden",
    "photography", "fishing", "fish", "bowling", "golf", "concert", "movie",
    "movies", "party", "wedding", "graduation", "interview", "adoption", "adopt",
    "surgery", "accident", "tournament", "contest", "festival", "parade",
    "roadtrip", "cruise", "volunteering", "class", "lesson", "practice",
})
_PLACE = re.compile(
    r"\b(?:in|at|to)\s+(?:the\s+)?([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,})?)\b")
_SLOT_WORDS = frozenset({
    "activity", "activities", "hobby", "hobbies", "sport", "sports", "book",
    "books", "movie", "movies", "song", "songs", "food", "foods", "dish", "dishes",
    "place", "places", "city", "cities", "country", "countries", "language",
    "languages", "instrument", "instruments", "pet", "pets", "animal", "animals",
    "game", "games", "event", "events", "feeling", "feelings", "emotion",
    "emotions", "color", "colors", "gift", "gifts",
})


def header_date(header: str) -> tuple[str | None, str]:
    m = re.search(r"\bon\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", header or "")
    if not m:
        return None, "NONE"
    day, mon, yr = int(m.group(1)), m.group(2).lower(), m.group(3)
    num = next((v for n, v in _MONTHS.items() if mon.startswith(n[:3])), None)
    if not num:
        return None, "NONE"
    try:
        return f"{yr}-{num:02d}-{day:02d}", "HEADER"
    except Exception:
        return None, "NONE"


def text_date(text: str) -> tuple[str | None, str]:
    for pat, kind in (
        (re.compile(rf"\b(\d{{1,2}})\s+({'|'.join(_MONTHS)})\w*,?\s+(\d{{4}})\b", re.I), "dmy"),
        (re.compile(rf"\b({'|'.join(_MONTHS)})\w*\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I), "mdy"),
    ):
        m = pat.search(text)
        if not m:
            continue
        g = m.groups()
        try:
            if kind == "dmy":
                return f"{g[2]}-{_MONTHS[g[1].lower()]:02d}-{int(g[0]):02d}", "TEXT"
            return f"{g[2]}-{_MONTHS[g[0].lower()]:02d}-{int(g[1]):02d}", "TEXT"
        except Exception:
            continue
    return None, "NONE"


def slots_of(question: str) -> set[str]:
    q = question.lower()
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", q) if w in _SLOT_WORDS}


def main() -> int:
    h9 = json.loads(H9.read_text(encoding="utf-8"))
    full = json.loads(AUDITF.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in full["records"]}
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    tuning_ids = set(split["tuning_qids"])

    frozen = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            frozen[json.loads(line)["qid"]] = json.loads(line)["context"]

    print("=" * 74)
    print("V4 MISSION 10 -- same-day event ambiguity, labelled BLIND")
    print("=" * 74)
    print("  labelling input : the question + the context the Reader was GIVEN")
    print("  never seen here : gold, the prediction, the correctness label")
    print(f"  questions       : {len(h9['rows'])}")
    print()

    # ---------------- BLIND LABELLING PASS ---------------------------------
    labelled: list[dict] = []
    for r in h9["rows"]:
        qid = r["question_id"]
        cq = (meta.get(qid) or {}).get("cache_qid", "")
        context = frozen.get(cq, "")
        if not context:
            labelled.append({"question_id": qid, "condition": "UNKNOWN",
                             "note": "no frozen context"})
            continue
        q = (meta.get(qid) or {}).get("question", "")

        lines = [l for l in context.split("\n") if l.strip()]
        # SPEAKER BUG: the regex used to be r"\]\s*([A-Z][a-z]+):" but the real format is
        #   [D10:3 on 8:56 pm on 20 July, 2023] (In reply to Melanie: "...") Caroline: text
        # so the name AFTER ']' is the reply-target, and the actual speaker comes after
        # the closing quote.  The old pattern caught 'Melanie' from the quote prefix and
        # labelled every speaker field as the reply target, which is why the
        # same-speaker test scored 0/426.
        _HDR = re.compile(r"^\[(D\d+:\d+) on [^\]]+\]\s*")
        _QUOTE = re.compile(r'^(?:\(In reply to[^)]*\)\s*)?(?:"[^"]*"\s*)?([A-Z][a-z]+):')
        events: list[dict] = []
        for i, line in enumerate(lines):
            m = _HDR.match(line)
            if not m:
                continue
            did = m.group(1)
            rest = line[m.end():]
            ms = _QUOTE.match(rest)
            spk = ms.group(1) if ms else ""
            body = rest[ms.end():] if ms else rest
            d, src = header_date(m.group(0))
            if d is None:
                d, src = text_date(body)
            acts = sorted({w for w in re.findall(r"[a-z]+", body.lower())
                           if w in _ACTIVITY})
            # PLACE BUG: _PLACE ran over the whole line including '(In reply to X:)',
            # so the reply-target's NAME was recorded as a place.  Restrict to the
            # body after the speaker prefix.
            places = [p for p in _PLACE.findall(body)]
            events.append({"turn": did, "idx": i, "speaker": spk, "date": d,
                           "date_source": src, "activities": acts,
                           "places": places, "text": body[:120]})

        dated = [e for e in events if e["date"]]
        if not dated:
            labelled.append({"question_id": qid, "condition": "UNKNOWN",
                             "note": "no restorable date in the given context"})
            continue

        # same-day groups
        by_date = defaultdict(list)
        for e in dated:
            by_date[e["date"]].append(e)
        same_day_groups = {d: g for d, g in by_date.items() if len(g) >= 2}

        # co-occurring: >=2 distinct events sharing a date
        cooccurring = bool(same_day_groups)

        # competing: among a same-day group, a DISTINCT event besides the one whose
        # activity/place matches the question's slot
        asked_slots = slots_of(q)
        q_low = q.lower()
        rival_pairs = []
        for d, g in same_day_groups.items():
            anchors = [e for e in g
                       if any(a in q_low for a in e["activities"])
                       or any(p.lower() in q_low for p in e["places"])
                       or any(s in q_low for s in asked_slots)]
            others = [e for e in g if e not in anchors]
            if anchors and others:
                rival_pairs.append({
                    "date": d,
                    "anchor_turn": anchors[0]["turn"],
                    "anchor_activities": anchors[0]["activities"],
                    "rival_turn": others[0]["turn"],
                    "rival_activities": others[0]["activities"],
                    "rival_places": others[0]["places"][:3],
                    "distance": abs(others[0]["idx"] - anchors[0]["idx"]),
                    "anchor_speaker": anchors[0]["speaker"] or None,
                    "rival_speaker": others[0]["speaker"] or None,
                    # BUG: this was bool(a and a == b), so two EMPTY speakers compared
                    # equal and 119/237 cases were reported as 'rival from the same
                    # speaker' without a single speaker having been extracted.  An
                    # unknown speaker must never count as agreement.
                    "same_speaker": bool(anchors[0]["speaker"]
                                         and anchors[0]["speaker"] == others[0]["speaker"]),
                })
        competing = bool(rival_pairs)

        # Four cells.  Adjacency MUST be measured in SOURCE coordinates, not in
        # frozen-context lines: the frozen context is ~24 lines and nearly every pair
        # in it is 'adjacent', which made two cells degenerate (563 / 0) and proved
        # nothing.  The [D16:6] header IS a source turn id (session 16, turn 6), so
        # adjacency = same session AND |turn index diff| <= 1, which is exactly what
        # M9 measured on the original dialogue.
        def _tid(t: str) -> tuple[int, int] | None:
            m = re.search(r"D(\d+):(\d+)", t or "")
            return (int(m.group(1)), int(m.group(2))) if m else None

        if competing:
            p = rival_pairs[0]
            ga, gr = _tid(p["anchor_turn"]), _tid(p["rival_turn"])
            adjacent = bool(ga and gr and ga[0] == gr[0] and abs(ga[1] - gr[1]) <= 1)
        else:
            ids = [t for t in (_tid(e["turn"]) for e in dated) if t]
            adjacent = any(a[0] == b[0] and abs(a[1] - b[1]) <= 1
                           for a, b in zip(ids, ids[1:]))
        if competing and adjacent:
            cell = "SAMEDAY_ADJACENT"
        elif competing:
            cell = "SAMEDAY_NOT_ADJACENT"
        elif adjacent:
            cell = "DIFFDAY_ADJACENT"
        else:
            cell = "DIFFDAY_NOT_ADJACENT"

        labelled.append({
            "question_id": qid,
            "condition": "PRESENT" if competing else "ABSENT",
            "cell": cell,
            "cooccurring_same_day": cooccurring,
            "competing": competing,
            "n_same_day_groups": len(same_day_groups),
            "asked_slots": sorted(asked_slots),
            "rival_pairs": rival_pairs[:4],
            "n_events": len(events), "n_dated": len(dated),
            "any_same_speaker_rival": any(p["same_speaker"] for p in rival_pairs),
        })

    # ---------------- JOIN OUTCOMES ONLY NOW -------------------------------
    conv_of = {r["question_id"]: r["conversation"] for r in h9["rows"]}
    outcome = {r["question_id"]: r["mutually_exclusive"] for r in h9["rows"]}
    for x in labelled:
        x["outcome"] = outcome.get(x["question_id"], "UNKNOWN")
        x["conversation"] = conv_of.get(x["question_id"], "")
        x["split"] = ("TUNING" if x["question_id"] in tuning_ids else "HOLDOUT")

    n_present = sum(1 for x in labelled if x["condition"] == "PRESENT")
    n_absent = sum(1 for x in labelled if x["condition"] == "ABSENT")
    n_unknown = sum(1 for x in labelled if x["condition"] == "UNKNOWN")
    print("CONDITION (labelled blind)")
    print(f"  PRESENT (competing same-day events) : {n_present}")
    print(f"  ABSENT                                : {n_absent}")
    print(f"  UNKNOWN (no restorable date)          : {n_unknown}")
    print()

    # ---------------- the table the mission asked for ----------------------
    def wilson(k: int, n: int, z: float = 1.96):
        """Wilson score interval -- honest at small n and near 0/1, unlike normal."""
        if n == 0:
            return (0.0, 0.0)
        p = k / n
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
        return (max(0.0, centre - half), min(1.0, centre + half))

    def block(rows, label):
        n = len(rows)
        if not n:
            return None
        cor = sum(1 for r in rows if r["outcome"] == "CORRECT")
        wro = sum(1 for r in rows if r["outcome"] == "INCORRECT")
        absn = n - cor - wro
        ans = cor + wro
        lo, hi = wilson(cor, n)
        return {"label": label, "n": n, "correct": cor, "wrong": wro,
                "abstain": absn,
                "acc": 100 * cor / n, "acc_lo": 100 * lo, "acc_hi": 100 * hi,
                "wrong_rate": 100 * wro / n,
                "abstain_rate": 100 * absn / n,
                "wrong_of_answered": (100 * wro / ans) if ans else 0.0,
                "answered": ans}

    print("=" * 74)
    print("CONDITION vs OUTCOME  (95% Wilson interval on accuracy)")
    print("=" * 74)
    print(f"  {'metric':<26}{'PRESENT':>18}{'ABSENT':>18}{'diff':>10}")
    pres = [x for x in labelled if x["condition"] == "PRESENT"]
    abst = [x for x in labelled if x["condition"] == "ABSENT"]
    bp, ba = block(pres, "PRESENT"), block(abst, "ABSENT")
    for key, name in (("n", "questions"), ("acc", "accuracy %"),
                      ("wrong_rate", "wrong rate % (of all)"),
                      ("abstain_rate", "abstain rate %"),
                      ("wrong_of_answered", "wrong % of answered")):
        pv = bp[key] if bp else 0
        av = ba[key] if ba else 0
        print(f"  {name:<26}{pv:>18.1f}{av:>18.1f}{pv - av:>+10.1f}")
    if bp and ba:
        print()
        print(f"  accuracy 95% CI  PRESENT [{bp['acc_lo']:.1f}, {bp['acc_hi']:.1f}]"
              f"   ABSENT [{ba['acc_lo']:.1f}, {ba['acc_hi']:.1f}]")
        overlap = not (bp["acc_lo"] > ba["acc_hi"] or ba["acc_lo"] > bp["acc_hi"])
        print(f"  intervals overlap: {overlap}"
              f"{'  <- NOT separable' if overlap else '  <- separable'}")
    print()

    # ---------------- four cells, M9's confound kept visible ---------------
    print("=" * 74)
    print("FOUR CELLS -- is it the DATE, the ADJACENCY, or both?")
    print("=" * 74)
    cells = defaultdict(list)
    for x in labelled:
        if x["condition"] == "UNKNOWN":
            continue
        cells[x["cell"]].append(x)
    print(f"  {'cell':<26}{'n':>6}{'correct':>9}{'wrong':>7}{'abstain':>9}{'acc':>8}")
    for cell in ("SAMEDAY_ADJACENT", "SAMEDAY_NOT_ADJACENT",
                 "DIFFDAY_ADJACENT", "DIFFDAY_NOT_ADJACENT"):
        b = block(cells.get(cell, []), cell)
        if not b:
            print(f"  {cell:<26}{0:>6}{'-':>9}{'-':>7}{'-':>9}{'-':>8}")
            continue
        print(f"  {cell:<26}{b['n']:>6}{b['correct']:>9}{b['wrong']:>7}"
              f"{b['abstain']:>9}{b['acc']:>7.1f}%")
    print()

    # ---------------- per-conversation + leave-one-out sensitivity ----------
    print("=" * 74)
    print("PER-CONVERSATION (questions are NOT independent -- 10 dialogues only)")
    print("=" * 74)
    print(f"  {'conv':<10}{'n':>5}{'PRESENT':>9}{'acc_pres':>10}{'n_abs':>7}"
          f"{'acc_abs':>10}")
    for cv in sorted({r["conversation"] for r in h9["rows"]}):
        sub_p = [x for x in pres if x.get("conversation") == cv]
        sub_a = [x for x in abst if x.get("conversation") == cv]
        bp_ = block(sub_p, cv); ba_ = block(sub_a, cv)
        print(f"  {cv:<10}{len(sub_p) + len(sub_a):>5}{len(sub_p):>9}"
              f"{(bp_['acc'] if bp_ else float('nan')):>10.1f}"
              f"{len(sub_a):>7}{(ba_['acc'] if ba_ else float('nan')):>10.1f}")
    print()

    print("LEAVE-ONE-CONVERSATION-OUT sensitivity on the accuracy gap")
    gaps = []
    for cv in sorted({r["conversation"] for r in h9["rows"]}):
        p2 = [x for x in pres if x.get("conversation") != cv]
        a2 = [x for x in abst if x.get("conversation") != cv]
        b2p, b2a = block(p2, ""), block(a2, "")
        if b2p and b2a:
            g = b2p["acc"] - b2a["acc"]
            gaps.append((cv, g))
            print(f"  drop {cv:<8} gap {g:+6.1f}pp   (n={b2p['n']}/{b2a['n']})")
    if gaps:
        worst = min(gaps, key=lambda t: t[1])
        print(f"  gap range {min(g for _, g in gaps):+.1f}pp .. "
              f"{max(g for _, g in gaps):+.1f}pp; "
              f"never vanishes (min {worst[1]:+.1f}pp dropping {worst[0]})"
              if all(g > 0 for _, g in gaps) else
              f"  the gap CHANGES SIGN or vanishes when dropping "
              f"{[c for c, g in gaps if g <= 0]}")
    print()

    # speaker kept SEPARATE, never merged into the condition
    spk = [x for x in pres if x.get("any_same_speaker_rival")]
    print(f"  SEPARATE FIELD, not merged: of {n_present} PRESENT cases, "
          f"{len(spk)} have a rival from the SAME speaker")
    print()

    # ---------------- decision gate ---------------------------------------
    print("=" * 74)
    print("DECISION GATE (pre-declared)")
    print("=" * 74)
    if not (bp and ba):
        print("  cannot evaluate -- a group is empty")
    else:
        gap = bp["acc"] - ba["acc"]
        sep = not (bp["acc_lo"] > ba["acc_hi"] or ba["acc_lo"] > bp["acc_hi"])
        # 'sep' means the intervals DO overlap (the names were inverted at one point,
        # which would have let a non-separable gap be read as a separable one).
        separated = not sep
        wrong_direction = gap > 0
        neg = [c for c, g in gaps if g <= 0]
        print(f"  measured gap (PRESENT - ABSENT) = {gap:+.1f}pp   "
              f"CI-separated = {separated}")
        if wrong_direction and separated:
            print(f"  PRESENT is MORE accurate by {gap:+.1f}pp and the intervals do not")
            print("  overlap.  So the condition does not predict failure at all -- it")
            print("  predicts SUCCESS.  Reporting it as a failure cause would be an")
            print("  inversion of the data.")
            print("  -> REJECT.  No intervention justified.")
        elif separated and gap < -5 and not neg:
            print(f"  PRESENT accuracy is LOWER by {gap:+.1f}pp, intervals do not")
            print("  overlap, and the gap survives every leave-one-out drop.")
            print("  -> same-day competing events ARE associated with failure across")
            print("     the benchmark.  Day-level disambiguation is a justified")
            print("     intervention candidate, to be proven by a paired A/B on holdout.")
        elif separated and gap < -5:
            print(f"  gap {gap:+.1f}pp and intervals separate, BUT dropping "
                  f"{neg} removes it.")
            print("  -> conversation-confounded.  Not a general cause yet.")
        else:
            print(f"  gap {gap:+.1f}pp and the confidence intervals OVERLAP, so the")
            print("  difference is not resolvable at this sample size.")
            print("  -> REJECT.  M9's pattern is LOCAL to the sampled failures; do NOT")
            print("     promote it to a production feature.")
        if wrong_direction:
            print()
            print("  DIRECTION CHECK: the sign of the effect is the WRONG WAY for a")
            print("  failure cause.  Whatever M9 found, it does not replicate as a")
            print("  benchmark-wide risk factor.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 10 -- same-day competing events, blind-labelled over 989",
        "baseline": {"head": "9e47fd6", "production_changed": False,
                     "evidence_binding": "unwired, preserved"},
        "labelling_was_blind": True,
        "split": {"tuning_convs": split["tuning_conversations"],
                  "holdout_convs": split["holdout_conversations"]},
        "n_present": n_present, "n_absent": n_absent, "n_unknown": n_unknown,
        "present": bp, "absent": ba,
        "cells": {c: block(v, c) for c, v in cells.items()},
        "same_speaker_rival_cases": len(spk),
        "rows": labelled,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
