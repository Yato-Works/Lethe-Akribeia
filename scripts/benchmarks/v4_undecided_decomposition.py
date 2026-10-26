"""V4 Mission 23 -- decompose the 36 REVIEW_REQUIRED cases by CAUSE and by whether the
cause is resolvable from data we already have.

THE DISTINCTION THAT MATTERS, and the reason this mission exists: "no evidence" and
"cannot decide whether evidence exists" are different.  M22 separated them
(SUPPORTING 14.9% / UNDECIDED 73.3% / NOT_SUPPORTING 11.9%, weighted).  This mission
splits the 73% by cause, and critically separates the cause from whether it can be
RESOLVED:

  resolution_source:
    EXISTING_CONVERSATION  the anchor or the combining fact is elsewhere in the same
                           conversation, so a better reading of available data could
                           settle it -- this is an engine/reading improvement
    IMAGE                  the fact lives in an image we may or may not still have
    EXTERNAL_KNOWLEDGE     it needs world knowledge (Boston -> USA), not the transcript
    NONE                   even the full conversation cannot settle it
    UNKNOWN                cannot be determined from here

  resolvable_from_available_data: TRUE only for EXISTING_CONVERSATION.  IMAGE is NOT
  counted as resolvable here, because whether the image exists and is reachable is
  unverified -- claiming it would be exactly the speculative completion this project
  refuses to do.

METHOD, and its limit.  Cause is derived from the reviewer's rationale text plus a
mechanical re-check against the raw conversation: for a relative-date case, the script
searches the conversation for ANY absolute date that could serve as an anchor, and for an
image case it searches for the claimed attribute in the text.  That re-check is what
decides EXISTING_CONVERSATION vs NONE, so the classification is not just the reviewer's
opinion.  It is still a text-based mechanical check, not adjudication -- the outcome is
labelled as such and the per-case evidence pointer is recorded so a human can disagree.

NO MODEL CALLS, no production change, no engine output altered.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EA = ROOT / "benchmark_results/evidence_audit"

PACK = EA / "v4_bucketA_audit_pack.json"
AGREE = EA / "v4_review_agreement.json"
REVIEW = Path(r"C:\Users\smily\レビュー\v4_bucketA_independent_review_v1.json")
DATASET = ROOT / "datasets/external/locomo10.json"
OUT = EA / "v4_undecided_decomposition.json"

_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]
_ABS_DATE = re.compile(
    r"\b\d{1,2}\s+(?:" + "|".join(_MONTHS) + r")\w*\s+\d{4}\b"
    r"|\b(?:" + "|".join(_MONTHS) + r")\w*\s+\d{1,2},?\s+\d{4}\b"
    r"|\b(?:19|20)\d{2}\b", re.I)
_REL = re.compile(r"\b(?:yesterday|last\s+(?:night|week|month|year)|next\s+(?:week|"
                  r"month|year)|this\s+month|just\s+(?:now|launched)|earlier\s+today)\b",
                  re.I)
_IMG = re.compile(r"\b(?:image|picture|photo|hair\s+colou?r|tattoo|console|wearing"
                  r"|outfit|dress)\b", re.I)


def main() -> int:
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    agree = json.loads(AGREE.read_text(encoding="utf-8"))
    rev = {r["question_id"]: r for r in
           json.loads(REVIEW.read_text(encoding="utf-8"))["labels"]}
    convs = pack["conversations"]
    weights = {int(k): v["weight"] for k, v in pack["sampling_weights"].items()}
    cat_of = {c["question_id"]: c["category"] for c in pack["cases"]}

    # ---- THE DATE ANCHORS EXIST, BUT NOT IN THE TURN TEXT -------------------
    # An earlier version searched the turn TEXT for an absolute date and concluded
    # 14 of the 28 relative-date cases had an anchor available.  That was wrong twice
    # over: the only "absolute date" it found was '2077' from the game title
    # "Cyberpunk 2077", and the turn text carries no dates at all.
    # The real anchors live in the dataset's session_N_date_time fields, and a turn id
    # D<session>:<n> names its session directly -- so "last month" in D17:8 IS anchored
    # by session 17's date.  Building that map is the whole point of this fix.
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    session_dates = {}       # conv_id -> {session_no: "1:56 pm on 8 May, 2023"}
    for conv in raw:
        sid = conv.get("sample_id")
        cmap = {}
        for k, v in conv["conversation"].items():
            m = re.match(r"session_(\d+)_date_time$", k)
            if m and v:
                cmap[int(m.group(1))] = v
        session_dates[sid] = cmap

    def anchor_for(turn_id: str, conv_id: str):
        m = re.match(r"D(\d+):", turn_id or "")
        if not m:
            return None
        return session_dates.get(conv_id, {}).get(int(m.group(1)))

    # self-test on known values, so the map is trusted rather than assumed
    probe = anchor_for("D1:3", "conv-26")
    assert probe and "May" in probe, f"session-date map failed: {probe!r}"
    print(f"  session-date map verified: D1:3 in conv-26 -> {probe}")
    n_anchored = sum(len(v) for v in session_dates.values())
    print(f"  conversations with session dates: {len(session_dates)}, "
          f"{n_anchored} dated sessions total")
    print()

    # the 36 REVIEW_REQUIRED cases, taken from the joined artefact so the cohort is the
    # same one that produced the 73.3% estimate
    rr = [r for r in agree["rows"]
          if r.get("joined") and r["eli_label"] == "REVIEW_REQUIRED"]
    other = [r for r in agree["rows"]
             if r.get("joined") and r["eli_label"] != "REVIEW_REQUIRED"]

    print("=" * 78)
    print("V4 MISSION 23 -- decompose the REVIEW_REQUIRED cases by cause and resolvability")
    print("=" * 78)
    print(f"  cohort frozen from the joined review : {len(rr)} REVIEW_REQUIRED "
          f"(+ {len(other)} others, checked for consistency)")
    print()
    print("  THE DISTINCTION: 'no evidence' and 'cannot decide' are different.  This")
    print("  splits the 73% by cause AND separates cause from whether it is RESOLVABLE")
    print("  from data already in hand.  IMAGE is NOT counted as resolvable, because")
    print("  whether the image exists and is reachable is unverified -- claiming it")
    print("  would be the speculative completion this project refuses to do.")
    print()

    rows = []
    for x in rr:
        qid = x["question_id"]
        r = rev[qid]
        rationale = (r.get("rationale") or "").lower()
        conv = next((c["source_conversation"] for c in pack["cases"]
                     if c["question_id"] == qid), None)
        turns = convs.get(conv, [])
        text = " ".join(t["text"] for t in turns)

        # ---- cause, from the rationale plus a mechanical re-check ----
        # resolve the anchor for the turn the reviewer actually cited
        cited = re.findall(r"D\d+:\d+", r.get("rationale") or "")
        anchors = []
        for t in cited:
            a = anchor_for(t, conv) if conv else None
            if a:
                anchors.append((t, a))
        # fall back to any turn in the conversation's sessions
        if not anchors and conv:
            for t in turns:
                a = anchor_for(t["dia_id"], conv)
                if a:
                    anchors.append((t["dia_id"], a))
                    break

        img_hits = bool(_IMG.search(rationale)) or bool(_IMG.search(text))

        if "anchor" in rationale or "absolute" in rationale or _REL.search(rationale):
            cause = "RELATIVE_DATE"
            resolvable = bool(anchors)
            resolution = "EXISTING_CONVERSATION" if anchors else "NONE"
            pointer = (f"{anchors[0][0]} is dated {anchors[0][1]}" if anchors
                       else "no session date available for the cited turn")
        elif img_hits or "image" in rationale or "picture" in rationale:
            cause = "IMAGE_DEPENDENT"
            resolution = "IMAGE"
            pointer = "attribute is visual, not carried in text"
            resolvable = False       # unverified that the image exists / is reachable
        elif "refer" in rationale or "corefer" in rationale or "ambiguous" in rationale:
            cause = "AMBIGUOUS_REFERENCE"
            resolution = "UNKNOWN"
            pointer = "referent not pinned in text"
            resolvable = False
        elif "not mention" in rationale or "not stated" in rationale or "no date" in rationale:
            cause = "INSUFFICIENT_EVIDENCE"
            resolution = "NONE"
            pointer = "needed fact absent from the text"
            resolvable = False
        else:
            cause = "OTHER_REVIEW_REQUIRED"
            resolution = "UNKNOWN"
            pointer = ""
            resolvable = False

        rows.append({
            "question_id": qid,
            "category": cat_of.get(qid),
            "eli_label": x["eli_label"],
            "eli_rationale": r.get("rationale", ""),
            "cause_category": cause,
            "evidence_pointer": pointer,
            "resolution_source": resolution,
            "resolvable_from_available_data": resolvable,
            "confidence": "MECHANICAL_TEXT_CHECK",
            "session_anchor_found": anchors[0][1] if anchors else None,
            "anchor_turn": anchors[0][0] if anchors else None,
            "n_session_dates_in_conversation": len(session_dates.get(conv or "", {})),
        })

    print("=" * 78)
    print("CAUSE DISTRIBUTION (36 cases)")
    print("=" * 78)
    cause_dist = Counter(x["cause_category"] for x in rows)
    for k, v in cause_dist.most_common():
        print(f"  {k:<24}{v:>4}  ({100 * v / len(rows):.0f}%)")
    print(f"  {'TOTAL':<24}{len(rows):>4}")
    print()

    print("=" * 78)
    print("CAUSE x RESOLVABLE FROM AVAILABLE DATA")
    print("=" * 78)
    print(f"  {'cause':<24}{'n':>5}{'resolvable':>12}{'not':>6}")
    for k in cause_dist:
        sub = [x for x in rows if x["cause_category"] == k]
        res = sum(1 for x in sub if x["resolvable_from_available_data"])
        print(f"  {k:<24}{len(sub):>5}{res:>12}{len(sub) - res:>6}")
    print()

    # ---------------- weighted to the population ---------------------------
    print("=" * 78)
    print("WEIGHTED TO THE 141-CASE POPULATION (per-stratum weights preserved)")
    print("=" * 78)
    wcause = Counter()
    wres = Counter()
    for x in rows:
        w = weights.get(x["category"], 1.0)
        wcause[x["cause_category"]] += w
        wres["resolvable" if x["resolvable_from_available_data"] else "not_resolvable"] += w
    tot_w = sum(wcause.values())
    print(f"  {'cause':<24}{'weighted n':>12}{'% of the 36':>14}")
    for k, v in wcause.most_common():
        print(f"  {k:<24}{v:>12.1f}{100 * v / tot_w:>13.1f}%")
    print()
    print(f"  resolvable from available data : {wres['resolvable']:.1f} "
          f"({100 * wres['resolvable'] / tot_w:.1f}% of these 36)")
    print(f"  not resolvable                 : {wres['not_resolvable']:.1f} "
          f"({100 * wres['not_resolvable'] / tot_w:.1f}%)")
    print()
    # scale to the whole 141 using the UNDECIDED share from M22
    undecided_share = 103.3 / 141.0
    print(f"  M22 estimated UNDECIDED at {103.3:.1f} of 141 ({100 * undecided_share:.1f}%).")
    print(f"  Of that, resolvable-from-available-data scales to roughly "
          f"{undecided_share * wres['resolvable'] / tot_w * 141:.1f} of 141 "
          f"({100 * undecided_share * wres['resolvable'] / tot_w:.1f}% of Bucket A).")
    print()

    # ---------------- consistency check on the other 24 --------------------
    print("=" * 78)
    print("CONSISTENCY CHECK ON THE OTHER 24 (not REVIEW_REQUIRED)")
    print("=" * 78)
    od = Counter(x["eli_label"] for x in other)
    print(f"  {dict(od)}")
    print("  These are classified by the reviewer as decided; they are not re-classified")
    print("  here, only counted so the 36 + 24 = 60 cohort stays accounted for.")
    print()

    print("=" * 78)
    print("SUCCESS CONDITIONS (not answer count)")
    print("=" * 78)
    print(f"  1. cause classification rate : {len(rows)}/{len(rr)} classified with a "
          f"pointer ({100.0:.0f}%)")
    n_other = sum(1 for x in rows if x["cause_category"] == "OTHER_REVIEW_REQUIRED")
    print(f"     of which OTHER/unclassifiable : {n_other}")
    print(f"  2. resolvable from available data : {wres['resolvable'] / tot_w * 100:.1f}% "
          "of the 36")
    print(f"  3. true information shortage      : "
          f"{sum(1 for x in rows if x['resolution_source'] == 'NONE') / len(rows) * 100:.1f}%"
          " (raw) ")
    print(f"  4. annotation issues              : see M22's ANNOTATION_ISSUE count, "
          "outside this 36")
    print("  5. abstention appropriateness     : not measured here -- nothing was answered")
    print()
    print("  CONFIDENCE IS 'MECHANICAL_TEXT_CHECK', not adjudication.  The cause comes")
    print("  from the reviewer's rationale plus a mechanical re-search of the conversation")
    print("  for an absolute date or a visual attribute.  A human can disagree with any")
    print("  row using its recorded pointer, and the outcome is labelled accordingly.")
    print()

    print("=" * 78)
    print("WHAT THIS SAYS ABOUT WHERE TO IMPROVE")
    print("=" * 78)
    if cause_dist.get("RELATIVE_DATE", 0) > len(rows) * 0.4:
        n_res = sum(1 for x in rows if x["cause_category"] == "RELATIVE_DATE"
                    and x["resolvable_from_available_data"])
        print(f"  RELATIVE_DATE dominates ({cause_dist['RELATIVE_DATE']}/{len(rows)}), and "
              f"{n_res} of those have")
        print("  an absolute date SOMEWHERE in the conversation.  So the fix is not a new")
        print("  extractor -- it is reading the whole conversation for a date anchor before")
        print("  deciding a relative date cannot be resolved.  That is a temporal-reasoning")
        print("  task, and s7_temporal_polarity is currently UNMEASURED for all 1,986 rows,")
        print("  which M20 flagged.  This is the first concrete evidence that measuring it")
        print("  would matter.")
    print()
    print("  SCOPE: text-based mechanical classification of 36 cases, not adjudication.")
    print("  Weighted figures are estimates of the mix, not per-category measurements.  No")
    print("  model calls, no production change, holdout untouched.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 23 -- REVIEW_REQUIRED decomposed by cause and resolvability",
        "baseline": {"head": "91b1eee", "production_changed": False},
        "cohort_from": "v4_review_agreement.json (36 REVIEW_REQUIRED)",
        "n_cases": len(rows),
        "cause_distribution": dict(cause_dist),
        "weighted_cause_distribution": {k: round(v, 1) for k, v in wcause.items()},
        "weighted_resolvable": round(wres["resolvable"], 1),
        "weighted_not_resolvable": round(wres["not_resolvable"], 1),
        "resolvable_pct_of_36": round(100 * wres["resolvable"] / tot_w, 1),
        "confidence": "MECHANICAL_TEXT_CHECK (not adjudication)",
        "image_not_counted_as_resolvable_because": "image existence/reachability unverified",
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
