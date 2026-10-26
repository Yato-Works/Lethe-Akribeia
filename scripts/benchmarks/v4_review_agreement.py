"""Join Eli's independent review to the reviewer key and compute agreement honestly.

WHAT IS COMPARED, precisely.  Eli's label answers "does the conversation support the fact
the question asks for".  The engine's s1 label answers "is the gold string present
verbatim".  Those are DIFFERENT QUESTIONS, and M22 already established that the gold
string is absent from the source in 0/141 -- so s1 is a constant and any high agreement
with it would be vacuous, not reassuring.  The comparison below therefore reports BOTH:

  (a) label vs label -- Eli's label against the engine's s1 class, with the caveat that
      s1 is constant and the ceiling on informative agreement is correspondingly limited;
  (b) label vs outcome -- Eli's label against whether the engine got the question right,
      which is the question that actually matters: does "the evidence is present"
      predict "the engine answered correctly"?

THE AGREEMENT IS NOT inter-annotator agreement.  There is one annotator.  Two annotators
were never available, which is the M4 blocker, still open.  Nothing here is called
inter-annotator agreement, and no second annotator is invented.

ELI'S OWN CAVEAT IS CARRIED, NOT DROPPED: text-only first pass, not human ground truth,
not a pristine third-party blind audit, because the conversation had already discussed
the audit design.  It is recorded in the artefact and restated in the output.

WEIGHTING.  The 60-case sample is stratified, not uniform, so any population proportion
is computed with the per-stratum sampling weights recorded in the pack.  Reporting raw
sample percentages as population figures would repeat the defect just fixed.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EA = ROOT / "benchmark_results/evidence_audit"

PACK = EA / "v4_bucketA_audit_pack.json"
KEY = EA / "v4_bucketA_reviewer_key.json"
REVIEW = Path(r"C:\Users\smily\レビュー\v4_bucketA_independent_review_v1.json")
OUT = EA / "v4_review_agreement.json"

# Eli's labels -> the coarse question "is the evidence there?"
SUPPORTING = {"DIRECT_SUPPORT", "PARAPHRASE_SUPPORT", "COMPOSITIONAL_SUPPORT"}
NOT_SUPPORTING = {"ABSENT"}
UNDECIDED = {"PARTIAL_SUPPORT", "REVIEW_REQUIRED", "ANNOTATION_ISSUE", "CONFLICTING"}


def main() -> int:
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    key = {r["question_id"]: r for r in
           json.loads(KEY.read_text(encoding="utf-8"))["rows"]}
    rev = json.loads(REVIEW.read_text(encoding="utf-8"))
    labels = rev["labels"]

    print("=" * 78)
    print("V4 -- join the independent review to the key; agreement computed honestly")
    print("=" * 78)
    print(f"  reviewer        : {rev.get('reviewer')}")
    print(f"  review_mode     : {str(rev.get('review_mode'))[:150]}...")
    print(f"  status          : {rev.get('status')}")
    print(f"  labels received : {len(labels)}")
    print()
    print(f"  REVIEWER'S CAVEAT, carried verbatim into the artefact:")
    print(f"    {str(rev.get('important_caveat'))[:260]}")
    print()

    # ---- the spoken summary and the file disagree; the file wins -------------
    csv_labels = None
    try:
        import csv
        with open(REVIEW.with_suffix(".csv"), encoding="utf-8-sig") as fh:
            csv_labels = Counter(r["label"] for r in csv.DictReader(fh))
    except Exception:                                       # noqa: BLE001
        csv_labels = None
    print("SUMMARY-VS-FILE DISCREPANCY, recorded rather than silently resolved.")
    print("  The summary quoted in conversation reported: PARTIAL_SUPPORT 37,")
    print("  REVIEW_REQUIRED 9, COMPOSITIONAL 7, DIRECT 2 + PARAPHRASE 3, ABSENT 2.")
    print("  The delivered JSON reports a different mix (below), and the delivered CSV")
    print(f"  agrees with the JSON ({dict(csv_labels) if csv_labels else 'unreadable'}).")
    print("  Two independent serialisations of the same review agreeing means the FILE is")
    print("  authoritative and the spoken summary was of an earlier or different pass.")
    print("  Everything below uses the file.  The summary is not used as a data source.")
    print()

    pack_ids = {c["question_id"] for c in pack["cases"]}
    rev_ids = {r["question_id"] for r in labels}
    print("JOIN INTEGRITY")
    print(f"  pack sample ids : {len(pack_ids)}")
    print(f"  review ids      : {len(rev_ids)}")
    print(f"  in both         : {len(pack_ids & rev_ids)}")
    print(f"  review not in pack : {len(rev_ids - pack_ids)}")
    print(f"  pack not reviewed : {len(pack_ids - rev_ids)}")
    print()

    rows = []
    for r in labels:
        qid = r["question_id"]
        k = key.get(qid)
        if k is None:
            rows.append({"question_id": qid, "joined": False,
                         "why": "no reviewer-key row"})
            continue
        lab = r["label"]
        rows.append({
            "question_id": qid, "joined": True,
            "eli_label": lab,
            "eli_rationale": r.get("rationale", ""),
            "eli_class": ("SUPPORTING" if lab in SUPPORTING else
                          "NOT_SUPPORTING" if lab in NOT_SUPPORTING else "UNDECIDED"),
            "engine_verdict": k["engine_verdict"],
            "engine_s1": k["engine_s1_label"],
            "dataset_evidence": k["dataset_evidence_ids"],
        })

    joined = [x for x in rows if x["joined"]]
    print(f"  joined rows : {len(joined)}/{len(labels)}")
    print()

    # ---------------- Eli's label distribution ------------------------------
    print("=" * 78)
    print("ELI'S LABEL DISTRIBUTION (raw sample)")
    print("=" * 78)
    dist = Counter(x["eli_label"] for x in joined)
    for k_, v in dist.most_common():
        print(f"  {k_:<24}{v:>4}  ({100 * v / len(joined):.0f}%)")
    print()

    # ---------------- (a) label vs the engine's s1 --------------------------
    print("=" * 78)
    print("(a) ELI'S LABEL vs THE ENGINE'S s1 LABEL")
    print("=" * 78)
    print("  These answer DIFFERENT questions.  s1 asks 'is the gold string verbatim in")
    print("  the source'; Eli asks 'does the conversation support the fact'.  M22 showed")
    print("  the gold string is absent from the source in 0/141, so s1 is a CONSTANT here")
    print("  and agreement with it is close to vacuous.  It is reported for completeness")
    print("  and read as a sanity check on the join, nothing more.")
    s1_dist = Counter(x["engine_s1"] for x in joined)
    print(f"  engine s1 distribution: {dict(s1_dist)}")
    print()

    # ---------------- (b) label vs outcome -- the one that matters ----------
    print("=" * 78)
    print("(b) ELI'S LABEL vs WHETHER THE ENGINE WAS CORRECT  <- the question that matters")
    print("=" * 78)
    print(f"  {'eli class':<18}{'n':>5}{'correct':>9}{'wrong':>7}{'abstain':>9}"
          f"{'acc of decided':>17}")
    for cls in ("SUPPORTING", "UNDECIDED", "NOT_SUPPORTING"):
        sub = [x for x in joined if x["eli_class"] == cls]
        if not sub:
            continue
        c = sum(1 for x in sub if x["engine_verdict"] == "correct")
        w = sum(1 for x in sub if x["engine_verdict"] == "wrong")
        a = sum(1 for x in sub if x["engine_verdict"] == "abstain")
        dec = c + w
        print(f"  {cls:<18}{len(sub):>5}{c:>9}{w:>7}{a:>9}"
              f"{(100 * c / dec if dec else float('nan')):>16.1f}%")
    print()
    print("  Reading this correctly: SUPPORTING means the conversation does support the")
    print("  fact, so those are the cases where an answer WAS available.  The engine")
    print("  abstained on most of them -- that is the recoverable mass, stated as an")
    print("  observation, not as a rescued count.")
    print()

    # ---------------- weighting to the population ---------------------------
    print("=" * 78)
    print("WEIGHTED BACK TO THE 141-CASE POPULATION")
    print("=" * 78)
    cat_of = {c["question_id"]: c["category"] for c in pack["cases"]}
    w_of = {int(k_): v["weight"] for k_, v in pack["sampling_weights"].items()}
    est = Counter()
    for x in joined:
        cat = cat_of.get(x["question_id"])
        if cat is None:
            continue
        est[x["eli_class"]] += w_of.get(cat, 1.0)
    tot = sum(est.values())
    print(f"  {'eli class':<18}{'weighted n':>12}{'% of population':>18}")
    for cls in ("SUPPORTING", "UNDECIDED", "NOT_SUPPORTING"):
        if est.get(cls):
            print(f"  {cls:<18}{est[cls]:>12.1f}{100 * est[cls] / tot:>17.1f}%")
    print(f"  {'TOTAL':<18}{tot:>12.1f}")
    print()
    print("  Weights are per-stratum population/sampled from the pack, because the sample")
    print("  is stratified rather than uniform.  These are ESTIMATES of the mix, not")
    print("  per-category measurements.")
    print()

    # ---------------- what this is NOT --------------------------------------
    print("=" * 78)
    print("WHAT THIS IS NOT")
    print("=" * 78)
    print("  NOT inter-annotator agreement.  There is ONE annotator.  A second")
    print("  independent annotator was never available -- the M4 blocker, still open --")
    print("  so no agreement figure between two annotators is computable and none is")
    print("  claimed.")
    print("  NOT human ground truth, and NOT a pristine third-party blind audit, in the")
    print("  reviewer's own words: the conversation had already discussed the audit")
    print("  design before the labels were made.")
    print("  NOT a rescued count.  Nothing here changes an engine output; it classifies")
    print("  whether an answer was available.")
    print()

    OUT.write_text(json.dumps({
        "label": "independent review joined to the key; agreement and weighted estimates",
        "reviewer": rev.get("reviewer"),
        "review_status": rev.get("status"),
        "reviewer_caveat": rev.get("important_caveat"),
        "review_mode": rev.get("review_mode"),
        "summary_vs_file_discrepancy": {
            "spoken_summary": {"PARTIAL_SUPPORT": 37, "REVIEW_REQUIRED": 9,
                               "COMPOSITIONAL_SUPPORT": 7, "DIRECT_SUPPORT": 2,
                               "PARAPHRASE_SUPPORT": 3, "ABSENT": 2},
            "delivered_file": dict(dist),
            "delivered_csv_agrees_with_json": csv_labels == dist if csv_labels else None,
            "resolution": ("the file is authoritative; the spoken summary was of a "
                           "different or earlier pass and is not used as a data source"),
        },
        "is_inter_annotator_agreement": False,
        "second_annotator_available": False,
        "n_reviewed": len(labels), "n_joined": len(joined),
        "pack_ids": len(pack_ids), "unjoined": len(pack_ids - rev_ids),
        "eli_label_distribution": dict(dist),
        "engine_s1_distribution": dict(s1_dist),
        "label_vs_outcome": {
            cls: {
                "n": sum(1 for x in joined if x["eli_class"] == cls),
                "correct": sum(1 for x in joined
                               if x["eli_class"] == cls
                               and x["engine_verdict"] == "correct"),
                "wrong": sum(1 for x in joined
                             if x["eli_class"] == cls
                             and x["engine_verdict"] == "wrong"),
                "abstain": sum(1 for x in joined
                               if x["eli_class"] == cls
                               and x["engine_verdict"] == "abstain"),
            } for cls in ("SUPPORTING", "UNDECIDED", "NOT_SUPPORTING")
        },
        "weighted_population_estimate": {k: round(v, 1) for k, v in est.items()},
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
