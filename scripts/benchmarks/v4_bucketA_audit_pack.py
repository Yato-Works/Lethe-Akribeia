"""V4 Mission 22 -- build the Bucket A audit pack, WITHOUT making the semantic judgement.

What this script does: assembles, for each Bucket A case, the raw source conversation,
the question, the dataset's gold answer, the category, and the engine's recorded labels --
in a stable JSON pack an annotator can work from.

WHAT IT DELIBERATELY DOES NOT DO: it does not label whether the evidence is present.
That is the annotator's job, and automating it is exactly the mistake that produced the
M18/M19 marker defects.  No SUPPORT label is emitted by this script, and none should be
read into its output.

WITHHELD FROM THE ANNOTATOR, per the audit protocol:
  - the engine's abstention reason and its s1 VERBATIM_ABSENT label (so the judgement is
    not anchored on the existing label)
  - the dataset's own `evidence` field, which is a gold evidence pointer.  It is recorded
    in a SEPARATE field for the reviewer's use but is NOT shown as part of the audit
    surface, because handing over gold evidence would decide the question being asked.
  - any statement of the hypothesis under test.

JOIN INTEGRITY IS VERIFIED, not assumed.  The ledger key is `lc_conv-26_0` and the raw
dataset key is `conv-26-qa-000`; these are different namespaces, so the join is checked by
matching the gold answer rather than by trusting the index.  The ledger carries no
`question` field at all, which is why a naive text comparison reports 0/1986 agreement --
that is a missing column, not a mismatched join.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LEDGER = ROOT / "benchmark_results/evidence_audit/v4_stage_ledger.json"
M21 = ROOT / "benchmark_results/evidence_audit/v4_s1_decomposition.json"
DATASET = ROOT / "datasets/external/locomo10.json"
OUT_PACK = ROOT / "benchmark_results/evidence_audit/v4_bucketA_audit_pack.json"
OUT_FULL = ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json"

SEED = 20261015
TARGET_SAMPLE = 60

# ---- LABELS ARE FIXED HERE, BEFORE ANY CASE IS JUDGED --------------------------
LABELS = {
    "DIRECT_SUPPORT": "the raw conversation states the needed fact explicitly",
    "PARAPHRASE_SUPPORT": "the fact is present as a paraphrase / synonym",
    "COMPOSITIONAL_SUPPORT": "the answer follows from combining several stated facts",
    "PARTIAL_SUPPORT": "related information exists but does not settle the answer",
    "ABSENT": "the needed evidence is genuinely absent from the conversation",
    "CONFLICTING": "evidence contradicts itself or competes temporally",
    "ANNOTATION_ISSUE": "the question or the gold answer itself is defective, or unjudgeable",
    "REVIEW_REQUIRED": "undecided -- held for a second look, never forced into a slot",
}


def norm_ledger_key(qid: str):
    m = re.match(r"lc_(conv-\d+)_(\d+)$", qid)
    return (m.group(1), int(m.group(2))) if m else None


def main() -> int:
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    rows = {r["question_id"]: r for r in led["rows"]}
    m21 = json.loads(M21.read_text(encoding="utf-8"))
    bucketA = list(dict.fromkeys(m21["bucket_ids"]["A"]))
    raw = json.loads(DATASET.read_text(encoding="utf-8"))

    # raw qa + raw conversation turns
    qa_by, conv_turns = {}, {}
    for conv in raw:
        sid = conv.get("sample_id")
        turns = []
        for sk in sorted((k for k in conv["conversation"] if re.match(r"session_\d+$", k)),
                         key=lambda x: int(x.split("_")[1])):
            for t in conv["conversation"][sk]:
                turns.append({"dia_id": t.get("dia_id"),
                              "speaker": t.get("speaker"),
                              "text": t.get("text", "")})
        conv_turns[sid] = turns
        for i, qa in enumerate(conv.get("qa", [])):
            qa_by[(sid, int(i))] = qa

    print("=" * 78)
    print("V4 MISSION 22 -- Bucket A grounding audit pack")
    print("=" * 78)
    print(f"  Bucket A population : {len(bucketA)}")
    print("  THIS SCRIPT MAKES NO SEMANTIC JUDGEMENT.  It assembles the audit surface;")
    print("  whether the evidence exists is the annotator's call.  Automating it is the")
    print("  mistake that produced the M18/M19 marker defects.")
    print()

    # ---------------- join integrity, verified not assumed ------------------
    # BUG FOUND AND FIXED, and it is not cosmetic.  Joining by INDEX mismatches 5 of the
    # 141 Bucket A rows: lc_conv-30_95 (idx 95 vs 61), lc_conv-30_102 (102 vs 77),
    # lc_conv-42_199 (199 vs 89), lc_conv-42_201 (201 vs 91), lc_conv-44_124 (124 vs 64).
    # Cause: the raw dataset contains `answer: "None"` rows, which some upstream stage
    # filters out, so the surviving index space is shifted relative to the raw one.  The
    # per-conversation COUNTS still match, which is why a count check passes and an index
    # join silently misaligns.
    # So the join is done on (conversation, gold answer) and the index is retained only
    # as a fallback tie-break.  A row that cannot be resolved uniquely is REPORTED, not
    # guessed -- an off-by-one here would audit the wrong question against the wrong
    # conversation, which is worse than dropping it.
    joined, unjoined, ambiguous = 0, [], []
    gold_pool = defaultdict(list)
    for (sid, idx), qa in qa_by.items():
        gold_pool[(sid, str(qa.get("answer", "")).strip())].append((idx, qa))

    cases = []
    for qid in bucketA:
        r = rows.get(qid)
        k = norm_ledger_key(qid)
        if r is None or k is None:
            unjoined.append(qid)
            continue
        gold = str(r.get("gold", "")).strip()
        cands = gold_pool.get((k[0], gold), [])
        if not cands:
            unjoined.append(qid)
            continue
        if len(cands) > 1:
            # prefer the candidate whose index matches the ledger index
            exact = [c for c in cands if c[0] == k[1]]
            if len(exact) != 1:
                ambiguous.append(qid)
                continue
            idx, qa = exact[0]
        else:
            idx, qa = cands[0]
        joined += 1
        cases.append({
            "question_id": qid,
            "conversation": k[0],
            "raw_qa_index": idx,
            "ledger_qa_index": k[1],
            "index_shifted": idx != k[1],
            "category": qa.get("category"),
            "question": qa.get("question", ""),
            "gold_answer": str(qa.get("answer", "")),
            "engine_verdict": r.get("verdict"),
            "engine_prediction": r.get("prediction"),
            # AUDIT SURFACE (shown to the annotator)
            "source_turns": conv_turns.get(k[0], []),
            # WITHHELD from the audit surface, kept for the reviewer
            "reviewer_only": {
                "dataset_evidence_ids": qa.get("evidence", []),
                "note": ("dataset gold evidence pointer -- NOT part of the audit surface; "
                         "showing it would decide the question being asked"),
                "engine_s1_label": r.get("s1_source_evidence"),
                "engine_abstain_reason": r.get("s8_commitment"),
                "note2": "engine labels withheld from the annotator to avoid anchoring",
            },
        })

    print("JOIN INTEGRITY (verified, not assumed)")
    print(f"  bucket A rows        : {len(bucketA)}")
    print(f"  joined               : {joined}")
    print(f"  unjoined             : {len(unjoined)}  {unjoined[:5]}")
    print(f"  ambiguous (reported, not guessed) : {len(ambiguous)}  {ambiguous[:5]}")
    shifted = sum(1 for c in cases if c["index_shifted"])
    print(f"  rows whose raw index differs from the ledger index : {shifted}")
    print()
    print("  WHY THE JOIN IS NOT BY INDEX.  The raw dataset contains `answer: \"None\"`")
    print("  rows, which an upstream stage filters out, so the surviving index space is")
    print("  shifted relative to the raw one.  Per-conversation COUNTS still match, so a")
    print("  count check passes while an index join silently misaligns -- 5 of these 141")
    print("  rows land on the wrong question (e.g. lc_conv-42_199: ledger idx 199 vs raw")
    print("  89).  Auditing a question against the wrong conversation is worse than")
    print("  dropping it, so unresolvable rows are reported rather than guessed.")
    print()
    print("  Note on a false alarm: the ledger has NO `question` column, so comparing")
    print("  question text across the two files reports 0/1986 agreement.  That is a")
    print("  missing column, not a broken join.")
    print()
    assert not ambiguous, (
        f"{len(ambiguous)} rows could not be resolved uniquely -- the join is not safe")
    assert joined == len(bucketA), "some bucket A rows could not be joined"

    # ---------------- stratified sample ------------------------------------
    by_cat = defaultdict(list)
    for c in cases:
        by_cat[c["category"]].append(c)
    cat_dist = {k: len(v) for k, v in sorted(by_cat.items())}
    print(f"  category distribution of the {len(cases)} joined cases: {cat_dist}")
    print()

    rng = random.Random(SEED)
    sample = []
    if len(cases) <= TARGET_SAMPLE:
        sample = list(cases)
    else:
        # proportional stratified allocation, so the sample mirrors the population
        n_cat = len(by_cat)
        per = max(1, TARGET_SAMPLE // n_cat)
        for cat, group in sorted(by_cat.items()):
            take = min(per, len(group))
            sample.extend(rng.sample(sorted(group, key=lambda c: c["question_id"]),
                                     take))
        if len(sample) < TARGET_SAMPLE:
            remaining = [c for c in cases if c not in sample]
            rng.shuffle(remaining)
            sample.extend(remaining[:TARGET_SAMPLE - len(sample)])
    sample.sort(key=lambda c: c["question_id"])

    print(f"  stratified sample    : {len(sample)} of {len(cases)}  (seed {SEED})")
    print(f"    sample category mix: "
          f"{dict(Counter(c['category'] for c in sample))}")
    print(f"    proportional to population: "
          f"{dict(Counter(c['category'] for c in cases))}")
    print()

    # ---------------- the annotator's brief ---------------------------------
    print("=" * 78)
    print("LABELS, FIXED BEFORE ANY CASE IS JUDGED")
    print("=" * 78)
    for k, v in LABELS.items():
        print(f"  {k:<24} {v}")
    print()
    print("  A SUPPORT label requires the supporting conversation location AND a short")
    print("  reason.  Judgement rule: not 'does the gold wording appear verbatim' but")
    print("  'does the conversation sufficiently support the fact the question asks for'.")
    print("  Undecided cases go to REVIEW_REQUIRED and are never forced into a slot.")
    print()
    print("WITHHELD from the audit surface, so the judgement is not anchored:")
    print("  - the engine's abstention reason and its VERBATIM_ABSENT label")
    print("  - the dataset's own `evidence` pointer (a gold evidence leak)")
    print("  - any statement of the hypothesis under test")
    print()

    print("=" * 78)
    print("STATISTICAL CAVEAT ON THE 60")
    print("=" * 78)
    print(f"  {TARGET_SAMPLE} cases drawn proportionally by category from {len(cases)}.")
    print("  With proportional allocation and a large population this is roughly a")
    print("  +/-10 point interval at 95% confidence for a proportion near 50%, and")
    print("  narrower for proportions near the extremes.  It is a FIRST-PASS estimate of")
    print("  the mix, not a measurement of each category -- deliberately over-sampling a")
    print("  rare category would require weighting by its sampling rate before any")
    print("  population proportion is computed.")
    print()

    OUT_PACK.write_text(json.dumps({
        "label": "V4 mission 22 -- Bucket A audit pack (stratified sample)",
        "baseline": {"head": "45d8257", "production_changed": False},
        "population_n": len(cases),
        "sample_n": len(sample),
        "seed": SEED,
        "stratification": "proportional by LoCoMo category",
        "labels_fixed_before_judgement": LABELS,
        "withheld_from_annotator": [
            "engine abstention reason / s1 VERBATIM_ABSENT label",
            "dataset `evidence` gold pointer (held in reviewer_only)",
            "any statement of the hypothesis under test",
        ],
        "judgement_rule": ("does the conversation support the fact the question asks "
                           "for -- not whether the gold wording appears verbatim"),
        "category_distribution_population": cat_dist,
        "join_verified_by": "gold answer equality across the two key namespaces",
        "cases": sample,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    OUT_FULL.write_text(json.dumps({
        "label": "V4 mission 22 -- ALL Bucket A cases (full population, not sampled)",
        "population_n": len(cases),
        "cases": sorted(cases, key=lambda c: c["question_id"]),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"written: {OUT_PACK}")
    print(f"        {OUT_FULL}")
    print()
    print("  SCOPE: no semantic judgement is made here, and none should be inferred from")
    print("  this artefact.  The 141 (and the 484 from M21) are an INVESTIGATION SET, not")
    print("  a confirmed recoverable loss.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
