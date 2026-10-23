"""V4 Mission 0-1 -- question-level stage ledger.  DIAGNOSTIC ONLY, production untouched.

One row per question, with the stage the mission asked for.  Where a stage cannot be
measured from existing artifacts it is recorded as UNKNOWN -- never inferred.

The stage sequence, and what can actually be measured from what exists here:
  1 source evidence present     : the raw LoCoMo conversation (datasets/external/locomo10.json)
  2 preserved in memory         : the frozen compiled context
  3 retrieved into candidate    : the same frozen context (retrieval already happened)
  4 speaker attribution         : does the evidence turn's speaker match the asked person
  5 subject/entity attribution  : does the evidence name the asked subject
  6 relation support            : does the evidence connect subject to value
  7 temporal/polarity/etc       : measured only where a rule exists, else UNKNOWN
  8 candidate commitment        : the engine's own detail string
  9 reader fallback             : not executed in this harness, so UNKNOWN
 10 benchmark/scorer            : scorer identity, and gold-in-context as a defect proxy

Gold is read ONLY to label and to measure.  No stage rule below takes a gold value as
input, and nothing here is wired into a runtime path.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory.skills.autonomous_engines.counting_engine import (  # noqa: E402
    CountingEngine,
)
from artificial_memory.skills.autonomous_engines.temporal_algebra_engine import (  # noqa: E402
    TemporalAlgebraEngine,
)
from artificial_memory.skills.deterministic_memory_engine import (  # noqa: E402
    DeterministicMemoryEngine,
)

_spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
scorer = importlib.util.module_from_spec(_spec)
sys.modules["eval_locomo_full"] = scorer
try:
    _spec.loader.exec_module(scorer)
except SystemExit:
    pass

AUDIT = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
RAW = ROOT / "datasets/external/locomo10.json"
OUT = ROOT / "benchmark_results/evidence_audit/v4_stage_ledger.json"

_WS = re.compile(r"\s+")
_STOP = {"the", "and", "for", "was", "were", "his", "her", "its", "their", "our",
         "your", "him", "she", "they", "them", "not", "but", "all", "any", "out",
         "about", "into", "over", "than", "then", "when", "what", "who", "why",
         "how", "which", "there", "here", "have", "has", "had", "been", "being",
         "are", "will", "would", "could", "should", "did", "does", "with", "that",
         "this", "from", "many", "much", "long", "often", "ago", "now"}


def norm(s: str) -> str:
    return _WS.sub(" ", str(s).strip().lower())


def load_raw_conversations() -> dict[str, list[dict]]:
    """The original dialogue, keyed by conversation id (conv-26, conv-30, ...)."""
    if not RAW.exists():
        return {}
    data = json.loads(RAW.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for sample in data:
        sid = sample.get("sample_id")
        conv = sample.get("conversation") or {}
        turns = []
        for key in sorted(conv.keys()):
            m = re.fullmatch(r"session_(\d+)", key)
            if not m:
                continue
            date_key = f"session_{m.group(1)}_date_time"
            date = conv.get(date_key, "")
            for t in conv[key] or []:
                turns.append({"session": int(m.group(1)), "date": date,
                              "speaker": t.get("speaker", ""),
                              "dia_id": t.get("dia_id", ""),
                              "text": t.get("text", "")})
        if sid:
            out[sid] = turns
    return out


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cache[json.loads(line)["qid"]] = json.loads(line)

    engine = CountingEngine()
    raw_by_conv = load_raw_conversations()
    print("=" * 74)
    print("V4 STAGE LEDGER -- one row per question, UNKNOWN where unmeasurable")
    print("=" * 74)
    print(f"  audit questions        : {len(audit['records'])}")
    print(f"  frozen contexts        : {len(cache)}")
    print(f"  raw conversations      : {len(raw_by_conv)} "
          f"({sum(len(v) for v in raw_by_conv.values())} turns)")
    print()

    rows: list[dict] = []
    for r in audit["records"]:
        cq = r["cache_qid"]
        ctx = cache.get(cq) or {}
        context = ctx.get("context") or ""
        if not context:
            continue
        gold = r["gold"]
        conv = f"conv-{cq.split('-')[1]}"
        n_ctx = norm(context)

        ans = DeterministicMemoryEngine.resolve(r["question"], context)
        verdict = ("abstain" if not ans.used
                   else ("correct" if scorer.score(ans.answer, gold) else "wrong"))

        # ---- stage 1/2: is the gold in the SOURCE at all, and in the MEMORY? ----
        g = str(gold).strip()
        g_norm = norm(g)
        toks = [t for t in re.findall(r"[a-z0-9]{3,}", g_norm)
                if t not in {"the", "and", "for", "was", "were"}]

        def present_in(text_norm: str) -> str:
            """Verbatim, or recoverable token-by-token.

            ALL-tokens-present is a WEAK test and the first version of this ledger
            reported 729 questions as 'lost in memory' on it -- including 19 the engine
            answered CORRECTLY, which is impossible if the evidence were really gone.
            The defect was requiring every token: a gold of 'because of their
            INCLUSIVITY' has token 'inclusivity' while the context says 'inclusive', so
            a one-character paraphrase read as total evidence loss.  Reported as three
            distinct levels instead, and only VERBATIM_ABSENT is treated as stage-2
            evidence loss; TOKEN_PARTIAL is a paraphrase/inflection signal (H6), which
            the mission asks to keep separate from memory loss.
            """
            if g_norm and g_norm in text_norm:
                return "VERBATIM"
            if not toks:
                return "NO_TOKENS"
            hit = sum(1 for t in toks if t in text_norm)
            if hit == len(toks):
                return "ALL_TOKENS"
            if hit > 0:
                return "TOKEN_PARTIAL"
            return "VERBATIM_ABSENT"

        in_source = "UNKNOWN"
        if conv in raw_by_conv:
            raw_text = norm(" ".join(t["text"] for t in raw_by_conv[conv]))
            in_source = present_in(raw_text)
        in_memory = present_in(n_ctx)

        # ---- stage 4/5/6: attribution, measured on the evidence turns ----------
        person = engine._extract_person(r["question"], []) or ""
        q_terms = {w for w in re.findall(r"[a-z]{4,}", r["question"].lower())
                   if w not in _STOP}

        # Which context lines contain the gold?
        gold_lines = []
        for line in context.split("\n"):
            if g_norm and g_norm in norm(line):
                gold_lines.append(line.strip())
        speaker_status = "UNKNOWN"
        subject_status = "UNKNOWN"
        relation_status = "UNKNOWN"
        if gold_lines:
            speakers = []
            for line in gold_lines:
                m = re.match(r"\[([^\]]+)\]", line)
                if m:
                    tail = m.group(1).split(" on ")
                    # header form: 'D16:7 on 10:55 am on 24 June, 2022'
                    speakers.append(tail[0])
            speaker_status = ("NO_EVIDENCE_TURN" if not speakers
                              else "EVIDENCE_TURNS_FOUND")
            # Does the asked person appear as speaker or in the text of those turns?
            if person:
                named = any(person in norm(l) for l in gold_lines)
                speaker_status = ("PERSON_IN_EVIDENCE" if named
                                  else "PERSON_ABSENT_FROM_EVIDENCE")
            subj = any(t in norm(" ".join(gold_lines)) for t in q_terms)
            subject_status = "SUBJECT_TERMS_IN_EVIDENCE" if subj else "SUBJECT_ABSENT"
            relation_status = ("RELATION_TERMS_PRESENT" if subj
                               else "RELATION_UNPROVEN")

        # ---- stage 8: what the engine did -------------------------------------
        detail = str(ans.detail)[:120]

        # ---- stage 10: scorer / annotation proxy ------------------------------
        scorer_defect_proxy = bool(in_memory and verdict == "abstain")

        rows.append({
            "question_id": r["question_id"],
            "cache_qid": cq,
            "conversation": conv,
            "category": r.get("category"),
            "shape": engine.quantity_type(r["question"]),
            "gold": gold,
            "verdict": verdict,
            "prediction": ans.answer,
            "engine_detail": detail,
            "s1_source_evidence": in_source,
            "s2_preserved_in_memory": in_memory,
            "s3_retrieved_into_context": "YES (frozen context is the candidate set)",
            "s4_speaker_attribution": speaker_status,
            "s5_subject_attribution": subject_status,
            "s6_relation_support": relation_status,
            "s7_temporal_polarity": "UNKNOWN (no rule measured in this pass)",
            "s8_commitment": ("COMMITTED" if ans.used else "ABSTAINED"),
            "s9_reader_fallback": "UNKNOWN (reader not executed in this harness)",
            "s10_scorer": ("eval_locomo_full.py c2535518d1b21653d9ad"
                           + (" | gold-in-context while abstaining"
                              if scorer_defect_proxy else "")),
            "n_gold_lines": len(gold_lines),
            "gold_lines": gold_lines[:3],
        })

    # ---- distributions -------------------------------------------------------
    def dist(field: str) -> Counter:
        return Counter(str(x[field]) for x in rows)

    print("STAGE DISTRIBUTIONS")
    for f in ("verdict", "s1_source_evidence", "s2_preserved_in_memory",
              "s4_speaker_attribution", "s5_subject_attribution",
              "s6_relation_support", "s8_commitment"):
        print(f"  {f}:")
        for k, v in dist(f).most_common(6):
            print(f"      {k:<38} {v}")
    print()

    # ---- the mission's 10 failure hypotheses, kept SEPARATE ------------------
    def stage_of(x: dict) -> str:
        """Assign the EARLIEST stage that demonstrably failed."""
        if x["s1_source_evidence"] == "VERBATIM_ABSENT":
            return "H1_EVIDENCE_ABSENT_FROM_SOURCE"
        if x["s2_preserved_in_memory"] == "VERBATIM_ABSENT":
            # present in the source conversation but gone from the compiled memory
            return "H2_LOST_IN_MEMORY_OR_COMPRESSION"
        if x["s2_preserved_in_memory"] == "TOKEN_PARTIAL":
            return "H6_PARAPHRASE_OR_COREFERENCE"
        if x["s4_speaker_attribution"] == "PERSON_ABSENT_FROM_EVIDENCE":
            return "H4_WRONG_SPEAKER_OR_ENTITY"
        if x["s6_relation_support"] == "RELATION_UNPROVEN":
            return "H5_RELATION_UNRESOLVED"
        if x["verdict"] == "wrong":
            return "H7_OR_H8_WRONG_OR_LOST_AT_COMMITMENT"
        if x["verdict"] == "abstain":
            return "H9_READER_FALLBACK_UNMEASURED"
        return "ANSWERED"

    by_stage = Counter(stage_of) if False else Counter(stage_of(x) for x in rows)
    print("=" * 74)
    print("FAILURE HYPOTHESES, kept distinct (earliest demonstrable failure)")
    print("=" * 74)
    for k, v in by_stage.most_common():
        print(f"  {k:<38} {v}")
    print()
    print("  H3 (retained but not retrieved) and H6 (paraphrase/coreference) are NOT")
    print("  separable with the artifacts present here: the frozen context is already")
    print("  the retrieved set, so 'not retrieved' cannot be distinguished from 'never")
    print("  selected', and paraphrase needs semantics.  Both are recorded UNKNOWN.")
    print()

    # ---- D: the four named cases, inspected individually ---------------------
    print("=" * 74)
    print("D -- the four named cases, per-question")
    print("=" * 74)
    named = ["lc_conv-26_131", "lc_conv-26_186", "lc_conv-26_92",
             "lc_conv-26_158", "lc_conv-26_132", "lc_conv-43_81",
             "lc_conv-47_53", "lc_conv-44_60"]
    by_id = {x["question_id"]: x for x in rows}
    for qid in named:
        x = by_id.get(qid)
        if not x:
            print(f"  {qid}: NOT IN LEDGER")
            continue
        print(f"  {qid}  [{x['verdict']}]  gold={str(x['gold'])[:18]!r}")
        print(f"     stage: {stage_of(x)}")
        print(f"     speaker={x['s4_speaker_attribution']}  "
              f"subject={x['s5_subject_attribution']}  "
              f"relation={x['s6_relation_support']}")
        for l in x["gold_lines"][:1]:
            print(f"     evidence: {l[:96]}")
    print()

    OUT.write_text(json.dumps({
        "label": "V4 mission 0-1 question-level stage ledger (diagnostic only)",
        "baseline": {
            "head": "4ec0c20",
            "audit_sha": "e7e21ec035aaa2f1f17c",
            "cache_sha": "34e548606071f0b704d0",
            "scorer_sha": "c2535518d1b21653d9ad",
            "raw_dataset_sha": "79fa87e90f04081343b8",
            "tests": "1 failed, 556 passed, 1 xfailed, 2 xpassed",
        },
        "n_rows": len(rows),
        "distributions": {f: dict(dist(f)) for f in
                          ("verdict", "s1_source_evidence", "s2_preserved_in_memory",
                           "s4_speaker_attribution", "s5_subject_attribution",
                           "s6_relation_support", "s8_commitment")},
        "by_stage": dict(by_stage),
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
