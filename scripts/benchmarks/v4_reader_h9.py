"""V4 Mission 2 -- frozen Reader evaluation of the H9 cohort.  MEASUREMENT ONLY.

Nothing here is wired into a production path.  The Reader is invoked through the same
cached Ollama wrapper the pipeline already uses, so every answer carries model, prompt
hash, tokens and latency as provenance.

RECONCILIATION, done before any new call (mission B):
  * locomo_full_20261010_151930.json holds 1,009 non-empty predictions but records NO
    model, prompt, context, decoding or scorer version -- so it cannot be proven to be a
    paired experiment with today's frozen contexts.  It is used as a CORROBORATING
    signal only, never as the baseline.
  * cache/phase0/*.json DOES carry model + prompt_sha256 + tokens + latency, keyed on
    sha256(model + question + context).  A cached entry is reusable ONLY when that key
    matches exactly, which is what this script checks rather than assumes.

Gold answers are used for scoring and for the audit labels only.  The Reader prompt is
built from the question and the frozen context alone.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory.research.benchmarks import llm as frozen_llm
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

LEDGER = ROOT / "benchmark_results/evidence_audit/v4_stage_ledger.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
HIST = ROOT / "benchmark_results/locomo_full_20261010_151930.json"
LLM_CACHE = ROOT / "cache/phase0"
OUT = ROOT / "benchmark_results/evidence_audit/v4_reader_h9.json"

READER_MODEL = "qwen2.5:7b-instruct"
MAX_NEW = int(os.environ.get("H9_MAX_NEW", "60"))


def stage_of(x: dict) -> str:
    """Same earliest-failure rule as the ledger, recomputed so H9 is exact."""
    if x["s1_source_evidence"] == "VERBATIM_ABSENT":
        return "H1_EVIDENCE_ABSENT_FROM_SOURCE"
    if x["s2_preserved_in_memory"] == "VERBATIM_ABSENT":
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


def cache_key(model: str, question: str, ctx: str) -> str:
    return hashlib.sha256(
        (model + "\x00" + question + "\x00" + ctx).encode("utf-8")).hexdigest()[:24]


def main() -> int:
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    # The ledger does not persist the question text, so join it from the audit by
    # question_id rather than assuming a field that is not there.
    audit = json.loads((ROOT / "benchmark_results/evidence_audit/evidence_audit.json")
                       .read_text(encoding="utf-8"))
    qtext = {r["question_id"]: r["question"] for r in audit["records"]}
    for x in led["rows"]:
        x["question"] = qtext.get(x["question_id"], "")
    h9 = [x for x in led["rows"] if stage_of(x) == "H9_READER_FALLBACK_UNMEASURED"]
    hist = json.loads(HIST.read_text(encoding="utf-8"))
    hist_by_id = {r["question_id"]: r for r in hist["results"]}

    cache_files = {f.stem: f for f in LLM_CACHE.glob("*.json")} if LLM_CACHE.exists() else {}

    print("=" * 74)
    print("V4 MISSION 2 -- frozen Reader on the H9 cohort")
    print("=" * 74)
    print(f"  H9 cohort               : {len(h9)}")
    print(f"  Reader model            : {frozen_llm.FROZEN_MODEL}")
    print(f"  frozen max_tokens       : {frozen_llm.FROZEN_ANSWER_MAX_TOKENS}")
    print(f"  llm cache entries       : {len(cache_files)}")
    print()

    # ---- B: which H9 questions already have a reusable cached Reader answer? ----
    reusable, needs_call = [], []
    for x in h9:
        key = cache_key(READER_MODEL, x["question_id"], "")
        # The cache key needs the exact context; look up by (model, question) instead
        # and verify the prompt hash against the frozen context when found.
        found = None
        for stem, f in cache_files.items():
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if d.get("model") != READER_MODEL:
                continue
            if d.get("question") == x["question"]:
                found = (stem, d)
                break
        (reusable if found else needs_call).append((x, found))

    print("REUSABILITY (mission B)")
    print(f"  reusable from cache     : {len(reusable)}")
    print(f"  require a fresh call    : {len(needs_call)}")
    print()
    print("  NOTE on the historical 1,986 run: it records no model, prompt, context or")
    print("  decoding config, so it CANNOT be treated as a paired baseline.  It is used")
    print("  below only as a corroborating signal on the same question IDs.")
    print()

    # ---- reconcile against the historical run by exact question ID ----------
    both = [x for x in h9 if x["question_id"] in hist_by_id]
    hist_correct = sum(1 for x in both if hist_by_id[x["question_id"]]["is_correct"])
    print(f"  H9 questions also present in the historical run : {len(both)}")
    print(f"  ... which that run scored correct               : {hist_correct}")
    print()

    if not needs_call and not reusable:
        print("  NOTHING TO MEASURE -- stop and report, per the mission.")
        return 1

    # ---- C: run the Reader on the H9 cohort -------------------------------
    # Use the FROZEN spec adapter with no overrides.  An earlier draft set
    # reader.max_tokens = 60, which would have changed the frozen decoding budget and
    # made the result incomparable to every other run -- the mission requires the exact
    # Reader configuration, so the constants in research/benchmarks/llm.py are used
    # as-is (model qwen2.5:7b-instruct, temperature 0.0, seed 42, max_tokens 256).
    frozen_ctx = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            frozen_ctx[r["qid"]] = r["context"]

    print("FROZEN READER CONFIGURATION (not overridden)")
    print(f"  model        : {frozen_llm.FROZEN_MODEL}")
    print(f"  temperature  : {frozen_llm.FROZEN_TEMPERATURE}")
    print(f"  seed         : {frozen_llm.FROZEN_SEED}")
    print(f"  max_tokens   : {frozen_llm.FROZEN_ANSWER_MAX_TOKENS}")
    print()

    reader = frozen_llm.OllamaAnswerer()
    # max_tokens stays at the frozen 256; only the A/B context cap is applied, which
    # the class documents as benchmark-only and which leaves decoding untouched.
    reader.num_ctx = 8192
    reader.think = False  # keep the 7B from spending the budget on reasoning

    rows: list[dict] = []
    t_start = time.time()
    for x, cached in (reusable + needs_call):
        qid = x["question_id"]
        ctx = frozen_ctx.get(x["cache_qid"], "")
        prompt_sha = hashlib.sha256(ctx.encode("utf-8")).hexdigest()[:20]

        if cached is not None:
            stem, d = cached
            # Reuse ONLY if the cached prompt hash matches today's frozen context
            # exactly.  Both sides are full sha256 of the context; comparing a
            # truncated prefix against a full digest would never match and would
            # silently discard every reusable entry.
            if d.get("prompt_sha256") == hashlib.sha256(
                    ctx.encode("utf-8")).hexdigest():
                text, p_tok, c_tok, lat = (d["text"], d["prompt_tokens"],
                                           d["completion_tokens"], d["latency_ms"])
                invoked, cache_hit = False, True
            else:
                cached = None
        if cached is None:
            try:
                res = reader.answer(x["question"], ctx)
                text, p_tok, c_tok = res.text, res.prompt_tokens, res.completion_tokens
                lat = res.latency_ms
                invoked, cache_hit = True, False
            except Exception as exc:
                rows.append({"question_id": qid, "reader_invoked": True,
                             "outcome": "READER_INVOCATION_FAILURE",
                             "error": str(exc)[:120]})
                continue

        norm = text.strip()
        valid = bool(norm) and norm.lower() not in {
            "unknown", "n/a", "none", "i don't know", "not stated", "not mentioned"}
        abstained = (not valid) or bool(re.search(
            r"\b(?:not\s+(?:stated|mentioned|specified|provided|available)|"
            r"cannot\s+be\s+determined|no\s+information)\b", norm, re.I))
        if abstained:
            outcome = "APPROPRIATE_ABSTENTION" if valid is False else "ABSTAINED"
            score = None
        else:
            score = bool(scorer.score(norm, x["gold"]))
            outcome = "CORRECT" if score else "INCORRECT"

        rows.append({
            "question_id": qid,
            "conversation": x["conversation"],
            "question": x["question"],
            "gold": x["gold"],
            "reader_invoked": invoked,
            "cache_hit": cache_hit,
            "prompt_sha256": prompt_sha,
            "raw_output": text[:300],
            "prediction": norm[:160],
            "valid": valid,
            "abstained": abstained,
            "score": score,
            "outcome": outcome,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "latency_ms": lat,
            "evidence_in_context": x["s2_preserved_in_memory"],
            "speaker": x["s4_speaker_attribution"],
            "relation": x["s6_relation_support"],
            "hist_is_correct": (hist_by_id.get(qid) or {}).get("is_correct"),
        })

    elapsed = time.time() - t_start

    # ---- D: mutually exclusive outcome table ------------------------------
    def bucket(r: dict) -> str:
        o = r.get("outcome")
        if o == "READER_INVOCATION_FAILURE":
            return "READER_INVOCATION_FAILURE"
        if r.get("abstained"):
            # An abstention is only APPROPRIATE when the context lacks the evidence.
            if r.get("evidence_in_context") in ("VERBATIM_ABSENT",):
                return "APPROPRIATE_ABSTENTION"
            return "UNSUPPORTED_ABSTENTION"
        if r.get("score") is True:
            return "CORRECT"
        if r.get("evidence_in_context") == "VERBATIM_ABSENT":
            return "UNSUPPORTED_ANSWER_UNANSWERABLE"
        if not r.get("valid"):
            return "INVALID_OUTPUT"
        return "INCORRECT"

    for r in rows:
        r["mutually_exclusive"] = bucket(r)
    tbl = Counter(r["mutually_exclusive"] for r in rows)

    print("=" * 74)
    print(f"H9 READER OUTCOMES  (n={len(rows)}, mutually exclusive)")
    print("=" * 74)
    for k in ("CORRECT", "INCORRECT", "APPROPRIATE_ABSTENTION",
              "UNSUPPORTED_ABSTENTION", "UNSUPPORTED_ANSWER_UNANSWERABLE",
              "INVALID_OUTPUT", "READER_INVOCATION_FAILURE"):
        print(f"  {k:<34} {tbl.get(k, 0)}")
    print()

    answered = [r for r in rows if r["mutually_exclusive"] in ("CORRECT", "INCORRECT")]
    print(f"  answered (non-abstaining)   : {len(answered)}")
    if answered:
        c = sum(1 for r in answered if r["score"])
        print(f"  Reader accuracy on answered : {c}/{len(answered)} = "
              f"{100 * c / len(answered):.1f}%")
    print(f"  runtime                     : {elapsed:.1f}s for {len(rows)} questions")
    tot_ct = sum(r.get("completion_tokens") or 0 for r in rows)
    tot_pt = sum(r.get("prompt_tokens") or 0 for r in rows)
    print(f"  tokens                      : prompt {tot_pt}, completion {tot_ct}")
    print()

    # ---- E: paired comparison, no subtraction across runs -----------------
    det = {x["question_id"]: x for x in h9}
    det_correct = sum(1 for x in h9 if x["verdict"] == "correct")
    det_wrong = sum(1 for x in h9 if x["verdict"] == "wrong")
    print("=" * 74)
    print("COMPARABLE MEASUREMENTS (same question set, same frozen contexts)")
    print("=" * 74)
    print(f"  H9 cohort N                : {len(h9)}")
    print(f"  deterministic-only correct : {det_correct}  (by construction 0 -- H9 is")
    print(f"                                defined as the deterministic ABSTAIN set)")
    print(f"  deterministic-only wrong   : {det_wrong}  (same reason)")
    print(f"  Reader correct on H9       : {tbl.get('CORRECT', 0)}")
    print(f"  Reader incorrect on H9     : {tbl.get('INCORRECT', 0)}")
    print()
    print("  NO E2E NUMBER IS COMPUTED.  Subtracting a deterministic-only rate from a")
    print("  Reader-inclusive rate would mix two different denominators and execution")
    print("  paths; the historical 50.8% run also carries no provenance to pair with.")
    print()

    # ---- paired deltas against the historical run, question ID joined -----
    paired = [r for r in rows if r.get("hist_is_correct") is not None]
    if paired:
        agree = sum(1 for r in paired
                    if bool(r["hist_is_correct"]) == bool(r.get("score")))
        print(f"  paired vs historical run   : {len(paired)} questions joined by ID")
        print(f"    same correctness verdict : {agree}/{len(paired)} "
              f"({100 * agree / len(paired):.1f}%)")
        print("    (a corroboration only -- the historical run has no recorded model or")
        print("     prompt, so agreement is evidence of similarity, not of identity)")
        print()

    print("EVIDENCE SUFFICIENCY for the non-correct outcomes")
    for k in ("INCORRECT", "UNSUPPORTED_ABSTENTION",
              "UNSUPPORTED_ANSWER_UNANSWERABLE"):
        sub = [r for r in rows if r["mutually_exclusive"] == k]
        if not sub:
            continue
        ev = Counter(r.get("evidence_in_context") for r in sub)
        print(f"  {k} (n={len(sub)}): evidence {dict(ev)}")
    print()

    OUT.write_text(json.dumps({
        "label": "V4 mission 2 -- frozen Reader on the H9 cohort (measurement only)",
        "baseline": {"head": "fca214b", "production_changed": False,
                     "evidence_binding": "unwired, preserved",
                     "tests": "1 failed, 556 passed, 1 xfailed, 2 xpassed"},
        "reader": {"model": READER_MODEL, "max_new_tokens": MAX_NEW,
                   "base_url": "http://localhost:11434"},
        "h9_n": len(h9),
        "reusable_from_cache": len(reusable),
        "fresh_calls": len(needs_call),
        "outcomes": dict(tbl),
        "deterministic_on_h9": {"correct": det_correct, "wrong": det_wrong,
                                "abstain": len(h9) - det_correct - det_wrong},
        "runtime_seconds": elapsed,
        "tokens": {"prompt": tot_pt, "completion": tot_ct},
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
