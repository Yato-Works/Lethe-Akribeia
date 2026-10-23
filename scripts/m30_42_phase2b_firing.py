#!/usr/bin/env python3
"""M30-42 Phase 2b: Firing logs + Reader-input hash across ablations.

For Bucket A 28:
  1. Instrument the three name mechanisms and count how often each FIRES.
  2. Run compile() under each ablation and sha256 the resulting context_text.

If context_text is byte-identical across ablations, then Reader, Answer
Extraction and scorer are necessarily identical (same input -> same output),
so any top-k difference did NOT propagate downstream.

Read-only wrt production. No LLM.
"""
import sys, json, re, hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory_gen.context.msc_compiler import MinimumSufficientContextCompiler
from artificial_memory.recall import evidence_scorer as es
from artificial_memory.recall import state_reconstructor as sr
from artificial_memory.recall import query_planner as qp

FIRE = {"es_actor_calls": 0, "es_actor_hit": 0,
        "sr_actor_calls": 0, "sr_actor_hit": 0,
        "planner_calls": 0, "planner_entities_nonempty": 0}

_ORIG_ES = es.UniversalEvidenceScorer.extract_target_actor
_ORIG_SR = sr.StateReconstructor._extract_target_actor


def inst_es(self, q):
    FIRE["es_actor_calls"] += 1
    if FIRE.get("disable_es_actor"):
        return None
    r = _ORIG_ES(self, q)
    if r:
        FIRE["es_actor_hit"] += 1
    return r


def inst_sr(self, q):
    FIRE["sr_actor_calls"] += 1
    if FIRE.get("disable_sr_actor"):
        return None
    r = _ORIG_SR(self, q)
    if r:
        FIRE["sr_actor_hit"] += 1
    return r


def load_gold():
    locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
    by_q = {}
    for conv in locomo:
        sid = conv.get("sample_id", "")
        for qa in conv.get("qa", []):
            by_q[(sid, qa.get("question", "").strip().lower())] = qa.get("evidence", []) or []
    return by_q


def make_records(turns):
    return [StructuredIR(
        entity=t.get("speaker", ""), property="statement", value=t.get("text", ""),
        source=t.get("speaker", ""),
        raw_content=f"[{t.get('dia_id','')}] {t.get('speaker','')}: {t.get('text','')}",
    ) for t in turns]


def main():
    bucket = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_bucketA_all.json").read_text(encoding="utf-8"))
    convs = bucket["conversations"]
    cases = {c["question_id"]: c for c in bucket["cases"]}
    inter = json.loads((PROJECT_ROOT / "benchmark_results/evidence_audit/v4_intervention_temporal.json").read_text(encoding="utf-8"))
    targets = sorted(r["question_id"] for r in inter["rows"] if r.get("group") == "TARGET")

    compiler = MinimumSufficientContextCompiler()
    ORIG_KNOWN = list(es.UniversalEvidenceScorer.KNOWN_ACTORS)
    ORIG_EXP = es.UniversalEvidenceScorer.SEMANTIC_EXPANSIONS
    ORIG_NAMED = set(qp.QueryPlanner.NAMED_ENTITIES)

    print("=" * 78)
    print("M30-42 Phase 2b: Firing logs + Reader-input hash")
    print("=" * 78)
    print()

    def reset():
        es.UniversalEvidenceScorer.extract_target_actor = _ORIG_ES
        sr.StateReconstructor._extract_target_actor = _ORIG_SR
        es.UniversalEvidenceScorer.KNOWN_ACTORS = ORIG_KNOWN
        es.UniversalEvidenceScorer.SEMANTIC_EXPANSIONS = ORIG_EXP
        qp.QueryPlanner.NAMED_ENTITIES = ORIG_NAMED

    def apply(dis):
        reset()
        # zero counters FIRST, then set disable flags (else the zeroing wipes them)
        for k in ("es_actor_calls", "es_actor_hit", "sr_actor_calls", "sr_actor_hit",
                  "planner_calls", "planner_entities_nonempty"):
            FIRE[k] = 0
        FIRE["disable_es_actor"] = "A" in dis
        FIRE["disable_sr_actor"] = "B" in dis
        if "C" in dis:
            qp.QueryPlanner.NAMED_ENTITIES = set()
        if "E" in dis:
            es.UniversalEvidenceScorer.SEMANTIC_EXPANSIONS = {}

    configs = [("Baseline", ""), ("A", "A"), ("B", "B"), ("C", "C"),
               ("E", "E"), ("F", "ABCE")]
    results = {}

    for label, dis in configs:
        apply(dis)
        # instrument
        es.UniversalEvidenceScorer.extract_target_actor = inst_es
        sr.StateReconstructor._extract_target_actor = inst_sr

        hashes = []
        for qid in targets:
            case = cases.get(qid)
            if not case:
                continue
            turns = convs.get(case.get("source_conversation", ""), [])
            if not isinstance(turns, list):
                continue
            p = qp.QueryPlanner()
            plan = p.plan(case.get("question", ""))
            FIRE["planner_calls"] += 1
            if plan.target_entities:
                FIRE["planner_entities_nonempty"] += 1
            try:
                pcc = compiler.compile(case.get("question", ""), make_records(turns))
                h = hashlib.sha256(pcc.context_text.encode("utf-8")).hexdigest()[:16]
            except Exception as e:
                h = f"ERR:{str(e)[:20]}"
            hashes.append(h)

        # restore instrumentation for SR/ES so the counter isn't double-wrapped
        results[label] = {
            "fire": dict(FIRE),
            "input_hash_join": hashlib.sha256("|".join(hashes).encode()).hexdigest()[:16],
            "hashes": hashes,
        }
        print(f"{label:10s} es_actor_hit={FIRE['es_actor_hit']:4d}/{FIRE['es_actor_calls']:4d}  "
              f"sr_actor_hit={FIRE['sr_actor_hit']:4d}/{FIRE['sr_actor_calls']:4d}  "
              f"planner_ent={FIRE['planner_entities_nonempty']:2d}/{FIRE['planner_calls']:2d}  "
              f"ctx_hash={results[label]['input_hash_join']}")

    reset()
    print()

    base_h = results["Baseline"]["input_hash_join"]
    print("=" * 78)
    print("READER-INPUT EQUIVALENCE")
    print("=" * 78)
    print(f"Baseline context hash: {base_h}")
    for label in ["A", "B", "C", "E", "F"]:
        same = results[label]["input_hash_join"] == base_h
        n_diff = sum(1 for a, b in zip(results[label]["hashes"], results["Baseline"]["hashes"]) if a != b)
        print(f"  {label:3s}: hash={'SAME' if same else 'DIFFERENT'}  questions_with_different_context={n_diff}/28")
    print()

    out = PROJECT_ROOT / "results/m30_42"
    out.mkdir(parents=True, exist_ok=True)
    (out / "firing_and_input_hash.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out / 'firing_and_input_hash.json'}")


if __name__ == "__main__":
    main()
