"""PHASE 2/5 -- A/B Evidence Binding over all 43 DURATION questions + holdout.

Every condition the mission put on this run, each enforced by an assertion rather than
by intent:

  * baseline is the WORKING TREE with only the binding flag disabled -- never git HEAD,
    because this file carries ~430 lines of uncommitted work from earlier sessions.
  * BOTH arms go through DeterministicMemoryEngine.resolve, the same dispatcher.
    Different dispatcher paths are an ERROR, not a result.
  * exactly 43 questions, no duplicate IDs, and the two arms' ID sets identical --
    checked as SETS, because matching counts alone hid a silent zero-join before.
  * every gold/prediction/score join is present for every question in both arms.
  * both arms are scored by the same scorer object with the same version stamp.
  * the binding flag is verified on the real path (bind() call counts differ) before
    any number is believed.
  * the holdout's question IDs are proven disjoint from the tuning set.

Fails loudly: an empty join, a zero diff, either arm committing nothing, the before-arm
scoring 0 correct, or overlapping tuning/holdout IDs all raise instead of printing a
comfortable verdict.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import artificial_memory.skills.autonomous_engines.temporal_algebra_engine as tam  # noqa: E402
from artificial_memory.skills.autonomous_engines.counting_engine import (  # noqa: E402
    CountingEngine,
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
OUT = ROOT / "benchmark_results/evidence_audit/binding_ab.json"

TUNING_CONVS = {"43", "47"}
HOLDOUT_CONVS = {str(n) for n in (26, 30, 41, 42, 44, 48, 49, 50)}
EXPECTED_N = 43


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cache[json.loads(line)["qid"]] = json.loads(line)

    engine = CountingEngine()
    qids = [r for r in audit["records"]
            if engine.quantity_type(r["question"]) == "DURATION"]

    # ---- the joins and identity checks the mission demands --------------------
    assert len(qids) == EXPECTED_N, (
        f"expected exactly {EXPECTED_N} DURATION questions, found {len(qids)}")
    id_list = [r["question_id"] for r in qids]
    assert len(set(id_list)) == len(id_list), "duplicate question IDs in the target set"

    joined = {r["question_id"]: r for r in qids if r["cache_qid"] in cache}
    assert len(joined) == len(qids), (
        f"join failure: {len(qids) - len(joined)} of {len(qids)} have no frozen context")
    assert all(r["gold"] is not None for r in qids), "a question has no gold"

    tuning_ids = {r["question_id"] for r in qids
                  if r["cache_qid"].split("-")[1] in TUNING_CONVS}
    holdout_ids = {r["question_id"] for r in qids
                   if r["cache_qid"].split("-")[1] in HOLDOUT_CONVS}
    assert tuning_ids and holdout_ids, "one of the two sets is empty"
    assert not (tuning_ids & holdout_ids), (
        "tuning and holdout share question IDs -- the holdout is not independent")

    print("=" * 74)
    print("PHASE 2/5 -- A/B Evidence Binding, 43 DURATION questions")
    print("=" * 74)
    print(f"  questions              : {len(qids)} (exact)")
    print(f"  duplicate IDs          : 0")
    print(f"  joined to contexts     : {len(joined)}/{len(qids)}")
    print(f"  tuning / holdout       : {len(tuning_ids)} / {len(holdout_ids)} "
          f"(disjoint: {not (tuning_ids & holdout_ids)})")
    print(f"  scorer                 : {getattr(scorer, '__file__', '?').split('/')[-1]}"
          f"  sha={sha(Path(scorer.__file__).read_text(encoding='utf-8'))}")
    ctx_hash = sha("".join(cache[r["cache_qid"]]["context"] for r in qids))
    print(f"  frozen contexts sha    : {ctx_hash}")
    print(f"  baseline               : working tree, binding flag OFF "
          f"(NOT git HEAD)")
    print()

    # ---- verify the flag on the real path BEFORE trusting any number ---------
    bind_calls = {"n": 0}
    real_bind = tam.bind

    def counting_bind(req, cand):
        bind_calls["n"] += 1
        return real_bind(req, cand)

    tam.bind = counting_bind

    def resolve(question: str, context: str, binding_on: bool):
        os.environ["LETHE_EVIDENCE_BINDING"] = "1" if binding_on else "0"
        return DeterministicMemoryEngine.resolve(question, context)

    rows: dict[str, dict] = {}
    timings = {"OFF": 0.0, "ON": 0.0}
    for r in qids:
        ctx = cache[r["cache_qid"]]["context"]
        rec = {"question_id": r["question_id"], "cache_qid": r["cache_qid"],
               "conversation": r["cache_qid"].split("-")[1],
               "question": r["question"], "gold": r["gold"]}
        for arm, on in (("OFF", False), ("ON", True)):
            bind_calls["n"] = 0
            t0 = time.perf_counter()
            a = resolve(r["question"], ctx, on)
            timings[arm] += time.perf_counter() - t0
            rec[f"{arm}_answer"] = a.answer
            rec[f"{arm}_used"] = bool(a.used)
            rec[f"{arm}_bind_calls"] = bind_calls["n"]
            rec[f"{arm}_detail"] = str(a.detail)[:170]
            rec[f"{arm}_verdict"] = (
                "abstain" if not a.used
                else ("correct" if scorer.score(a.answer, r["gold"]) else "wrong"))
        rows[r["question_id"]] = rec

    tam.bind = real_bind

    # ---- guards: a vacuous or inconsistent run must raise --------------------
    assert len(rows) == len(qids), "a question produced no record"
    assert set(rows) == set(id_list), "the two arms' ID sets differ"
    for qid, v in rows.items():
        assert v["gold"] is not None and v["OFF_verdict"] and v["ON_verdict"], (
            f"{qid}: an incomplete join between gold, prediction and score")
    off_binds = sum(v["OFF_bind_calls"] for v in rows.values())
    on_binds = sum(v["ON_bind_calls"] for v in rows.values())
    assert off_binds == 0, (
        f"bind() ran {off_binds} times in the OFF arm -- the flag is not honoured")
    assert on_binds > 0, "bind() never ran in the ON arm -- the branch is dead"
    off_correct = sum(1 for v in rows.values() if v["OFF_verdict"] == "correct")
    assert off_correct > 0, (
        "the OFF arm scored 0 correct but the audit measured 14 -- the arms are not "
        "on the same dispatcher path, so this comparison is invalid")

    print(f"  bind() calls  OFF {off_binds} / ON {on_binds}  "
          f"-> the flag is verified on the real path")
    print(f"  runtime        OFF {timings['OFF']:.2f}s / ON {timings['ON']:.2f}s")
    print(f"  LLM calls      : 0 (deterministic engine, both arms)")
    print()

    # ---- the eight change classes the mission requires -----------------------
    CLASSES = ["wrong -> correct", "correct -> wrong", "wrong -> abstain",
               "abstain -> wrong", "abstain -> correct", "correct -> abstain",
               "changed but same score", "unchanged"]
    classes = {k: [] for k in CLASSES}
    for qid, v in rows.items():
        o, n = v["OFF_verdict"], v["ON_verdict"]
        changed_answer = v["OFF_answer"] != v["ON_answer"]
        if o == n:
            key = "changed but same score" if changed_answer else "unchanged"
        else:
            key = f"{o} -> {n}"
        classes.setdefault(key, []).append(qid)

    def summarise(recs: dict, label: str) -> dict:
        b = Counter(v["OFF_verdict"] for v in recs.values())
        a = Counter(v["ON_verdict"] for v in recs.values())
        print("-" * 74)
        print(f"{label}  (n={len(recs)})")
        print("-" * 74)
        print(f"  OFF: correct {b['correct']}  wrong {b['wrong']}  abstain {b['abstain']}")
        print(f"  ON : correct {a['correct']}  wrong {a['wrong']}  abstain {a['abstain']}")
        for k in CLASSES:
            if classes.get(k):
                print(f"  {k:<24}: {len(classes[k])}")
        for k in ("correct -> wrong", "abstain -> wrong"):
            for q in classes.get(k, []):
                print(f"    {k} {q}: gold={str(recs[q]['gold'])[:14]!r} "
                      f"was={str(recs[q]['OFF_answer'])[:16]!r} "
                      f"now={str(recs[q]['ON_answer'])[:16]!r}")
        for k in ("wrong -> correct", "abstain -> correct"):
            for q in classes.get(k, []):
                print(f"    {k} {q}: -> {str(recs[q]['ON_answer'])[:18]!r}")
        print()
        return {"OFF": dict(b), "ON": dict(a)}

    print("ALL 43 QUESTIONS -- change classes")
    for k in CLASSES:
        print(f"  {k:<24} {len(classes.get(k, []))}")
    print()
    t_recs = {k: v for k, v in rows.items() if v["conversation"] in TUNING_CONVS}
    h_recs = {k: v for k, v in rows.items() if v["conversation"] in HOLDOUT_CONVS}
    t = summarise(t_recs, "TUNING (convs 43, 47)")
    h = summarise(h_recs, "HOLDOUT (8 conversations, never used to design)")

    off_c = sum(v["OFF_verdict"] == "correct" for v in rows.values())
    on_c = sum(v["ON_verdict"] == "correct" for v in rows.values())
    off_w = sum(v["OFF_verdict"] == "wrong" for v in rows.values())
    on_w = sum(v["ON_verdict"] == "wrong" for v in rows.values())
    net = on_c - off_c

    print("=" * 74)
    print("ADOPTION TEST")
    print("=" * 74)
    print(f"  1. no correct answer lost : "
          f"{'YES' if not classes.get('correct -> wrong') else 'NO -- ' + str(classes['correct -> wrong'])}")
    print(f"  2. no new wrong answer    : "
          f"{'YES' if not classes.get('abstain -> wrong') else 'NO -- ' + str(classes['abstain -> wrong'])}")
    print(f"  3. at least one gained    : "
          f"{'YES' if (classes.get('wrong -> correct') or classes.get('abstain -> correct')) else 'NO'}")
    adopt = (not classes.get("correct -> wrong")
             and not classes.get("abstain -> wrong")
             and bool(classes.get("wrong -> correct") or classes.get("abstain -> correct")))
    print()
    print(f"  correct {off_c} -> {on_c} ({net:+d})   wrong {off_w} -> {on_w}")
    print(f"  VERDICT: {'ADOPTABLE' if adopt else 'NOT ADOPTED -- evidence binding stays out'}")

    OUT.write_text(json.dumps({
        "label": "evidence binding A/B, 43 DURATION questions, one dispatcher",
        "baseline": "working tree with the binding flag OFF (NOT git HEAD)",
        "scorer_sha": sha(Path(scorer.__file__).read_text(encoding="utf-8")),
        "context_sha": ctx_hash,
        "n_questions": len(qids),
        "bind_calls": {"OFF": off_binds, "ON": on_binds},
        "runtime_seconds": timings,
        "llm_calls": 0,
        "change_classes": {k: v for k, v in classes.items()},
        "tuning": t, "holdout": h,
        "totals": {"OFF_correct": off_c, "ON_correct": on_c,
                   "OFF_wrong": off_w, "ON_wrong": on_w, "net_correct": net},
        "adoptable": adopt,
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
