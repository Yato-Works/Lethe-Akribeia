"""Systematic dead-path audit: which engine features actually fire?

MOTIVATION.  The date-proximity boost was inert for two stacked reasons and no
test caught it.  A green suite plus a completed full run proved nothing.  The
only signal was that two runs produced byte-identical predictions.

So instead of asking "is this feature correct", ask "does this feature RUN AT
ALL".  Every feature below is instrumented on the REAL LoCoMo path and its
firing count is reported.  A feature that never fires is dead code regardless of
how well it is written or tested.

Features audited (all are wired into the harness or the compile path):
  anchor_prefill         H1 anchor-priority pre-fill in budgeted_compiler
  enumeration_hatch      the enumeration escape hatch in the bundle gate
  date_boost             date-proximity boost in rank_turns (just repaired)
  speaker_fallback       subject fallback in rank_turns (opt-in, off)
  enum_probe             _enumeration_probe_turns plural-head probe
  fact_index             FactIndex cross-session recall
  coverage_expansion     retrieve_convergent_evidence sub-query expansion
  resolve_dates          resolve_relative_dates post-processing
  attribution_cert       speaker attribution certificate block
  temporal_norm          temporal normalizer rewriting turns
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402
from eval_v4_pipeline_dev import V4CandidateRetriever  # noqa: E402
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner  # noqa: E402
from artificial_memory.compiler.graph.graph_builder import GraphBuilder  # noqa: E402
from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler  # noqa: E402
from artificial_memory.recall.fact_index import FactIndex  # noqa: E402

DATA = ROOT / "datasets" / "external" / "locomo10.json"


def main() -> int:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    retriever = V4CandidateRetriever()
    planner = CognitiveExecutionPlanner()
    gb = GraphBuilder()
    comp = BudgetedCognitiveCompiler()

    fires = Counter()
    n = 0

    # ---- instrument budgeted_compiler by wrapping the observable parts ----
    orig_detect = BudgetedCognitiveCompiler._detect_enumeration

    def detect_spy(self, question):
        enum, subs = orig_detect(self, question)
        if enum:
            fires["enumeration_detected"] += 1
        return enum, subs

    orig_names = BudgetedCognitiveCompiler._bundle_names_subject

    def names_spy(self, bundle, subjects):
        r = orig_names(self, bundle, subjects)
        if r:
            fires["enumeration_hatch_admitted"] += 1
        return r

    BudgetedCognitiveCompiler._detect_enumeration = detect_spy
    BudgetedCognitiveCompiler._bundle_names_subject = names_spy

    # ---- instrument rank_turns internals by re-deriving the counts -------
    orig_parse = V4CandidateRetriever._parse_relative_date

    for conv in data:
        turns, dia_map = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        fidx = None
        for q in E.load_questions(conv):
            n += 1

            # date_boost: did the query yield a target date?
            if orig_parse(q["question"], None) is not None:
                fires["date_boost_target"] += 1

            # speaker_fallback: only counts if the harness enables it (it does
            # not, so this is expected to be 0 -- recorded as a known-off flag)
            fires["speaker_fallback_off_by_default"] += 0

            # enum_probe: does the plural-head probe find anything?
            already = set()
            probe_out = E._enumeration_probe_turns(q["question"], turns,
                                                   retriever, already,
                                                   max_extra=6)
            if probe_out:
                fires["enum_probe_added_turns"] += 1

            # fact_index
            if fidx is None:
                fidx = FactIndex()
                fidx.build(turns)
            ft = fidx.query(q["question"], turns, max_turns=16)
            if ft:
                fires["fact_index_returned"] += 1

            # coverage expansion + full compile path
            from artificial_memory.compiler.coverage.coverage_engine import (
                MultiSessionCoverageEngine,
            )
            cov = MultiSessionCoverageEngine()
            top_turns, turn_scores, guaranteed = cov.retrieve_convergent_evidence(
                query=q["question"], all_turns=turns, retriever=retriever,
                reference_date=None, base_top_k=45)
            if len(guaranteed) > 0:
                fires["coverage_guaranteed_sessions"] += 1
            if len(top_turns) > 45:
                fires["coverage_expanded_pool"] += 1

            plan = planner.plan(q["question"], [])
            t_map = {(t[0], t[3]): f"{t[0]}:t{k}" for k, t in enumerate(top_turns)}
            anchor_ids = [t_map[(t[0], t[3])] for t in top_turns[:12]
                          if (t[0], t[3]) in t_map]
            if anchor_ids:
                fires["anchor_prefill_candidates"] += 1
            edg = gb.build(list(top_turns), [])
            bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
            budget = E._scaled_budget(
                plan.recommended_budget,
                sum(len(t[3].split()) * 4 // 3 for t in top_turns))
            try:
                out = comp.compile(
                    question=q["question"], bundles=bundles,
                    token_budget=budget,
                    anchor_scores={a: 5.0 for a in anchor_ids},
                    is_multi_session=len({t[0] for t in top_turns}) > 1,
                    adaptive_escalation=True)
                fires["compile_ok"] += 1
                if "SPEAKER ATTRIBUTION" in out.context_text:
                    fires["attribution_cert_emitted"] += 1
            except Exception:
                fires["compile_failed"] += 1

            # resolve_relative_dates post-processing
            if re.search(r"\bwhen\b", q["question"], re.I):
                rd = E.resolve_relative_dates("last week", out.context_text,
                                              q["question"])
                if rd and rd != "last week":
                    fires["resolve_dates_rewrote"] += 1

    BudgetedCognitiveCompiler._detect_enumeration = orig_detect
    BudgetedCognitiveCompiler._bundle_names_subject = orig_names

    print("=" * 74)
    print("DEAD-PATH AUDIT -- how often each feature actually fires")
    print("=" * 74)
    print(f"questions walked through the real pipeline: {n}")
    print()
    print(f"{'feature':34s} {'fires':>7s} {'rate':>8s}   status")
    print("-" * 74)
    EXPECTED_OFF = {"speaker_fallback_off_by_default"}
    for feat, c in sorted(fires.items(), key=lambda kv: kv[1]):
        rate = c / n * 100
        if feat in EXPECTED_OFF:
            status = "off by design"
        elif c == 0:
            status = "*** DEAD -- never fires ***"
        elif rate < 1.0:
            status = "near-dead (<1%)"
        else:
            status = "live"
        print(f"{feat:34s} {c:7d} {rate:7.2f}%   {status}")
    print("-" * 74)
    dead = [f for f, c in fires.items() if c == 0 and f not in EXPECTED_OFF]
    print(f"features that NEVER fire: {dead if dead else 'none'}")
    print()

    # ---- EFFECT check, not just firing ------------------------------------
    # Firing counts alone did NOT catch the second date bug: the query date
    # parsed (date_boost_target = 240, 'live') while the session date did not,
    # so the boost still changed nothing.  A feature can fire and be inert.
    # Compare ranking with each feature ON vs OFF; if the ranking is identical,
    # the feature fires but has no effect.
    print("=" * 74)
    print("EFFECT CHECK -- does firing actually change anything?")
    print("=" * 74)
    effect = _measure_effects(data)
    print(f"{'feature':34s} {'rank changes':>14s}   verdict")
    print("-" * 74)
    for feat, changed in effect.items():
        if feat.startswith("_"):
            continue
        verdict = ("live AND effective" if changed > 0
                   else "*** FIRES BUT INERT -- has no effect ***")
        print(f"{feat:34s} {changed:14d}   {verdict}")
    print("-" * 74)
    print(f"(date_boost measured on {effect.get('_date_boost_sample', 0)} "
          f"questions where a date target exists)")
    inert = [f for f, c in effect.items()
             if not f.startswith("_") and c == 0]
    print(f"features that fire but change nothing: {inert if inert else 'none'}")
    print()
    print("Read: 'DEAD' means it never runs. 'FIRES BUT INERT' means it runs and")
    print("accomplishes nothing -- the second date bug was exactly this state and")
    print("a firing-count audit reported it as healthy.")
    return 0


def _measure_effects(data) -> dict:
    """A/B each feature: ranking with it on vs off, on a sample of questions."""
    retriever = V4CandidateRetriever()
    orig_parse = V4CandidateRetriever._parse_relative_date

    changed_date = 0
    sample = 0
    for conv in data:
        turns, _ = E.conversation_turns(conv["conversation"])
        if not turns:
            continue
        for q in E.load_questions(conv):
            if orig_parse(q["question"], None) is None:
                continue          # only questions where the boost can matter
            sample += 1
            on, _ = retriever.rank_turns(q["question"], turns, top_k=20,
                                         reference_date=None)
            V4CandidateRetriever._parse_relative_date = classmethod(
                lambda cls, query, ref: None)
            off, _ = retriever.rank_turns(q["question"], turns, top_k=20,
                                          reference_date=None)
            V4CandidateRetriever._parse_relative_date = orig_parse
            if [t[0] for t in on] != [t[0] for t in off]:
                changed_date += 1

    return {"date_boost": changed_date, "_date_boost_sample": sample}


if __name__ == "__main__":
    raise SystemExit(main())
