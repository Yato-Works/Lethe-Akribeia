# Lethe-Akribeia — current architecture vs the proposed 5-layer design

Purpose of this document: state what already exists before the reverse-engineering
sprint lands, so the sprint is judged on the REAL gap and not on a gap that was
closed two versions ago.  Every line below was read from the source on 2026-10-13.

## What the proposed design asked for

```
1. Evidence Store      raw text, speaker, turn id, event, time, negation, provenance
2. Multi-Channel Retrieval   lexical + entity + temporal + graph relations fused
3. Deterministic Reasoning   dates, numerals, enumerations, causality, relations, fusion
4. Evidence Verifier   question match, speaker, chronology, contradiction, sufficiency
5. Answer Committer    commit with evidence, or abstain
```

## What already exists (measured, not assumed)

| proposed layer | existing code | size | status |
|---|---|---:|---|
| Evidence Store | `answer_committer.parse_turns` / `Turn` | ~1,190 LOC | id + speaker + text + date. **negation and provenance are NOT explicit** |
| Multi-Channel Retrieval | `compiler/closure/retriever.py::MultiViewRetriever` | 341 LOC | **5 views + RRF fusion, already built** |
| Deterministic Reasoning | 7 engines in `autonomous_engines/` | 3,860 LOC | counting 1,064 / event_action 1,053 / entity_attr 663 / temporal_algebra 638 / arith_diff 420 / relational 245 / anchor 136 / boolean 110 |
| Evidence Verifier | `closure/closure_engine.py` + `boolean_verifier.py` | 634 LOC | SATISFIED / PARTIAL / CONFLICTING / UNRESOLVED states |
| Answer Committer | `answer_committer.py` | 1,187 LOC | commit-or-abstain with evidence turn ids |

## The retrieval layer is further along than assumed

`MultiViewRetriever` already fuses five views by weighted Reciprocal Rank Fusion:

```
lexical      1.00
predicate    1.10
entity       1.00
temporal     1.05
neighborhood  0.85     (adjacency-weighted, delta==1 gets 2.0)
```

`EvidenceClosureEngine` (524 LOC) then runs a **multi-round evidence-closure loop**
with `_evaluate_requirements` returning four-valued status, `_derive_expansion_queries`
formulating follow-up queries, and `_build_evidence_graph` constructing a verifiable
graph.  `DeterministicQueryPlanner` (172 LOC) plans the query.

This means the "hybrid retrieval" idea is not a gap.  The gap is DOWNSTREAM.

## The real gap, stated precisely

Not missing layers — **broken boundaries between layers**.  Three specific ones,
each already visible in the 987-failure atlas:

1. **Evidence Store → Retrieval.** Speaker attribution and event binding are not
   carried as first-class fields, so retrieval can return the right turn about the
   WRONG event or the WRONG speaker.  This is exactly the `when_date` cluster
   (104 failures): the engine found a real relative anchor in a turn about a
   different event.  Graphiti's bi-temporal model is aimed at this disease.
   Lethe's `Turn` has a date but no notion of which EVENT a fact was valid for.

2. **Retrieval → Reasoning.** Items of one answer scattered across turns are never
   unioned.  The plural-list cluster (311 failures, the largest) is not a retrieval
   failure — the evidence IS retrieved, nothing aggregates it.  This is where
   HippoRAG's multi-hop/PPR thinking and the counting engine's set aggregation meet.

3. **Reasoning → Verifier → Committer.** 288 failures have gold fully present in
   context and every one abstains: evidence present, no engine owns the shape.
   Plus 294 where an engine claims and is wrong.  The verifier's four-valued status
   exists but is not consulted before every commit.

## So the sprint's question is reframed

NOT "what capability are we missing" — five layers exist.
Instead, for each candidate project:

- which ALGORITHM repairs one of the three named boundaries above,
- which parts are `[DETERMINISTIC-FEASIBLE]` vs `[LLM-DEPENDENT]`,
- and can the fix be measured as `FIXED > BROKE` on a frozen cohort keyed by
  question_id, with the matched count asserted before any number is reported.

The last point is not decoration.  This session already retracted an engine gain
that the scorer had manufactured (commit e6a305f).  Any sprint result will be
priced the same way.

## Standing rules for the sprint output

- License and reuse boundary per project: learn-from vs copy-with-attribution.
- No published score may be quoted without its conditions (backbone, judge, date).
- LoCoMo numbers from LLM-judge systems are NOT comparable to Lethe's deterministic
  scorer until the judging method is stated and aligned.  Do not average them in.
