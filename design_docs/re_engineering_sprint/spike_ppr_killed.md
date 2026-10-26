# RE Sprint spike — PPR over the evidence graph: HYPOTHESIS KILLED

Date: 2026-10-13.  This experiment exists to be negative evidence.  It is kept
because a spike that kills an idea is worth more than one that confirms a
preference, and because the unpaired version of it would have been reported as a
win.

## The claim being tested

From HippoRAG (arXiv:2405.14831), whose ablation showed PPR is the mechanism:
replacing PPR with query-node probability was worse on all three datasets, and
adding a node's direct neighbours **without** PPR was worse still than the query
nodes alone.  The authors' conclusion: the graph traversal IS the gain.

Applied to Lethe: the existing evidence graph (`_build_evidence_graph`) carries
only `DIALOGUE_FLOW` (adjacent turns in a session) and `SAME_SPEAKER_REMENTION`.
It has **no entity-sharing edge**, so two turns that both mention `Bailey` are
not connected unless adjacent.  Adding that edge and running PPR looked like a
direct answer to the 311-failure plural-list cluster.

## The unpaired run was worthless

`spike_ppr_entity_edges.py` first reported:

```
cohort n = 40
gold evidence turns reached by PPR top-k : 37
```

and stopped.  That number means nothing, because reaching 37 of 40 is only
evidence if the **current retriever** reaches fewer.  A control that loses to
nothing proves nothing.

## The paired control

`spike_ppr_vs_baseline.py`, identical cohort / questions / frozen contexts /
seeds / top-k, three conditions:

| condition | gold turns reached |
|---|---:|
| **A — current `MultiViewRetriever` (5-view weighted RRF)** | **38** |
| B — PPR over the existing graph (no entity edge) | 36 |
| C — PPR over the graph **plus** the entity-sharing edge | **32** |

Questions where C reached gold and A did not: **1** (`lc_conv-30_44`).

**VERDICT: PPR is worse than the retrieval Lethe already has.  Do not build it.**

## Why the paper's result did not transfer

HippoRAG is evaluated on **MuSiQue, 2WikiMultiHopQA, HotpotQA** — external
corpora where passages are encoded in isolation and nothing links two passages
that share an entity.  That is the exact condition under which adding entity
edges + traversal helps.

LoCoMo is different in the decisive way: its turns are **already in one
conversational structure**.  `MultiViewRetriever` already exploits that with five
views, including a `neighborhood` view that is adjacency-weighted, and the
speaker view chains a subject's turns.  The relational structure HippoRAG has to
*synthesise* is already present.  Adding entity edges on top over-connects the
graph — every turn mentioning any proper noun becomes mutually reachable, mass
diffuses, and precision drops.  C is worse than B, which is the signature of that
over-connection.

**Transferable lesson, not a transferable result:** a graph-traversal win on a
corpus with no conversational structure is not a win on one that already has it.
The 311-failure cluster is therefore **not** a retrieval problem, which the
existing retriever's 38/40 corroborates independently.  The union has to happen
downstream, in the reasoning/aggregation layer, where nothing currently unions
scattered items.  Candidate I2 is downgraded from "PPR over the graph" to
"aggregation over already-retrieved turns".

## The trap this spike nearly walked into

`retrieve()` regenerates turn ids as `f"{sid}:t{index}"` internally
(`retriever.py:164` raises `KeyError` on the ids it synthesises).  Comparing
those candidate ids against atlas gold ids (`D7:18`) would have silently
zero-matched, reported condition A as reaching **zero** turns, and handed
condition C a fabricated 37-point win.

The comparison is positional instead — turns are passed in a known order, `D7:t8`
is decoded back to index 8 — and the decode is asserted rather than trusted:
ids not in `sid:tN` form abort, out-of-range indices abort.  This is the same
class of defect as the two scorer guards already committed this session
(`e6a305f`, and the atlas join in `ebb0c75`): a join that silently matches
nothing reports a confident number built on no data.

## What this changes

- **I2 (HippoRAG PPR) — rejected** on this cohort.  Recorded so it is not
  re-proposed without new evidence.
- The 311 plural-list cluster moves to the **reasoning→committer** boundary, not
  retrieval.  The retrieval already surfaces the evidence (38/40); the failure is
  that nothing unions it.
- Remaining candidate: **I1, bi-temporal fields with rule-based invalidation**
  (Graphiti), targeting the 104 `when_date` failures.  That one attacks a
  different boundary — store→retrieval — which the PPR result says is *not* where
  the largest cluster lives.

## Artifacts

```
scripts/benchmarks/spike_ppr_entity_edges.py       unpaired run (kept as the trap)
scripts/benchmarks/spike_ppr_vs_baseline.py        paired control, the real result
```

Cohort: 40 questions, `shape` in (what_thing, what_person), `gold_in_context`
full, `rescored_correct` false, from `failure_atlas.jsonl` joined to
`locomo_context_cache_rules.jsonl`.  No LLM calls.  No production code changed.
