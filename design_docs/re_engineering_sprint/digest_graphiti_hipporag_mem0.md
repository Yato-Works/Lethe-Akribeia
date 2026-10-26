# RE Sprint digest 1 — Graphiti (Zep), HippoRAG, Mem0: what is stealable without an LLM

Gathered 2026-10-13 directly from primary sources (arXiv + Zep docs + a production
PR with measured numbers).  The 6-way subagent fan-out died on HTTP 429 before
producing digests, so these three were gathered by hand.  Sources are cited at
the end; nothing here is inferred without a citation.

---

## 1. Graphiti / Zep — the bi-temporal model

**Source:** arXiv:2501.13956 (Zep paper), help.getzep.com, mintlify.wiki/getzep,
github.com/getzep/graphiti PR #1492.

### The model

Four fields on every entity edge, split into two independent timelines:

| field | meaning |
|---|---|
| `valid_at` | when the fact became true **in the world** |
| `invalid_at` | when it stopped being true in the world (null if still true) |
| `created_at` | when the system **learned** the fact |
| `expired_at` | when the system learned it no longer holds |

**Contradiction never deletes.**  On conflict the old edge gets
`invalid_at = new_edge.valid_at` and `expired_at = utc_now()`; a new edge is
created for the superseding fact.  History stays queryable, which is what makes
"who was CEO *in 2023*" answerable after the fact changed.

Edge ordering on conflict is by `valid_at`: the edge that is **earlier in
real-world time** is the one invalidated.

### Why this is directly aimed at Lethe's `when_date` boundary

Lethe's 104-failure `when_date` cluster is: the engine finds a real relative
anchor — in a turn about a **different event**.  `lc_conv-26_0` asked when
Caroline went to the LGBTQ support group and got "the Tuesday before 20 July
2023" from a turn about something else.

That is exactly a missing **valid-time axis**.  Lethe's `Turn` carries a session
date, but a fact has no notion of *which event it was true for*.  Graphiti's
answer: bind each fact to the event it came from, and when two facts about the
same subject conflict, let the earlier one expire at the later one's valid_at.
A question about event E then cannot be answered by a turn about event F,
because F's fact carries F's valid_at and E's query filters on it.

### The load-bearing caveat — measured, not speculated

Graphiti's own production PR #1492 measured the bi-temporal axis breaking:

```
1,384 edges from 264 historical-backfill episodes
valid_at   = today midnight UTC (LLM hallucination)   771  56%
valid_at   = legitimate in-text date                  610  44%
invalid_at = today midnight UTC (hallucination)        100   7%
```

**56% of the differentiator was hallucinated by the LLM** because ambiguous
past-tense sentences collapsed to "valid as of today", erasing the temporal
axis.  The PR fixed it with prompt changes alone.

**This is the single most important finding of the sprint for Lethe.**  The
bi-temporal *data model* is pure bookkeeping and `[DETERMINISTIC-FEASIBLE]`.
The *invalidation decision* is LLM-dependent and, in production, was wrong more
often than it was right.  Lethe should take the data model and implement the
invalidation decision with rules — which is where Lethe is already strong.

---

## 2. HippoRAG / HippoRAG 2 — PPR is the part worth having

**Source:** arXiv:2405.14831 (HippoRAG v1), github.com/OSU-NLP-Group/HippoRAG.

### The two-part split

HippoRAG maps three components onto brain regions:

- **Neocortex (LLM):** OpenIE — turns passages into schemaless KG triples.
- **PHR (dense encoder):** synonymy edges between similar noun phrases, above a
  cosine threshold.
- **Hippocampus (PPR):** given query nodes as seeds, distribute probability mass
  through the graph, then aggregate node probability over the passage×node
  matrix to rank passages.

Query nodes are seeded with **equal** probability; everything else zero.
`node specificity` (a local IDF analogue: 1/|passages the node appears in|)
multiplies each query node's probability *before* PPR.
Hyperparameters tuned on MuSiQue: synonymy threshold 0.8, PPR damping 0.5.

### The ablation says PPR is the mechanism, not decoration

Table 5 of the paper, replacing components:

| ablation | result |
|---|---|
| PPR → query-node probability × specificity only | **worse on all 3 datasets** |
| PPR → above **plus** direct-neighbour distribution | **worse still than query nodes alone** |
| remove node specificity | considerable drop on MuSiQue and HotpotQA, ~none on 2Wiki |
| remove synonymy edges | **largest drop on 2Wiki** |

The authors' own conclusion: adding a node's neighbourhood *without* PPR is
worse than using the query nodes alone.  **The graph traversal IS the gain.**
That matters because traversal is `[DETERMINISTIC-FEASIBLE]` — PPR needs no
LLM at all.

### Why this fits Lethe's largest cluster

Lethe already builds graphs: `compiler/graph/dependency_graph.py` and the
closure engine's `_build_evidence_graph`.  Lethe already fuses five retrieval
views by weighted RRF in `MultiViewRetriever`.

The 311-failure plural-list cluster is items of one answer **scattered across
turns** — a multi-hop join that nothing performs.  RRF fuses *rankings of the
same candidates*; it does not traverse.  PPR over a rule-built graph would
start from the subject's turns and accumulate probability onto the turns that
share entities with them, which is precisely the union the cluster needs.

Default backbone in the paper is GPT-3.5-turbo-1106 with Contriever/ColBERTv2 —
so their headline numbers are not reproducible without an LLM, but the PPR layer
does not need one.

---

## 3. Mem0 — and why its 92.5 must not be compared to Lethe's score

**Source:** arXiv:2504.19413.

### The update pipeline

Two phases.  Extraction pulls candidate facts; the **update phase** decides what
to do with each one:

For each candidate fact: retrieve the top `s` semantically similar existing
memories (`s`=10 in their eval), then present candidate + neighbours to the LLM
as a function call, and **the LLM chooses one of four ops**:

```
ADD     no semantically equivalent memory exists
UPDATE  augment an existing memory with complementary info
DELETE  remove memories contradicted by new information
NOOP    no modification needed
```

Config: `m`=10 previous messages, `s`=10 similar memories, **GPT-4o-mini** for
every LLM operation, Neo4j for Mem0^g.

### The reason its number is not Lethe's number

Mem0's evaluation is explicitly **LLM-as-a-Judge**, judged by gpt-4o-mini,
**10 independent runs, mean ± 1 std**.  The paper states the motivation: lexical
metrics like F1 give a relatively high score to "Alice is born in July" against
gold "Alice was born in March" because the remaining tokens overlap.

Lethe's scorer is deterministic containment with two contradiction guards.
**These are different instruments.**  A 92.5 measured by a judge that tolerates
rephrasing and a 49.5 measured by strict containment are not two readings of the
same quantity.  Comparing them directly — or averaging them — would repeat
exactly the mistake this session spent two commits correcting.

LoCoMo headline figures: Mem0 +26% relative over OpenAI on the judge metric;
Mem0^g ~+2% over base Mem0.  DMR: Zep 94.8% vs MemGPT 93.4% (arXiv:2501.13956).

### What IS deterministic in Mem0

The four-op taxonomy is a clean description of a problem Lethe also has:
contradicted facts must not be deleted but superseded, and near-duplicate facts
must not be added twice.  Lethe's abstain-or-commit committer is a coarser
version of the same decision.  The *ops* are `[DETERMINISTIC-FEASIBLE]`; the
*choice among them* is LLM-dependent in Mem0.

---

## Ranked integration candidates

Ordered by: touches a named boundary, `[DETERMINISTIC-FEASIBLE]` core, small
enough to measure on a frozen cohort.

**I1 — Bi-temporal fields on evidence turns** *(Graphiti)*
Target: the `when_date` boundary, 104 failures.
Change: give each extracted fact a `valid_at` (the event it was true for) and
keep superseded facts instead of dropping them.  Invalidation by **rule** — the
earlier real-world fact expires at the later one's valid_at — never by LLM.
Graphiti's own 56%-hallucination measurement is the argument for doing the
decision deterministically rather than trusting a model.
Experiment: the 104 `when_date` failures, keyed by question_id.  Success is
`FIXED > BROKE` with the matched count asserted, and **zero** regression on the
other 1,882.

**I2 — PPR over the rule-built evidence graph** *(HippoRAG)*
Target: the retrieval→reasoning boundary, 311 failures (the largest cluster).
Change: seed PPR from the question's entity/date nodes over the existing
evidence graph, aggregate node mass back onto turns, and union the top turns'
items for plural questions.  PPR needs no LLM; the graph already exists.
Experiment: the 311 plural-list failures.  Also measure the node-specificity
term, since the ablation shows it matters.

**I3 — Four-op supersede instead of abstain-or-drop** *(Mem0)*
Target: the reasoning→committer boundary, 288 abstentions with gold fully
present.
Change: when an engine finds evidence but the shape is not owned, decide
ADD/UPDATE/DELETE/NOOP deterministically and commit only on a verified ADD.
This is where the self-verification gate already lives; the taxonomy makes the
refusal reasons explicit instead of a single "no skill could commit".

**Explicitly NOT recommended now:** Cognee's ECL pipeline (no digest yet — the
subagent that ran longest also died on 429) and Letta's memory tiers (no digest).
Both need a second pass before they can be judged.  Do not read their absence
from this list as a rejection.

---

## Honesty notes

- The subagent fan-out returned **no digests** — all six hit HTTP 429.  Tasks
  1–4, 6 died in ~50s each after a few API calls; task 5 (Cognee) ran 1054s and
  45 API calls and still died.  Everything above was gathered by hand from
  primary sources.  Letta and Cognee are **unresearched**, not dismissed.
- The HippoRAG abstract's numbers render as bare `%` in the arXiv HTML ("by up
  to %", "- times cheaper").  Those figures are **UNVERIFIED** here; the
  ablation table's *direction* is what this digest relies on, not the magnitudes.
- No published score in this document is comparable to Lethe's deterministic
  scorer.  Mem0's is LLM-judged (gpt-4o-mini, 10 runs, mean±std); Zep's DMR
  number is reported on a different benchmark.  Both are recorded for context.
- Licenses: Graphiti is Apache-2.0 (LICENSE verified).  HippoRAG and Mem0
  licenses **UNVERIFIED** in this pass.  Learning an algorithm is always fine;
  copying code requires the license check first.  Do that before any reuse.

## Sources

- https://arxiv.org/abs/2501.13956 — Zep, temporal KG for agent memory
- https://arxiv.org/abs/2504.19413 — Mem0 paper
- https://arxiv.org/abs/2405.14831 — HippoRAG v1
- https://mintlify.wiki/getzep/graphiti/concepts/temporal-model — bi-temporal fields
- https://getzep.com/blog/beyond-static-knowledge-graphs — invalidation semantics
- https://github.com/getzep/graphiti/pull/1492 — the 56% valid_at hallucination measurement
- https://github.com/getzep/graphiti — Apache-2.0 LICENSE verified
- https://github.com/OSU-NLP-Group/HippoRAG — code
- https://github.com/mem0ai/mem0 — code
