# RE Sprint — consolidated decision

Five systems researched.  All five published numbers are LLM-judged or on a
different benchmark, so **none is comparable to Lethe's 49.5%** deterministic
score.  What was extracted instead is the algorithm, tagged by whether it can run
with no model at all.

| system | license | headline | why it is NOT comparable to Lethe |
|---|---|---|---|
| Mem0 | UNVERIFIED | LoCoMo 92.5 (vendor) | LLM-as-a-Judge, gpt-4o-mini, 10 runs mean±std |
| Zep/Graphiti | **Apache-2.0** | DMR 94.8 vs MemGPT 93.4 | different benchmark (DMR), LLM-backed |
| Letta | **Apache-2.0** | LoCoMo 74.0 (3rd-party mirror) | gpt-4o-mini as answerer AND judge; adversarial excluded; 1,540 q |
| Cognee | **Apache-2.0** | BEAM 0.79 / 0.67 | BEAM not LoCoMo; LLM judge; **n=20, 10M number in-sample** |
| HippoRAG | UNVERIFIED | multi-hop QA | MuSiQue/2Wiki/HotpotQA — no conversational structure |

Licenses verified for Graphiti, Letta, Cognee (all Apache-2.0, permissive, patent
grant, retain notices, mark modified files).  HippoRAG and Mem0 still
UNVERIFIED — **check before any code reuse.**  Learning an algorithm is always
fine; copying needs the license first.

---

## What the five actually gave us

### 1. The plural-list cluster is an IDENTITY problem, not a retrieval problem

**Cognee's UUID5 identity scheme** — `uuid5(NAMESPACE_OID, "ClassName:v1|v2")`
with values lowercased and spaces→`_`.  Verified pure stdlib, no model, no
network, no randomness (`verify_cognee_determinism.py`): the same entity across
different surface spellings collapses to one identity, distinct entities do not
collide.

This is the piece the 311-failure cluster was missing.  Today a rule engine has
no principled way to decide two mentions are the same entity — "Bailey" in D7:18
and "a cat named Bailey" in D13:4 are matched heuristically.  A keyed hash
**removes that decision entirely.**  Combined with a union rule for list-valued
slots, `Melanie's pets' names` becomes a set-union over one identity key.

**This is now the top candidate**, and it explains why the PPR spike failed: the
retriever already surfaces the evidence (38/40), and the missing step was never
traversal — it was knowing that two turns name the same thing.

### 2. Supersede-and-keep beats overwrite (both Graphiti and Cognee agree)

Two independent systems, same discipline:
- **Graphiti:** on conflict, old edge gets `invalid_at = new_edge.valid_at`,
  `expired_at = now`; nothing is deleted; history stays queryable.
- **Cognee:** `valid_to` + `close_node()` — replaced fact is stamped superseded
  and stays in the graph; **supersession never affects node identity**.

For the 111-failure attribute cluster this makes "what did they *used to*
prefer" and "what do they prefer *now*" both answerable from a temporal ordering,
instead of whatever the last write happened to be.

**And Graphiti's own production measurement is the warning that makes this
buildable:** on 1,384 edges, `valid_at` was "today midnight UTC" — an LLM
hallucination — in **771 cases (56%)**.  The data model is bookkeeping; the
*decision* was model-dependent and wrong more often than right.  Lethe does the
bookkeeping and makes the decision by rule.

### 3. Turn-bound provenance as a traversal, not a heuristic

**Cognee's `chunk_attachment="all"`** links every extracted node to the chunk
that produced it by a shared `id`, so "which turn said this" is one hop instead
of a guess.  Directly targets the 104-failure `when_date` cluster, whose failure
mode is binding a real anchor to the wrong turn.

A deterministic rejection rule follows: if the value's source turn and the turn
the answer binds to disagree, **abstain rather than commit.**

### 4. Letta's most useful signal is an argument AGAINST specialized machinery

Letta's LoCoMo result (74.0, third-party mirror) is a **filesystem agent** — the
history is written to files and the agent is given `grep`, `search_files`,
`open`/`close`, `answer_question`.  It beats the comparison table's Mem0ᵍ 68.4,
Zep 66.0, best RAG 61.0, LangMem 58.1.

**Read that carefully.**  "Give the agent a filesystem and let it grep" scored
highest.  That does not argue for elaborate memory architecture; it argues that
**retrieval quality matters more than memory cleverness** — which corroborates
the PPR spike independently: Lethe's 5-view RRF is already strong at 38/40, and
adding structure on top made it worse.

Still liftable from Letta, all `[DETERMINISTIC-FEASIBLE]`:
- **Memory blocks with per-block char caps** — a labeled, capped, always-visible
  slot whose contents are rewritten by policy.  The `chars_current`/`chars_limit`
  pair is a ready-made eviction budget: over budget → consolidate or abstain.
- **Shared read-only blocks** — one writer, many readers, with `read_only`
  guards.  Maps 1:1 onto Lethe's per-entity attribute state: a deterministic
  resolver writes, engines read, no drift.
- **Agent-immutable append-only archival** — inserts are cheap and
  reversible-by-id, in-place edits forbidden.  Prevents the silent-rewrite class
  of retrieval bugs.
- **Offline consolidation** (the deterministic analogue of sleep-time compute,
  arXiv:2504.13171) — a per-session pass that runs the 7 engines once and writes
  consolidated state, so per-question answering is lookup.  Same amortization the
  paper demonstrates, minus the LLM.

### 5. What is genuinely not liftable

- **Graph extraction** (Cognee): deciding which spans are entities, their type,
  coreference, and predicate direction.  Cognee's own keyless path substitutes
  GLiNER — a trained span-labeller over a closed label bank, ~750 MB, and
  explicitly "a demo" of the enterprise product.  Not a rule engine, not free of
  model weights.  **This is the irreducible model dependency**, and it is exactly
  one step of eleven.
- **Contradiction adjudication** (Cognee), **LLM-mediated summarisation**
  (Letta), **OpenIE triples** (HippoRAG).

---

## Ranked candidates

| # | idea | source | targets | deterministic |
|---|---|---|---|---|
| **1** | **UUID5 entity identity + union-on-append merge** | Cognee | **311** plural-list, 111 attribute | ✅ verified stdlib |
| **2** | **Bi-temporal supersede-and-keep + rule invalidation** | Graphiti+Cognee | 111 attribute, 104 when_date | ✅ bookkeeping only |
| **3** | **Turn-bound provenance via chunk→fact edges** | Cognee | 104 when_date | ✅ shared id |
| 4 | Resident-set gate + cold-tier quarantine | Letta | 157, 104 | ✅ |
| 5 | Offline per-session consolidation | Letta | 288, 311 | ✅ |
| — | PPR over evidence graph | HippoRAG | 311 | **KILLED** — worse than current retriever |

**Candidate 1 is the one to build first.**  It attacks the largest cluster, it
was verified before being believed, and it explains the PPR failure rather than
competing with it.  Its kill criterion is pre-declared: if recall gain < 2pp AND
duplicate-node rate is unchanged, the cluster is not an identity problem and the
idea dies too.

## Standing rules carried forward

- Every claim passes `scripts/benchmarks/integration_gate.py`: FIXED / BROKE /
  SPURIOUS on frozen cohorts keyed by `question_id`, matched count asserted,
  against the **983/1986 = 49.5%** baseline.
- Pre-declared kill criteria.  A negative result is a deliverable.
- Report deltas with the scorer generation named on each side.
- No published score above is comparable to Lethe's, in either direction.  Do not
  average them, do not quote them as a target.
