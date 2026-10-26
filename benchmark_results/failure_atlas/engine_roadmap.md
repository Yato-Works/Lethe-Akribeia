# Engine Roadmap — ranked from the failure atlas

Source: `benchmark_results/failure_atlas/`, built by `build_failure_atlas.py`.
All 1,974 joined questions re-scored with the fixed scorer (`c449aa0`).

```
joined           1,974
  correct          987
  failures         987
```

## Ranking method

Each candidate is scored on: how many failures it touches, how many have gold
**fully present** in the frozen context (i.e. fixable without retrieval work),
whether an existing engine already owns the shape, implementation size, and
whether a small test can measure it.

**Important distinction, kept throughout:** the failure COUNT is an upper bound on
what a fix could touch. It is NOT a predicted score gain. A fix that recovers half
of its cluster is a good fix; the cluster size alone proves nothing.

## Candidate counts (measured)

| # | candidate | failures touched | gold fully in context |
|---|---|---:|---:|
| C1 | plural-list union across turns | 166 (+145 in C3) = **311** | 117 + ~145 |
| C2 | `when_date` picks the wrong turn's date | 104 | 101 |
| C3 | evidence present, no engine owns the shape | 433 (145 list, 288 single) | 433 |
| C4 | attribute resolver answers wrong | 111 | 87 |
| C5 | gold absent from context (retrieval/E) | 157 | 0 |

C1 and C3's list portion are **the same disease**: a plural question whose items
are scattered across several turns, where nothing unions them. Counted separately
above, treated as ONE fix below.

---

## Priority 1 — Plural-list union engine

**Touches 311 failures** (166 partial enumerations + 145 list-gold abstentions).
117 of the partial set and essentially all 145 abstentions have gold fully in
context, so this needs **no retrieval work**.

**Reproducible failure** (`lc_conv-26_52`):

```
Q : What are Melanie's pets' names?
GT: 'Oliver, Luna, Bailey'
predicted (stored run): 'Luna, Oliver'
committer now: abstain — 'no deterministic skill could commit an answer'
evidence turns: D7:18, D13:4, D13:6
```

The items exist, in three separate turns. The prediction took two and dropped
`Bailey`; the current engine takes none.

**Expected behaviour.** For a plural question, collect the item set across all
turns belonging to the queried subject, union them, and answer with the union —
not with whichever single turn the retriever ranked first.

**Why this is first.** One fix addresses the single largest cluster, the evidence
is already present, and `counting_engine` / `relational_traverser` already do
partial-set aggregation — so this is an **extension of an existing engine**, not a
new one. Same principle as `d166f19`.

**Risk.** Over-union: pulling in items from other people or other topics. The
speaker-attribution fixes from `9e0abe9` bound this, and the fix must key on the
question's subject. Measured on the cluster's first 60: current engines answer
14 correctly, 15 wrongly, 31 abstain — so the bar to clear is low.

**Test.** Re-run the same 311 questions; success is `FIXED > BROKE` with zero
regression on `tests/unit`.

---

## Priority 2 — `when_date` binds to the wrong turn

**Touches 104 failures**, 101 with gold fully in context.

**Reproducible failure** (`lc_conv-26_0`):

```
Q : When did Caroline go to the LGBTQ support group?
GT: '7 May 2023'
committer: "extracted relative temporal anchor 'The Tuesday before 20 July 2023'"
```

The engine found a real relative anchor — in a turn about a different event.
Related: `lc_conv-26_9` extracts `'The week before 9 June 2023'` where the gold
**is** that value, so the anchor resolution works and only the *turn selection*
is wrong.

**Expected behaviour.** Bind the anchor to the turn whose content matches the
question's event, and abstain when the best-scoring date turn is about a
different event. `temporal_algebra_engine` resolves dates well; it does not
currently verify the turn is about the asked event.

**Risk.** Over-abstention on legitimate date questions. Must be measured as
`FIXED vs BROKE` on the same 104.

---

## Priority 3 — Single-value extraction with no engine

**Touches 288 failures**, all with gold fully in context. Currently **every one
abstains**.

**Reproducible failure** (`lc_conv-26_71`):

```
Q : What book did Melanie read from Caroline's suggestion?
GT: '"Becoming Nicole"'
predicted (stored run): 'The Book of Joy'     <- wrong book
committer now: abstain
evidence: D7:11, D13:1
```

**Expected behaviour.** A constrained single-value extractor: the question names a
relation ("read from X's suggestion"), so find the turn carrying that relation and
return its object. Narrower and safer than the generic-attribute branch, which
measured 19.1% precision and is why it is gated.

**Risk.** Highest of the three, because this is the shape that produced the
14%-precision generic branch. Must land with the self-verification gate already
in place, or not land at all.

---

## Priority 4 — Attribute resolver precision

**Touches 111 failures** where the resolver answers but is wrong, 87 with gold
fully in context. Lower priority than 1–3 because it is a *precision* problem:
the engine already claims, so the work is deciding when not to.

**Reproducible failure** (`lc_conv-26_19`):

```
Q : What do Melanie's kids like?
GT: 'dinosaurs, nature'
committer: "extracted liked objects: ['our last one', 'a ...']"
```

**Expected behaviour.** Abstain when the extracted objects are pronouns or
deictic phrases rather than concrete nouns.

---

## Priority 5 — Gold absent from context (157)

**No fix available in the engines.** These are retrieval, memory-representation,
or genuinely-absent-in-conversation cases. The atlas infers absence from the
*compiled context*, which is weaker than absence from the *conversation* — this
needs the raw conversation checked before any of these are called class E.
**Do not spend engine effort here yet.**

---

## What is explicitly NOT claimed

- No score improvement is predicted for the full 1,986-question benchmark.
- The 987 failure count is **not** 987 available points. It is the size of the
  pool a fix would work in.
- The live-engine check (114 of 987 already answerable) is a **capability
  ceiling** for the current engines, not a measured pipeline gain.
- The stored run's `code_revision` is `null`, so it is not a valid current
  baseline. Any before/after must be measured fresh.
