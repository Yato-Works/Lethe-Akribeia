# ECE (Evidence Closure Engine) — FROZEN, NOT REJECTED

Status as of this commit: **experiment frozen**. The effect could not be
demonstrated, but the experiment was not taken to a verdict. Do not resume
work here until the budget-isolation experiment has been run (step 3 below).

## What was measured

Slice: LoCoMo conv-30 (data index 1), questions 0-19, baseline vs ECE.

| metric | baseline | ECE |
|---|---|---|
| accuracy | 15/20 (75.00%) | 15/20 (75.00%) |
| delta | — | **+0.00pt** |
| newly correct | — | 0 |
| regressions | — | 0 |

Fallback activity on that slice: fired on 11 of 20 questions, adding
23-28 candidate turns each (~270 turns total).

Reader context tokens on those 11 questions: pinned at ~1394-1400, net change
**-23 tokens** across all 11 (individual questions ranged -8 to +6).

Artifacts (all runs preserved):
- `benchmark_results/experiment_ece_gate2/locomo_ece_gate2_conv-30_off0_20261011_175956.json`
- `benchmark_results/experiment_ece_gate2/locomo_baseline_gate2_conv-30_off0_20261011_175956.json`

## What the code does now (all committed, do not undo without re-measuring)

1. `80e74f8` — `eval_ece_gate2.py` used `V4CandidateRetriever()` without
   importing it, so the runner raised `NameError` on the first question.  The
   artifacts in `experiment_ece_gate2/` predate this file's current state and
   were never re-verified against it.  Import added from
   `eval_v4_pipeline_dev` so both modes rank turns with the same frozen
   retriever.
2. `80e74f8` — runner-side fallback: when the engine terminates with
   `no_further_expansion_possible`, re-query the pool for each unresolved
   requirement.  This compensates for `_derive_expansion_queries` having no
   handler for `req_direct_fact`, which `closure/planner.py` does emit and
   which was the top unmet requirement (28 of 50 on conv-30).  Emitted as
   `fallback_turns_added` so its effect is measurable.
3. `6d971aa` — fallback turns are now ANCHORED with their own rank score.
   Before this, adding 12 turns to conv-30_9 changed `context_tokens` by 0,
   because unanchored turns lose every knapsack selection round.

## What is NOT established

The claim "budget saturation is the cause" is a hypothesis, not a result.
Net token change of -23 is equally consistent with all three of:

- (a) ECE's evidence selection is actively harmful — it is evicting useful
  turns. Under this hypothesis, raising the budget would NOT help.
- (b) the reader window is too small to admit anything new. Under this one,
  raising the budget would help.
- (c) both.

To distinguish (a) from (b), a per-question diff of the compiled context is
needed: which turns entered, which were evicted, and whether any evicted turn
was one the baseline got right.

## Recommended order (owner's decision: adopt C)

1. **Freeze** — done by this file. No further ECE edits; the frozen code and
   the artifacts above are the reference point.
2. **Move to cat1 enumeration** — the confirmed weakness (multi-session fact
   enumeration). This is about final answer quality, not about completing ECE.
3. **When there is headroom, isolate the budget variable** — same candidate
   pool, same reader, same questions, budgets 1400 / 2400 / 4000. Confirm the
   7B model's real context limit and how output tokens are accounted.  This
   decides whether ECE is salvageable before any more evidence plumbing.

## Contamination audit of the LoCoMo path (pre-baseline-run clearance)

Full-corpus sweep: every string literal in every prompt/answer-building
function was checked against all 1,427 LoCoMo gold answers and all 414
LongMemEval gold answers (`tests/unit/audit_contamination.py`).

Result: **VERDICT: CLEAN**.

The reader prompt retains only `'Yes'` / `'Likely no'`, which are generic
yes/no answer vocabulary, not benchmark-specific answers, and zero
question-content branches.

The audit was then narrowed to the six modules the LoCoMo harness actually
imports — `budgeted_compiler`, `coverage_engine`, `execution_planner`,
`graph_builder`, `fact_index`, `research.benchmarks.llm` — and re-run against
every gold answer of length >= 10 chars:

  all six modules: 0 hits

So no contamination is reachable from the LoCoMo evaluation path.

### Found but OUT OF SCOPE (not in the LoCoMo path, recorded for honesty)

A repo-wide sweep (beyond the evaluation path) surfaced 80 lines containing a
gold answer string. Most are docstring/comment examples or generic vocabulary
lists. Two are worth flagging even though they are unreachable from LoCoMo:

1. `src/artificial_memory/recall/proposition_integrity_gate.py` hardcodes
   LoCoMo's own characters and their attributes (`Caroline` vs `Melanie`,
   `charity race`, `Grand Canyon`, `black and white bowl`) as bespoke
   detection patterns. It is benchmark-specific tuning, but it is imported
   only by `context/msc_compiler.py` (a different benchmark), never by the
   LoCoMo harness. **Not affecting any number reported here; should be
   generalised before that module is used for a public score.**

2. `src/artificial_memory/compiler/certificates/temporal_certificate.py`
   maps keyword -> canned answer strings ("cousin"+"wedding" ->
   "my cousin's wedding"). Those strings are not LoCoMo gold answers (verified:
   0 exact and 0 substring matches), and the module is not in the LoCoMo path,
   but the pattern itself is the cheat shape and should not be extended.

## Separate artifact found while triaging cat1 (do not confuse with the above)
Re-scoring the 1986-question run `locomo_full_20261010_151930.json` with the
CURRENT scorer flips 107 questions to PASS:

| | value |
|---|---|
| recorded | 1009/1986 = 50.81% |
| re-scored | 1116/1986 = 56.19% |
| delta | +107 questions, **+5.39pt** |
| by category | cat1 15, cat2 55, cat3 6, cat4 21, cat5 10 |

**This is not an improvement.** The predictions are byte-identical; the run
predates the current scorer. 5 of the 107 were cat1 list-gold questions marked
FAIL only because the answer listed the same items in a different order
(`'Paris, Rome'` vs `'Rome, Paris'`), which the current scorer handles.

So there are now TWO independent inflation artifacts, both pointing the same
direction and both noise:

1. contaminated reader prompt (gold answers embedded) — up to 2.72pt
2. stale scorer — +5.39pt on re-scoring

A clean comparison requires one full run with the de-contaminated prompt AND
the current scorer together. Until that run exists, no LoCoMo headline number
is trustworthy, and neither 50.81% nor 56.19% should be quoted as progress.

## Note on _scaled_budget

Its floor is 1400 (`floor: int = 1400` in
`scripts/benchmarks/eval_locomo_full.py`), which is why every measured
context lands at ~1394-1400 regardless of pool size. Saturation is therefore
not uniform: `_scaled_budget(550, 8000) -> 1400` but
`_scaled_budget(550, 12000) -> 1440`, and it only reaches the 2400 cap at a
~20000-token pool. Any budget-isolation experiment must vary the floor/cap
deliberately, not assume the current scaling is the only variable.
