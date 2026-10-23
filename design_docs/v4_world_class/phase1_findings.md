# Phase 1 Findings — Leak Removal, Strict Recall, and the True Baseline

> Date: 2026-10-06 · Reader: `qwen2.5:7b-instruct` (Ollama, temp 0, seed 42) · Dev-250 · Budget 1300

---

## 1-1. Leakage removed (inference boundary)

| File | Change |
|------|--------|
| `scripts/benchmarks/eval_v4_pipeline_dev.py` | Aggregation mode no longer branches on gold `question_type`; it is derived from question syntax + memory-conditioned aggregation demand. Abstention reference label is computed **after** inference, grading-only. |
| `src/artificial_memory/compiler/eval/phase0_runner.py` | Inference path (`run` + helpers) accepts only `question` / `question_date` / `haystack_*`. Gold labels enter only the grader and the offline recall measurement. |
| `tests/unit/test_no_label_leak.py` | New regression suite: (a) AST scan of every inference method for gold-label tokens, (b) behavioural invariance — two items differing **only** in gold fields must produce byte-identical prompts/predictions, (c) frozen-reader signature guard. **5 passed.** |

Gold-label tokens are now absent from the inference path by construction and by test.

## 1-2. Evidence recall made strict (structural sets)

Replaced `f"[{sid}" in context_text` string containment with session-id **set
operations** over the compiled node metadata, and split the metric in three:

* `recall_retrieved` — gold sessions present in the candidate evidence graph.
* `recall_selected` — gold sessions with ≥1 turn selected after the budget (L2).
* `gold_turn_recall` — turn-level, `N/A` (LongMemEval-S has no turn-level gold).

| Type | Recall@Retrieved | Recall@Selected | N |
|------|------------------|-----------------|----|
| knowledge-update | 100.0% | 91.0% | 39 |
| multi-session | 100.0% | 81.0% | 67 |
| single-session-assistant | 100.0% | 100.0% | 28 |
| single-session-preference | 100.0% | 93.3% | 15 |
| single-session-user | 100.0% | 100.0% | 35 |
| temporal-reasoning | 100.0% | 77.3% | 66 |
| **Overall** | **100.0%** | **87.1%** | 250 |

`recall_retrieved` is 1.0 by construction because the current pipeline indexes
the full haystack into the evidence graph; the meaningful signal is
`recall_selected`. Loopback errors: **0**.

## 1-3. The true (leak-free) baseline

Command:

```
python -m artificial_memory.compiler.eval.phase0_runner \
  --dataset datasets/external/longmemeval_s_cleaned.json \
  --split benchmark_results/splits/lme_dev_qids.json \
  --budget 1300 --output-dir benchmark_results/phase0 --cache-dir cache/phase0
```

| Metric | Value |
|--------|-------|
| **Accuracy** | **31.2%** (78/250) |
| 95% bootstrap CI | [25.6%, 37.2%] |
| Mean prompt tokens | 1756 |
| Acc / 1k tokens | 0.18 |
| Errors | **0** (would be recorded with `status="error"` + traceback) |
| Deterministic overrides | 16/250 |
| LLM cache hits | 234 (bit-for-bit reproducible) |

Per type: KU 56.4% · multi-session 7.5% · ss-assistant 53.6% · preference 0.0% ·
ss-user 71.4% · temporal 16.7%.

### This is far below the expected 65–72 %, and why

The old **74.0 %** is not the leak-free baseline for this pipeline; two effects
compound:

1. **Leak removal** — gold `question_type` previously selected the aggregation
   path and prompts for the whole multi-session/temporal slice.
2. **`phase0_runner` is a weaker pipeline than the old `eval_v4` script.**
   `eval_v4` used a focused candidate retriever (IDF + proper-noun + user-turn
   boosts, top sessions) plus an *exhaustive aggregation mode* and a budget
   boost to 2000 tokens. `phase0_runner` feeds **all** ~491 turns to the budgeted
   compiler with plain overlap anchors and a hard type-branch prompt stack.

Concrete failure modes observed in single-question diagnostics:

* **multi-session** ("total cost of Lola's vet visit and flea medication", gold
  `$75`): the compiled context was 1293 tokens of an unrelated home-closing
  conversation — generic "total cost" words out-ranked the gold sessions.
* **preference**: `[USER PROFILE & PREFERENCES]` was **empty**; the gold session
  `answer_f7b22c66` was not selected, and the profile certificate emitted junk
  extractions ("general prefers it would be perfect").
* **temporal**: the ordering certificate named the **wrong** first trip
  (extraction/date error), so the reader inherited a wrong answer.

Session-level recall (87.1 %) is high while accuracy (31.2 %) is low, i.e. the
bottleneck is now **which turns** are selected and certificate correctness —
not session presence. This is an L2/L3 problem, matching the design-doc analysis.

## Phase 3 attempt — Coverage-Aware Selector (REJECTED)

Implemented `compiler/selectors/coverage_selector.py` (submodular greedy with
λ·sessions + μ·entities). Measured at the same budget:

| Selector | multi recall_selected | temporal recall_selected |
|----------|-----------------------|--------------------------|
| budgeted BCC (baseline) | 81.0% | 77.3% |
| coverage selector | **37.4%** | **31.9%** |

Root cause: the selection budget is already **saturated** (mean 1296/1300 tokens,
~8 turns). A per-new-session bonus cannot add turns — it *displaces* the deeper,
higher-relevance selection and drops gold sessions. A non-displacing coverage
post-pass is also impossible (no leftover budget). Improving multi/temporal
recall therefore needs **re-allocation (Phase 4 dynamic budget) or better
per-turn relevance**, not an unconstrained coverage term.

Kept as `--selector coverage` (experimental) for reproducibility; not adopted.

## Phase 2 — Soft routing (evaluated: no significant change)

`--soft-routing` composes the prompt from continuous demand signals
(temporal / aggregation / state-update / preference / assistant-recall); several
may co-activate, and the compiler's multi-session behaviour is driven by
aggregation demand rather than the hard classifier label. The hard type branch
remains the default so the Phase-1 baseline stays reproducible.

| Metric | Phase 1 (hard) | Phase 2 (soft) |
|--------|----------------|----------------|
| Accuracy | 31.2% (78/250) | 31.2% (78/250) |
| multi-session | 7.5% | 6.0% |
| temporal-reasoning | 16.7% | 21.2% |
| single-session-user | 71.4% | 68.6% |
| single-session-assistant | 53.6% | 50.0% |
| preference | 0.0% | 0.0% |
| Mean tokens | 1756 | 1811 |
| Recall@Selected | 87.1% | 86.7% |
| Errors | 0 | 0 |

**McNemar (paired, n=250): b=7, c=7, exact two-sided p = 1.0000 (not
significant).** Soft routing therefore satisfies the Phase-2 acceptance
criterion — *no significant accuracy regression* — but yields no significant
improvement either; temporal +4.5 pp is offset by losses elsewhere, all within
noise. Output: `benchmark_results/phase2/`.

## Reproduce

```
# leak regression
python -m pytest tests/unit/test_no_label_leak.py -q

# strict recall only (no LLM, ~5 min)
python -m artificial_memory.compiler.eval.phase0_runner --dataset datasets/external/longmemeval_s_cleaned.json \
  --split benchmark_results/splits/lme_dev_qids.json --budget 1300 --output-dir benchmark_results/phase0 --skip-llm

# Phase 2 variant (writes elsewhere; compare with --compare-to)
python -m artificial_memory.compiler.eval.phase0_runner ... --soft-routing --output-dir benchmark_results/phase2 \
  --compare-to benchmark_results/phase0/lme_dev250_v4_label_free.json
```

---

# Failure anatomy of the V4 pipeline run (LLM-free census)

Source run: `benchmark_results/pipeline_v4/lme_dev250_v4_pipeline.json`
(250 dev questions, **183/250 = 73.2%**, 67 failures, deterministic overrides included).

Tool: `python scripts/benchmarks/diag_fail_anatomy.py`
(`ALL_TYPES=1` for all 67 failures; default covers multi-session + knowledge-update;
`DUMP_QID=<qid>` dumps the rebuilt compiled context for one question).

## Method

For every failed question the pipeline is re-run **without the reader LLM call**:
ingest → planner → multi-channel retrieval → co-processor → per-path compilation
(aggregation sampler vs. graph/budgeted compile). The rebuilt context is then
checked, without any model, against the haystack:

| signal | meaning |
|---|---|
| `gt_in_ctx` | ground-truth string (or its numeric variants) occurs verbatim in the compiled context |
| `gold in candidates` | gold `answer_session_ids` survived candidate retrieval |
| `gturns k/n` | k of n gold-session turns survive verbatim into the context |
| `gtbear(in)` | of the gold turns that actually contain the GT value, how many are in the context |

Buckets (mutually exclusive, decided in this order):
`retrieval_miss` → `selection_miss` / `partial_gt_bearing` →
`computed_gt_unverifiable` → `committer_bug` → `reader_miss_unverified` → `reader_miss`.

## Result — all 67 failures

| bucket | n | share | meaning |
|---|---|---|---|
| `reader_miss` | 36 | 54% | GT verbatim in the compiled context → reader/prompt is the bottleneck |
| `computed_gt_unverifiable` | 15 | 22% | GT is a computed/paraphrased value (count, sum, difference, paraphrase); gold turns are present, but answerability is not decidable without a model |
| `selection_miss` | 8 | 12% | a gold turn that carries the GT exists but was **dropped by compilation/selection** |
| `reader_miss_unverified` | 7 | 10% | short numeric GT; the verbatim hit may be a date/number coincidence and no GT-bearing gold turn was compiled |
| `retrieval_miss` | 1 | 1% | gold session never entered the candidate set |

By question type (bucket order as above: computed / reader / reader_unverified / retrieval / selection):

| type | computed | reader | reader_unv. | retrieval | selection |
|---|---|---|---|---|---|
| multi-session | 7 | 23 | 7 | 0 | 2 |
| knowledge-update | 2 | 3 | 0 | 0 | 1 |
| temporal-reasoning | 5 | 6 | 0 | 1 | 3 |
| single-session | 1 | 4 | 0 | 0 | 2 |

Path split of failures: aggregation path 30, graph path 37.

## Conclusions

1. **Candidate retrieval is not the bottleneck.** Gold sessions reach the
   candidate set in 66/67 failures (`retrieval_miss` = 1).
2. **Reader-side failure dominates:** `reader_miss` + `reader_miss_unverified`
   = 43/67 (64%), and a further 15 `computed_gt_unverifiable` failures are
   counting/arithmetic/paraphrase questions where predictions like
   `28` vs `99`, `$225+$120=$345` vs `$495` also point at the reader.
   At most 9/67 (13%) are decidable pipeline defects.
3. **`selection_miss` (8) is the one decidable pipeline defect:** gold sessions
   are in the candidate set, but the GT-bearing turn is dropped by budgeted
   compilation / per-session sampling. Representative cases:
   `eace081b` (Oahu, 0/7 bearing turns compiled), `36b9f61e` ($2,500),
   `9ee3ecd6` (100), `gpt4_5dcc0aab` (white Adidas sneakers, 0/3).
4. **Caveats of the LLM-free check:** for numeric GTs a verbatim hit can be a
   coincidence (`"3"` matches `2023/02/15`), which is why short-GT hits are
   split into `reader_miss_unverified`; computed answers (sums, counts) are
   reported as `computed_gt_unverifiable` rather than allocated.

Full per-question table: `benchmark_results/pipeline_v4/diag_fail_anatomy_all.txt`.

---

# Phase 3 — Reader × Memory Engine co-optimization (experiment infrastructure)

Research axes (independent, one variable per experiment, no benchmark
memorization): **A** reader-eliciting context design, **B** evidence
selection, **C** inference-cost reduction. Chosen success criterion:
accuracy AND efficiency together — report Accuracy / Tokens / Latency for
every condition.

## Conditions

| cond | change (exactly one) | artefact |
|---|---|---|
| `R0` | frozen baseline — prompt byte-identical to the original pipeline | `benchmark_results/pipeline_v4/lme_dev250_v4_pipeline.json` (and `--out` re-runs under current code) |
| `R1` | evidence-first ordering — same evidence blocks, relevance order only | `benchmark_results/pipeline_v4/lme_dev250_v4_R1.json` — **rejected (-2.8pp)** |
| `R3` | provenance — R0 order kept; agg Event headers stamped `[session id]` (graph path already has `[sid on date]` headers; per-turn full-date tags were dropped as redundant, +500 tok vs +90 tok) | `benchmark_results/pipeline_v4/lme_dev250_v4_R3.json` |
| R2, R4, R5 | not yet implemented (`--cond` rejects them) | — |

Note: `per_session_sampler.py` changed after the original 250-run baseline
(06:12), so paired comparisons use an `--out` R0 re-run under current code;
the original artefact stays frozen for historical reference.

## Infrastructure (Phase 0)

- `eval_v4_pipeline_dev.py --cond R0|R1 [--out PATH]` — condition routing,
  per-condition checkpoints, artefacts never overwrite the frozen baseline.
- `build_prompt()` + `reorder_evidence_first()` — prompt variants in one
  place; R0 pinned by `tests/unit/test_r0_prompt_frozen.py` (golden
  literals) so an accidental prompt edit fails before any benchmark run.
- `cache/v4_contexts/<cond>/<qid>.json` — frozen reader inputs for
  Reader-side sweeps (R5): replay identical contexts without recompiling.
- `scripts/benchmarks/report_v4_summary.py RUN [RUN2] [--out MD]` —
  type-wise accuracy/tokens/latency list; with two runs a paired comparison
  (exact McNemar, flips listed).
- `scripts/benchmarks/classify_reader_fails.py --run ... [--cond ...]` —
  hypothesis labels (output_format / aggregation / temporal / entity /
  conflict / extraction) for reader-side failures. Hypotheses for audit,
  never final verdicts.

## Validation performed

- `pytest tests/unit/test_r0_prompt_frozen.py test_lme_prompt_integration.py
  test_no_label_leak.py test_no_answer_injection.py` → **23 passed**.
- R0 fast-loop (3 qids) reproduces baseline predictions 3/3.
- R1 fast-loop: block multiset identical, only order differs; token count
  unchanged for the reordered question.

## Result — R1 (evidence-first ordering) vs R0, paired n=250

Source: `benchmark_results/pipeline_v4/lme_dev250_v4_R0rerun.json` (current
code, 185/250 = 74.0%, mean 1324 tok) vs `lme_dev250_v4_R1.json`
(178/250 = 71.2%, mean 1324 tok). Full report:
`benchmark_results/pipeline_v4/report_R0_vs_R1.md`.

| metric | R0 | R1 | delta |
|---|---|---|---|
| Accuracy | 74.0% | 71.2% | **-2.8 pp** |
| Mean tokens | 1324 | 1324 | +0 (content-preserving confirmed) |
| Mean latency | 1050ms | 1032ms | -17ms |
| McNemar | — | — | A-only 11, B-only 4, exact p = 0.1185 |

Per type: knowledge-update **-10.3pp** (92.3→82.1), multi-session **-4.5pp**,
temporal -1.5pp net; all other types identical. Reorder was applied to
~81% of prompts (the rest had no lexical signal / single block).

**Verdict: R1 rejected.** Chronological order is itself evidence — breaking
it to chase lexical relevance destroys the reader's ability to resolve
updates (knowledge-update collapsed: latest-vs-superseded requires temporal
order) and to run elapsed-time arithmetic. The chronological timeline must
be PRESERVED in any following context-design condition; attribution and
structuring should be added *within* the time-ordered layout.

Next condition (R3): keep R0 ordering, attach provenance/structure
(source session, timestamp, speaker) to each evidence block — Experiment A's
attribution half, tested alone.

## Result — R3 (provenance tags on agg Event headers) vs R0, paired n=250

Source: `lme_dev250_v4_R0rerun.json` vs `lme_dev250_v4_R3.json`.
Report: `benchmark_results/pipeline_v4/report_R0_vs_R3.md`.

| metric | R0 | R3 | delta |
|---|---|---|---|
| Accuracy | 74.0% (185) | 74.4% (186) | **+0.4 pp** (+1 question) |
| Mean tokens | 1324 | 1342 | +18 (agg-only cost; graph path unchanged) |
| Mean latency | 1050ms | 1082ms | +32ms |
| McNemar | — | — | A-only **0**, B-only 1, exact p = 1.0000 |

- multi-session +1.5pp (40.3→41.8); all other types identical.
- **Zero regressions**: no question that R0 answered correctly was lost.
- Scope verified from the context cache: agg prompts changed 7/7, graph
  prompts changed 0/35 at run start (graph already carries `[sid on date]`).
- Design note: a first variant stamped `[sid | date]` per turn (+491 tok on
  one question) — redundant because the Event header already carries the
  timestamp; stamping `[sid]` once per event cut the cost to +94.

**Verdict: R3 accepted as non-regressing (not significant on accuracy
alone, p=1.0).** Provenance costs ~1.4% tokens, never hurts, and is kept
as a building block for auditability / provenance-preserving designs; its
accuracy effect may compound with later conditions.

## Result — R2 (compact evidence) vs R0, paired n=250

Source: `lme_dev250_v4_R0rerun.json` vs `lme_dev250_v4_R2.json`.
Report: `benchmark_results/pipeline_v4/report_R0_vs_R2.md`.

| metric | R0 | R2 | delta |
|---|---|---|---|
| Accuracy | 74.0% (185) | 74.4% (186) | +0.4 pp |
| Mean tokens | 1324 | 1252 | **-72 (-5.4%)** |
| Mean latency | 1050ms | 927ms | **-123ms (-11.7%)** |
| McNemar | — | — | A-only 3, B-only 4, exact p = 1.0000 |

Per type: multi-session +1.5pp **with -129 tokens**, preference +6.7pp
(n=15), single-session-user -2.9pp (1 regression: `75499fd8` — the
zero-overlap filter dropped the evidence turn whose words did not lexically
overlap the question). All other types identical.

**Verdict: R2 accepted under the accuracy-and-efficiency criterion.**
Accuracy statistically indistinguishable (p=1.0) while cutting ~5% tokens
and ~12% latency. Caveat recorded: lexical filtering is a ceiling probe and
can drop paraphrased evidence (see `75499fd8`); it should ship only with a
safety net (keep-at-least-k blocks / semantic overlap), not raw zero-overlap.

## Phase-1 ladder summary (single-variable conditions vs R0, n=250)

| cond | accuracy | Δacc | Δtokens | Δlatency | verdict |
|---|---|---|---|---|---|
| R0 (current code) | 74.0% | — | — | — | baseline |
| R1 evidence-first order | 71.2% | -2.8pp | +0 | -17ms | **rejected** (temporal order is signal) |
| R2 compact evidence | 74.4% | +0.4pp | **-72** | **-123ms** | **accepted** (efficiency win) |
| R3 provenance tags | 74.4% | +0.4pp | +18 | +32ms | **accepted** (0 regressions) |

Integration step (Phase 6 discipline): test R2+R3 (`--cond R23`) against R2 —
adding provenance on top of R2 is exactly one variable.

## Result — R23 (R2+R3 integration)

Source: `lme_dev250_v4_R2.json` vs `lme_dev250_v4_R23.json`.
Report: `benchmark_results/pipeline_v4/report_R2_vs_R23.md`.

| metric | R2 | R23 | delta (provenance on top of R2) |
|---|---|---|---|
| Accuracy | 74.4% (186) | 74.4% (186) | +0.0 pp (A-only 2, B-only 2, p=1.0) |
| Mean tokens | 1252 | 1267 | +15 |
| Mean latency | 927ms | 962ms | +35ms |

Provenance on top of compact evidence is **accuracy-neutral** (±2 flips
each way); it costs +15 tokens (less than standalone R3's +18 because the
filter also removes some tagged events). Its value is auditability, not
measured accuracy — decide inclusion on that basis.

## Final ladder (paired, n=250, current code)

| cond | accuracy | Δacc vs R0 | Δtokens vs R0 | Δlatency vs R0 | verdict |
|---|---|---|---|---|---|
| R0 | 74.0% (185) | — | — | — | baseline |
| R1 | 71.2% (178) | -2.8pp | +0 | -17ms | rejected |
| R2 | 74.4% (186) | +0.4pp | **-72 (-5.4%)** | **-123ms (-11.7%)** | accepted (efficiency) |
| R3 | 74.4% (186) | +0.4pp | +18 | +32ms | accepted (0 regressions) |
| R23 | 74.4% (186) | +0.4pp | -57 (-4.3%) | -88ms (-8.4%) | accepted (R2 savings + provenance) |
| H1 anchor prefill | 72.0% (180) | **-2.0pp** | -7 | -66ms | **rejected** (2 fixed, 7 regressed) |
| **R2 + safety floor** | **74.8% (187)** | **+0.8pp** | **-65 (-4.9%)** | **-88ms (-8.4%)** | **best observed; efficiency condition** |

Accuracy differences of ±1–2 questions are within noise (McNemar never below
p=0.12 across the ladder); the robust finding is **R2-class compaction:
same-or-better accuracy with ~5% fewer tokens and ~8–12% lower latency**.
Under the chosen criterion (accuracy AND efficiency), the shipping
candidate is **R2 + safety floor (74.8%, 1259 tok, 962 ms)** — or R23 if
provenance auditability is wanted at +15 tokens.

## Next phases (open)

- **R4** decomposed reader (extract → answer, two calls) — accuracy axis A.
- **B axis**: fix the `selection_miss` cases (GT-bearing turns dropped by
  compilation) — one change at a time, full-250 regression each.
- **C axis**: budget ladder (900/1300/1800/2200) accuracy-token curve and
  cheap-reader routing.

## R5 — Reader sweep over frozen contexts (DONE)

`scripts/benchmarks/r5_reader_sweep.py --cond <cond>` replays the exact
frozen prompts from `cache/v4_contexts/<cond>/` against several readers;
only the model changes. Deterministic overrides (3) are copied per model.

| model | acc | Δ vs 7b | mean tok | mean lat | notes |
|---|---|---|---|---|---|
| qwen3:4b (think=False) | **84.8%** (212) | **+10.8pp** | 1326 | 5121ms | 4.8x latency |
| llama3.1:8b-instruct-q4_K_M | **82.8%** (207) | **+8.8pp** | 1254 | 1359ms | +27% latency |
| qwen2.5:7b-instruct (baseline) | 74.0% (185) | — | 1324 | 1073ms | reproduces R0 rerun **exactly** (sanity ✓) |
| qwen2.5:1.5b | 69.6% (174) | -4.4pp | 1324 | 366ms | 1/3 latency (cheap-reader candidate) |

Per type (acc): llama3.1 multi-session **73.1%** vs 7b 40.3%; qwen3
temporal 92.4%, assistant 92.9%, user 97.1% (knowledge-update 84.6% below
7b's 92.3%).

**Findings**

1. **The reader is the bottleneck.** Same contexts, +8.8~10.8pp from a
   different reader — corroborating the failure anatomy (reader_miss
   dominant) and the doctrine that runtime gains must survive a weaker
   reader but ceiling gains come from the reader.
2. The dragging category (multi-session 40.3%) is NOT intrinsically hard:
   llama3.1:8b reaches 73.1% on identical inputs.
3. Cost ladder: 1.5b (366ms, -4.4pp) → 7b → llama3.1 (1359ms, +8.8pp) →
   qwen3:4b (5121ms, +10.8pp) — the C-axis routing tradeoff, measured.

**Two harness bugs found and fixed during R5 (disclosure):**

- *Empty-answer scorer bug*: `score_answer` substring rule made `""` match
  every gold string (`"" in "2"` is True). First qwen3 attempt returned 236
  empty answers (server-side thinking consumed the frozen `num_predict=256`)
  and scored a bogus 94.0%. Fixed: empty answer → False; `think` knob added
  to `OllamaAnswerer` (`num_ctx` precedent) and qwen3 arm rerun with
  `think=False` → 84.8%, empty=0. **Re-scoring all five ladder runs with the
  guard: identical counts (185/178/186/186/186) — no published number moved.**
- *Abstention auto-pass (pre-existing, NOT changed)*: `eval_v4_pipeline_dev`
  passes `pcc_is_abstention=is_abstention_reference` (the reference label)
  whereas official `evaluate_item` passes `bool(pcc.is_abstention)`. All 15
  abstention-reference questions therefore score correct in every run; 12
  answered with abstention keywords (would pass anyway), 3 gave substantive
  answers that would fail keyword-only scoring. Absolute accuracies are
  inflated by ≤3 questions (~1.2pp) **equally for every condition and every
  sweep arm, so all deltas/comparisons are unaffected**. Fixing it would
  change published absolutes — decision deferred to the benchmark owner.

### R5b — does R2 survive a different reader? (llama3.1:8b, paired n=250)

`r5_reader_sweep.py --cond R2 --models llama3.1:8b...` vs the R0 arm:

| metric | R0 | R2 | delta |
|---|---|---|---|
| accuracy | 82.8% | 81.6% | -1.2pp (A-only 9, B-only 6, McNemar p=0.607) |
| mean tokens | 1254 | 1188 | **-66 (-5.3%)** |
| mean latency | 1359ms | 1279ms | -79ms |

- **Efficiency generalizes**: the ~5% token cut is reader-independent (it is
  a property of the context, and it reproduced on a second reader).
- **Accuracy effect does not replicate** (7b: +1q, llama3.1: -3q, both
  within noise) → R2 must be sold as an *efficiency* condition, never as an
  accuracy condition.
- **The `75499fd8` regression reproduced**: `Golden Retriever.` (R0) →
  `I don't know.` (R2) on llama3.1 as well — the zero-overlap filter really
  does drop paraphrased evidence, independent of reader. The safety-net
  requirement (keep-at-least-k / semantic overlap) is confirmed, not anecdotal.

## Selection-miss autopsy (hygiene step, before R6)

`scripts/benchmarks/diag_selmiss.py` tracks every GT-bearing turn through
retrieval → candidates → anchors → bundles → compiled context for the 8
`selection_miss` failures of the R0 rerun; `trace_knapsack.py` replays
`compile()`'s greedy with pick logging.

| class | n | root cause (verified, not hypothesised) |
|---|---|---|
| G — knapsack starvation | 5 (graph) | cert reserve eats up to 39% of budget (290/750); submodular trap: the turn that first covers the query words wins the density race, the ANSWER turn is redundant (`new_words=0`) and loses; oversized anchors (288–776 tok — `max_bundle_tokens=150` only caps *adjacent additions*, never the anchor itself) either crowd or exceed `effective_budget` entirely |
| S — sampler starvation | 2 (agg) | `turns_per_session=2` cap + generic gate + round-robin packing drop a retriever rank-6 / score-53 turn |
| R — candidate drop | 1 | `top_sids = sessions of the top-18 TURNS` — the gold session's best turn ranked 30 |

### H1 — anchor-priority prefill: REJECTED (paired n=250)

`--cond H1` pre-filled the top-5 anchor bundles before density fill
(cost-free, same budget). Report: `report_R0_vs_H1.md`.

| metric | R0 | H1 | delta |
|---|---|---|---|
| Accuracy | 74.0% (185) | 72.0% (180) | **-2.0pp** (A-only 7, B-only 2, McNemar p=0.180) |
| Mean tokens | 1324 | 1318 | -7 |

- Fixed: `3249768e` (rank-1 Absinthe bundle, 0/1 → 1/1 bearing turns in
  context), `e982271f`. assistant type +7.1pp.
- Regressed 7: knowledge-update -5.1pp, temporal -3.0pp, preference -6.7pp —
  static anchor rank favours question-term overlap, which for
  knowledge-update means OLD-state mentions displace the latest-state turn.
- **Verdict: rejected.** Static rules cannot tell which high-rank evidence
  is the answer — exactly the deficit R6's Evidence Sufficiency judge must
  close (the 6 remaining selection-miss cases are R6 escalation targets).

## Hygiene item 2 — R2 safety floor

`filter_evidence` now never lets the context fall below
`MIN_EVIDENCE_BLOCKS = 3` (earliest dropped blocks reinstated in original
order), targeting the paraphrase-drop regression (`75499fd8`). R2 rerun
with the floor: `lme_dev250_v4_R2.json` (the previous artefact is kept as
`lme_dev250_v4_R2_nosafety.json`).

| metric | R0 | R2 (no safety) | **R2 + safety floor** |
|---|---|---|---|
| Accuracy | 74.0% (185) | 74.4% (186) | **74.8% (187)** |
| Mean tokens | 1324 | 1252 | **1259 (-65 vs R0)** |
| Mean latency | 1050ms | 927ms | **962ms (-88 vs R0)** |

Paired vs R0 (`report_R0_vs_R2safety.md`): +0.8pp (A-only 2, B-only 4,
p=0.688); temporal **+3.0pp**, multi +1.5pp, user -2.9pp. The floor also
turned two old-R2 failures into passes (`4dfccbf7`, `gpt4_e061b84g`).

**Honest notes**: (a) the floor did NOT rescue its target `75499fd8` — the
kept set was already ≥3 blocks, so nothing was reinstated; a
claim-aware drop rule remains an R6 target; (b) accuracy deltas are still
within noise (p=0.69): this ships as an **efficiency condition with no
accuracy loss** at -4.9% tokens / -8.4% latency.
