# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-10-04

### Added
- **Auditable Results Registry & Automated Verification** (`benchmark_results/RESULTS_REGISTRY.json`, `scripts/benchmarks/verify_registry.py`):
  - Machine-readable registry tracking 21 core headline benchmark metrics across LoCoMo 1,540, LongMemEval 500, BEAM 500K, and McNemar model sensitivity tests.
  - One-command automated verifier (`verify_registry.py`) confirming 21/21 PASS (100% consistency against raw artifacts).
- **Single Frozen Generic Matcher (`matcher-v2`)** (`LoCoMoAdapter.score_binary`):
  - Unified live execution and offline re-scoring under a single, fully generic rule-based matcher with WideSlicer stemming.
  - Completely purged all dataset-specific keyword enumerations (`HIGH_SPECIFICITY`, `liberal`, `national park`, phrase lists, etc.) and dead code (`evaluate_refined_gate`).
- **Two-Tier Holdout Isolation Protocol** (`benchmark_config/holdout.yaml`):
  - Formal division into Tier 1 (Strict Clean Holdout: `conv-48`, 191 Qs) with zero diagnostic history, and Tier 2 (Validation Split: `conv-42`, 199 Qs).
- **Expanded Unit Test Suite**: Reached 328 unit tests (740+ total tests, 100% passing).

### Changed
- Clarified evaluation protocol taxonomy across READMEs: distinguishing official benchmark datasets and official LoCoMo Token F1 harness from deterministic in-house binary extraction matchers.

## [0.3.0] - 2026-10-01

### Added
- **HippoRAG-Style Personalized PageRank (PPR) Associative Memory Graph** (`artificial_memory/recall/ppr_graph.py`):
  - Implemented neurobiologically-inspired associative memory layer over deterministic Entity-Turn bipartite graph.
  - Zero Write LLM calls, Zero Recall LLM calls: Pure power-iteration stationary distribution activation spreading.
  - Wired into `MinimumSufficientContextCompiler` (`ppr_retrieval=True`) to interleave associative multi-hop evidence behind top lexical hits.
- **Allen's Interval Algebra & Minute-Level Temporal Precision** (`artificial_memory/temporal/interval_algebra.py`, `artificial_memory/recall/temporal_resolver.py`, `artificial_memory/skills/answer_committer.py`):
  - Implemented Allen's 13 interval relations (`before`, `meets`, `overlaps`, `during`, etc.) and `TimeInterval` algebraic engine.
  - Upgraded temporal resolver and committer to preserve `HH:MM` timestamp precision, deterministically resolving same-day event order without LLM coin flips.
- **Derivation Scaffolding Engine** (`artificial_memory/context/derivation_scaffold.py`):
  - Injects deterministic calculation and order traces (`[DERIVATION SCAFFOLD: ...]`) into compiled MSC context.
  - Directly breaks the Gold-Context Ceiling where 7B readers fail on arithmetic/ordering despite evidence presence.
- **Bi-Temporal State Model (Valid Time vs Assertion Time)** (`artificial_memory/core/ir/structured.py`):
  - Added `valid_from`, `valid_until`, and `assertion_time` to `StructuredIR`.
  - Implemented `is_valid_at(target_time)` point-in-time querying and `supersede()` for conflict-free state evolution.
- **Category-Bounded Sampling & Stop Sequences** (`artificial_memory/research/benchmarks/llm.py`, `locomo_adapter.py`, `longmemeval_adapter.py`):
  - Category-specific output token ceilings and stop sequences without violating the strict `["self", "question_text", "context"]` leakage boundary test (`test_phase8_leakage.py`).
  - Physically suppresses runaway reasoning loops and verbose preambles on 1.5B / 7B small reader models.
- **Unit Test Suite Expansion**: Expanded suite from 214 to 264 tests (100% passing across all 264 unit tests and leakage boundary tests).
- **Deterministic Speaker Attribution & Pronoun Binding** (`artificial_memory/compiler/speaker_normalizer.py`):
  - Normalizes ambiguous first-person pronouns ("I", "my", "me") into explicit speaker entities ("Caroline", "Caroline's").
  - Deterministically binds kinship and possession terms ("my sister", "my dog") to the speaking actor context.
  - Eradicates conversational Actor Attribution Failure in Adversarial (LoCoMo Cat 5) and Single-Hop (Cat 4) scenarios.
- **Conversational SPO Extraction Engine** (`artificial_memory/compiler/ir_extractor.py`):
  - Expanded `UniversalIRExtractor` with Patterns H through K covering daily dialogue predicates (Actions/Events, Preferences/Tastes, Possession/Kinship, Occupation/Identity).
  - Reduced unstructured statement fallbacks from ~65% down to structured Subject-Predicate-Object triplets without LLM calls.
- **Compound Temporal Algebraic AST & Normalizer Integration** (`artificial_memory/temporal/interval_algebra.py`, `context/temporal_normalizer.py`):
  - Implemented `TemporalASTNode` hierarchy (`DatePointNode`, `OffsetNode`, `OrdinalWeekdayNode`, `IntervalSpanNode`) and `resolve_ast()` algebraic evaluator.
  - Added deterministic natural-language expression parser `parse_compound_temporal_expression()` handling compound relative offsets ("2 weeks after 25 December 2022") and ordinal weekdays ("the third Monday of June 2023").
  - Integrated into `TemporalNormalizer` (Rule 19) for zero-leak resolution of complex temporal bounds.
- **Expanded Zero-Reader Committer Skills** (`artificial_memory/skills/answer_committer.py`):
  - Added `commit_derivation_scaffold()`: Directly commits count and summation queries from derivation scaffolds (0 ms, 0 tokens, 100% arithmetic precision).
  - Added `commit_single_hop_fact()`: Directly commits unambiguous single-hop entity attributes (occupations, locations, favorites, names) with confidence >= 0.88.
  - Added master `commit_answer()` dispatcher prioritizing proof-carrying certificates, derivation scaffolds, temporal spans, and single-hop facts before Reader fallback.
- **Multilingual Foundation & LocaleProvider** (`artificial_memory/i18n/locale_provider.py`):
  - Decoupled hardcoded English calendar tokens and month mappings into pluggable `LocaleProvider` interface.
  - Implemented `EnLocaleProvider` and `JaLocaleProvider` for robust CJK/multilingual expansion.

## [0.2.0] - 2026-09-28

### Changed
- **Project Renaming**: Renamed from `Artificial Memory` to **`Lethe Akribeia`** (*"Forgetting is not deletion. It is loss of resolution."*).
- **Package Version**: Bumped to `0.2.0` with CLI aliases `lethe` and `am`.
- **Honest Baseline Publication**: Published full 1,540-question official LoCoMo benchmark results (Deployed 7B Reader: 50.9% official F1, Evidence Retrieval: 81.1%).

### Added
- **Frontier Reader Validation Probe**: Validated the Decoupled Context-Reader harness by freezing Lethe compiled context and swapping Reader from 7B to Gemini 3.6 Flash on a 10-question probe (+25.46pp F1, +25.32pp Binary Hit).
- **Deterministic Co-Processors**:
  - `CHRONOS`: Calendar & temporal arithmetic co-processor for relative date grounding.
  - `AggregationCoProcessor`: Deterministic counting and list aggregation co-processor.
  - `AnswerCommitter`: Answer contract enforcement preventing hallucinations.
- **Error Ceiling & Failure Census**: Three-way partitioning of failures on 1,540 questions: Retrieval Failure (8.7%), Commitment Failure (2.1%), and Reasoning Gap / Unresolved (24.5%).
- **Corrected Content+Date Retrieval Oracle**: Audited and replaced flawed ID-matching oracle with content+timestamp grounding.

- **AM-decides layer, slice 1: deterministic temporal committer** (`skills/answer_committer.py`)
  - Decides temporal answers without the reader: IDF-weighted evidence-turn selection, anchor on the strongest key term, answer-shape aware (date vs duration vs year), and it prefers the annotation `TemporalNormalizer` produced over the provenance header date - the exact mistake measured in 45 of 184 wrong temporal answers.
  - `parse_turns` understands both context dialects: LoCoMo `[D4:5 on 27 June, 2023] (In reply to …) Caroline: …` and LongMemEval `[answer_280352e9 on 2023/05/30 (Tue) 17:27] user: …`. The first version silently dropped every LoCoMo turn whose header was followed by a space.
  - `extract_certificate_answer` commits the runtime's own `[Temporal Calculation: … Exactly 7 days passed …]` / `[Temporal Ordering: … from first to last: …]` certificate instead of making the model redo arithmetic the memory system already did.
  - `MIN_TURN_SCORE = 0.80` is a swept default, not a guess: below ~0.60 the committer loses to the reader.
  - Measured on all 321 LoCoMo temporal questions: **42.68% → 47.98%** (normaliser rules 16-18) **→ 49.22%** (with the committer), where 111 questions (34.6%) are answered at **70.3%** with **0 ms and 0 LLM calls** and the category's wall clock fell 2.25 s/Q → 1.48 s/Q. On LongMemEval temporal-reasoning the certificate branch claims 56/133 at 92.9% against a 91.1% reader (+0.75pp).
- **`context/temporal_normalizer.py` rules 16-18**: `last <season>` now resolves the year from the *end* of the season (the previous rule returned the wrong year for every reference date inside or after the named season), plus `N years ago → (YYYY)` and `N months ago → (Month YYYY)`. Expansion rates on the rebuilt cache: `this week` 41.5% → 100%, `next week` 0% → 100%, `last <season>` 0% → 100%, `last <weekday>` 89.8% → 100%.
- **`recall/answer_shape.py`**: deterministic classification of the answer form a question demands (polarity / choice / attribute). **Default off and documented as a refuted hypothesis** - see "Fixed" below - but kept as the diagnostic that shows which 63 of the 96 open-domain questions are attribute-shaped.
- **Measurement tooling** (the loop that made the above cheap - no LLM calls, seconds per iteration):
  - `scripts/benchmarks/commit_sweep.py` - "does the committer beat the reader on the questions it claims?", paired against a reader artefact, `--suite locomo|lme`, `--sweep` for the threshold curve. It is what decided both the temporal threshold and the two rejected arms. `--report-out` now stores the run as a raw-count artefact, and `--qtype all` measures every LongMemEval capability in one pass.
  - **The committer's numbers are now first-class scorecard KPIs** (`apex_scorecard.py` §6 "Deterministic memory engine"): *deterministic coverage*, *commit accuracy*, *reader on the same set*, *commit delta*, *fallback rate* and *LLM calls saved*, read from `benchmark_results/committer_metrics/*.json`. Totals are re-derived from the stored counts, never averaged percentages, and a missing or corrupt artefact degrades to `NOT MEASURED` instead of failing the report (`tests/unit/test_scorecard_committer.py`). First stable measurement, at threshold 0.80: LoCoMo temporal **34.0% coverage at 70.6%** against a 67.0% reader (+1.25pp net), LongMemEval **11.8% coverage at 93.2%** against 94.9% (-0.20pp, i.e. the certificate branch is at parity, not ahead), 168 questions answered with **0 LLM calls**.
  - `scripts/benchmarks/_partial_cmp.py` - restricts a baseline to exactly the questions a *running* arm has answered, so a partial number is never compared against a full-run headline.
  - `scripts/benchmarks/_answerability_ceiling.py` - per category: oracle misses vs GT-verbatim vs GT-words-in-context vs unreachable, which is what separates "retrieval's problem" from "reader's problem".
- `scripts/run_locomo_1540.py --commit-temporal` / `--answer-shape-gate` (both **default off**, so the frozen path stays reproducible).
- **AM-decides layer, step 2: model sensitivity** (`scripts/benchmarks/model_sensitivity.py`, scorecard §7)
  - Pairs two frozen-cache runs that differ in exactly one field - `--model` - joins them question by question and buckets every shared question into both-correct / A-only / B-only / both-wrong. Only the 4-way **counts** are stored; accuracies, delta and the exact two-sided McNemar p-value are re-derived by the scorecard, so a stored artefact and the published table can never disagree.
  - Three controls guard "only the model changed": per-question token count and oracle recall must match (both runtime-side), `--claims` cross-checks the offline claims cache against what the runs retrieved, and the committer's *decision* (`committed` vs `reader`) and its *extraction point* (`turn`/`anchor`) are compared as two separate controls. Mismatches are counted, sampled and printed with `!!`, never averaged away.
  - **Zone mode is three-valued**, and only one value may assert the invariant: `system` (both arms ran the committer → the deterministic zone must score identically under both readers, else it is a determinism bug), `counterfactual` (neither did → that row is the *readers'* accuracy on the code's questions, held out of the zone table), `mixed` (an A/B of the committer, warned as not a reader swap).
  - `--restrict-both-reader` builds a **repeat control**: the same reader across two runs that differ only by the committer flag - the run-to-run noise floor every delta has to clear. Measured on LoCoMo: **0/210 correctness flips at 7B**, **4/1431 at 1.5B** (the smaller reader is the noisier one; still an order of magnitude below the deltas below).
  - `run_model_sensitivity_arms.ps1` runs the arms (`-SkipLme`, `-Skip15bArms`, `-Recheck7bSystemArm`, `-Recheck7bLmeArm`) and logs each to `benchmark_results/logs/*.log`.
  - **Measured, LoCoMo (1,540, both arms rebuilt under current code), 7B → 1.5B: 64.7% → 64.5% (+0.13pp, p=0.95)** - the runtime absorbs the reader swap over the whole suite, with the deterministic zone **70.6% = 70.6% (0 flips)**. The exception is temporal reasoning, category 2 (321 questions): **50.2% → 43.6% (+6.54pp)**, against 67.0% → 57.8% (−9.17pp) on the reader-only pairing - i.e. the committer absorbs most of the reader downgrade *inside* the capability it owns, and nothing outside it. A second 7B reader-only arm would change nothing here: the repeat control shows the two 7B runs agree on **210/210** questions with 0 text differences.
  - **Measured, LongMemEval 500, 7B → 1.5B (both arms built under current code): 81.6% → 80.0% (+1.60pp, p=0.40)** - again the runtime absorbs the swap, with every control at zero (0/500 token drift, 0/500 oracle drift, claims cache agrees 500/500). LLM zone +2.72pp (p=0.18). **The same comparison paired against the *stored* 7B arm read −4.20pp (p=0.044) - the "smaller reader wins" result was entirely an artefact of pairing across two pipeline builds.**
  - **Pairing failures are reported, not hidden.** The stored 7B LongMemEval arm predates the current MSC compiler - recompiling its questions offline (deterministically) reproduces today's arm on every probe while the stored one differs on 340/500 token costs and 5 oracles - so `-Recheck7bLmeArm` rebuilds it rather than pairing across builds. The stored 7B LoCoMo commit arm claims 111 questions against today's 109, which shows up as 4 engine + 15 anchor mismatches in the system pair (`-Recheck7bSystemArm` purges it, and the purged pairing is the one §7 now reports).
  - Scorecard §7 renders it: per-slice table, pooled row **one pairing per cohort** (two pairings of the same question set can never double-count), zone table with the held-out arithmetic spelled out, controls, KPI definitions, `NOT MEASURED` when nothing is stored. Tests: `tests/unit/test_model_sensitivity.py`, `tests/unit/test_scorecard_sensitivity.py`.
  - `scripts/benchmarks/finalize_model_sensitivity.ps1` - waits for the arms, pairs each one the moment it lands, and re-renders the scorecard, so a 45-minute arm does not have to be babysat.
- **AM-decides layer, step 3a: failure-ceiling census** (`scripts/benchmarks/failure_ceiling.py`)
  - Asks the question Step 2 opened: a score of 64.7% limits the runtime only if the missing 35.3% is semantic reasoning. Every wrong answer of a frozen run is bucketed from stored evidence, with **no new model run** - the run artefact, the second reader run and the compiled-context cache are enough.
  - **LoCoMo 1,540, deployed 7B reader (544 failures):** retrieval **211 (13.7%)**, commit **32 (2.1%)**, unresolved **301 (19.5%)** - 89 reader-sensitive, 212 reader-insensitive. The buckets partition the failures exactly, per category as well as in total.
  - **Reader accounting, the 2x2 that bounds any reader work:** both right 863, reference only 133, other reader only 131, both wrong 413. Only **264 of 1,540 questions (17.1pp) change outcome with the reader at all**, and 131 of those are won by the *smaller* reader.
  - **B (context noise), C (reasoning) and E (genuinely semantic) are deliberately not separated.** From stored evidence they share one signature - "the answer is in the context and the reader missed it" - so the mass stays in one *unresolved* bucket with the signals that do discriminate attached (the other reader's verdict, and whether the ground-truth string survives verbatim in the compiled context: 79 of 301). One controlled experiment closes the gap: re-read the same questions with the gold evidence only.
  - Two defects the census surfaced that the score alone did not: **6 committed questions were answered wrong with no answer session in the context at all** - the committer over-claims, which is a threshold defect and the reason `commit` is tested before `retrieval`; and **42 questions filed as retrieval failures were answered correctly by the 1.5B reader**, i.e. oracle false negatives (or parametric luck) that make the retrieval bucket an upper bound.
  - Tests: `tests/unit/test_failure_ceiling.py` (6), including that the buckets partition the failures and that `unresolved` is never allocated to a hypothesis.
- **AM-decides layer, step 3a.5: oracle false-negative audit** (`scripts/benchmarks/oracle_fn_audit.py`) - the retrieval oracle turned out to be an **id** test, and the ids do not mean what they say.
  - The LoCoMo oracle is `any(ev_id in pcc.context_text for ev_id in evidence_ids)`, but the context compiler numbers the turns it emits in its own group order: `D1:3` in a compiled context is not `D1:3` in `locomo10.json` (across LoCoMo 1,540 every rendered gold id carried the right session date and the **wrong speaker** - one index out). So the flag compares labels, not turns.
  - Re-tested against the dataset by content and session date over the whole slice: **165 false positives** (flagged retrieved, gold turn not in the context) and **172 false negatives** (flagged missing, gold turn present) - **21.9% of the flags are wrong on one side or the other**, while the aggregate barely moves (id-based 80.6% vs content-based 81.1%). A per-question flag this unreliable cannot carry a per-question bucket.
  - The 42 disputed questions resolve to **23 genuine oracle false negatives** (the gold turn is in the context), **16 true retrieval failures** (no evidence, yet the other reader answered), 3 ambiguous. Projection only, nothing reclassified: with a content oracle the retrieval bucket falls **211 → 134 (13.7pp → 8.7pp)** and the unresolved mass rises 301 → 378, because 108 questions move out of retrieval and 31 move in.
  - **The matcher is lenient, and that touches the score itself.** Of the 16 true retrieval failures the 1.5B "answered", 15 predictions do not contain the gold string at all (`"Yes; it's classical music"` scored correct for `"Yes."`). Across the run, **137 correct answers in each arm** are scored correct without the gold string and with <50% word overlap (open-domain 22.9%, multi-hop 12.4%, single-hop 9.4%, temporal 0.3%) - only 68 of them are the same questions in both arms. Under a strict rule the absolute score would be **55.8% / 55.6%** instead of 64.7% / 64.5% - but the *delta* stays **+0.13pp**, so the Step 2 conclusion (7B ≈ 1.5B) is invariant to the matcher's leniency while the absolute number is not.
  - No score, grade or ceiling artefact was changed by this audit; it is read-only and reports the discrepancy instead.
  - Tests: `tests/unit/test_oracle_fn_audit.py` (5), pinning that evidence is located by content and date rather than by label, and that a reader answering without evidence is a true retrieval failure.
- **AM-decides layer, step 3b: the retrieval oracle is a content test** (`locomo_adapter._evidence_present`, `scripts/benchmarks/recount_oracle.py`)
  - The flag used to ask whether the gold evidence **id** appeared in the compiled context. The context compiler numbers the turns it emits in its own group order, so `D1:3` in a context is not `D1:3` in `locomo10.json` - across LoCoMo 1,540 every rendered gold id carried the right session date and the **wrong speaker**. The oracle is now asked of the evidence itself: is this turn's text present (>=80% content-word overlap, which survives the compiler's condensation) in a context that also carries that turn's session date.
  - Effect on the failure ceiling, recomputed from the frozen cache without re-running anything: **retrieval 211 -> 134 questions (13.7pp -> 8.7pp)**, unresolved 301 -> **378 (24.5pp)**. So a third of the "retrieval is failing" mass was the measurement, not the retriever - and the reader accounting is unchanged (264 reader-determined), as it must be: the oracle is a property of the context, not of the reader.
  - Existing runs are corrected offline into `<stem>.oraclefix.json` (the original is never overwritten, and keeps its old flag as `oracle_recall_by_id`, which every future run now records too). 172 flags went False -> True, 165 True -> False, identically in both arms.
  - Tests: `tests/unit/test_locomo_oracle_presence.py` (5), including that the same words in a *different session* do not count and that a paraphrased turn does not count.
- **AM-decides layer, step 3c: the LoCoMo scoring contract** (`scripts/benchmarks/locomo_scorer.py`) - the published number is a *policy*, so the policy is printed next to it.
  - The scorecard's number is a **binary hit rate under Lethe's own matcher** (substring in either direction, one word when the gold has three or fewer, 33% word overlap). LoCoMo's protocol is **graded F1** with its own normalisation (lowercase, drop punctuation, drop a/an/the/and) and per-category rules. Same runs, three policies: **lethe_current 64.7% / 64.5%**, **lethe_strict 35.0% / 34.0%**, **official_f1 50.9% / 38.1%**.
  - **The Step 2 conclusion is metric-dependent and is now stated per metric.** Under the hit rate the readers are interchangeable (+0.13pp); under the official F1 the 7B is **+12.80pp** ahead. Mechanism: the 1.5B answers **11.7 words** on average against the 7B's **5.4**, and **50.0% of its hits score below 0.5 F1** versus 21.3% for the 7B - the binary rule forgives the verbosity and vagueness that F1 charges for. So "the reader is not the bottleneck" was a property of the matcher, and the smaller reader's problem is at least partly *answer shape*, not only capability.
  - Part of the step-3a.5 "leniency" list is **not** a bug: the official harness itself truncates an open-domain gold at the first ``;``, so a bare ``"Yes."`` *is* the correct answer to ``"Yes; it's classical music"``. Official rules re-implemented from `snap-research/locomo task_eval/evaluation.py`: category 3 gold truncated, category 1 partial F1 over sub-answers, categories 2/3/4 plain F1, category 5 the refusal check.
  - **The gap is answer shape, not knowledge.** Decomposing the official F1 into its parts: the 1.5B's **recall is 0.543 against the 7B's 0.553** (it finds the same tokens) while its **precision is 0.356 against 0.525** (it pads them). So under the official protocol the small reader is not missing facts, it is burying them - which is why half of its hits score below 0.5 F1 and why the binary matcher never noticed.
  - **Confirmed by a one-line A/B** (`run_locomo_1540.py --concise-answer`, same model, same cache, same questions, one extra instruction; tag `temporal321_rules_commit_15b_concise`): mean answer length **11.7 → 9.3 words**, precision **0.356 → 0.450**, hits scoring below 0.5 F1 **50% → 31%**, and the official-F1 gap to the 7B **+12.80pp → +7.04pp**. So **~45% of the measured reader gap was formatting**, recoverable without touching the runtime's memory at all. The residual is not pure capability either: forcing brevity cost the 1.5B recall (0.543 → 0.499) and cost it binary hits (64.5% → 60.8%), so what remains is a mix of "the small model needs more words to cover the answer" and genuine capability.
  - The lever this exposes is a **runtime-side answer-format contract** - a deterministic trimmer that keeps the value-bearing span and drops the padding. It costs no model calls, it works for any reader, and F1 pays for it directly: the deployed 7B still loses 0.075 of precision to padding on its own.
  - Nothing published was changed: the scorecard still carries the hit rate it has always carried, and this tool is additive, per-run-set and per-category.
  - Tests: `tests/unit/test_locomo_scorer.py` (6), pinning the official normalisation, the category-3 truncation, the graded multi-hop score (one of two sub-answers = 0.5), and that every policy is always reported.
- **AM-decides layer, step 3d: the answer contract, measured and rejected** (`scripts/benchmarks/answer_contract.py`)
  - Six safe transformations (markdown, boilerplate prefixes, repeated sentences and 5-grams, trailing "According to...", category-3 polarity, date canonicalisation) applied **offline to stored predictions** - no model call - and scored with **Lethe's own matcher imported from the adapter** rather than a re-implementation (self-check: it agrees with the stored `is_correct` on 99.7% of the run), alongside the official F1. Every rule is measured separately, and the two numbers that decide shippability are printed together: how much F1 it recovers and **how many answers it breaks**.
  - **Result: it recovers nothing.** Deployed 7B - 44 of 1,540 answers change, hit rate and F1 both move 0.00pp. Verbose 1.5B - +0.01pp F1. Concise 1.5B - +0.02pp. The category-3 polarity rule breaks one answer rather than fixing one.
  - Why: the padding is not boilerplate. The 1.5B's 11.7 words are a sentence that *contains* the value, so there is no safe substring to delete. Recovering the F1 gap needs a mechanism that **selects** the value-bearing span - an extraction step that makes a choice, not a trim - which is a different mechanism with a different risk profile.
  - The lever that did work is prompt-side and free: `--concise-answer` recovered +5.8pp of the 12.8pp gap. Recorded so the trimmer is not built twice.
  - Tests: `tests/unit/test_answer_contract.py` (7), pinning that no rule can empty an answer and that each one fires only when something survives.
- **AM-decides layer, step 3e: the gold-context ceiling** (`scripts/benchmarks/gold_context_cache.py`, `gold_context_ceiling.py`)
  - The 378 unresolved questions are replayed through the frozen runner with **only the gold evidence turn(s) and nothing else** - 66 words per question against the production context's 1,590, i.e. the distractor budget is **24x the evidence**. Same prompt, same matcher, same verifier; the only change is the context. Both readers were run.
  - **The result inverts the roadmap.** Recovered: **15 questions (4.0%)** for both readers, **10 (2.6%)** for the 7B alone, **56 (14.8%)** for the 1.5B alone, and **297 (78.6%)** for neither. On the official F1 the *smaller* reader scores higher on this subset (12.5% vs 7.0%), so reader choice is not a monotone lever here - and temporal is the extreme case: **98 of 99** temporal questions fail even with the gold turn in front of the reader.
  - **The decisive check:** on **272 of those 297 (92%) the gold answer is not in the labelled evidence at all** (not verbatim, under 80% word overlap). A reader cannot be wrong about a fact that is not in front of it, so this cell is not a reasoning failure - it is questions whose answer has to be **derived** (temporal arithmetic, a join across turns) or whose LoCoMo evidence label under-specifies the evidence. So: retrieval and context construction have a ~4-7% ceiling on this bucket, a bigger reader buys ~2.6%, and the mass sits in a deterministic **derivation** layer.
  - Caveat recorded: the gold context is built from LoCoMo's *labelled* evidence ids, so the 92% mixes "needs derivation" with "label under-specifies". The follow-up that separates them is cheap and reuses this tooling: replay the same 378 with the evidence question's whole session as context.
  - Tests: `tests/unit/test_gold_context.py` (4), pinning that the gold context is the evidence and nothing else, that the experiment's input is exactly the census's unresolved set, and that the four cells are the four outcomes.

### Changed
- **Scorecard §8 "Failure ceiling"** - the census is now published, not just measured: one row per bucket with both denominators (share of slice, share of failures), the reader accounting, the gold-context outcome split, and a provenance table that keeps **both** oracle revisions on the page (legacy label oracle 211 / 1,540 retrieval failures; corrected content+date oracle 134 / 1,540, unresolved 378 / 1,540). The earlier figure is not deleted - the revision is stated instead, because a published number that quietly changed is worse than one that says it moved.
- **The published LoCoMo retrieval recall is now the corrected one: 80.6% → 81.1%** (accuracy unchanged at 65.2%). `collect_locomo1540` prefers the `.oraclefix.json` copy, which is the same run with the content-based flag; the original artefact is kept and still carries its old flag as `oracle_recall_by_id`.
- **The published LongMemEval score is now the frozen *default* configuration: 75.80% → 81.6%** (scorecard §2: that row's grade **C → B**; retrieval recall 99.2% → 98.2%, tokens/Q 6185 → 3653). The old artefact `grand_longmemeval_report_coder7b_apex_ctx8192.json` was **not a default run**: it was produced with a `--window-cap 48` A/B override, i.e. `selection_window_cap = 48` against the frozen default of 24. Re-verified by recompiling its questions offline at cap 48, which reproduces its token counts and oracle flags exactly (16280/9074/5120… and 9/9 oracle agreement on a fresh sample), while cap 24 reproduces today's arm; the dataset has not changed since 9/19 and no code change is responsible (reverting `msc_compiler.py` + `temporal_normalizer.py` to HEAD leaves today's numbers byte-identical). **The finding is that a *half-size* context is 5.8pp better for the 7B reader: 6.2k tokens/Q of retrieved context bought 1.0pp of oracle recall and cost 5.8pp of answers.** `_find_longmemeval_report()` now prefers the default run; the override artefact is kept, not overwritten, and stays cited by the stored §7 pairing.
- **§6's LongMemEval committer row is re-measured against that default run** (`commit_sweep.py --llm-run grand_longmemeval_report_7b_apex_ctx8192_postfix`, offline replay, no LLM calls): coverage 11.8% (59) and commit accuracy 93.2% are unchanged - code does not depend on the reader - but the paired reader on the same 59 questions drops 94.9% → 89.8%, so the commit delta moves **−0.20pp → +0.40pp** and the committer is now *ahead* of the reader on the questions it claims rather than at parity. The old baseline was part of the cap-48 artefact, i.e. the previous "at parity, not ahead" verdict was a property of the override, not of the runtime.

### Fixed
- **The answer committer was not deterministic across processes.** `_key_terms` ranked a `set` with a stable sort and `_anchor_position` walked its `matched` list in set order, so which of two equally-weighted terms won a tie depended on `PYTHONHASHSEED`: the *same* command scored 77/78/79/80 of the 321 LoCoMo temporal questions across four identical runs, and the new scorecard KPI inherited that noise. Both tie-breaks now break by name, and `tests/unit/test_answer_committer.py` pins the two mechanisms plus a four-hash-seed end-to-end run. Measure anything with a metric that moves and you are measuring the hash seed.
- **`FROZEN_MODEL` drift**: the constant, its comment, `benchmark_config/apex_config.yaml` and `tests/test_phase8_leakage.py` named three different readers (`qwen2.5:7b-instruct` / `qwen2.5-coder:7b` / `phi4-mini:latest`). The yaml now matches the code, and the test asserts the *invariants* plus yaml↔code equality so the next drift fails CI instead of silently changing published numbers.
- **Two unit tests encoded the pre-guard `AnswerVerifier` contract** (integrity override firing on an empty proposition list, which is how the LoCoMo path turned correct answers into abstentions). They now pass a proposition - the input the feature was designed for - and `test_integrity_override_is_skipped_without_propositions` pins the LoCoMo contract.
- **Optional-dependency tests now skip with a reason** instead of failing with `ModuleNotFoundError` from inside `embeddings.py`: `tests/test_phase8_arena.py` and the three `TestLeakageGuard` cases guard on the `vector` extra, which CI installs and a lean venv does not.
- Dead code and lint debt in the files this work touched: `other_ent` in `locomo_adapter.py`, an unused `TemporalResolver` import, unused test imports, E402 on the `sys.stdout.reconfigure` line. **Not** touched: the no-op "false refusal recovery" branch in `answer_verifier.py:328-340` (it computes `is_refusal` and then `pass`es) - it is harmless but still dead, and removing it is a separate change.

### Known issues (measured, not fixed here)
- `ruff check src/ tests/` still reports **375 pre-existing errors in `src/`** (337 auto-fixable), so the CI lint step cannot be green at HEAD. One-liner: `ruff check src/ tests/ --fix`, but that is a whole-repo reformat.
- **The ordering certificate drops the time of day.** `[Temporal Ordering: 'a' occurred on 2023-03-15. 'b' occurred on 2023-03-15. The event that happened first is 'b'.]` is a coin flip: on the real cache three same-date pairs ("Samsung Galaxy S22 vs Dell XPS 13", "tomatoes vs marigolds", "smart thermostat vs mesh network") all name the wrong winner while the reader is right. The resolver's verdict is *not* inverted - `scratch/_ordering_cert_audit.py` confirms the verdict matches the dates on 20/20 distinct-date cases - so the committer abstains on ties and self-checks the winner otherwise. The upstream fix is to carry the `HH:MM` that the LongMemEval headers already have into the certificate.
- The numbered `[Chronological Order of <X>: …]` certificates list turns that are not reliably the events the question asks about ("the six museums I visited" → entries 1-4 are turns about unrelated routines), so they cannot be transcribed and are left to the reader.

### Added
- **v0.2.0 Phase 8.12 - Abstention dataset repair (arena-2) + abstention semantics audit**
  - **Dataset defect (root cause of the Phase 8.9 abstention collapse)**:
    `_plot_abstention` planted its near-miss distractor on
    `PROJECTS[index + 7]`.  With 20 questions and 20 projects that shift is
    surjective over `PROJECTS`, so *every* queried project eventually received
    a debugger from some other question's plot -- 7 of 20 abstention questions
    had real evidence in the history (e.g. the `nectar` question planted
    "atlas ... set the debugger to tokei", directly contradicting S5's "never
    a debugger").  Phase 8.9's `gate_wide` did not fabricate "tokei"; it
    surfaced a planted assertion and the scorer booked it as
    `false_positive`
  - **Repair**: new `PROJECTS_UNQUERIED` pool (20 entities, disjoint from
    `PROJECTS`) supplies abstention distractors, making evidence-absence a
    structural guarantee.  `ARENA_VERSION` -> `v0.2.0-arena-2`, frozen hash
    regenerated (`eed68804...`)
  - **Invariant tests** (`tests/test_v02_phase7.py`): pool disjointness, plus
    a history grep asserting no turn says "for project <P>, the user set the
    debugger to" for any abstention project (pre-repair 7 offenders ->
    post-repair 0)
  - **`--abstention-audit`** (`scripts/deepseek_ab_small.py`): Part 1 counts
    asserting vs denying memories per abstention question (post-repair:
    `abstention_evidenced=0`, `structurally_unwinnable=true`); Part 2 sweeps
    `score_floor` for the dose-response
  - **Floor dose-response** (20 abstention + 9 GT questions, pool=100,
    window=450): floor 0.0 -> ctx 83.5 / GT 0.5128; floor 2.0 -> 78.0 /
    **0.5385** (retention-neutral-to-positive, the knee); floor 3.0 -> 37.7 /
    0.4872; floor 4.0 -> 2.2 / 0.3333; floor 6.0 -> 0.1 (95% empty) / 0.2564;
    floor 8.0 -> 0.0 (100% empty) / 0.1538
  - **Finding**: a query-independent *lexical* floor cannot implement
    abstention -- abstention and evidence questions share the same lexical
    profile, so emptying the context costs 85% of GT retention.  Abstention
    must come from evidence *presence* (denial memories), not evidence
    *strength*.  `score_floor` is retained as a cost lever only (default 0.0)
  - Evidence: `abstention_audit_20260919_170618.json`,
    [ADR-0003](docs/adr/0003-abstention-evidence-absence.md)

### Added
- **v0.2.0 Phase 8.11 - Score-floor abstention lever + three-layer forensic audit**
  - `SessionGate.score_floor`: memories whose own BM25 is below the floor are
    excluded individually (not backfilled), so the gate may return *fewer*
    than pool_size memories; an empty context is the correct answer to a
    question the store cannot evidence
  - Dose-response (LLM-free, 20 abstention + 9 GT questions, pool=100):
    floor=0.0 abstention_ctx=84.8 / gt_retention=0.513; floor=1-2 78.0 /
    0.539 (retention unchanged); floor=3 38.1 / 0.487; floor=8 0.0 (100%
    empty) / 0.128.  Usable window: floor in [1, 2] trims filler with zero
    retention cost
  - **Fix (silent no-op #1)**: `SessionGate.filter` returned early when
    `len(memories) <= pool_size`, which disabled the floor entirely whenever
    the candidate window was small (measured: floor=8 produced
    `floor_drops=0`, 60 consecutive validation runs were evidence-free
    no-ops).  Floor evaluation now runs BEFORE the size short-circuit
  - **Fix (silent no-op #2)**: swapping `engine.gate` did not invalidate the
    CSA2 plan cache -- plans seeded under floor=0.0 kept replaying under
    floor=8.0 (`DeepSeekRecallEngine.recall` now flushes the plan cache when
    the gate type/pool/max_sessions/score_floor signature changes)
  - **Fix (validation harness)**: a per-floor `SessionGate` built without
    `corpus_provider` computes IDF over the *pool*, not the store -- a
    different scorer from the shipped one; harnesses must wire
    `corpus_provider=runtime._store_snapshot` to measure production
    behaviour
  - **Dataset finding (misattribution)**: the S4 abstention questions are
    *not* unevidenced -- S6 contains "For project <P>, the user set the
    debugger to <X>" for all 20 projects, directly contradicting S5's
    "only discussed scheduling and never a debugger".  The Phase 8.9
    abstention accuracy collapse (0.95 -> 0.00) was therefore partly
    misattributed: gate_wide surfaced the S6 answer and the scorer marked
    answering as `false_positive`.  The floor lever is mechanically
    correct (dose-response above) but the abstention dataset must be
    repaired before it can validate the fabrication hypothesis
  - Evidence: `floor_validation_20260919_153600.json` (valid run; the 60
    earlier no-op evidence files were deleted, see comment in
    `scripts/deepseek_ab_small.py` history)
### Added
- **v0.2.0 Phase 8.10 - Session-granular cheap-recall cost curve**
  - `SessionGate.max_sessions`: top-K whole sessions retained, pool-size cap
    removed (sessions kept whole; deterministic tie-break on first-index)
  - LLM-free cost audit: 95.2% cost reduction (pool=10) / 43.7% retention at
    pool=100 / 43.7% retention, 1394 mean tokens at pool=60 (max_sessions=1
    gives 95.6% of the accuracy at 4.6% of tokens)
  - `scripts/deepseek_ab_small.py`: `--cost-audit` (pool x K grid),
    `--gate-audit` (funnel), `--window-sweep`
  - Finding: the pool=60 "free win" was **refuted** under stratified
    sampling -- the winning run's question set was easy-category only
    (`factual_recall` + `multi_session_recall`); on a stratified 40-question
    set pool=60 scores 0.50 / 0.613 (accuracy / GT coverage) vs 0.55 / 0.725
    at pool=100. The `max_sessions` lever is inactive at pool=100 (the
    pool_size cap binds first); real leverage is at pool <= 60 OR K=1
  - Evidence: `benchmark/results/deepseek_ab/session_cost_audit_*.json`,
    `production_20260919_093444.json` (pool=60), `production_20260919_095444.json`
    (pool=100), `production_20260919_095745.json` (both, same stratified set)
- **v0.2.0 Phase 8.9 - Production A/B validation (3,000-call real-LLM run)**
  - `scripts/deepseek_ab_production.py`: chunked three-arm production
    experiment -- 200 questions x 5 repeats x 3 arms (`classic` / `cache` /
    `session`) = 3,000 answer calls in 30 crash-isolated chunks (evidence
    flushed per chunk, `failures=0` across all 30)
  - Medium-scale confirmation (40 questions, 600 calls): session-gate
    accuracy 1.0 vs classic 0.0, GT coverage 1.0, repeat stability 1.0,
    plan reuse 960/1000 (96%); cost: p50 93 ms vs 32 ms, tokens 1,991 vs
    511 (the correctness-for-cost trade-off that motivates Phase 8.10)
  - Budget audit: every one of the 1,800 answer contexts <= 2,000 tokens
    (0 violations); `violation=None` clarified as "key absent from
    summary", re-verified from raw traces
  - Window/scorer sweep evidence retained under
    `benchmark/results/deepseek_ab/` (see ADR-0002 for the full chain)
- **v0.2.0 Phase 8.8 - Session-granular gate: runtime wiring + real-LLM validation**
  - `SessionGate` in `recall/retrieval_cache.py`: session-level BM25-style
    scoring where a session's *maximum* evidence score is assigned to every
    memory in it -- session membership is the recall-bearing structure, not
    per-memory lexical similarity
  - `RuntimeConfig.retrieval_strategy="deepseek_session"`: session gate +
    widened candidate window wired into the runtime (opt-in, additive);
    `DeepSeekRecallEngine` gained a configurable candidate window
    (`candidate_window`, default = frozen recent-50)
  - 10-question real-LLM validation: accuracy 0.0 -> 0.6, GT coverage
    0.1 -> 0.75, repeat stability -> 1.0 at 440 -> 100 candidate reduction
- **v0.2.0 Phase 8.6/8.7 - Bottleneck forensics + cheap-scorer sweep**
  - GT-funnel audit (`--gate-audit`): store -> candidates -> gate ->
    selected attribution; finding: the frozen candidate window (recent-50
    of 440 memories) loses 89% of ground-truth evidence *before* the gate
    runs -- the original Jaccard gate was innocent (fired 0 times)
  - `--window-sweep`: candidate-window decision table (50/100/250/450)
  - Cheap-scorer sweep (Jaccard / BM25 / embedding / hybrid / session):
    memory-level scorers keep destroying 62-86% of evidence at useful pool
    sizes; only session-granular scoring achieves non-destructive reduction
    (the "right unit of retrieval was not a Memory" finding, ADR-0002)

### Fixed
- `memory/vector_search.py`: bounded retry (50-400 ms exponential backoff)
  around the atomic `os.replace` in `_save_index`; on Windows, antivirus
  scanners briefly hold a shared-read lock on freshly written
  `.tmp`/FAISS index files, making `os.replace` fail with WinError 5 even
  though permissions are correct.  This flaked 5 `test_phase8_arena`
  tests intermittently (all passed in isolation); with the retry the full
  suite is stable: 435 passed, 11 skipped, 0 failed.  No semantic change.

- **v0.2.0 Phase 8.4 - DeepSeek Raid (DeepSeek V4.1-inspired retrieval acceleration)**
  - `recall/retrieval_cache.py`: three DeepSeek V4.1 systems patterns
    translated to memory-runtime mechanisms:
    - `RetrievalPlanCache` (CSA2 analogue): FULL / REINDEX / REUSE
      candidate-set reuse for repeated query families, with deterministic
      punctuation-stripping signatures, insertion-order eviction and TTL
    - `HierarchicalGate` (HSI analogue): cheap lexical Jaccard pre-filter
      narrowing the candidate pool before semantic ranking; order-stable
      tie-breaks keep recorded runs replayable
    - `EphemeralStore` (SWA Bounded Replay analogue): bounded session
      scratch state that is discarded, never persisted, and rebuilt from
      the source log through a replay closure
  - `recall/deepseek_engine.py`: `DeepSeekRecallEngine` composing the plan
    cache + gate on top of the frozen `BasicRecallEngine` (subclass, never
    in-place modification)
  - `RuntimeConfig.retrieval_strategy`: opt-in switch (`"classic"` default =
    frozen behavior; `"deepseek"` wires the raid engine); S0-S3 baseline
    players and the frozen Scorer are untouched
  - `tests/test_deepseek_raid.py`: 23 tests covering tiering, gating,
    replay, composition and runtime wiring (full suite: 423 passed,
    11 skipped, zero regressions)
- **v0.2.0 Phase 8.5 - DeepSeek Raid A/B experiment harness**
  - `scripts/deepseek_ab_small.py`: three-arm controlled E2E over the
    10-question Phase 8.3.4 subset -- `classic` (frozen baseline) vs
    `cache` (CSA2 plan cache) vs `gate` (plan cache + HierarchicalGate);
    EphemeralStore stays out of every arm so deltas attribute cleanly
  - `AMv020Player`: `retrieval_strategy` config passthrough (default
    `"classic"` keeps frozen behaviour) and `memory_ids` in Answer.metadata
    for selection-level determinism evidence
  - Evidence per arm: frozen-Scorer aggregates, retrieval latency
    p50/p95, context tokens, FULL/REINDEX/REUSE counters, gate traffic,
    repeat-selection stability, cross-arm selection agreement
  - Dry-run validation findings: REUSE freezes selection across repeats
    (stability 1.0) where classic re-ranking drifts (0.0); gate is a no-op
    at pool_size=64 with 20-candidate pools (use a smaller pool to arm it)
  - First real-LLM run (phi4-mini, frozen config, gate-pool=10):
    - quality identical across arms (answer_accuracy 0.1 = abstention only;
      phi4-mini fails all knowledge questions under the 2,000 budget, so
      raid mechanisms are quality-neutral here -- no recall lost to REUSE)
    - cache arm: retrieval p50 32.1 ms vs classic 42.9 ms (~25% faster);
      plan tiering full=10 / reuse=240 (96% reuse)
    - gate arm: pool narrowed 200 -> 100 candidates, context tokens
      522 -> 262.6 (-50%), retrieval p50 25.8 ms (fastest); selection
      agreement vs classic drops to 0.0 (gate changes candidate pools by
      design), repeat-selection stability stays 1.0
    - cross-arm agreement confirms determinism: cache replays stable
      selections, classic re-ranks drift (stability 0.05)
    - evidence: benchmark/results/deepseek_ab/summary_20260918_161229.json

### Added
- **v0.2.0 Phase 1 - Memory Evolution Core**
  - Runtime memory state vector (`MemoryStateVector`) with `importance` /
    `confidence` / `currentness` / `future_utility` / `contradiction_risk`
    and the deterministic `StateSignalCalculator` behind it
    (`state_signals.py`)
  - New first-class lifecycle operations in `MemoryEvolutionEngine`:
    `REINFORCE` (usage/evidence raises confidence), `REINTERPRET` (new
    higher-order interpretation keeping the original evidence), `RESTORE`
    (rebuild the richest available resolution from stored versions),
    `KEEP` and `REJECT` (explicit retention decisions; reject means DORMANT,
    never deletion)
  - `evolution_policy.py`: proposal -> validation -> commit -> audit split.
    `EvolutionProposal` is validated by `EvolutionGovernor` (9 deterministic
    rules) before any mutation; AI-proposed mutations are proposals, not
    truth (plan #26)
- **v0.2.0 Phase 2 - Temporal Management & Contradiction Handling**
  - `temporal_validity.py`: explicit validity windows
    (`valid_from` / `valid_until`) with half-open interval semantics,
    monotonic invalidation and `apply_supersession_bounds` (plan #21)
  - `supersession.py`: `SupersessionManager` with explicit
    `SUPERSEDES` edges, bidirectional chain walking and current-head
    resolution; history is preserved, never overwritten
  - `contradiction_edges.py`: explicit `CONTRADICTS` edges with severity,
    audit events on both sides, and temporal resolution of genuine state
    changes (`apply_temporal_resolution`)
  - `lifecycle_transitions.py`: validated `MemoryStatus` state machine
    (`ALLOWED_TRANSITIONS`), `InvalidTransitionError`, and transition
    history read back from the audit log
  - `provenance.py`: evidence chain walking (Semantic -> Episode ->
    Conversation -> Message) and `explain()` combining validity,
    provenance, supersession, contradictions and audit history (plan #22)
  - Governor dispatch for `CONTRADICT` / `SUPERSEDE` / `TRANSITION`;
    Phase 1's `CONTRADICT` routing placeholder is now a real dispatch
- Tests: `tests/test_v02_phase1.py` (34 tests) and
  `tests/test_v02_phase2.py` (35 tests)
- **v0.2.0 Phase 3 - Reconstruction Layer**
  - `graph.py`: deterministic memory-graph traversal (plan #15 / #16).
    Breadth-first expansion with visited-set cycle guards, total ordering
    (strength, association id, memory id) and path/seed/strength bookkeeping;
    supporting relations (RELATED / ELABORATES / SUMMARIZES / CAUSES /
    FOLLOWS) are expandable while CONTRADICTS / SUPERSEDES are never
    implicitly expanded
  - `evidence.py`: LLM-free evidence ranking (plan #17) combining lexical
    query coverage (Latin words + CJK bigrams), graph proximity, the Phase 1
    state vector and temporal validity, with contradiction and non-current
    penalties; every item carries human-readable inclusion reasons;
    `keep_ids` pins caller-supplied seed chains into the package
  - `reconstruction.py`: the full pipeline "candidates -> graph expansion ->
    provenance expansion -> temporal filter -> ranking -> contradiction
    checks -> multi-memory synthesis -> evidence package" (plan #15),
    producing an auditable `ReconstructedEvidencePackage` with chains,
    conflicts (resolved vs unresolved), provenance traces and deterministic
    text synthesis; `reconstruct_multi_hop()` recovers known A -> B -> C
    chains
  - Fixed a SQL operator-precedence bug in
    `SQLiteMemoryStore.get_associations` / `PostgresMemoryStore`:
    the `association_type` filter silently leaked to only one side of the
    `OR`, making type-filtered queries return unrelated associations
- Tests: `tests/test_v02_phase3.py` (53 tests)
- **v0.2.0 Phase 4 - Context Allocator**
  - `context/allocator.py`: the context window as a constrained resource
    (plan #18). Every candidate receives one of five representations
    (`FULL / COMPRESSED / SUMMARY / TAG / OMIT`) chosen by priority, so the
    allocator optimizes *useful information per token* rather than raw
    context volume
  - Priority covers every plan #18 factor: query relevance, importance,
    confidence, temporal relevance, contradiction risk and resolution; token
    cost enters through the representation choice
  - Greedy budget allocation with a strict downgrade path (FULL ->
    COMPRESSED -> SUMMARY -> TAG); COMPRESSED / SUMMARY texts come from
    stored `MemoryVersion`s (never generated on the fly) and parts always
    report the *effective* level after degradation; TAGs are synthesized
    deterministically from the memory's own tokens plus its id
  - Every OMIT carries an auditable reason (below relevance floor /
    superseded / budget exhausted); `ContextAllocation` reports token usage,
    the representation distribution and utility-per-token stats
  - `allocate_from_reconstruction()` consumes a Phase 3 evidence package
    directly, closing the retrieval -> reconstruction -> allocation pipeline
  - Fixed an import cycle (`context` -> `memory` -> `compiler` ->
    `context`) by importing the memory modules lazily inside the allocator
- Tests: `tests/test_v02_phase4.py` (28 tests)
- **v0.2.0 Phase 5 - Background Reflection**
  - `memory/reflection.py`: memory management outside the critical response
    path (plan #13 / #14). The `BackgroundReflector` runs the full pipeline —
    candidate selection -> evidence retrieval -> review / state assessment ->
    proposed mutation -> policy validation -> commit -> audit
  - Seven triggers (`idle` / `session_end` / `low_load` / `memory_pressure`
    / `contradiction_detected` / `scheduled` / `unresolved_clusters`); review
    is bounded (`max_candidates`), prioritizing frequently-used, stale,
    low-utility and low-confidence memories
  - Five reflection workers, each producing an `EvolutionProposal` — never a
    direct mutation (plan #26): `REINFORCE` (frequently accessed + valuable),
    `MERGE` (duplicate clusters via lexical overlap), `REINTERPRET`
    (deterministic synthesis from related evidence), `ARCHIVE` (`TRANSITION`
    to archived — never deletion), `KEEP` (explicit retention for rarely
    accessed high-value memories)
  - All proposals flow through the Phase 1 `EvolutionGovernor` (the exact
    reason the gate was built first); commits write auditable evolution
    events with `triggered_by="reflection"`; `dry_run` mode reviews without
    committing; `ReflectionReport` is fully serializable
  - LLM reviewer hook: the heuristic proposer can later be replaced by an
    LLM submitting through the same proposal interface and validation gate
- Tests: `tests/test_v02_phase5.py` (16 tests)
- **v0.2.0 Phase 6 - Predictive Recall**
  - `recall/predictive.py`: `TopicPredictor` ranks the topics a conversation is
    likely to move to next from three deterministic, LLM-free signals
    (association links into the topic, topic recency, query/topic term
    overlap); a zero-total case predicts nothing rather than guessing
  - `PrefetchCache` + `PredictiveRecallEngine.prefetch()` warm the likely
    topics into a TTL- and size-bounded cache; `recall()` serves them only
    when the conversation actually moves there (cache hit -> `reused`,
    otherwise a normal recall) — deferred context injection, never eager
    injection of speculative context
  - Prefetched memories are chosen by the Phase 1 `StateSignalCalculator`
    (`future_utility`), so prediction cost and memory value share one model
  - `PrefetchResult` / `stats()` report fetched, reused, hit rate and
    evictions, keeping the cost/benefit of prediction measurable
  - Fixed an import cycle introduced by the new module (recall ->
    `memory.__init__` -> `memory.compiler` -> `compiler` -> `context` ->
    recall) by importing `memory.evidence` / `memory.state_signals` lazily,
    following the Phase 4 `allocator.py` cycle-guard convention
- Tests: `tests/test_v02_phase6.py` (19 tests)
- **v0.2.0 Phase 7 - Benchmark Infrastructure (Track B Arena)**
  - `research/benchmarks/arena.py`: deterministic, LLM-free Track B dataset —
    one shared 200-question conversation history (10 categories x 20
    questions) planted into 8+ session buckets, with question-level ground
    truth expressed as term groups ("must mention X and Y, not Z") so scoring
    needs no judge model (plan #33 / #34)
  - Categories map 1:1 to the capability axes: factual recall,
    multi-session recall, temporal reasoning, knowledge updates,
    contradiction handling, indirect recall, distractor resistance,
    abstention, compression recovery, reflection/reinterpretation
  - Session buckets are data-derived (`day // SESSION_INTERVAL_DAYS`): one
    scenario per bucket with unique ids, turns merged and day-ordered;
    cross-session follow-ups (`+15` / `+30`) may open trailing sessions
    beyond the base 8-session grid
  - `write_dataset` / `read_frozen_hash` / `verify_freeze`: hash freeze so
    results are only comparable against an unchanged dataset (Fairness Rule
    8); the frozen dataset is committed at `benchmark/dataset/arena.json`
    with its sha256 (`84dd96f224ea6fe8...`)
  - Dataset integrity fixes during implementation: early-session facts are
    clamped to day >= 0 (no `session--1` bucket), per-day scenarios no
    longer collide on one bucket id, and `required_sessions` is derived
    from the real distinct evidence sessions instead of a hardcoded claim
- Tests: `tests/test_v02_phase7.py` (28 tests)
- **v0.2.0 Phase 8.1 - Arena Core** (`Benchmark_Plan.txt` Rev.2 spec freeze)
  - `research/benchmarks/player.py`: `Player` protocol (`ingest` / `answer` /
    `finalize` / `reset_costs`), `Answer` (full last-assistant-message text per
    the Answer Extraction Protocol), and `CostReport` covering the mandatory
    write-side cost (write LLM calls / tokens / latency) plus read-side
    retrieval, context construction, answer tokens and total cost —
    Total Cost = Write LLM + Retrieval + Context Construction + Answer LLM
  - `ControlledPlayer` (fixed 2,000-token Official Context Budget, budget
    violations recorded) vs `RealSystemPlayer` (native context management,
    actual tokens measured) — the two arenas are never mixed
  - `research/benchmarks/runner.py`: `ArenaRunner` with the §9 Repeat
    Protocol (3 answer repeats with majority vote, >= 5 latency samples with
    p50/p95), per-run output tree (manifest.json, metrics/latency/tokens CSV,
    failures.jsonl, traces, environment.txt) and `create_arena_manifest`
    reusing `research/experiments/manifest.py`
  - `research/benchmarks/players.py`: skeleton of the five Controlled
    players (Simple RAG / MemoryBank-style / MemGPT-style / Mem0-style /
    AM v0.2.0) with real ingest (embedding + store) and deterministic mock
    answer generation until the fixed LLM is wired in Phase 8.2
  - Robustness fixes found while integrating: `VectorSearchEngine` now
    tolerates a corrupt/empty `metadata.json` (rebuilds instead of crashing)
    and writes its index atomically; Controlled players get isolated
    ephemeral FAISS index dirs (the shared repo-level `vector_index/`
    previously bled memories across players and runs); `AMv020Player` was
    calling the runtime facade's async `remember` without awaiting it (a
    silent no-op — AM ingested nothing) and non-existent
    `recall_adaptive`/`build_context` methods; it now drives
    `asyncio.run(remember(...))` and uses `recall()` + `RecallResult`
    - Tests: `tests/test_phase8_arena.py` (31 tests; budget enforcement,
    repeat protocol, failure recording, real-player native behavior,
    non-LLM write cost semantics, Mem0 write-side token accounting,
    index dir isolation + cleanup)
    - Tests: `tests/test_phase8_leakage.py` (9 tests; Ground Truth Sentinel
    never reaches the LLM prompt, answer API accepts text+context only,
    frozen config / shared instance / config hash stability assertions)
  - Tests: `tests/test_phase8_scorer.py` (27 tests; §10 synthetic case table,
    normalization, repeat-majority + tie-break determinism, pure-function
    invariance, aggregate math, run-replay determinism + hash-mismatch guard,
    scores.csv header validation)
  - Live smoke: `tests/smoke_llm_answer.py` (phi4-mini real run,
    `f5317d523d06a65d...` config hash)
- **v0.2.0 Phase 8.2 - Fixed LLM / Leakage Boundary / Evaluation Boundary**
  - `research/benchmarks/llm.py`: single shared `OllamaAnswerer` for all five
    Controlled players with a frozen configuration (model, temperature 0.0,
    seed 42, prompts, max output tokens) hashed into
    `config_sha256` / `prompt_sha256` for the run manifest
  - Generation signature is `answer(question_text, context)` only, so ground
    truth / forbidden terms / abstention flags cannot reach the prompt by
    design (Leakage Boundary); extraction is a second frozen entry point
    `extract(exchange_text)` for Mem0-style write-side strategies
  - Frozen model changed to `phi4-mini:latest` before the freeze: the original
    candidate `qwen3:4b` ignores both `think: false` and `/no_think` on this
    Ollama install and emitted 300-500 reasoning tokens into `content`
    (~26 s/call), which would have poisoned forbidden-term scoring; recorded in
    `Benchmark_Plan.txt` §5 as a pre-freeze LLM configuration decision
  - Mock answer generation removed: all five players now retrieve real
    context and answer through the shared frozen LLM (`_generate_answer`);
    deterministic abstention is used only when no answerer is injected
    (unit-test mode), so a run cannot silently produce ground-truth-flavoured
    answers
  - Mem0-style player now implements its observed OSS core: single-pass
    ADD-only LLM extraction on write plus semantic + BM25-style keyword +
    entity multi-signal fusion retrieval (0.5 / 0.3 / 0.2)
  - Write-side cost truthfulness fixes: `CostReport.add_write_cost()` now
    counts LLM calls only (`calls=0` for embedding-only ingest and
    deterministic consolidation/paging, latency still recorded), and Mem0-style
    no longer double counts per-turn extraction latency inside E2E write
    latency — so `write_llm_calls` / `write tokens` actually separate
    LLM-write strategies from non-LLM ones
  - Leakage Guard tests (`tests/test_phase8_leakage.py`, 9 tests): a
    `GROUND_TRUTH_SENTINEL_9f83a` planted into dataset ground truth never
    appears in any recorded prompt, `answer` accepts only text + context, and
    the frozen config values / hash stability / shared instance are asserted
  - Arena tests grew to 31 (write-cost semantics regression guards)
  - Ephemeral player index dirs are now tracked in-process and removed at
    interpreter exit (`atexit`), so repeated benchmark runs no longer leave one
    FAISS index per player ingest behind in the system temp dir; a regression
    test asserts per-ingest isolation and removal
  - `tests/smoke_llm_answer.py`: live smoke of the frozen LLM
    (real Ollama answer + one end-to-end player question, manually run)
- **v0.2.0 Phase 8.3 - Scoring & E2E Validation (8.3.1-8.3.4)**
  - `research/benchmarks/scorer.py`: judge-free Scorer per the frozen §10 spec
    (SCORING_SPEC_VERSION `v0.2.0-scoring-1`): NFKC+lowercase+whitespace
    normalization (no punctuation stripping so `f#` survives), word-boundary
    term matching, synonym term groups, forbidden-term zeroing, abstention
    lexicon + frozen `ABSTENTION_TEXT`, outcome table
    (pass/partial/fail/false_positive), blank answers stay out of the
    false-positive bucket
  - Scorer is a pure post-step, fully separated from the runner:
    `score_run_dir()` replays a persisted run (`traces.jsonl` + frozen
    dataset) into `scores.csv` / `aggregates.csv` / `aggregates.json`,
    rejects a dataset-hash mismatch, and can re-score old traces with an
    improved scorer without invalidating the run
  - Official repeat reduction per §10: majority answer by exact text with the
    shared deterministic `majority_index` tie-break; all repeat scores kept
    for `score_consistency`
  - §10 aggregates: answer_accuracy, partial/false_positive/forbidden rates,
    mean coverage, abstention_accuracy, per-category accuracy
  - Scorer unit tests (`tests/test_phase8_scorer.py`, 27): the full §10
    synthetic case table (case/compound/boundary/synonym/symbol), repeat
    reduction determinism, aggregate math, replay determinism, hash-mismatch
    rejection
  - `tests/smoke_e2e_arena.py` (8.3.4): 10 questions (one per category) x
    Simple RAG x 1 repeat with the shared frozen answerer, then replay scoring;
    the §12 output tree (manifest/metrics/latency/tokens/traces/scores) is
    asserted complete and `failures.jsonl` now always exists (empty when clean)
  - Critical retrieval bug found by the E2E (regression test added): all three
    vector-index baselines queried `BasicRecallEngine`, which never touches
    the vector index (insertion-order candidates + word-overlap ranking) —
    the dataset's target fact ranked #1 (0.808 cosine) in the index the
    players had built, yet was never returned, so Simple RAG scored 0.1/10
    before the fix and 6.5/10 after. Baselines now retrieve through
    `VectorSearchEngine` directly (Simple RAG / MemoryBank: dense top-20;
    MemGPT: main-context turns first, then dense archival hits; Mem0 keeps
    its own multi-signal fusion), and baseline context is built by a plain
    budget-capped join instead of the AM `ContextAllocator` (SUT machinery
    must not re-rank a baseline's retrieved lines)
  - E2E observation recorded for 8.3.5+ (not a harness bug): Simple RAG
    asserts concrete values on abstention questions (false_positive) and
    misses cross-session multi-hop evidence within top-20 dense retrieval

## [0.1.0] - 2026-09-15

### Added
- Initial release of Artificial Memory / Context Runtime
- Core memory system with 6 resolution levels (RAW to DEEP_LONG_TERM)
- Progressive memory compression (Forget = Resolution Down)
- Adaptive-resolution recall with provenance tracking
- Topic-based organization with automatic classification
- Memory IR / Context IR formal definitions
- Vector search with FAISS and pgvector backends (`vector` extra)
- Shared SentenceTransformer model cache and lazy module exports (import time reduced from 19s to 0.3s)
- Temporal queries, belief management, and contradiction detection
- Memory integrity checking and self-healing
- Federated multi-agent memory exchange and enterprise governance
- MCP interface for AI agents with 7 verified tools and real-agent cross-session E2E support
- CLI, WebSocket, and REST API
- Reproducible benchmark harness (deterministic recall, Ollama LLM QA, MCP smoke test)
- Docker, docker-compose, and Kubernetes Operator with Helm chart

### Changed
- Pinned `psycopg[binary]` and `mcp>=1.0,<2.0` for stability
- CI enhanced with matrix testing, vector/mcp extras, and non-blocking mypy strict adoption

### Fixed
- Fixed MCP tool response nested Pydantic `MemoryIR` JSON serialization
- Fixed `memory_inspect` event loop deadlock in MCP tools
- Fixed PostgreSQL test timeout with fast connectivity check and graceful skip
- Resolved all Ruff linting issues and majority of type annotations

### Infrastructure
- Docker and docker-compose support
- GitHub Actions CI/CD pipeline
- Pre-commit hooks configuration
- Comprehensive test suite (113 tests)
- Type checking with mypy (strict mode)
- Linting with ruff

---

## Release Process

1. Update version in `pyproject.toml`
2. Update this CHANGELOG.md
3. Create a git tag: `git tag v<version>`
4. Push tag: `git push origin v<version>`
5. GitHub Actions will build and publish release

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

## License

MIT License - see [LICENSE](LICENSE) for details.