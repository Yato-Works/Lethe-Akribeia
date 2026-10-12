# LoCoMo CLEAN BASELINE — 46.32% (first uncontaminated measurement)

Run: `locomo_full_20261011_195837.json`, 1986 questions, ~90.7 min.
Conditions, all verified clean before the run:

- reader prompt de-contaminated (`fa00a9c` — 20 hardcoded gold answers removed)
- current scorer
- anti-cheat guard VERDICT: CLEAN (`04102cd`)
- LoCoMo evaluation path audited against all 1,427 gold answers: 0 hits
  (`52a4734`)

**This is the first LoCoMo number produced with no contamination and no stale
scorer. Treat it as the baseline.**

## Score

| category | clean | contaminated | delta |
|---|---|---|---|
| cat1 single-hop | 22.0% (62/282) | 21.3% (60/282) | +0.7pt |
| cat2 multi-hop | **68.2% (219/321)** | 37.7% (121/321) | **+30.5pt** |
| cat3 temporal | 30.2% (29/96) | 25.0% (24/96) | +5.2pt |
| cat4 open-domain | 59.2% (498/841) | 67.1% (564/841) | **-7.8pt** |
| cat5 adversarial | **25.1% (112/446)** | 53.8% (240/446) | **-28.7pt** |
| **TOTAL** | **46.32% (920/1986)** | 50.81% (1009/1986) | -4.49pt |

"Contaminated" = `locomo_full_20261010_151930.json`, same scorer, same
conditions, reader prompt still carrying 20 gold answers.

## The per-category pattern is the finding

The headline barely moved (-4.49pt) but the *shape* changed violently, and the
shape is the evidence for what the contamination was doing:

- **cat5 adversarial -28.7pt.** cat5 answers are "this information does not
  exist, say you don't know". 59.2% of clean cat5 predictions are
  `I don't know.` (264/446) — which is *correct behaviour* for the category.
  The contaminated prompt instead told the model to answer, so it produced
  fluent hallucinations and scored them as correct.
- **cat2 multi-hop +30.5pt.** the reverse: removing the prompt's answer hints
  forced real evidence chaining, which the pipeline is actually good at.

So the contamination was not uniformly inflating. It was *distorting the
category profile* — buying cat5 points with hallucination while suppressing
cat2 points that the pipeline earns legitimately.

## Direct evidence (147 questions)

147 cat5 questions are FAIL under clean and OK under contaminated. The
predictions show the mechanism outright:

    Q:     What was grandma's gift to Melanie?
    GT:      necklace
    CLEAN:   I don't know.
    CONTAM:  a necklace

    Q:     Why did Melanie choose the adoption agency?
    GT:      because of their inclusivity and support for LGBTQ+ folks
    CLEAN:   I don't know.
    CONTAM:  because they help LGBTQ+ folks with adoption.

The clean run abstains because the evidence is absent. The contaminated run
answers correctly because the answer was in the prompt. That is not
capability, that is leakage.

## What the baseline says about real weaknesses

Ranked by correctable volume, using the CLEAN numbers:

1. **cat1 22.0% (220 failures).** Known list-gold enumeration problem; ~40% of
   all questions have list golds. Largest single addressable block.
2. **cat4 59.2% (343 failures).** Open-domain; largest category by count.
3. **cat5 25.1% (334 failures).** Two sub-populations: legitimate abstentions
   the scorer may be penalising, and genuine retrieval misses. **Must be split
   before attacking** — 264 of 446 are already `I don't know.`, so the
   addressable subset is much smaller than 334.
4. **cat2 68.2% (102 failures)** and **cat3 30.2% (67 failures)** — smaller,
   and cat3 is now above its contaminated value.

## Do not compare against

- 50.81% / 56.19% — both produced under contamination and/or a stale scorer.
- Any LongMemEval dev/holdout number quoted without its scorer revision.

---

## End-to-end result of the second-pass recall (measured, 1,986 questions)

Full run with `--second-pass` (`4cf32dd`), artifact
`locomo_full_20261011_230313.json`, vs this baseline
`locomo_full_20261011_195837.json`:

| category | baseline | second-pass | delta |
|---|---|---|---|
| cat1 single-hop | 22.0% (62) | 22.3% (63) | +0.4pt |
| cat2 multi-hop | 68.2% (219) | 68.8% (221) | +0.6pt |
| cat3 temporal | 30.2% (29) | 30.2% (29) | ±0 |
| cat4 open-domain | 59.2% (498) | 59.1% (497) | -0.1pt |
| cat5 adversarial | 25.1% (112) | 26.2% (117) | +1.1pt |
| **TOTAL** | **46.32% (920)** | **46.68% (927)** | **+0.35pt** |

**Conversion efficiency: evidence retention +1.59pt produced only +0.35pt of
score — 22%. 78% of the evidence gain was wasted.**

Per-question accounting:
- 38 questions flipped to correct
- 31 questions flipped to wrong
- net +7

Character of the flips:
- the 31 regressions: 19 of 31 were only correct via scorer tolerance (neither
  exact nor substring matches)
- the 38 gains: 36 of 38 were only correct via scorer tolerance

Both sides of the ledger are dominated by tolerant partial matches, so the net
+7 sits inside the scorer's noise band. +0.35pt is not a claim of improvement.

### Dilution hypothesis — TESTED AND REJECTED

The obvious explanation for 38 gains / 31 regressions is that 60 extra turns
dilute signal density so the knapsack drops turns it previously kept.
Measured on cat1 (n=281), off vs on:

| | off | on |
|---|---|---|
| avg context lines | 33.1 | 41.8 |
| avg subject lines | 26.1 | 35.2 |
| evidence retained | 45.9% | 50.9% |
| **signal density (subject/context)** | **0.79** | **0.84** |

Density went UP, not down. The added turns were the queried subject's own
turns, which is exactly what an enumeration question wants. Dilution is
therefore not the mechanism.

**Honest position:** the second pass genuinely improves evidence retention, and
the reader does not reliably convert that into correct answers. This is the same
wall measured earlier — 100% evidence retention still produced only 40%
accuracy. The bottleneck is now unambiguously the READER, not retrieval and not
the compiler.

---

## Strict grading audit — the verdict on the second-pass recall

Tool: `artificial_memory/compiler/eval/audit_strict_grade.py`, rubric pinned by
`tests/unit/test_audit_strict_grade.py`.  Arms graded by one fixed rule set;
nothing about the treatment leaks into grading.

Rubric: STRICT (exact) / NEAR (containment, list golds excluded) / PARTIAL
(token overlap >=0.5 or fuzzy >=0.85, which is what the production scorer
accepts) / WRONG.  A PARTIAL credit is deliberately NOT counted as an
improvement.

| | baseline | second-pass |
|---|---|---|
| strict | 20.8% (414) | 21.3% (424) |
| near | 9.3% (184) | 9.6% (190) |
| partial | 29.6% (405) | 28.9% (412) |
| wrong | 49.5% (983) | 48.3% (960) |

| McNemar tier | baseline | 2-pass | delta | b | c | p | |
|---|---|---|---|---|---|---|---|
| STRICT only | 414 | 424 | +0.50pt | 17 | 27 | 0.1742 | not significant |
| **STRICT+NEAR** | **598** | **614** | **+0.81pt** | 16 | 32 | **0.0293** | **SIGNIFICANT** |
| STRICT+NEAR+PARTIAL | 1003 | 1026 | +1.16pt | 29 | 52 | 0.0140 | SIGNIFICANT |

REAL changes (those leaving or entering the tolerant band):
- 32 real improvements, 11 real regressions, **net +21 questions**
- STRICT-only is not significant on its own: the gain is not just extra exact
  matches, it is wrong -> near/strict conversions (16 wrong->strict, 9 wrong->near)

**VERDICT: the second-pass improvement is REAL.** Under a rubric that is
stricter than production — it refuses subset credit for enumeration golds, which
the production scorer had been granting — the effect gets *stronger*, not weaker
(p 0.0140 on the full tolerant band vs p 0.0649 under the earlier looser rubric).

That is the opposite of what an inflated-scorer artefact looks like.

### Two real bugs the strict rubric exposed in the OLD scorer

1. **Subset credit in the tolerant band.** `grade("Rome", "Paris, Rome")` is
   PARTIAL under the production scorer (token overlap 1.0) but is an incomplete
   answer. The strict rubric refuses it at every tier.
2. **Prose misread as a list.** A comma is not sufficient to make an
   enumeration gold: `"Likely no, though she likes reading"` was treated as a
   list, which blocked the containment credit it legitimately earns. The strict
   rubric requires short, non-clausal items.

Both are recorded here because they mean some previously reported numbers were
crediting incomplete enumeration answers.

---

## Value analysis of the second-pass recall — NOT WORTH KEEPING AS DEFAULT

Cost/benefit across all five categories, under the FIXED production scorer and
the strict rubric.  Existing artifacts only, no new model runs.

**Cost:** +45% total run time (63.3 -> 91.8 min) for ~60 extra candidate turns
per question.

**Accuracy, fixed production scorer:**

| category | n | baseline | 2-pass | delta | p | |
|---|---|---|---|---|---|---|
| single-hop | 282 | 27.66% | 28.72% | +1.06pt | 0.5465 | ns |
| multi-hop | 321 | 68.22% | 68.85% | +0.62pt | 0.6831 | ns |
| temporal | 96 | 35.42% | 36.46% | +1.04pt | 1.0000 | ns |
| open-domain | 841 | 61.47% | 60.88% | **-0.59pt** | 0.4990 | ns |
| adversarial | 446 | 26.01% | 27.80% | +1.79pt | 0.1338 | ns |
| **TOTAL** | **1986** | **48.54%** | **48.99%** | **+0.45pt** | **0.3619** | **ns** |

**Strict rubric (STRICT+NEAR):**

| category | n | baseline | 2-pass | delta | p | |
|---|---|---|---|---|---|---|
| single-hop | 282 | 10.64% | 10.28% | **-0.35pt** | 1.0000 | ns |
| multi-hop | 321 | 28.04% | 27.41% | **-0.62pt** | 0.6171 | ns |
| temporal | 96 | 21.88% | 22.92% | +1.04pt | 1.0000 | ns |
| open-domain | 841 | 40.78% | 41.97% | +1.19pt | 0.0525 | ns |
| adversarial | 446 | 19.73% | 20.63% | +0.90pt | 0.3865 | ns |
| **TOTAL** | **1986** | **28.80%** | **29.41%** | **+0.60pt** | **0.0961** | **ns** |

**Not one category is significant under either scoring rule.**

### The decisive test: enumeration questions, where it was supposed to work

Restricted to the 528 questions whose gold is an enumeration (the exact shape
the second pass was built for):

| | baseline | 2-pass | delta | p | |
|---|---|---|---|---|---|
| enumeration golds (n=528) | 32.77% | 32.95% | **+0.19pt** | 1.0000 | ns |

The mechanism was justified by cat1's list-gold weakness. On exactly that
subset it moves nothing.

### Verdict

The second-pass recall **improves evidence retention by +1.59pt (reproducible,
measured without an LLM) but has not been shown to improve accuracy anywhere** —
not overall, not per category, and not on the subset it was designed for — at
+45% runtime cost.

It stays behind `--second-pass` (off by default) as an experiment, and is no
longer a candidate for the default pipeline. This retracts the earlier
"REAL NET +21 questions / SIGNIFICANT" reading from 18d6b4d, which the scorer
bug had manufactured.

**What is still true:** retrieval and the compiler are not the binding
constraint on score. Evidence retention improved and accuracy did not follow.
That leaves the reader, or the reader-plus-scorer pair, as the place to look —
but that is now a hypothesis with clean evidence behind it, not a diagnosis.

---

## A2 step 1 — enumeration output contract: `count` is noise

Owner's plan: change ONE clause, measure it, decide.  First clause tested is
`--enum-contract count` ("count the distinct items in the evidence first, then
list exactly that many; re-read if the list is shorter than the count").

Slice: same 150 questions, same seed, `off` vs `count`.  Elapsed 7.2 min both.

| | accuracy | b | c | discordant | p | |
|---|---|---|---|---|---|---|
| off | 100/150 (66.67%) | 3 | 3 | 6 | 0.6831 | ns |
| count | 100/150 (66.67%) | | | | | |

3 questions flipped to correct, 3 flipped to wrong, net 0.

**Not only null — the regressions are worse in kind.**  Two of the three losses
are degradations, not swaps:

    GT   : A painting inspired by sunsets with a pi...
    off  : abstract painting          (partially right)
    count: I don't know.              (gave up)

    GT   : Likely no
    off  : No                          (right, scored)
    count: Yes, Caroline would still want to pursue...   (wrong AND confident)

The second is exactly the failure mode recorded earlier when the prompt told
the model to reason: a correct hedged answer flipped to a confident wrong one.
**`count` reproduces that failure mode and is rejected.**

This is the second time "elaborate the reader prompt" has been measured and
failed (the first being "推論せよ").  The accumulating evidence is that prompt
elaboration is not the lever — which is itself the useful finding.

Remaining untested clauses: `evidence`, `attribute`.  Given two consecutive
null/negative results from prompt elaboration, these are lower priority than
the reader-plus-scorer pair, where the strict-rubric work already found a real
defect (227 misclassified enumeration golds, 51 still over-firing on 'and').

---

## cat4 diagnosis, and one more scorer bug it exposed

cat4 is 841 of 1,986 questions (largest category) and regressed -0.59pt under
the second pass.  Shape of its 327 failures:

| failure shape | n | share |
|---|---|---|
| answered wrongly | 220 | 67.3% |
| abstained | 107 | 32.7% |
| enumeration gold | 92 | 28.1% |

Accuracy does **not** degrade with gold length (1-2 words 67.0%, 3-5 words
57.6%, 6-10 words 58.3%, 11+ words 61.6%), so long answers are not the problem.
The abstentions are not laziness either — their golds are specific unrecoverable
details (`"a cup with a dog face on it"`, `"Trans Lives Matter"`,
`"Finding Freedom"`), which a 7B reader will not reproduce verbatim.

### The -0.59pt cat4 regression is a SCORER artifact, now fixed

Diffing the two arms per question surfaced this pair:

    GOLD : Dramas and emotionally-driven films
    base : dramas and emotionally-driven films    -> credited
    2pass: dramas, emotionally-driven films       -> WRONG

Same two items, different separator. The harness list-branch split gold items on
**commas only**, so an `and`-enumerated gold stayed a single chunk, never
reached the `>= 2` permutation check, and fell through to containment — which is
gated off for list golds. The comma-rephrased correct answer scored WRONG.

Fixed by splitting gold items on commas and `and`, mirroring `_items()` in the
strict rubric.

A/B against `HEAD`'s scorer on the stored predictions (not a hand
reconstruction — the committed file was imported directly):

| arm | newly credited | newly failed | net |
|---|---|---|---|
| baseline | 24 | 0 | +24 |
| 2-pass | 27 | 1 | +26 |

Every newly credited case is a legitimate re-separation (`Bach and Mozart` /
`Bach, Mozart`; `education and infrastructure` / `education, infrastructure`).
The single newly-failed case is correct behaviour: gold
`"To set goals, track achievements, and find areas for improvement"` vs pred
`"to stay on track, visualize goals and tokenize successes"` — `tokenize
successes` is not in the gold, and the old scorer had credited it through loose
word overlap.

**Both arms gain, the second pass gains slightly more, so this does not change
the arm comparison. Verdict unchanged: NOT SIGNIFICANT at every tier.**

### What cat4 actually is

The largest category is dominated by **specific-detail recall the reader cannot
reproduce verbatim**, not by enumeration and not by length. That is a different
problem from cat1's list-gold coverage, and neither the retrieval work nor the
reader prompt work has addressed it. Recorded as the next distinct target rather
than folded into the enumeration thread.

---

## Outstanding issues — status

Closed in this pass, all measured rather than assumed:

| # | issue | resolution |
|---|---|---|
| 1 | `_scaled_budget` magic numbers unproven | The pool-scaling half measured as **dead code** (0 of 1,986 pools reach the 11667-token engagement point; it produced the constant 1400) and was removed (`f245e68`). The `+3.5pt` I credited to it in `58b1c0b` actually came from the *other* half of that commit — `_scaled_budget()` raising the floor in the harness. Credit was mis-attributed and is now recorded as such. |
| 2 | deterministic hold-out partitioning (catalog #12) | Every artifact now carries `code_revision`, full `config`, and a `run_salt` (`2e71c33`), so any number is traceable to what produced it and a tuned-on partition is visible. |
| 3 | `context_tokens` / latency not recorded | Four fields added per question: `context_tokens`, `latency_seconds`, `gold_is_list`, `abstained` (`dd3b62b`). Retention and cost are now readable from the artifact alone — no separate no-LLM script. |
| 4 | `is_list_gold` gerund over-firing | Gerund-led `and` golds are enumerations, closing a real **subset leak** (`score('education', 'Improving education and infrastructure')` was `True`) — `be1e71e`. |

Still open, bounded and non-measurement-distorting:

5. `temporal_certificate.py` maps keyword -> canned answer strings. Verified
   unreachable from the LoCoMo path and not a LoCoMo gold, but the shape is
   catalog surface #3 (Stubbed Implementation). Must not be extended.
6. `_CLAUSE_LEAD`'s verb list was derived from 91 corpus cases and will not
   generalise to unseen phrasing. Known limitation of the fix in #4.
7. `proposition_integrity_gate.py` and `adaptive_search.py` cast hardcoding —
   already fixed (`f87d1a6`, `1bf6854`). Listed so the sweep reads as complete.

## Measurement system — current state

| property | status |
|---|---|
| gold answers in any prompt | 0 — enforced by `tests/unit/audit_contamination.py`, which fails the build |
| enumeration classification | single source of truth (`is_list_gold`) shared by scorer and audit |
| scorer/rubric agreement | pinned by `test_production_scorer_matches_rubric_on_and_enumerations` |
| rubric drift | pinned by `test_rubric_tiers_are_stable` |
| significance | McNemar mandatory on every reported delta (`audit_strict_grade.py`) |
| provenance | revision + config + salt on every artifact |
| tests | 419/419 |

Every number reported from here is reproducible from its artifact.

---

## All full runs re-scored under the current scorer

The scorer changed four times this session, so recorded `is_correct` values are
not comparable across artifacts.  Every full run re-scored under one rule set:

| artifact | recorded | re-scored |
|---|---|---|
| `locomo_full_20261010_133824` (contaminated prompt) | 50.81% | **56.70%** |
| `locomo_full_20261010_151930` (contaminated prompt) | 50.81% | **56.65%** |
| `locomo_full_20261010_115839` | 47.28% | 52.82% |
| `locomo_full_20261011_230313` (second pass) | 46.68% | 50.35% |
| `locomo_full_20261011_195837` (clean baseline) | 46.32% | 49.75% |
| `locomo_full_20261012_014752` (clean, current) | 49.75% | 49.75% |

**Paired McNemar, newest vs each, n=1986:**

| comparison | delta | b | c | p | |
|---|---|---|---|---|---|
| vs contaminated `133824` | **-6.95pt** | 306 | 168 | 0.0000 | SIG |
| vs contaminated `151930` | **-6.90pt** | 306 | 169 | 0.0000 | SIG |
| vs `115839` | -3.07pt | 273 | 212 | 0.0064 | SIG |
| vs second-pass `230313` | -0.60pt | 45 | 33 | 0.2127 | ns |
| vs clean baseline `195837` | ±0.00pt | 0 | 0 | 1.0000 | ns |

### What contamination was actually worth

**-6.95pt, p=0.0000, 306 questions that only the contaminated prompt got right.**
The gap between the honest 49.75% and the contaminated 56.70% is real, large,
and highly significant — this is not a rounding artefact.

Note the direction of the correction: the contaminated runs recorded 50.81% at
the time, and re-scoring them under the *current, stricter* scorer lifts them to
56.70%. Two effects compound there — the contaminated prompt genuinely produced
answers the clean pipeline does not (that is the -6.95pt, the real contamination
gain), and the older scorer was also more permissive. Both inflate, and they
point the same way.

### Current honest standing

**49.75% (988/1986)**, clean prompt, current scorer, no contamination, guard
CLEAN, 419/419 tests.

| category | acc | |
|---|---|---|
| single-hop | 28.0% | 79/282 |
| multi-hop | 68.2% | 219/321 |
| temporal | 35.4% | 34/96 |
| open-domain | 63.9% | 537/841 |
| adversarial | 26.7% | 119/446 |

Caveat on this artifact: it began before the run-manifest and per-question
diagnostic fields landed, so it carries the old six fields and no manifest. Its
revision is therefore not self-recorded. The next run will be.

---

## Oracle ceiling — "can this engine emit the correct evidence for EVERY question?"

Eli's proposal: get the engine to a state where it surfaces the right evidence
100% of the time, so the remaining gap is provably the reader, and a 120B reader
would then be expected to score near 100%. That would make the engine's quality
model-independent.

Two new instruments, both no-LLM, both on the **real** LoCoMo path (the harness's
own retriever -> graph_builder -> form_bundles -> budgeted_compiler.compile).

NOTE: the pre-existing `scripts/benchmarks/oracle_funnel.py` measures a different
path (`msc_compiler` / `StateReconstructor`) that the LoCoMo harness never calls,
so its numbers do not describe this pipeline. These were written fresh.

### `scripts/benchmarks/oracle_ceiling_locomo.py`

Stage funnel over the 1,955 questions with identifiable gold evidence:

| stage | result | meaning |
|---|---|---|
| candidate pool | 77.2% | retriever put it in the pool |
| bundled | 77.2% | it received an EvidenceBundle |
| **compiled context (real run)** | **35.0%** | it reached the reader |
| **oracle (gold forced to top anchor)** | **36.6%** | survives when told exactly where it is |

Per category, real vs oracle headroom: cat1 +6.4pt, cat3 +4.5pt, cat2 +0.9pt,
cat4 +0.6pt, cat5 +0.5pt.

**Forcing the gold turn to the top anchor with score 900 buys only +1.6pt.**
Anchoring is not the constraint.

### `scripts/benchmarks/diag_budget_vs_gate.py`

Holds everything fixed and only raises the token budget:

| budget | evidence in context |
|---|---|
| 1400 | 35.0% |
| 2400 | 35.0% |
| 4000 | 35.0% |
| 6000 | 35.0% |
| 8192 | 35.0% |

**5.8x the budget, +0.0pt.** A perfectly flat line.

### Conclusion

The binding constraint is the **bundle-selection gate** in
`budgeted_compiler.compile`, not retrieval, not anchoring, and not the token
budget. Two independent manipulations agree: forcing the gold to the top anchor
gains +1.6pt, and 5.8x the tokens gain +0.0pt. If the budget were the limit,
raising it would have moved the number; if anchoring were the limit, oracle
forcing would have recovered far more than 1.6pt.

So Eli's target state — "correct evidence for every question" — is **not
reachable by retrieval or anchoring alone, and is not reachable by buying tokens
either.** The gate is what has to change.

What the gate currently requires of a bundle (all three, or it is skipped):
  1. an anchor score, OR
  2. lexical overlap with the question, OR
  3. continuation of an already-selected session

Evidence that satisfies none is dropped. That is a deliberate precision choice,
and the enumeration escape hatch (`9091af6`) was built to relax it for
enumeration questions — but it measured inert, because the evidence it needed to
admit never became a candidate in the first place.

The honest framing for the 120B-reader idea: **the engine cannot yet be shown to
emit the correct evidence for every question, so a 120B reader's score would not
yet be interpretable as a pure reader measurement.** Closing the gate gap is the
prerequisite, and it is now isolated to one function.

---

## Reconciled accounting — the retrieval gap is 0.9%, and the 21pt is gone

Eli asked two arithmetic questions before Oracle Search was designed on top of
these numbers. Both are answered by a single classified pass
(`scripts/benchmarks/diag_reconcile_audits.py`).

**Q1 — the missing 7.** One population, classified once:

| | n |
|---|---|
| total questions | 1986 |
| temporal gold (resolved from session header) | 277 |
| no evidence field | 8 |
| gold has no content words (`yes`/`no`/…) | 33 |
| annotation related to the answer | **1455** |
| annotation unrelated to the answer | **213** |

`1455 + 213 + 277 + 8 + 33 = 1986`, exactly. The earlier "247 unrelated / 1709
testable / 7 missing" came from two audits classifying with different rules; the
unrelated count is 213, not 247.

**Q2 — the 0.2% denominator.** It was computed on the retrieval audit's own
population (1,955 questions with ≥40-char evidence), not on the 1,668 the
annotation audit used. Mixing them is what produced the "0.2–11.9pt range".

**Reconciled result, one population, one rule** (ANY gold content word present
in the retrieved pool):

| population | n | evidence NOT in pool |
|---|---|---|
| annotation supports the answer (**valid retrieval tests**) | 1455 | **13 (0.9%)** |
| annotation unrelated to the answer | 213 | 3 (1.4%) |

**The 21pt does not exist. The honest retrieval gap is 0.9%.** The 11.9pt
strict-rule figure and the 0.2pt loose-rule figure were both real measurements
of different things — neither was the complement of the oracle test's 78.9%.

**Population for Oracle Search: the 1,455 annotation-valid questions.**

## Oracle Search — what perfect retrieval would buy

`scripts/benchmarks/oracle_search.py`. The gold **answer** is never shown; only
the annotated evidence turn, appended to the normal context. Same model, same
prompt builder, same scorer. Per-question extra token cost recorded.

Smoke test (6 questions): normal 50.0% → oracle 66.7% (+16.7pt), 1 fixed, 0
broken, 2 still wrong with gold evidence in hand.

No answer leakage, verified by inspection: the fixed question is "Where did
Caroline move from 4 years ago?" where the normal arm answers "Caroline's home
country" and the oracle arm answers "Sweden". The model derived that from the
evidence turn; an injected answer string would not require derivation.

Full 1,455-question run in progress.

---

## ORACLE SEARCH RESULT — the decision experiment

`benchmark_results/oracle_search_20261012_170747.json`, n=1455 annotation-valid
questions, same model / same prompt / same scorer, gold **answer** never shown —
only the annotated evidence turn appended to the normal context.

| | normal | + gold evidence | delta |
|---|---|---|---|
| **overall** | **48.38%** | **62.68%** | **+14.30pt** |
| cat1 single-hop | 30.1% | 52.8% | +22.7pt |
| cat2 multi-hop | 29.0% | 74.2% | **+45.2pt** |
| cat3 temporal | 37.1% | 45.7% | +8.6pt |
| cat4 open-domain | 66.3% | 83.4% | +17.1pt |
| cat5 adversarial | 27.4% | 29.7% | +2.2pt |

McNemar, paired: b=52 (broke), c=260 (fixed), discordant=312,
**p≈0, SIGNIFICANT**. Cost: median 61 extra tokens per question.

**260 questions fixed by evidence alone. 543 (37.3%) still wrong WITH the gold
evidence in the context.**

### What this settles

Both directions of the earlier debate now have numbers:

1. **Retrieval is worth investing in — for cat2 and cat4.** cat2 gains +45.2pt
   from evidence alone, which means its 29.0% normal accuracy is almost entirely
   a retrieval problem. Same for cat4 (+17.1pt). For those categories the
   engine's evidence selection, not the reader, is the binding constraint.

2. **The reader is the constraint for cat5.** +2.2pt only. cat5 is the
   adversarial category, whose correct behaviour is to *abstain* when the
   information does not exist. Handing the reader more evidence does not help it
   decide "this is not stated" — that is a reader/scorer judgement problem.

3. **The 52 broken questions are a real cost of adding evidence**, concentrated
   in cat5 (30) and cat4 (15). Several are cases where the model had the right
   hedged answer and the extra evidence pushed it to a confident wrong one —
   the same failure mode already on record for "tell the model to reason".

### Consequence for the plan

The earlier framing "the reader is the bottleneck" was too coarse. The accurate
statement is category-dependent:

| category | binding constraint |
|---|---|
| cat2 multi-hop | **retrieval** (+45.2pt available from evidence alone) |
| cat4 open-domain | **retrieval** (+17.1pt) |
| cat1 single-hop | **mixed** (+22.7pt) |
| cat3 temporal | **mixed** (+8.6pt) |
| cat5 adversarial | **reader / scorer** (+2.2pt — evidence cannot help) |

So retrieval work is justified, but it must be aimed at cat2/cat4, and the 52
regressions show evidence injection is not free — a real retrieval improvement
must be measured against this regression risk, not just the gain.

Note also the ceiling: even with perfect evidence on the annotation-valid
population, accuracy is 62.68%, not 100%. 543 questions survive. Those are the
reader's, and they are now the largest single block of failures.

---

## Absolute-date retrieval fix — FIRST REAL SCORE IMPROVEMENT (+1.11pt, p=0.0005)

Found by following the Oracle Search result, which said cat2/cat4 are
retrieval-bound. Every cat2 question that evidence alone fixes is temporally
scoped, which led to the date-proximity boost — and to two stacked bugs in it.

**Bug 1** (`8138cad`): `_parse_relative_date` handled only relative expressions
("3 weeks ago"). An absolute mention ("in April 2022") parsed to None. Worse, the
`if not reference_date` guard sat at the top of the function and the harness
always passes `reference_date=None`, so absolute dates were unreachable, not
merely unhandled. 240 of 1,986 questions (cat4 132, cat5 59, cat2 37) lost the
boost silently.

**Result of bug-1 fix alone: 100% IDENTICAL predictions across all 1,986
questions.** Not "no improvement" — the feature never executed.

**Bug 2** (`0171927`): the boost compares the parsed query date to the SESSION
date with `_re.search(r"(\d{4})/(\d{2})/(\d{2})", date)`. LoCoMo session headers
are prose — `"10:31 am on 13 October, 2023"` — so the ISO pattern never matches,
`session_dt` is never built, and the 3.0x/2.0x/1.4x proximity boost is dead code
no matter how well the query date parses.

Verified by A/B on one question, boost on vs off, before the fix:

    'What did Mel and her kids paint in their latest project in July 2023?'
    OFF: session_8 (15 Jul) / session_11 (14 Aug) / session_18 (20 Oct)
    ON : session_8 (15 Jul) / session_11 (14 Aug) / session_18 (20 Oct)  IDENTICAL

After the fix the ON arm returns session_8 / session_8 / session_8 — all
15 July 2023, the target month.

### Result

| | baseline | + date fix | delta |
|---|---|---|---|
| **TOTAL** | **49.75%** | **50.86%** | **+1.11pt** |
| cat1 single-hop | 28.0% | 28.7% | +0.7pt |
| cat2 multi-hop | 68.2% | 69.5% | +1.2pt |
| cat3 temporal | 35.4% | 36.5% | +1.0pt |
| cat4 open-domain | 63.9% | **65.5%** | **+1.7pt** |
| cat5 adversarial | 26.7% | 26.9% | +0.2pt |

McNemar, paired, n=1986: b=8 (worse), c=30 (better), discordant=38,
**p=0.0005 SIGNIFICANT**. No category regresses.

**Mechanism confirmed: 30 of 30 improved questions contain an absolute date.**
85 of 1,986 predictions changed (vs 0 after bug-1 alone), which is the direct
evidence the code now runs.

This is the first change in the session that improves the honest end-to-end
score with statistical significance — and it required noticing that a
"successful" full run producing byte-identical predictions meant the feature
was inert. 432 tests were green the whole time; none of them exercised the
session-date path.

---

## cat2 is NOT a multi-hop problem — it is temporal, and the offset fix was rejected

Acted on the Oracle Search signal that cat2 is retrieval-bound (+45.2pt
available from evidence alone). Measured cat2's 98 remaining failures before
building anything.

**cat2 failure shape:**

| shape | n | share |
|---|---|---|
| temporal (`When…` / `How long…` / `How many … ago`) | 63 | 64.3% |
| other | 35 | 35.7% |
| abstained | 41 | 41.8% |
| answered wrongly | 57 | 58.2% |

**63 of 63 temporal questions have a date/duration as the gold answer.** cat2's
failures are not multi-hop chaining at all — they are temporal reasoning and
temporal *formatting*. The multi-hop plan was aimed at the wrong target.

### Attempted fix: offset-date resolution — REJECTED, 0 gain / 15 regressions

17 of the 63 temporal failures share a number with the gold but carry or lack an
offset expression:

    GT  : The Friday before 23January, 2022
    PRED: 23 January 2022

Hypothesis: teach `resolve_relative_dates` offset arithmetic ("the Friday before
X" → X − 1 day) so the prediction lands on the gold's shape.

**Result: 0 questions improved, 15 regressed.** The offset arithmetic converts
predictions that were ALREADY CORRECT into wrong dates:

    GT  : The week before 1 January 2023      (already scoring correct)
    PRED: the week before 1 January 2023
    RES : 25 December 2022                    (now wrong)

The gold phrases the offset but still *names* the anchor date, so "the week
before 1 January 2023" is the correct answer — and my arithmetic rewrote it into
a different date. Reverted.

Also found en route: the corpus writes offsets with no space — `"23January"`,
`"3June"` — so a whitespace-requiring pattern never matches the gold's own
shape. Worth knowing, but it does not rescue the approach.

**Why this was the right thing to refuse.** The remaining 12 offset cases are a
*scorer* question, not an engine one: the prediction carries the right date and
the gold adds an offset phrase. I deliberately gated numeric equality on
`_is_list` in `ff0227f` precisely so that "2 May, 2022" could not be credited
for "The week before 2 May, 2022". Loosening that to rescue these 12 would be
score inflation dressed as a fix — exactly the mistake this session has already
retracted twice.

**Honest position on cat2:** its failures are temporal, and the ones that look
rescuable are blocked by a deliberate anti-inflation rule, not by a bug. The
real temporal weakness is the 73% with genuinely wrong dates (e.g. GT "In 2013"
vs PRED "2012-06-03", GT "two weeks" vs PRED "2"), which is reader-side date
reasoning — the thing the earlier "tell the model to reason" prompt also failed
to fix.
