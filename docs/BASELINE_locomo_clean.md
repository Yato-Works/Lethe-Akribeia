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

## Outstanding issues carried into the next run

Ordered by how much they distort a reported number. Items 1-3 are correctness
of measurement; 4-6 are known engine debt.

1. **`_scaled_budget` magic numbers have no derivation** (catalog surface #8,
   Parameter Overfitting). `ratio=0.12, cap=2400, floor=1400` were tuned
   against observed behaviour, on the knob most likely to be re-tuned into the
   test. Needs provenance or a measured justification.
2. **Deterministic hold-out partitioning** (catalog surface #12). Hold-out
   slices are pure functions of `--conv-idx`/`--offset`, so anyone iterating on
   those numbers tunes against a knowable partition. Needs a per-run salt,
   logged after completion.
3. **`context_tokens` / per-question latency are not recorded** in the full-run
   artifacts, so retention and cost had to be measured with separate scripts.
   The harness should emit them.
4. **`is_list_gold` still over-fires on `and` for gerunds**
   (`"Taking walks and hiking"` is an enumeration; a verb-led part is a clause).
   The verb list in `_CLAUSE_LEAD` was derived from 91 corpus cases and will not
   generalise to unseen phrasing.
5. **`temporal_certificate.py` maps keyword -> canned answer strings**
   (`"cousin"+"wedding" -> "my cousin's wedding"`). Verified NOT reachable from
   the LoCoMo path and not a LoCoMo gold, but the shape is catalog surface #3
   (Stubbed Implementation) and must not be extended.
6. **`proposition_integrity_gate.py`** had LoCoMo's cast hardcoded; fixed in
   `f87d1a6`. `adaptive_search.py` grounding tags likewise fixed in `1bf6854`.
   Listed so the sweep is complete, not because they are still open.
