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
