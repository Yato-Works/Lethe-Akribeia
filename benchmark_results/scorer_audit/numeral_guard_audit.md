# Audit — NUMERAL CONTRADICTION guard on the LoCoMo scorer

Date: 2026-10-13
Scorer: `scripts/benchmarks/eval_locomo_full.py::score`
Prior related fix: `c449aa0` (day-of-month contradiction, same disease, different unit)

## What was measured

Every stored prediction/gold pair across all 46 `benchmark_results/locomo_full_*.json`
runs was scored with the guard bypassed (OLD) and the guard active (NEW).

```
runs scanned                        : 46
total pred/gold pairs               : 17,604
correct BEFORE the guard            : 9,059
correct AFTER  the guard            : 9,027
spurious credits REMOVED            : 32
real credits BROKE                  : 0        <-- must be 0
VERDICT                              : clean fix
```

## The defect

The either-direction 50% token-overlap rule shares one content word and credits
the answer while the disagreeing numeral is never compared:

| pred | gold | old verdict |
|---|---|---|
| `two dogs` | `three dogs` | CORRECT |
| `5 dogs` | `2 dogs` | CORRECT |
| `ten times` | `two times` | CORRECT |
| `four times, 4` | `three times` | CORRECT |
| `2 weeks` | `three weeks` | CORRECT |
| `5 months` | `4 months` | CORRECT |
| `0 years` | `three years` | CORRECT |
| `2 years ago` | `10 years ago` | CORRECT |
| `22 August, 2023` | `August 2022` | CORRECT |

Measured on the 382 pairs that state conflicting numerals, 32 (8.4%) were
credited correct.  The single-word overlap (`dogs`, `weeks`, `times`, `years`)
carries the credit and the differing count is invisible to the rule.

`c449aa0` closed the day-of-month instance of this.  Quantities, durations, and
years were still unguarded.

## The fix

`_numeral_contradiction(p, g)`, placed immediately after `_date_contradiction`
and therefore BEFORE every credit rule.

Fires only when:
- both sides state a quantity, and
- the quantity sets are DISJOINT.

Two channels: `_quantity_values` (cardinals word or digit, excluding bare
4-digit years) and `_year_values` (explicit year comparison, so
`22 August, 2023` vs `August 2022` is caught).

Narrow by construction: a prediction naming several quantities where one matches
gold is NOT a contradiction and keeps its credit.

## Both-direction verification

`scripts/benchmarks/check_numeral_guard.py` — 29 cases, all pass.

```
must-be-wrong       : 17     the defect cases, now WRONG
must-be-correct     : 10     legitimate credits, still CORRECT
guard-must-be-silent:  2     pairs the guard must not claim as contradictions
```

The `guard-must-be-silent` set pins the blast radius.  Two cases first reported
as failures — `twice` vs `two times`, and `three times` vs `often` — were
diagnosed (`diag_numeral_false_positive.py`) as `_numeral_contradiction == False`
on both: the guard was already silent and those verdicts are the pre-existing
scorer's own behaviour.  They were moved into the silent set, not papered over.

Regression: `490 passed, 1 xpassed` (was `488 passed, 1 xfailed`).

## RETRACTION — a previously reported engine gain was spurious

`benchmark_results/rnd_counting_discovery_after_20261013.json` recorded
`lc_conv-44_17` as FIXED, contributing the "FIXED 1 / BROKE 0" reported earlier
for the counting cohort.

```
question : How many times did Audrey and Andrew plan to hike together?
gold     : 'three times'
engine   : 'four times, 4'
old scorer : CORRECT     (shared the word 'times')
new scorer : WRONG       (numerals {4} vs {3} are disjoint)
```

That FIXED was manufactured by the defect this audit removed.  Re-measured with
the corrected scorer (`remeasure_engine_claims.py`):

```
counting discovery   (n=6)  FIXED=0  BROKE=1  SPURIOUS=0
counting after-final (n=6)  FIXED=0  BROKE=1  SPURIOUS=0
counting after-final2(n=6)  FIXED=0  BROKE=0  SPURIOUS=0
temporal binding     (n=9)  FIXED=0  BROKE=0  SPURIOUS=0
-----------------------------------------------
TOTAL                       FIXED=0  BROKE=2  SPURIOUS=0
```

The real engine gain on these cohorts is **zero**, not one.  The earlier claim is
withdrawn.

## What is NOT claimed

- No full-benchmark score improvement is predicted.  This is a scorer integrity
  fix, and it moves the score DOWN by 32 spurious credits across stored runs.
- The 987-failure atlas counts are unchanged by this work; this audit touches
  scoring, not the atlas.
- The 32 removed credits are removed from STORED RUNS, which are not the current
  baseline (`code_revision` was null).  They bound how much past self-reported
  progress was inflated, not the current engine's score.
- The counting/temporal cohorts are 6 and 9 questions.  Zero gain on that sample
  is evidence those particular engines did not help; it is not proof no fix in
  that area can help.

## Artifacts (new files only, nothing existing modified)

```
scripts/benchmarks/eval_locomo_full.py                     guard added
scripts/benchmarks/measure_scorer_numeral_defect.py        measures the defect
scripts/benchmarks/check_numeral_guard.py                  both-direction tests
scripts/benchmarks/measure_numeral_guard_ab.py             before/after on real data
scripts/benchmarks/remeasure_engine_claims.py              re-scores recorded cohorts
scripts/benchmarks/diag_numeral_false_positive.py          traces the two false positives
tests/unit/test_locomo_eval_scorer.py                      guard regression tests
benchmark_results/scorer_audit/numeral_guard_audit.md      this document
benchmark_results/scorer_audit/numeral_guard_ab.json       measured before/after
```
