# Real baseline — measured, after both scorer guards

Date: 2026-10-13.  Everything below is re-scored output, nothing inferred.

## The honest current number

Newest full run `locomo_full_20261012_211601.json` (code_revision `0171927`, 1,986 questions):

```
stored / self-reported accuracy   : 1010 / 1986 = 50.9%
RESCORED with today's scorer      :  983 / 1986 = 49.5%
delta                             :  -27 questions  (-1.4 pt)
```

**The real baseline is 49.5%, not 50.9%.**  The 27-question gap is the two
contradiction guards removing credits the old scorer handed out for wrong dates
and wrong counts.

## The guards account for the whole difference — verified, not assumed

`scripts/benchmarks/isolate_scorer_delta.py` re-scored that same run under five
scorer configurations:

| variant | correct | vs stored | correct→wrong | wrong→correct |
|---|---:|---:|---:|---:|
| stored flags (run's own report) | 1010 | — | — | — |
| both guards OFF | 1010 | **+0** | **0** | **0** |
| date guard only | 986 | −24 | 24 | 0 |
| numeral guard only | 986 | −24 | 24 | 0 |
| both guards ON (current) | 983 | −27 | 27 | 0 |

With both guards disabled the scorer reproduces the stored flags **exactly** —
zero flips.  So `score()` itself has not drifted, and the entire −27 is
attributable to the guards.  (The two guards overlap by 24 questions: most
conflicting-numeral pairs are also conflicting dates, which is why 24+24 > 27.)

This matters because `rescore_all_runs.py`, aggregating across all 46 runs,
reported 505 correct→wrong **and** 480 wrong→correct.  A one-directional guard
cannot produce an even split.  It turned out that split is an artifact of
comparing runs written by DIFFERENT scorer generations: runs from 10/10–10/11
predate both guards and were under-credited by +41…+58, while runs from 10/12
carry the guards' credits and are over-credited by −27.  **Within any single run
the flip is one-directional.**  Comparing a run's stored flags to today's scorer
is only meaningful run-by-run, never in aggregate.

## Per-run effect, newest full runs

| run | rev | stored | rescored | delta | rescored acc |
|---|---|---:|---:|---:|---:|
| 20261010_115839 | null | 939 | 985 | +46 | 49.6% |
| 20261010_133824 | null | 1009 | 1067 | +58 | 53.7% |
| 20261011_195837 | null | 920 | 961 | +41 | 48.4% |
| 20261011_230313 | null | 927 | 974 | +47 | 49.0% |
| 20261012_014752 | null | 988 | 961 | −27 | 48.4% |
| 20261012_193230 | 8138cad | 988 | 961 | −27 | 48.4% |
| 20261012_211601 | 0171927 | 1010 | 983 | −27 | **49.5%** |

The pre-10/12 runs are not comparable to the post-10/12 runs under one scorer;
their `code_revision` is null and their flags were written before the guards.
They are reported for provenance, not as a trend line.

## What this means for the reverse-engineering sprint

Any integration claim must be measured as `FIXED / BROKE / SPURIOUS` on a frozen
cohort, against **983/1986 as the baseline**, with the matched question_id count
asserted before any number is reported.  A change that is not visible above the
27-question noise floor the guards themselves introduced has not been measured.

Reported deltas must state which scorer generation produced each side.  An
"improvement" that crosses the guard boundary is a scoring artifact, not a gain.

## Artifacts

```
scripts/benchmarks/rescore_all_runs.py       re-scores all 46 runs
scripts/benchmarks/isolate_scorer_delta.py   proves the guards explain the whole delta
benchmark_results/scorer_audit/rescored_all_runs.json
```
