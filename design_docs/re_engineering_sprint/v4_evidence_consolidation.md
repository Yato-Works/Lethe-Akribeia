# V4 Evidence Consolidation — M5 through M13

Status: **current hypothesis group closed with NO-GO.** Not "nothing was learned" —
seven directions were measured and none produced a reproducible improvement
hypothesis, which is knowledge the next round must not re-spend effort rediscovering.

This document is a re-derivation, not a recollection. Every number below was read back
from a committed artefact by `scripts/benchmarks/v4_evidence_consolidation.py`. Where an
artefact does not carry a claim, the claim is marked NOT-IN-ARTEFACT rather than
restated. Three of the numbers originally stated from memory turned out to be missing or
mis-keyed, and the script refused to fabricate them.

---

## 1. What is established

| # | Finding | Mission | Denominator |
|---|---|---|---|
| 1 | Adding candidates does not help. Top-50 found exactly **1** more gold-like string than top-20. | M5 | 75/90 → 76/90 |
| 2 | Adding context does not help. Full source was **net zero**: 11 rescued, 11 regressed (3 of them controls), at +21% prompt tokens. | M6 | stratified 90 |
| 3 | Question-term evidence selection is **harmful** — it breaks correct answers. | M3 | 90 |
| 4 | The Reader answers 805/989 and is **85.1%** accurate on what it actually answers. | M2 | 989 |
| 5 | The wrong value is a **real content error**, not a surface-form error. | M7 | 54 |
| 6 | The wrong value maps to a **different event** on the source in 26/28. | M8 | 28 |
| 7 | Both the correct event **and** the rival sit inside the frozen context the Reader actually saw, in 23/28. | M11 | 28 |
| 8 | A competing event is visible about as often when the Reader is **correct** (14/16 = 87.5%) as when it is wrong (23/28 = 82.1%). | M11 | 28 vs 16 |
| 9 | Same-day competing events do **not** predict failure across the benchmark: 69.6% vs 69.1% accuracy, confidence intervals overlap. | M10 | 989 |
| 10 | Position, lexical overlap and speaker agreement do not separate chosen-wrong from chosen-right, and the two that show a gap point the **wrong way**. | M12 | 23 vs 14 |
| 11 | Predicate satisfaction does not separate them either — under a **low-power instrument**. | M13 | 17 vs 9 |

The single most load-bearing fact in the table is **#8**. It is the reason the whole
event-binding direction closed: the rival really is in front of the Reader, and it is in
front of the Reader about as often when the answer is right.

---

## 2. What is NOT established

This section exists because the earlier wording over-claimed, and the correction matters.

**"Observable features cannot explain the wrong answers" is not supported.** That is a
universal claim. What the evidence supports is narrower and must be stated as:

> Under the features, instruments and sample sizes tried in M5–M13, the failure group
> and the control group were not stably separated.

That distinction is not pedantry. Two separate ceilings bound the negative:

- **Sample size.** M12/M13 used 17–23 failures against 9–14 controls. A handful of cases
  moves the result.
- **Instrument power.** M13's `satisfies_predicate` returned POSSIBLE for 22 of 26
  measured pairs and **NO zero times**. A judge that calls almost everything a
  satisfying candidate cannot detect a real relationship. So M13's −16pp is a *weak*
  negative: evidence against the hypothesis *under a low-power instrument*, not proof
  the relationship is absent.

*Failing to detect an effect is not the same as the effect being absent.* Both missions
reported the sign reversal honestly (the gaps ran the wrong way for the hypothesis), but
a sign reversal under a weak instrument is not a refutation.

Also not established:

- Any claim about **why** the Reader preferred a candidate. Internal reasoning was never
  observed in any mission. Every audit measured what the *output* chose.
- Generalising M6's net-zero from 90 stratified questions to all 1,986.
- Reading M10's null result as "the M9 pattern is impossible." It shows the pattern does
  not replicate *under that condition definition*.

---

## 3. A reporting hazard worth recording

M7 carries a field named `gold_in_context`, and it is TRUE for **54/54**. The 54-case
cohort was selected *because* the gold string was verbatim in the context, so 100% is
true by construction and carries no information.

The "0/28" figure reported for M7 is a **different** metric: whether the gold's words
appear in the Reader's raw *output*. Those two point in opposite directions and must
never be quoted interchangeably. Consolidation is exactly where such a substitution
would have become permanent, so it is written down here instead.

---

## 4. Where the apparatus is missing

Not "the hypothesis failed" — the *measuring instrument* does not exist for these:

| Gap | Consequence |
|---|---|
| No second independent annotator | Inter-annotator agreement is unmeasurable. The M4 blocker is still open and was never resolved. |
| No way to observe the Reader's own selection criterion | "Why this candidate" is permanently out of reach for a black-box reader. |
| No predicate judge with adequate power | M13 could not have found a real effect even if one existed (NO occurred 0/26). |
| No external intervention probes | Causal predictions remain entirely untested. |

---

## 5. The untested causal probes — the real gap

This is the most important omission, and it is fixable without any new apparatus.

Everything in M5–M13 is **observational**: features were measured and correlated with
outcomes. Nobody ever *intervened*. The Reader's choice can be probed causally from
outside, without touching its internals:

1. **Remove one candidate.** Drop one of the two competing utterances and re-ask. If the
   answer flips to the other, the choice was caused by co-presence. If it does not
   change, co-presence is not the driver.
2. **Reorder the utterances.** Swap the two and re-ask. M12 measured position
   correlationally and got −29pp in the *wrong* direction; an actual swap tests
   causality instead of correlation.
3. **Paraphrase and re-ask.** Same meaning, different surface form. This tests whether
   the M12 overlap result (+21pp) reflects lexical sensitivity or genuine confusion.

These are precisely the probes that M12's observational gaps cannot substitute for, and
they require **no new instrument** — only new Reader calls on already-frozen contexts.

---

## 6. Valued asset, deliberately unspent

```
unused holdout conversations : 26, 30, 41, 42, 44, 48
unused holdout questions     : 553
split unit                   : conversation (no dialogue shared with tuning)
```

Fixed in M10 *before* any measurement, so it was never used to design anything. It
should not be consumed while no intervention candidate exists.

---

## 7. The decision

**NO-GO on the current hypothesis group.** Seven directions were measured; none yielded
a reproducible, generalisable improvement hypothesis. No new rule is invented and
nothing is promoted to production.

The three intervention probes in §5 are the honest next research target — and note that
they are *causal* tests, not more feature correlation. If those also come back null, the
correct conclusion is that the reader's selection is not steerable by context surgery at
all, which is a substantially different and more final statement than anything M5–M13
can currently support.

Before any probe is run, three things must be fixed in advance:

1. What observation would count as *explaining* a failure the existing hypotheses could not.
2. How agreement is evaluated — which requires resolving the missing-annotator problem.
3. Which result resumes the research and which result sustains NO-GO.

Fixing these first is what prevents "read 23 cases and see what turns up."
