# V4 Program — from a deflated measurement to engine coverage

Status: **program document + measurement correction.** This replaces the score
interpretation in `README.md` for research purposes. It contains no capability
claim that was not measured on the pinned official harness in this session.

Every number below was produced by `scripts/benchmarks/score_locomo_official.py`
(the pinned upstream `third_party/benchmarks/locomo/task_eval/evaluation.py`,
`metric=f1`) over stored predictions. Nothing was regenerated, edited or dropped.

---

## 1. The premise to correct first

The starting belief was "we are far behind Mem0 (92.5 on LoCoMo)". That comparison
is not available yet, and quoting it as a target is unsafe:

| system | published | judge / protocol | comparable to our F1? |
|---|---|---|---|
| Mem0 | LoCoMo 92.5 | LLM-as-a-Judge, gpt-4o-mini, 10 runs | no |
| Zep / Graphiti | DMR 94.8 | different benchmark, LLM-backed | no |
| Letta | LoCoMo 74.0 | third-party mirror, gpt-4o-mini answerer **and** judge | no |
| Cognee | BEAM 0.79 / 0.67 | BEAM, LLM judge, n=20 | no |
| Lethe | official token F1 | deterministic matcher, local 7B reader | — |

`design_docs/re_engineering_sprint/consolidated_decision.md` already records this.
The only fair comparison route in this repository is the **P2 arena**
(`scripts/benchmarks/p2_external_arena.py`): every player shares one frozen answer
LLM, one frozen extraction LLM, one frozen embedding model, a 2,000-token budget
and one scorer. Until that runs with a real LLM, "we are behind" is a feeling.

---

## 2. Measurement: the current run, on the official protocol

Run `benchmark_results/locomo_full_20261023_010055.json` (rev `4ed8d4e`,
1,986 questions, `--cond R7`), re-shaped to the official `question_id` scheme
(`lc_conv-26_52` -> `conv-26-qa-052`) and scored by upstream code:

| category | n | official F1 |
|---|---:|---:|
| multi-hop | 282 | 32.03% |
| temporal | 321 | 59.90% |
| open-domain | 96 | 22.12% |
| single-hop | 841 | 52.20% |
| adversarial | 446 | **0.00%** |
| **ALL** | **1,986** | **37.41%** |
| ALL (no cat5) | 1,540 | **48.24%** |

The v0.3.0 artifact (`locomo1540/temporal321_rules_commit_7b_postfix.json`,
cats 1-4 only) scored on the same protocol:

| category | v0.3.0 | current | delta |
|---|---:|---:|---:|
| multi-hop | 33.97% | 32.03% | -1.94 |
| temporal | 56.33% | 59.90% | **+3.57** |
| open-domain | 24.76% | 22.12% | -2.64 |
| single-hop | 59.31% | 52.20% | **-7.11** |
| ALL (no cat5) | **51.89%** | **48.24%** | **-3.65** |

So the R7 rebuild improved temporal reasoning and lost more on single-hop. That
regression is a concrete, measured target — not a mystery.

**The 48.7% "LoCoMo" headline is not a LoCoMo score.** It comes from the in-house
binary matcher, which for cat5 uses `adversarial_answer` as gold (see §3). The
official-comparable numbers are the table above.

---

## 3. The measurement defect (found this session, verified)

### 3.1 cat5 gold is the adversarial distractor

`scripts/benchmarks/eval_locomo_full.py:97`:

```python
gold = qa.get("answer", qa.get("adversarial_answer", ""))
```

LoCoMo category 5 stores **no** `answer`; it stores only `adversarial_answer`. The
pinned official harness (`evaluation.py:217-221`) scores cat5 as:

```python
if 'no information available' in output.lower() or 'not mentioned' in output.lower():
    score = 1
else:
    score = 0
```

Consequence: the in-house matcher **rewards emitting the wrong option**. In the
current run, 126 cat5 questions are "correct" only because the prediction matched
the distractor; upstream scores all of them 0. The category is therefore reported
at 28.3% in-house while its official value is **0.00%**.

This also retroactively explains `docs/BASELINE_locomo_clean.md`'s observation
that the contaminated prompt "produced fluent hallucinations and scored them as
correct" on cat5. Part of that was leakage; part of it was this inversion.

### 3.2 Refusal lexical form does not match the contract

Of 436 refusals, 248 are on cat5 — the category where refusal is *always* the
correct behavior. The system says `I don't know.`, which upstream does not accept.

Measured projection, produced by rewriting **only** the refusal string on the
stored predictions (no other change) and re-scoring with upstream code:

| variant | cat5 | ALL (1,986) | cats 1-4 |
|---|---:|---:|---:|
| current | 0.00% | 37.41% | 48.24% |
| refusal form -> `not mentioned` | 55.61% | **49.91%** | 48.27% |

**+12.50 pt, for free, with no change to any category 1-4 answer.**

This is a *format* fix, not a capability gain, and must be reported as one. It
also must not be implemented by rewriting predictions inside the scorer — that
would be metric manipulation. The system itself emits the canonical refusal.

### 3.3 The holdout denominator needs one definition

`benchmark_config/holdout.yaml` counts cats 1-4 only (conv-42: 199, conv-48: 191,
total 390). The official wrapper counts all categories (499). Both are correct
for their stated scope; every report must say which one it uses.

Artifacts: `benchmark_results/official_locomo_score_v4r7_aligned.json`,
`..._v4r7_refusal_form.json`, `..._v030_rules_commit.json`.

### 3.4 M0 status — implemented

`scorer_protocol: locomo_official_v1` is now the harness default:

- `load_questions` no longer falls back to `adversarial_answer`; the distractor
  is carried in its own field and never compared as gold.
- `score_question` applies the upstream category-5 refusal contract, and is
  deliberately **stricter** than the normaliser so the in-house number can never
  be more lenient than the official one.
- The harness emits a canonical refusal (`not mentioned`) whenever the reader
  declines, and records `predicted_answer_raw` + `refusal_normalized` so the
  format change stays auditable.
- Tests: `tests/unit/test_locomo_cat5_protocol.py` (21 cases), existing
  `test_locomo_eval_scorer.py` unchanged and green.

Re-scoring the frozen 2026-10-23 run with the new protocol reproduces the
projection exactly: binary **1,089/1,986 = 54.83%** (cat5 248/446 = 55.6%),
official F1 **49.91%**.

---

## 4. Where the loss is

Current official F1, with the engine-recoverable share from
`benchmark_results/failure_atlas/engine_roadmap.md`:

| cluster | n | status | owner |
|---|---:|---|---|
| plural / list union across turns | 311 | gold in context for 262 | CountingEngine (extension) |
| evidence present, no engine owns the shape | 288-433 | all gold in context | none (new shape) |
| `when_date` binds wrong turn | 104 | 101 gold in context | TemporalAlgebra + binding check |
| attribute resolver wrong | 111 | 87 gold in context | EntityAttributeResolver (precision) |
| gold absent from context | 157 | 0 recoverable | retrieval — **not now** |
| cat5 refusal (n=446) | 446 | refusal always correct | sufficiency gate (unwired) |

In-house decomposition of the same run: committer claims 312 (59.3% precise),
reader answers 1,674 (46.7%), 436 refusals (0.2%). List-gold questions: 521 at
36.9% binary.

### 4.5 M1 result — deterministic refusal gates are a measured NO-GO

Three gates were built and priced on the frozen contexts, paired against the
stored run, dev scope (1,487 questions), official F1. Delta = (category-5 newly
refused) - (category-1-4 credit lost).

| gate | rule | best delta | cat5 F1 | single-hop F1 |
|---|---|---:|---:|---:|
| `subject_absent` | best-matching line never names the subject | **-1.11 pt** | 54.0 -> 63.2 | 52.5 -> 46.8 |
| `predicate_object` | no subject-turn binds the asked predicate + objects | **-5.34 pt** | 54.0 -> 71.8 | 52.5 -> 39.5 |
| `predicate_absent` | subject missing from the line's opening clause | worse (superset) | — | — |

Every variant buys category-5 recall with single-hop precision. The cause is
structural, not a tuning failure: this corpus distributes one fact across several
turns ("charity race" in one, "realize" in another), so a single-turn binding test
refuses questions the Reader answers correctly by joining turns. **Surface-string
attribution gating cannot be the refusal mechanism.** The remaining options are a
Reader-side sufficiency verifier (one extra call) or a proposition-level IR path;
both are new measurements, not a retry of this one.

Artifacts: `benchmark_results/refusal_gate/*.json`,
`scripts/benchmarks/measure_refusal_gate.py`.

### 4.6 B1 measurement — the list cluster is compiler-bound before it is engine-bound

Measured on the current pipeline (frozen contexts, 0 LLM), list-gold cats 1-4
(n=445):

| group | n | correct |
|---|---:|---:|
| committer claims | 97 | 59 (60.8% precision) |
| reader answers the rest | 348 | 114 (32.8%) |
| multi-hop lists (cat1) | 160 | 21 (13.1%) |

Gold-item availability, matched exactly against the text:

| category | items in compiled context | items in full conversation |
|---|---:|---:|
| cat1 (multi-hop lists) | 40.1% | **64.6%** |
| cat4 (single-hop lists) | 64.7% | 79.5% |
| cat2 (temporal lists) | 55.7% | 2.6% (derived dates; the compiler adds them) |

For cat1: 76 of 160 questions lose at least one item that exists in the
conversation; **129 available items are dropped by retrieval/compilation**; the
"all items present" ceiling is 25 questions in the compiled context versus **62**
in the conversation. `--second-pass` moves this by **+1 item / +1 question** at an
unchanged token budget, so the existing enumeration pass is not the lever — the
budget/selection is.

Consequence: B1's order is inverted. Build the **enumeration recall** step first
(B0), keeping the union engine for after the items are actually in the context.
B0 is measurable at 0 LLM calls (item recall), which makes it the cheapest
high-volume lever in the program.

Three variants were then frozen on the same 160 questions (`--budget-scale` is a
measurement knob; 1.0 is unchanged):

| variant | items in context | fully answerable | mean tokens |
|---|---:|---:|---:|
| baseline | 205/511 (40.1%) | 25/160 | 1,442 |
| budget x2 | **253/511 (49.5%)** | **32/160** | 2,887 |
| `--second-pass` | 206/511 (40.3%) | 26/160 | 1,442 |
| `--second-pass` + budget x2 | 251/511 (49.1%) | 30/160 | 2,888 |

So the token budget is a real but partial lever (+48 items, +7 answerable
questions, for 2x tokens), and the existing second pass is not a lever at any
budget. §4.7 records the candidate pool to attribute what is left.

### 4.7 The loss is attributed, and two compiler defects are now fixed

With the pool recorded (`--freeze-pool`), cat1's 707 gold items classify as:

| bucket | items | share | stage that could fix it |
|---|---:|---:|---|
| IN_CONTEXT | 234 | 33.1% | already available |
| **POOL_ONLY** | **127** | **18.0%** | compilation selection |
| CONVERSATION_ONLY | 32 | 4.5% | retrieval |
| ABSENT | 314 | 44.4% | paraphrase / genuinely absent (exact-match upper bound) |

So selection, not retrieval, is the addressable stage. The budget–recall curve on
the 160 list questions:

| budget | items in context | fully answerable | mean tokens |
|---|---:|---:|---:|
| x1 | 205/511 (40.1%) | 25 | 1,442 |
| x2 | 253/511 (49.5%) | 32 | 2,887 |
| x3 | 263/511 (51.5%) | 36 | 4,331 |
| x5 | 290/511 (56.8%) | 42 | 7,203 |

The ceiling from the conversation is 64.6%, so the budget explains nearly all of
the remaining gap, at a cost that climbs steeply after x2.

Two real defects were found on the way, both in the selection path, and both are
now fixed behind an opt-in switch so that the default path stays byte-identical:

1. **`_bundle_names_subject` ignored the speaker label.** A first-person turn
   ("I hiked to the forest") contains no name, so the enumeration escape hatch
   was blind to exactly the turns it was written for. The speaker now counts.
2. **Admission was re-dropped by the second gate.** The escape hatch let a
   bundle past the first gate, and the next gate ("no new query words, no anchor
   score -> skip") removed it. Enabling admission therefore moved recall by
   exactly zero. Both gates now honour it.

With those fixed, admitting the queried person's turns still changes nothing at
x1 or x2 — because the budget, not admissibility, is binding: of the dropped
items, 65 sit only in the subject's turns, 31 in both, 9 only in the other's, and
they do not fit. Tests: `tests/unit/test_subject_admission.py` (7 cases).

### 4.8 B0 reader-level outcome: the costed gain is real but small, and the cheap fix is rejected

Item recall is a proxy. The only number that counts is the reader's score, so the
2x-budget variant was run end to end on cat1 (282 questions, 7B reader, 20 min),
then compared per question with the stored baseline under official F1:

| scope | questions | official F1 | delta | improved / worsened |
|---|---:|---:|---:|---:|
| all cat1 | 282 | 32.03% -> 34.39% | **+2.36pt** | 62 / 42 |
| **dev only** | **224** | **34.99% -> 36.66%** | **+1.68pt** | 47 / 35 |

The full-set figure is flattered by the held-out conversations; the dev figure is
what may be claimed, and it is +1.68pt on 14% of the benchmark for **2x the
tokens and 2x the latency**. Binary accuracy did not move at all (86/282 in both
runs) — the gain exists only in partial credit. Not promoted.

The cheap version was also tested and rejected: admitting the subject's turns
*with* a gain bonus (8 and 16) moved item recall by +1 item, lost one
fully-answerable question, and in the traced case ("Where has Melanie camped?")
it dropped `mountains` from a context that previously contained it. A boost that
displaces real evidence is not a fix.

**B0 conclusion.** Enumeration recall is budget-bound, the budget is expensive,
and the reader converts only about a fifth of the incremental item recall into
score (+9.4pt item recall -> +1.68pt dev F1; a proportional conversion would have
been ~+8pt). This is the same conversion wall M6 measured. The next attempt
should construct the list answer from the evidence rather than show the reader
more of it.

### 4.9 The answer-provenance gate: first net-positive intervention (+4.33pt dev)

**The idea that failed three times was keyed on the QUESTION.** Ask "does the
subject have a turn containing the question's predicate?" and the answer is often
no for a correctly answered question, because this corpus spreads one fact across
turns. Key the test on the ANSWER instead and the signal is sharp: in the
adversarial category the reader returns the *other person's* fact, so the
answer's own wording appears only in the other speaker's turn.

Rule: name the single speaker the question asks about; take the answer's content
stems; find the best-matching dialogue line; if that line belongs to someone else,
the answer did not transfer and becomes a refusal. It never rewrites an answer.

Measured on dev (1,487 questions, official F1, paired against the stored run):

| threshold | F1 before | F1 after | delta | adversarial | single-hop |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 50.56% | 54.66% | +4.10 | 54.0 -> 79.8 | 52.5 -> 49.4 |
| **0.7** | **50.56%** | **54.89%** | **+4.33** | **54.0 -> 79.5** | 52.5 -> 49.9 |
| 0.9 | 50.56% | 54.72% | +4.16 | 54.0 -> 78.6 | 52.5 -> 50.0 |

Full set (1,986, reported not tuned): 49.91% -> 52.59% (**+2.68pt**). The cost is
concentrated in single-hop (-2.6) and temporal (-1.05); the gain is +25.5 on the
adversarial category.

Guards that make it safe, each of which was a measured failure when absent:
date-shaped answers are out of scope (enforcing provenance on dates cost 7.8pt on
temporal); questions naming no single speaker, or naming both, are out of scope;
an answer whose wording is not traceable in the context is out of scope.

Shipped as `src/artificial_memory/skills/refusal_gate.py` (12 synthetic tests),
opt-in via `--refusal-gate`, and verified to reproduce the measurement **exactly**
(+4.331pt) through the shipped module rather than a copy of its logic.

**Holdout leak — recorded, not hidden.** While computing the full-set number, the
`--scope` filter was inverted, so one run measured ONLY the 499 held-out
questions (it showed +1.81pt). The filter is fixed. Consequence: for *this gate*
the holdout is no longer clean, so it can neither be tuned on nor cited as
validation. The design number is the dev figure (+4.33pt). The 553-question
M5-M13 holdout remains unspent and is a different slice.


---

## 5. What is already measured as not working (do not re-spend)

| direction | result | mission |
|---|---|---|
| more candidates | top-50 found +1 gold | M5 |
| more context | net 0, +21% tokens | M6 |
| PPR over evidence graph | worse than the current retriever | spike |
| ordering rules | every rule rescues the same 6 | M17 |
| prompt elaboration (`count`/`evidence`/`attribute`) | 3 consecutive failures | commits `6ec219f` etc. |
| abstention prompt clause | 0 gain / 2 worse (n=40) | `6ec219f` |

The last row used n=40 and `p=0.48`; it is a **weak** negative, and it tested a
*prompt* intervention. It does not test a deterministic decision.

---

## 6. V4 tracks

### Track A — Protocol & refusal (weeks 1-2)

1. `scorer_protocol: locomo_official_v1` in the run manifest: cat5 scored by the
   refusal contract, `adversarial_answer` kept as a *reported field*, never as gold.
2. The system emits the canonical refusal string when it decides to refuse.
   Raw prediction retained in the artifact (`predicted_answer_raw`).
3. Re-baseline; every earlier number is re-labelled with its protocol.
4. Sufficiency decision (not a prompt clause): refuse when the evidence does not
   support the asked attribution. Start from the existing, never-wired
   `recall/evidence_sufficiency_gate.py` — but its six hardcoded English patterns
   are a *baseline for the gate shape*, not a general mechanism (audit §1.1).

Metric: cat5 refusal recall up, cats 1-4 F1 down by less than 0.5 pt.

### Track B — Engine coverage (weeks 2-6)

One shape at a time, each measured with `integration_gate.py` on frozen contexts:

| order | engine | target cluster | entry condition |
|---|---|---|---|
| B0 | enumeration recall in the compiler (0 LLM) | 129 dropped cat1 items (4.6) | item recall up, token budget flat |
| B1 | plural/list union with speaker scope + completeness gate | 311 | precision on the cluster >= 55% |
| B2 | date binding verifier (anchor + event match, else abstain) | 104 | zero regression on already-correct dates |
| B3 | constrained single-value relation extraction | 288 | self-verification gate present |
| B4 | attribute resolver precision (deictic/pronoun filter) | 111 | abstain is allowed to be the answer |

These are extensions of existing engines, not new subsystems. `entity_identity.py`
(UUID5 identity) exists and has **no importers** — B1 is where it earns its place.

### Track C — Fair comparison (parallel, weeks 2-3)

Run the P2 arena with a real LLM, all five players, one scorer. Two deliverables:
a comparable table, and an `AM current` player so the comparison uses today's
engine rather than `AMv020Player`.

---

## 7. Guards (non-negotiable)

1. **Dev / holdout.** 553 questions (convs 26, 30, 41, 42, 44, 48) and conv-48's
   191 cats-1-4 questions stay unspent until a method is frozen. One shot.
2. **Integration gate.** FIXED / BROKE / SPURIOUS, matched count asserted, scorer
   generation named on both sides. No bare "accuracy went up".
3. **Synthetic tests.** New engines are tested on `tests/unit/synthetic` shapes
   before any benchmark contact.
4. **No benchmark vocabulary.** The ontology leak guard (`tests/unit/hygiene`)
   stays green; no question-id, topic or speaker-name branches.
5. **Cross-benchmark.** Any engine that helps LoCoMo must not regress LME dev250
   (`lme_dev250_v4_R7.json`, 88.0% — dev, not holdout).
6. **One change, one measurement.** Bundles are not measurable and will be split.

---

## 8. Milestones and targets (official F1, 1,986 questions)

| # | milestone | gate | target |
|---|---|---|---|
| M0 | protocol v2 + canonical refusal + re-baseline | refusal form only | **49.9%** — DONE (binary 54.83%, official 49.91%) |
| M1 | deterministic attribution gate | dev delta >= 0 | **REJECTED** (-1.11 to -5.34pt; see 4.5) |
| M2 | enumeration recall for list questions (0 LLM) | item recall up, cats2-4 non-regression | measured: +1.68pt dev for 2x tokens — **not promoted** (4.8) |
| M3 | construct the list answer from evidence (union engine) | precision >= 55% on fully-available items | +25-42 cat1 questions answerable in-context |
| M4 | Reader-side sufficiency verifier (one extra call) | cats1-4 loss < 0.5 pt | cat5 75%+, overall 52-54% |
| M5 | date binding (temporal 59.9% -> 66%+) | zero date regressions | 57-59% |
| M6 | single-value extraction (single-hop 52% -> 58%+) | precision >= 55% | 60-63% |
| M7 | holdout validation, once | frozen method | report honestly |

"World-class" is only claimed after Track C produces a same-LLM, same-scorer
comparison and after M5. Until then the honest statement is: *the engine is 3.65 pt
below its own v0.3.0 artifact on official F1, and 12.50 pt of that gap class is a
pure protocol defect.*

---

## 9. What would falsify this program

- The sufficiency gate cannot separate cat5 from answerable questions at better
  than 60% precision (then refusal stays a format fix and nothing more).
  **Partially observed:** all three surface-string gates are net-negative (§4.5),
  so the remaining test is the Reader-side verifier, not another string rule.
- Plural/list union lands below 55% precision on its cluster (the 14%-precision
  generic branch is the precedent).
- Dev gains do not survive the 553-question holdout (then the engines were fitted).
- P2 shows the memory strategy is not the differentiator at all, in which case the
  program pivots to reader and context quality rather than engine count.
