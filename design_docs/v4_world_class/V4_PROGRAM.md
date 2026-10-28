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

### 3.5 Artifacts outside this log: the R8.1 dev-250 run

`benchmark_results/pipeline_v4/lme_dev250_v4_R8.1.json` reports 250/250 (100%) on
LongMemEval **dev-250** and is the artifact behind the tag
`v4-r8.1-perfect-score`. It is not part of the measured program above and must
not be cited as a benchmark result:

- it is scored on **dev**, the split used for development, not on holdout;
- **56 of 250 answers (22.4%)** are deterministic overrides, so the run measures
  the override table as much as the reader;
- the program's own dev-250 reference is `lme_dev250_v4_R7.json` (88.0%), which is
  the number the cross-benchmark guard in §7 uses;
- 100% on all six question types of a 250-question dev slice is a saturation
  signal, not a capability signal. A score at the ceiling on tuned data carries
  no information about transfer.

Standing rule: dev saturation is a reason to stop tuning on that split, not a
result to publish. The tag stays for provenance; the claim does not.

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

Full set (1,986, reported not tuned), through the shipped module: 49.91% ->
**53.77% (+3.87pt)**. The cost is concentrated in single-hop (-2.9) and temporal
(-1.4); the gain is +24.9 on the adversarial category.

Guards that make it safe, each of which was a measured failure when absent:
date-shaped answers are out of scope (enforcing provenance on dates cost 7.8pt on
temporal); questions naming no single speaker, or naming both, are out of scope;
an answer whose wording is not traceable in the context is out of scope.

Shipped as `src/artificial_memory/skills/refusal_gate.py` (12 synthetic tests),
opt-in via `--refusal-gate`, and verified to reproduce the measurement **exactly**
(+4.331pt) through the shipped module rather than a copy of its logic.

**Further variants were tried and all were worse** — the gate is at a dev-local
optimum, and the search stops here rather than becoming threshold fitting:

| variant | rule added | dev delta |
|---|---|---:|
| **plain (shipped)** | — | **+4.33** |
| margin | the subject's own overlap must be < 0.7 x threshold | +3.68 |
| solo | the question must name exactly one person | +3.46 |
| actor | the subject must be the grammatical actor | +2.97 |
| any | provenance OR "answer not in the evidence at all" | +2.39 |
| untraceable | answer's wording appears nowhere in the evidence | -0.44 |

The trade-off is genuine rather than a tuning miss: every guard that protects
single-hop (-2.6pt of cost) removes more adversarial recall (+25.5pt of gain)
than it saves. `untraceable` is not even a hallucination detector in practice —
it mostly fires on answers that were already refusals.

A final parameter was swept on the merged full run (committer off): requiring the
answer to carry more content terms before the gate may fire. Minimum terms 1 / 2 /
3 give 54.94% / 54.54% / 53.02%, because the single-hop false refusals are
disproportionately one-term answers but so are the adversarial catches. The
shipped default stays at 1.

### 4.10 Cross-benchmark guard: LongMemEval, measured in both modes

The program requires that anything helping LoCoMo does not regress LongMemEval.
The LME dev contexts were compiled for the purpose (250 questions, 0.81 s each,
100% oracle recall, mean 6,932 tokens), then joined with the stored R7 dev run.

The first version of this guard was **vacuous**: the gate's parser required a
capitalised name, and LME renders `[session date] user: ...`, so it read 0 of 128
contexts. A guard that cannot run is not a pass. The parser now accepts role
labels, and first-person questions can be attributed to the user under an
explicit switch (`allow_roles`), which makes the gate applicable to agent memory:

| mode | contexts parsed | fires | LME accuracy | delta |
|---|---:|---:|---:|---:|
| **name-only (shipped default)** | 128/128 | 0 | 82.03% -> 82.03% | +0.00 |
| roles | 128/128 | 2 | 82.03% -> 80.47% | **-1.56** |

**Verdict: the shipped configuration passes; role mode is rejected as a default.**
The assistant legitimately restates the user's facts, so "the answer's wording
comes from the other role" is not evidence of misattribution in agent memory —
and the two questions it refused were correct. The guard is now a real measured
negative rather than a vacuous pass, and the shipped gate stays name-only, where
its provenance premise actually holds: two people with distinct identities.

Two defects were found while building this guard, both fixed and tested:

- `DateArithmeticEngine.calculate_recency` subtracted a `datetime` from a `date`
  and raised TypeError, aborting the entire 250-question cache build on one
  question. Normalised at the arithmetic boundary; 3 regression tests.
- `str.splitlines()` splits on U+2028 / U+0085, which occur inside LongMemEval
  messages, cutting JSONL rows in half. The LME reader splits on `"\n"` only.

Guard artifact: `scripts/benchmarks/measure_refusal_gate_lme.py`
(`benchmark_results/refusal_gate/lme_guard.json`).


---

### 4.11 The Reader-side support verifier: a pilot's +4.8 is +0.45 at scale

The one refusal mechanism left untested was a second model call that audits the
answer against the same evidence. The pilot (n=40) projected +4.8pt. The full dev
run is decisive: 1,173 answered questions, one extra 7B call each (~2 s), and the
pinned official evaluator on the same frozen predictions.

| policy | refused | official F1 | delta | won (cat5) / lost (cat1-4) |
|---|---:|---:|---:|---:|
| **gate (0 LLM calls)** | 133 | **54.89%** | **+4.33** | 87 / 31 |
| verifier | 290 | 51.00% | +0.45 | 77 / 113 |
| gate OR verifier | 377 | 53.42% | +2.86 | 132 / 138 |
| gate AND verifier | 46 | 52.47% | +1.91 | 32 / 6 |
| verifier beyond the gate | 244 | 49.09% | -1.47 | 45 / 107 |

**REJECTED.** The verifier buys 77 adversarial refusals and pays 113 correct
answers across categories 1-4 — it refuses inference and derived-date questions
that the benchmark rewards. Combining it never beats the gate alone: the union is
worse (+2.86), and the intersection is not additive (+1.91).

The pilot's +4.8 was small-sample noise on 8 questions per category. This is the
clearest demonstration in the program of why a pilot is not a promotion, and it is
recorded for that reason. The deterministic gate remains the only refusal
mechanism: equal or better, at zero LLM cost.

Artifacts: `benchmark_results/refusal_gate/policies_dev.json`,
`scripts/benchmarks/price_refusal_policies.py`.

### 4.12 P2 arena: the shipped runtime is last, and Simple RAG wins

The controlled arena compares five players on 200 planted-fact questions over a
520-turn history, with one frozen answer LLM, one frozen extraction LLM, one
2,000-token budget and one scorer. This is the fair comparison objective (3)
asked for, and it is the first time it has been run with a real LLM.

| player | accuracy | write tokens | read tokens | abstention rate |
|---|---:|---:|---:|---:|
| **Simple RAG** | **51.0%** | 0 | 485k | 27.0% |
| Mem0-style | 42.5% | 61.6k | 387k | 26.5% |
| Zep-style | 35.0% | 61.6k | 591k | 42.5% |
| HippoRAG-style | 26.0% | 61.6k | 628k | 61.5% |
| **AM v0.2.0** | **11.5%** | 0 | 431k | **86.5%** |

Per category, AM v0.2.0 scores 0% on factual recall, multi-session recall,
knowledge updates, temporal reasoning, distractor resistance, contradiction
handling and compression recovery; 15% on reflection; and 100% on abstention —
the one category that rewards refusing. **Its 11.5% is almost entirely the
abstention category.**

Two findings matter more than the ranking:

1. **The shipped runtime answers only 13.5% of questions.** This is not a subtle
   retrieval gap; the runtime almost always declines. Whatever the R7 research
   pipeline does (53.77% official F1 on LoCoMo), it is not what ships.
2. **The LLM-extraction baselines spend 61.6k write tokens and lose to a player
   that spends zero.** Write-side cleverness did not pay for itself here.

Caveats, stated because they bound the claim: the arena is synthetic and plants
sentences like "For project atlas the user said their preferred editor toolchain
is ripgrep", which lexical retrieval can match; `AM v0.2.0` is the runtime facade,
not the R7 pipeline; and one run is one run.

**Why the runtime abstains — diagnosed at 0 LLM calls.** Replaying the full
440-turn user history through the runtime facade and recalling the first three
factual questions:

| question | retrieved | what the context actually contains |
|---|---|---|
| project atlas, editor toolchain | 20 memories | the right project, the **wrong attribute** ("keeps an untouched backup") |
| project beacon, editor toolchain | 20 memories | beacon does not appear at all |
| project cinder, editor toolchain | 20 memories | the right project, the **wrong attribute** |

The planted facts are template-identical ("For project X, the user ..."), so
dense similarity barely separates them: the query ranks same-project siblings
above the asked relation. Simple RAG uses the same k=20 and scores 51%, so the
loss is not the retrieval budget — it is how the runtime stores and ranks
memories. That is a product defect, and it is the concrete next target.

**A lexical channel would recover most of it (0 LLM calls).** IDF-weighted lexical
recall@20 over the same 440 user turns, asking only whether the gold value
appears in the top 20:

| category | lexical recall@20 | AM accuracy today |
|---|---:|---:|
| knowledge updates | 100% | 0% |
| compression recovery | 100% | 0% |
| temporal reasoning | 100% | 0% |
| abstention (must refuse) | 100% | 100% |
| factual recall | 85% | 0% |
| distractor resistance | 85% | 0% |
| multi-session recall | 80% | 0% |
| contradiction handling | 0% | 0% |
| indirect recall | 0% | 0% |
| reflection / reinterpretation | 0% | 15% |

The three categories at 0% need reasoning rather than retrieval, so they are a
separate problem. For the rest, the evidence is reachable by a lexical ranker that
the runtime does not have. Blending lexical and dense scores in the runtime's
recall is the next engine change, and it can be measured at 0 LLM calls before any
arena rerun.

### 4.13 The fix: an IDF-weighted lexical channel in the runtime (+44pt, paired)

Root cause, read from the code after the diagnosis: `BasicRecallEngine` built its
candidate pool from resolution buckets (40 episodes + 20 semantic, `limit=40`),
**independently of the query**, and ranked what it got with unweighted word-set
overlap. On a topic with 440 facts the asked fact was usually not a candidate at
all, and common words ("the user") outvoted the rare one ("atlas").

The change, behind `RuntimeConfig.lexical_recall` (default `False`, so the frozen
baseline is byte-identical):

- the candidate pool also includes the topic's active memories (capped);
- ranking adds an IDF-weighted overlap over stemmed terms, normalised to 0..1 and
  scaled by 2.0, so a shared rare term outweighs many shared common ones.

Measured, paired on the identical 50-question arena subset (`--limit 5`, same
questions, same LLM, same budget):

| configuration | accuracy | retrieval ms | read tokens |
|---|---:|---:|---:|
| lexical off (shipped default) | **10.0%** | 16.3 | 106k |
| lexical on | **54.0%** | 70.2 | 366k |

And at 0 LLM calls, gold-in-context over all 200 questions rose from **11% to
75%** (§4.12 table).

The cost is honest: 4x retrieval latency and 3.4x read tokens, because the context
now contains relevant evidence instead of 20 short irrelevant memories. That is
the mechanism working, not a regression.

**Full 200-question confirmation** (same arena, same LLM, one run each):

| player | accuracy | read tokens | retrieval ms |
|---|---:|---:|---:|
| **AM v0.2.0 with the lexical channel** | **53.0%** | 1,463k | 73.2 |
| Simple RAG | 51.0% | 485k | 9.7 |
| Mem0-style | 42.5% | 387k | 20.9 |
| Zep-style | 35.0% | 591k | 2.7 |
| HippoRAG-style | 26.0% | 628k | 25.4 |
| AM v0.2.0 as shipped | 11.5% | 431k | 17.7 |

Per category, the same change turns 0% into: factual recall 100%, multi-session
recall 100%, knowledge updates 85%, indirect recall 85%, distractor resistance
80%, compression recovery 70%. **Abstention falls from 100% to 10%** — the system
now answers instead of declining, which is the whole point and also the new
problem: it now answers questions that should be refused. That is the same
sufficiency question this program met on LoCoMo's adversarial category, and it is
the next target.

**Do not overclaim this.** The failure was diagnosed *on this arena* and the fix
was validated *on this arena*; one run each. §4.14 measures the same mechanism
out-of-sample on real dialogue, where the direction holds and the magnitude does
not. The honest statement is "the diagnosed mechanism is real and worth 41.5pt
here", not "the product is now world-class".

Tests: `tests/unit/test_lexical_recall.py` (5 cases, including one that asserts the
off path is unchanged). Flag: `--am-lexical` in the arena runner.

### 4.14 Out-of-sample: the lexical channel helps on real dialogue too (+16.7pt)

The arena result was in-sample for the mechanism. To test it on a corpus it was
not designed against, the SHIPPED runtime was run over a LoCoMo conversation
(conv-26: 419 turns, 60 questions), asking one question at 0 LLM calls: does the
recalled context contain the gold answer?

| channel | gold reached | mean tokens | retrieval ms |
|---|---:|---:|---:|
| dense-only | 0/60 (0.0%) | 1,072 | 14 |
| **lexical** | **10/60 (16.7%)** | 1,964 | 80 |

Direction generalises: dense-only reaches nothing on real dialogue, the lexical
channel reaches some. Magnitude does not: 16.7% here versus 75% on the synthetic
arena, because real dialogue paraphrases and spreads a fact across turns. The
runtime also remains far below the R7 pipeline's retrieval — that is the honest
gap between "the product" and "the research harness".

Artifact: `scripts/benchmarks/probe_runtime_recall_locomo.py`.

### 4.15 Date binding as a deterministic engine: rejected by its own ceiling

B2 was "bind the date anchor to the turn whose content matches the event". Its
ceiling was measured first, on the 318 temporal questions:

| measure | count |
|---|---:|
| gold anywhere in the compiled context | 199 / 318 (62.6%) |
| **gold in the single best event-matching turn** | **74 / 318 (23.3%)** |
| answered correctly by the current pipeline | 194 / 318 (61.0%) |

A deterministic single-turn binding would be **worse than doing nothing** (23%
ceiling against a 61% pipeline). The temporal failures are not "the anchor was
attached to the wrong turn"; they are multi-turn synthesis that the pipeline
already does better than any lexical rule. B2 is closed before any code was
written.

### 4.16 The runtime's "answers too much" is not a lexical-threshold problem

After the lexical fix the runtime answers instead of refusing, and the arena's
abstention category fell 100% -> 10%. A cheap gate was tested: use the best
lexical match between question and recalled memory as a sufficiency score. It
does **not** separate the populations — abstention questions sit at 0.67 (mean,
median, p10 and p90 all 0.67), answerable at 0.63 mean / 0.70 median. The
unanswerable questions are lexically just as close as the answerable ones; what
differs is whether the specific fact is stated, which is the same sufficiency
question LoCoMo raised and the same one a score threshold cannot answer.

### 4.17 The committer was costing points: it is now opt-in (+1.16pt)

The deterministic committer answers a question only when an engine claims it, and
the reader answers the rest. Its own claims were never A/B'd against the reader on
the current pipeline. They are now: the same 312 questions, the same frozen
contexts, the reader answering instead (312 model calls, one run), paired official
F1.

| | committer answers | reader answers | delta |
|---|---:|---:|---:|
| all 312 claimed | 51.30% | **61.61%** | **+10.31pt** |
| multi-hop (36) | | | -5.20 |
| temporal (155) | | | -2.91 |
| open-domain (3) | | | -33.33 (n=3, noise) |
| single-hop (90) | | | **+23.95** |
| adversarial (28) | | | **+64.29** |

The committer is better on multi-hop and temporal and much worse on single-hop,
which dominates. On the full 1,986 the swap is **+1.62pt before the gate** (49.91
-> 51.53) and **+1.16pt with it** (53.77 -> **54.94%**).

The previous default rested on a 321-question cat2 probe where the committer agreed
with the reader on 75, uniquely solved 7 and lost 7 — a tie *on temporal only*.
Over all 312 claims, including single-hop, the reader wins by more than it loses.

Changed: `--committer` is opt-in (default off), `--no-committer` is a no-op kept
for compatibility, and every run manifest records `committer`. The committer's
provenance value is unchanged — it can still be enabled where a deterministic
answer trail is worth more than the points.

A middle policy was also priced: reader first, and the committer only where the
reader refused (36 questions were rescued that way). It measures **54.62%**, i.e.
worse than switching the committer off entirely (54.94%). The rescued answers cost
more than they earn, so the committer is off, not "off except when the reader
gives up".


**Consequence for the program:** the distance to "world-class" is not a
benchmark-recipe problem, it is a *product wiring* problem. The next engine
measurement must be "R7 retrieval + compile behind the arena player", so the
comparison prices the engine that actually wins on LoCoMo.

Artifacts: `benchmark/results/p2_external/controlled/controlled_20261024_234709/`.

**Holdout leak — recorded, not hidden.** While computing the full-set number, the
`--scope` filter was inverted, so one run measured ONLY the 499 held-out
questions (it showed +1.81pt). The filter is fixed. Consequence: for *this gate*
the holdout is no longer clean, so it can neither be tuned on nor cited as
validation. The design number is the dev figure (+4.33pt). The 553-question
M5-M13 holdout remains unspent and is a different slice.

### 4.18 A selective committer by engine source: observed, not promoted

If the committer only wins on some engines, maybe it should be enabled per source.
The 312 claims split by the engine that produced them, priced with official F1
against the reader on the same questions:

| source | n | committer | reader | reader - committer |
|---|---:|---:|---:|---:|
| temporal algebra | 94 | 69.6% | 69.4% | -0.2 |
| date span | 81 | 67.0% | 72.2% | +5.2 |
| event action resolver | 75 | 23.8% | 51.3% | +27.5 |
| set aggregator | 21 | 34.2% | 45.2% | +11.0 |
| entity attribute resolver | 16 | 15.1% | 59.6% | +44.5 |
| boolean verifier | 9 | **66.7%** | 48.9% | **-17.8** |
| single-hop fact | 7 | 14.3% | 57.1% | +42.9 |
| counting engine | 4 | **58.3%** | 0.0% | **-58.3** |
| relational traverser | 3 | 54.2% | 55.6% | +1.4 |
| disjunctive resolver | 2 | **100.0%** | 50.0% | **-50.0** |

The committer wins on three engines totalling **15 questions** (n = 9 / 4 / 2). A
selective policy would be worth about **+0.25pt** overall — inside the noise of a
15-question sample. Recorded, **not implemented**: promoting it would be fitting to
five correct answers.

### 4.19 List completion: the ceiling is +2.16pt and the cheap rule loses

The union engine was priced before being built, on dev only (316 list-gold
questions):

| variant | dev cohort F1 | delta | overall equivalent |
|---|---:|---:|---:|
| current | 49.19% | — | — |
| **oracle completion** (append every gold item present in the context) | **59.37%** | **+10.17** | **+2.16** |
| sibling completion (append other members of the same comma list, subject-only) | 44.59% | **-4.60** | -0.98 |

So a *perfect* item extractor is worth +2.16pt overall, and the cheap heuristic
loses 4.6pt because it adds wrong items faster than it adds right ones — the same
over-union failure the failure atlas warned about. The existing aggregation engine
scores 34.2% F1 on its own claims (§4.18).

**M4 stays closed**: the prize is +2.16pt, the naive rule is negative, and closing
the gap needs a materially better item extractor, not a union wrapper around the
current one.

### 4.20 Single-value extraction: a +9.15pt prize that no deterministic rule reaches

M7 was the last engine in the plan: extract a single value for open-domain and
single-hop questions. Its ceiling was measured first, on dev cats 3-4 (687
questions): if every question whose gold tokens are present in the compiled
context were answered exactly, how much would official F1 gain?

| measure | value |
|---|---:|
| dev cat3+4 questions | 687 |
| questions where the gold is present in context but the pipeline is wrong | 446 |
| current official F1 | 52.10% |
| oracle single-value extraction | **71.91%** |
| **ceiling** | **+19.81pt on the cohort = +9.15pt on dev overall** |

So the prize is real and large — the evidence is already in the context and the
reader fails to convert it. But the ceiling says nothing about whether a rule can
reach it, and the evidence that it cannot is already in this repository:

| extractor | F1 on its own claims | reader on the same questions |
|---|---:|---:|
| entity attribute resolver (16 claims) | 15.1% | 59.6% |
| single-hop fact resolver (7 claims) | 14.3% | 57.1% |
| set aggregator (21 claims) | 34.2% | 45.2% |
| event action resolver (75 claims) | 23.8% | 51.3% |

Every deterministic extractor is one third to one quarter of the reader on the
same questions. **M7 is closed**: the bottleneck is extraction capability, not a
missing rule, and a new deterministic extractor would have to beat a 52-60%
baseline that the existing ones miss by 40 points.

This is the program's central negative result, stated plainly: on this corpus,
deterministic memory engines can **veto** an answer (the provenance gate, +3.87pt)
or answer a **narrow declared shape**, but they cannot **extract** answers better
than a 7B reader that is already reading the same evidence.

### 4.21 The R7 chain priced in the arena: complementary with the lexical channel, and the union is quantified

§4.17 named the next measurement: "R7 retrieval + compile behind the arena
player".  `scripts/benchmarks/am_current_player.py` is that player: at ingest
it keeps the shared history as raw pipeline turns (0 write-LLM); at answer it
runs the R7 chain (planner -> convergent coverage -> candidate pool ->
fact-index recall -> graph bundles -> budgeted compiler) and answers from the
compiled context with the frozen shared answerer.

**One wiring defect was found and fixed on the way.**  The compiler's token
unit is `word_tokens_floored` (words x 4/3); the arena's shared contract is
the estimator `len(text) // 4`.  Measured on this dataset a 1400-unit budget
compiles to ~2,367 arena tokens -- every context violated the 2,000-token
contract without any check complaining.  `compile_r7_context` now
re-compiles with a measured correction (up to twice) and drops whole lines
as the last resort: **200/200 contexts at or under 2,000 arena tokens**
(mean 1,965), at a reachability cost of only -0.5pt.

**0-LLM gold-reachability, all 200 compiled contexts**
(`scripts/benchmarks/probe_am_current_ctx.py`, artifact
`benchmark_results/am_current/probe_ctx.json`):

| channel | answerable reach | indirect | reflection | temporal |
|---|---:|---:|---:|---:|
| R7 chain full budget | 124/180 (69%) | 20% | 30% | 70% |
| lexical top-k (re-implementation) | 120/180 (67%) | 0% | 0% | 100% |
| hybrid (R7@1200 + lexical top-30 fill) | 128/180 (71%) | 10% | 30% | 100% |

The two channels hold **opposite category profiles**.  The hybrid restores
temporal completely, but indirect stays near zero and the union gains only
+2pt at this level.

**End-to-end, one run each, shared frozen LLM**
(`benchmark/results/p2_external/controlled/controlled_20261025_135405` and
`..._140731`; the comparison table in 4.12/4.13 was re-used as stored):

| player (full 200) | accuracy |
|---|---:|
| Simple RAG | 51.0% |
| **AM hybrid** (R7@1200 + lexical top-30 fill) | **54.0%** |
| AM v0.2.0 with the lexical channel | 53.0% |
| **AM current** (R7 chain) | 46.5% |
| Mem0-style | 42.5% |
| Zep-style | 35.0% |
| HippoRAG-style | 26.0% |

Paired per question (200 common official questions,
`scripts/benchmarks/price_am_current.py`):

- hybrid vs lexical: 54.0 vs 53.0 (+1.0pt) -- both pass 77, lexical-only 29,
  hybrid-only 31, both fail 63;
- hybrid vs R7 chain: 54.0 vs 46.5 (+7.5pt).

**Where the wins live.**  The R7 chain solves the arena's abstention
collision: 18/20 (subset) and **19/20 with the hybrid** correct refusals,
against **2/20 for the lexical runtime** -- the category the runtime broke
when the lexical channel taught it to answer (4.13: abstention 100 -> 10%).
Compact deterministic-but-auditable context is itself an abstention engine.
The lexical side still owns indirect recall (16-17 of 20, R7 chain 0)
and distractor resistance.

**Why the union does not realise its ceiling.**  Oracle union of the two
stored runs is 131/200 = **65.5%**; the hybrid realizes 54.0%.  The mismatch
is concentrated where the lexical-only wins lived: indirect (16 lost) and
distractor (11 lost).  Diagnosis from the answers: the hybrid's context
contains decoy lines (e.g. `ctags`, `difftastic`, `Aarhus`) that the raw
top-30 fill admitted, and the reader picks them.  The runtime's own lexical
wins come from *its* recall stack (store + episodes + topic actives), not
from a raw top-k over raw turns -- so the honest next lever is to fill the
remainder with the runtime's actual recall rather than this
re-implementation.

**Two structural walls, shared by every player**: contradiction handling
0/20 and reflection 0/20 across all seven players, and temporal 0-2/20 in
every E2E despite ~70-100% reachability.  These are conversion failures
(the evidence is present, the reader does not synthesise), not retrieval
failures -- the arena's analogue of the M6 conversion wall on LoCoMo.

Artifacts: `benchmark_results/am_current/probe_ctx.json`,
`benchmark_results/am_current/probe_hybrid*.json`,
`scripts/benchmarks/am_current_player.py` (9 unit tests,
`tests/unit/test_am_current_player.py`),
`scripts/benchmarks/probe_hybrid_ctx.py`,
`scripts/benchmarks/price_am_current.py`.

**Standing after this session**: LoCoMo official F1 54.94% unchanged; the
arena now has two new measured rows -- the R7 chain (46.5%) and the hybrid
(54.0%, tied with the lexical runtime at +1.0pt paired) -- and the union
ceiling (65.5%) plus the two conversion walls are recorded as the next
targets.  "World-class" is still not claimed.

### 4.22 Hybrid v2 (the runtime's own recall stack as fill): A/B'd and rejected

Hypothesis from 4.21's diagnosis: the lexical runtime's unique wins (indirect,
distractor) come from *its* recall stack, not from a raw top-k, so a hybrid
whose fill forgoes the re-implementation and drives the same facade the
AM v0.2.0 player answers from should realise more of the 65.5% union ceiling.

`AMHybridPlayer` gained `hybrid_fill: "runtime"` (registered as "AM hybrid
v2"): at ingest the same deterministic facade (`RuntimeConfig.lexical_recall`)
remembers the user turns; at answer, after the R7 compile at budget A, the
runtime's level-2 recall output fills the remainder
(`fill_context_from_lines`).

**0-LLM reachability (200 contexts, `probe_hybrid_ctx.py`):**

| variant | answerable reach | reflection |
|---|---:|---:|
| R7 chain | 124/180 (69%) | 6/20 |
| lexical top-k re-implementation | 120-127/180 (67-71%) | 0-6/20 |
| hybrid, raw fill | 128/180 (71%) | 6/20 |
| **hybrid, runtime fill** | **142/180 (79%)** | **20/20** |

The highest reachability measured in this program -- and the wall shared by
all seven players (reflection) looked solved in reach terms.

**End-to-end, one run each (50-question paired subset):**

| player | accuracy |
|---|---:|
| hybrid, raw fill | **54.0%** |
| AM v0.2.0 lexical | 54.0% |
| **hybrid, runtime fill** | **44.0%** |

Paired: raw-fill wins 5 questions, runtime-fill wins **0**, same questions.
The conversion regression is diagnosed from the answers: the fill carries in
planted **near-miss distractor lines**, which the reader then answers with
(`abstention-00/01/02`: fluent wrong answers where the raw fill refuses; a
decoy tool where `delta` was asked).  The 0-LLM reachability metric could
not see this: it only checks gold-term presence, never *negative* evidence
(what the extra lines invite the reader to answer wrongly).

**A routing policy was priced before being built (0 LLM).**  "Refuse when
the R7 compile found no evidence, else take the hybrid's answer": of the 6
questions where R7 abstained and the hybrid answered, the hybrid was CORRECT
on 3 and the veto would fix 1.  Net negative -- **not implemented**.  The
hybrid already inherits the compile's refusal through its own context
(abstention 19/20 vs lexical 2/20), so the veto is redundant where it helps
and destructive where it fires.

**Recorded negatives:**
- `hybrid_fill: "runtime"` -- rejected as a default (44.0 vs 54.0 paired on
  the same questions).  Reachability is not conversion; a fill that raises
  reachability by carrying more lines can lower accuracy by moving decoys in.
- R7-abstention veto of a non-abstaining player -- priced at 0 LLM, net
  negative (3/6 correct answers broken, 1/20 abstentions fixed).

**Standing change: none.  AM hybrid with the raw top-30 fill remains the
arena leader at 54.0% (paired +1.0pt over the lexical runtime; oracle union
65.5%).**  The honest openings this measurement leaves:

1. The reachability metric must grow a *negative-evidence* dimension: does
   added context invite a wrong answer where the correct move is refusal.
   Reachability alone has now been shown to rank two fills in the opposite
   order of their E2E accuracy.
2. The two conversion walls (contradiction, reflection) are the only
   categories no player ever solves -- on the arena as on LoCoMo, the reader
   has the evidence and does not synthesise.

Artifacts: `benchmark_results/am_current/probe_hybrid_rt.json`,
run dirs `controlled_20261025_161439` (hybrid v2, 50q),
`controlled_20261025_140439`/`140731` (raw fill 50q/200q), priced with
`scripts/benchmarks/price_am_current.py`.

### 4.23 The negative-evidence metric: implemented, and its first result is a limit, not a win

Opening 4.22 left: the reachability metric could rank two fills in the
opposite order of their E2E accuracy.  The fix attempted here is the
cheapest possible two-sided metric: every arena question already carries a
`forbidden` list (the decoy vocabulary the official scorer itself uses), so
the negative-evidence dimension is "does a forbidden term appear in the
context" (`forb_hit` state in `probe_hybrid_ctx.py`) -- no new vocabulary,
no leakage, measurement only.

Validated expectation: `forb_hit` on the planted abstention temptation
should be high exactly where the E2E refusals break, separating the fills
the probe previously ranked by gold-reach alone.

**Measured (200 compiled contexts, artifacts
`benchmark_results/am_current/probe_hybrid_metric.json` and `probe_ctx.json`):**

| variant | abstention temptation (forb-hit) | E2E abstention correct |
|---|---:|---:|
| R7 chain (compile) | 20/20 | 20/20 |
| hybrid, raw fill | 20/20 | 19/20 |
| hybrid, runtime fill | 15/20 | ~1/5 (subset) |
| lexical runtime | 20/20 | 2/20 |

**The hypothesis is falsified: forbidden-term presence does NOT separate
the fills' refusal behaviour.**  R7 and the raw-fill hybrid carry the
temptation in every abstention context and still refuse almost always,
while the runtime fill carries it *less* and breaks.  The measured
differentiator is the **line shape**, not the term presence: raw turns keep
the near-miss assertion embedded inside a conversational utterance ("for
project aurora, ... in session-x"), whereas the runtime's semantic-content
lines are already fact-shaped sentences that read *directly* as an answer
to the question's form -- a decoy that arrives pre-formatted for the reader
to adopt.

Recorded honestly, this is the metric's limit: `gold-reach x forb-hit`
explains necessary conditions (evidence present / decoy present) but not
conversion.  Conversion of BOTH directions (answering and refusing) runs
through phrasing fidelity, consistent with every conversion-wall result in
this program (M6, 4.11, 4.20, 4.21).  For future measurement the report
must carry both sides, but the E2E stays the arbiter of any player change.

**Standing change: none.**  AM hybrid with the raw top-30 fill stays the
arena leader (54.0%).  No player change is justified by this measurement --
which is itself the disciplined outcome the program requires: a metric was
extended, its prediction failed, the extension is kept as a reported
dimension, and nothing was tuned against the failed prediction.

Artifacts: `benchmark_results/am_current/probe_hybrid_metric.json`,
`probe_hybrid_ctx.py` (forb dimension added).

### 4.24 Conflict surfacing: a priced, two-iteration measurement, and a new arena leader

**Diagnosis correction.**  The §4.22 statement that contradiction_handling
is a *conversion* wall is wrong: the probe metric shows the category is a
**reach wall of a new kind** — gold group 0 (the two planted values) is in
every compiled context, but gold group 1 (a conflict marker word) NEVER
reaches any context of any player, because nothing in anyone's pipeline
surfaces that the two reports conflict.  The reader never states a conflict
it was never shown; with a single-value or refusal answer it scores
partial/fail (§10 covers group 1 in the answer text, and it never appears).

**Priced before being built (0 LLM,
`scripts/benchmarks/price_conflict_detector.py`,
`benchmark_results/am_current/conflict_pricing.json`).**  Hypothesis: a
generic deterministic detector — two turns asserting different values for
the same (entity, property) pair, extracted with a fixed relational
lexicon — can surface the conflict from the turns alone.  Measured: the
detector captures the planted pair under one key **20/20**; the resulting
annotation would fire on **0/180** questions of the other nine categories,
so the added context text can reach only questions whose correct answer is
"both, and they conflict".  The scorer's `CONFLICT_MARKERS` vocabulary is
read nowhere in player code (Leakage Boundary); the annotation is built
only from the detected turns.

**Iteration 1 (measured, one 50-question run): wording failed.**  The
annotation `...: SQLite vs Redis; which is correct is unverified.` reached
the reader, but its phrasing steered the frozen reader into a refusal:
contradiction stayed **0/5** (answers: "I don't know"), overall 54.0%
(paired delta 0 vs the leader).  Recorded as a negative: an annotation
whose content says "unresolved" converts to abstention, not to the
dual report the category semantics call for.

**Iteration 2 (measured): the phrasing states the semantics.**
`NOTE: conflicting reports for project P (prop): V1 (day d) vs V2 (day
d). Both are on record and neither is confirmed; report both values
rather than choosing one.`

End-to-end, one run each:

| run | AM hybrid (raw fill) | AM conflict | paired |
|---|---:|---:|---|
| 50 questions | 54.0% | **62.0%** | contradiction 0/5 → 4/5, zero exclusive losses |
| 200 questions | 54.0% | **60.0%** | **12 wins / 0 losses**, contradiction 0/20 → 12/20 |

No other category moved (abstention stays 19/20 — the annotation never
fires there).  Residual 8/20 diagnosed from answers: the reader still
turns the note into a refusal on 6 and emits "could be either X or Y"
without a conflict marker on 2 (group 1 miss → partial).  Reflection
remains the only untouched wall (0/20 for every player).

**Standing change: AM conflict is the new arena leader at 60.0%**
(full-200, +6pt paired over the hybrid raw leader; honest caveats: one
run each on the frozen qwen2.5:7b reader; the mechanism lives in the
benchmark player, not the shipped runtime — porting the conflict
surfacing (and the compile-style refusal) into
`artificial_memory` runtime is the next product-side opening; the
remaining 8 contradiction conversions and reflection are the next
measured targets).

Artifacts: `benchmark_results/am_current/conflict_pricing.json`;
run dirs `controlled_20261025_164953` (full-200),
`controlled_20261025_164727`/`164412` (50q iter-2/iter-1), priced with
`price_am_current.py` against `controlled_20261025_140731`/`140439`.

### 4.25 M14 — Runtime Port & Parity: the shipped runtime can now surface conflicts

Opening 4.24 left: the M13 mechanism lived in the benchmark player only.
Before touching anything, a full read-path survey of
`src/artificial_memory` established three facts: the actual reader context
is assembled by `EnhancedContextBuilder` (`context/builder.py:116-247`)
from current-state + per-level recall memories; the shipped runtime has
**zero conflict logic anywhere in the read path** (`ConflictStateManager`
exists at `core/ir/conflict_state.py` but is only exercised by research
benchmarks, and its latest-wins `active_candidate` pick would have to be
replaced, not reused, to keep both values); insertions inside the recall
engine or `LLMManager` carry the highest blast radius (frozen-semantics
suites pin the engine; the deterministic gate consumes the context
blob).

**Design: parity by construction, single source of truth.**  The canonical
implementation now lives in the product,
`src/artificial_memory/recall/conflict_surface.py`
(`index_assertions` / `detect_conflicts` / `note_line` /
`query_gated_keys` — generic relational extractor over
`(content, provenance)` documents), and the benchmark player
(`am_current_player.py`) is a *thin adapter* over it.  Any future edit to
the mechanism lands in both worlds with one commit.

**Product wiring (opt-in, default off).**  `RuntimeConfig` gained
`conflict_surfacing: bool = False`; `EnhancedContextBuilder` (constructor
flag) collects candidate memories during `_build_context_parts`, detects
conflicting `(entity, property)` keys, and appends one provenance-bearing
NOTE `ContextPart` (HIGH tier, priority 0.9 — the measured behaviour
influenced this choice: the annotation must survive budget selection to
convert anything) **only when the query names both key tokens** — the
same 0/180 false-fire gate the arena pricing measured.  No value is ever
merged or deduplicated on the write path; supersession and temporal
validity stay untouched (the gate operates on the already-validity-filtered
candidate pool).

**Provenance parity note (bootstrap honesty).**  one measurement trap was
caught during the port: the refactor initially drifted the note surface
from `SQLite (day 45)` to `SQLite (45)` — the M13-measured string.  The
provenance label is now caller-owned (`"day 45"` in the benchmark, an
ISO date in the runtime) and the M13-measured surface is pinned by test,
so unit-green never silently means text-drifted again.

**Tests (24 new/updated, 109 regression passed).**
`tests/unit/test_conflict_surface.py`: both values keyed with provenance;
plan-21 note semantics; query gate inert without entity+property; runtime
builder fires only when flagged and stays frozen otherwise; annotation
inert on unrelated queries; write path keeps both memories; config default
off.  `tests/unit/test_am_current_player.py`: AST enforcement that the
player can never import the scorer's marker vocabulary (the Leakage
Boundary is now mechanical, not just a comment) + the M13 surface pin.
Regression: `test_core`, `test_phase2`, `test_deepseek_raid`,
`test_llm_deterministic_gate`, `test_refusal_gate` — all green.

**Standing change: none to any measured number.  New standing: the shipped
runtime can express conflict surfacing behind an explicit config flag with
zero behavioural change by default.**  What is deliberately NOT claimed:
a product-side E2E arena run of the flag-on runtime as a *player* (the
torchlit question of whether the product's own read path converts like
the player's compile does — the facade path differs from the R7 compile,
and that parity gap is the next measured opening), and the residual-8
diagnosis of 4.24.  Both stay honestly open.

Next measured steps (unchanged from the session plan): (a) residual-8
triage (6 reader-refusals vs 2 either-or-phrasings) — price any wording
change at 0 LLM before a reader run; (b) flag-on product player E2E
(parity measurement, not a standing change until paired); (c) reflection
0/20 remains untouched as its own investigation.

### 4.26 Phase-4 triage (0 LLM): the reflection wall is an abstraction step, not evidence loss

Per the M14 plan, reflection was investigated independently of the
contradiction fix — never assumed to be the same problem.  Answer autopsy
of all 20 reflection_reinterpretation questions in the full-200 runs
(both `controlled_20261025_140731` and `controlled_20261025_164953`):

- **Every one of the 20 answers asserts a concrete value** ("The user
  consistently shows a preference for the ClickHouse database for project
  atlas"), with coverage 0/1 in all 20.  Zero refusals — this wall is NOT
  the abstention-style refusal wall seen elsewhere.
- The plot asks to reinterpret three concrete observations ("reverts the
  build configuration", "copies files by hand", "keeps a backup") into
  the *abstract trait* they jointly evidence — the frozen gold marker is
  exactly such a trait word (`stability` / `minimal` / `measurement`,
  arena.py `THEME_MARKERS`).  The reader reports the most salient
  concrete fact instead of the conclusion the observations support.
- Verdict: **evidence is reached, the evaluation condition is standard
  coverage, and the missing capability is interpretive generalisation**
  (concrete observations -> named trait).  It is therefore NOT the same
  problem as the contradiction surfacing fix (which supplied *absent
  information*; here the *conclusion* must be derived, not carried in).
- Overfitting guard: a deterministic observation->trait lexicon would be
  hard-coding the scorer's own vocabulary (the literal trait words live
  in the gold markers) — rejected without measurement, per program
  guards.  The honest priced lever is reader-side: a system-prompt
  instruction to name the recurring pattern at the abstraction level
  ("one change one measurement" on the frozen prompt surface), and/or
  exposing the three observations adjacent to each other in the compile,
  which 0-LLM probe budget checks can price first.

### 4.27 Flag-on product E2E: the transfer was priced through three real gaps, then measured

Opening 4.25 left: is the port's flag-on product read path, without the
R7 compile, actually converting?  Answering that honestly cost three
*measured* 0-LLM pricing iterations
(`scripts/benchmarks/price_runtime_conflict.py`, 200 contexts each run),
every one of which exposed a real research-vs-product boundary the unit
tests could not see:

1. **Budget race (measured: 0/320 notes).**  With a 400+-memory
   candidate pool the annotation was appended LAST and lost the HIGH-tier
   allocation race.  Fix measured into `EnhancedContextBuilder`: the note
   is detected over current-state memories and its tokens are reserved
   *before* any part allocation.  Regression-tested against a crowded
   pool and a 1200-token budget.
2. **One-sided ingest (measured: 0 notes again).**  The AM v0.2.0 ingest
   protocol remembers user turns only; in every planted conflict the
   second report is an *assistant* turn ("a teammate reported..."), so
   the product store never contained a contradicting key.  The flag
   player now ingests both speakers — as the product's own conversation
   compiler does.
3. **The product fetch window (measured: 0 notes again).**
   `LLMManager.chat` shapes its context from
   `store.get_memories(limit=100)` — the most-recently-updated 100 rows.
   With ~880 memories the older planted pairs sat outside the window.
   The flag player pages the active corpus for detection; the paired
   baseline pipes identically, so the delta isolates the flag.

**Pricing end-state (0 LLM, 200 contexts):** note fires **20/20** on
contradiction contexts and **0/180** on the other nine categories —
identical to the benchmark-side pricing.  No collateral fire exists on
the product path either.

**Product read path, E2E (one run, 50 shared questions, same conditions):**

| player | accuracy | contradiction |
|---|---:|---:|
| AM runtime (flag off) | 40.0% | 0/5 |
| AM runtime conflict (flag on) | **42.0%** | **1/5** |

Paired: flag-on wins 1 (contradiction-00: "you should assume the system
runs on both Redis and SQLite" — 2/2 coverage), flag-off wins 0; every
non-contradiction category unchanged.  Honest diagnosis of the residual:
the reader still folds the note into a refusal on 4/5 — the difference
from the benchmark player's 4/5 conversion is a *context composition*
gap: the R7 compile surrounds the note with the raw planted turn lines
(evidence in the reader's own register), while the product builder's
current-state/recall mix fills its allocation with compressed,
unrelated-by-query state lines (measured mean ctx ≈1260 tokens).  The
flag reached the reader in both worlds; the reader's company differs.

**Standing change: none.  `RuntimeConfig.conflict_surfacing` stays
opt-in (default off).**  What is now MEASURED on the product path: the
flag converts at least one contradiction question with zero exclusive
losses, collateral is bounded by construction (pricing 0/180), and the
parity frontier has moved from "does the runtime have the mechanism" to
"name the context-composition differences that bind the reader's
report-obligation".  Full-200 E2E was skipped deliberately: pricing
already bounds the collateral at 200-scale, so more reader runs would
only re-measure the known path gap, not test the flag.

Artifacts: `benchmark_results/am_current/runtime_conflict_pricing.json`;
runs `controlled_20261025_175540` (paired 50q), plus two honestly-recorded
junk runs (`..._174830`, `..._175229`) whose values were overwritten by a
player-name collision (`AMv020Player` hard-codes its name; the flag arms
now name themselves from the flag) — do not cite their csv rows.

### 4.28 M15.1 composition audit: pair-reach was measured, fixed, and proven NOT the binding constraint

Phase 1+2 (read-only, 0 LLM, `scripts/benchmarks/
audit_context_composition.py`).  Baseline artifact
`context_composition_audit.json` saved *before* any change (copy:
`..._baseline_noteonly.json`).  Stages per contradiction question:
store → per-recall-level → allocation → final selection.

- The planted pair is in the store 20/20, and the recall engine finds
  evidence at every level 20/20 — but the *allocated parts contain zero
  recall:* parts.  Cause, measured: the builder dedupes recall results
  that are already current-pool members, and every planted line is a
  current member, so recall can never resurface them.
- Allocation is a recency window: ~57 of ~520 current lines win an
  `ORDER BY updated_at DESC` race.  Which planted lines make it is a
  question-ordering artifact, not relevance: for the E2E slice
  (contradiction-00..04 = earliest-scenario pairs) 0–1 lines survived;
  for 05..19 (recently ingested scenarios) both did — that is where the
  baseline's 15/20 "co-presence" came from.  The window has no unique
  tie-break, so pair appearance is a run-to-run lottery at the boundary.
- The NOTE itself fired 20/20 and carried both values inside its chain
  in every context.  Mean final ctx ≈1308 tokens.

Phase 3 — the ONE change chosen on this evidence: **the conflict part
carries its source assertion lines** (one immutable reserved HIGH part
per key: NOTE + raw planted lines with provenance labels).  Support:
`conflict_entries()` added to the canonical module (first document per
distinct value, with indices); `EnhancedContextBuilder._append_conflict_parts`
composes note + lines.  A query-relevance re-sort of the current pool
was considered and REJECTED: it reaches non-flag behavior and re-risks
the frozen semantics.

Measured at 0 LLM, same metric basis (both values on non-note lines +
NOTE present):

| | baseline (note-only) | note carries lines |
|---|---:|---:|
| co-presence | 15/20 | **20/20** |
| both values at allocation | 19/20 | **20/20** |
| note firing (contradiction / other) | 20/20, 0/180 | unchanged |
| mean final ctx tokens | 1307.7 | 1283.1 |

Phase 4 — minimal paired product E2E (one run, 50q, same conditions):
AM runtime 40.0% → AM runtime conflict 42.0%, identical to 4.27 — the
same single conversion (contradiction-00), zero exclusive losses
(`controlled_20261025_180855`).  0-LLM verification at the player's real
budget shows the reader SAW the note plus both raw assertion lines with
labels on 01..04 — and still answered "I don't know".

**Conclusion, stated without hedging:** the composition gap named in
4.27 ("the pair lines are missing from the product context") was real,
was fixed, and was NOT the binding constraint on the product reader.
M15.1 achieved its structural goal and honestly delivers no reader gain
beyond the known +2pt.  The remaining explain-candidates, each a future
1-change-1-measurement: (a) provenance distinctness — the product labels
both values "2026-10-25" (single-day ingest), while the benchmark's
"day 45" vs "day 48" reads as two separate reports; (b) ≈1100 tokens of
unrelated recency noise surrounding the annotation; (c) reader register.
Bench 4/5 and product 1/5 stay different-path results, never equated.
`RuntimeConfig.conflict_surfacing` remains default-off.

Tests: 2 new (`conflict_entries` mapping; composed part test), 27 unit
green.  Full-suite run 1093 passed / 11 skipped; 2 failures triaged —
`test_create_all_players` pinned an exact factory count and was updated
to the real contract (baseline five present + unique names), and
`test_temporal_algebra_duration_years_now` fails on committed code
untouched by this work (wall-clock-dependent; recorded OPEN).

Artifacts: `context_composition_audit.json` (post-change) and
`..._baseline_noteonly.json` (frozen baseline);
`runtime_conflict_pricing.json` (re-run, unchanged firing profile).

### 4.29 M15.2 provenance distinguishability: measured, and the result is NULL

Phase 1 (0-LLM investigation).  The benchmark's "day 45" style labels
come from the dataset's OWN data (`ArenaTurn.day` → a fixed epoch, fully
deterministic — nothing fabricated).  The product write path threw that
away: `facade.remember` hard-stamped `valid_from=now()`, so every label
the builder prints — NOTE chain and source lines — collapsed to the
single ingest date "2026-10-25", identical for both reports.  Storage is
a pass-through (`create_memory` saves `valid_from` as provided), and the
builder label derives from `valid_from`, so a write-side change makes
the provenance distinct with NO builder change.  Recall's time decay
reads `updated_at`, which stays the ingest time — measured, not assumed.

Phase 2 (the ONE change).  `facade.remember(..., occurred_at=None)`
(caller-given occurrence time; default None keeps every existing call
byte-identical), plus a `provenance_days` ingest knob on the flag player
passing the scenario's own timestamps; the arm is named
"AM runtime conflict dt" (flag-off guard verified: dt alone never leaks
into the baseline arm).  0-LLM pricing compared conflict vs dt over 200
real product contexts (`price_provenance_dt.py`):
NOTE firing 20/20 & 0/180 — unchanged; co-presence 20/20 — unchanged;
mean ctx 1299/max 1303 tokens — byte-equal; labels distinct **20/20**
(control 0/20, e.g. "Redis (2026-02-22) vs SQLite (2026-02-19)" — two
data-backed time points).  A probe-side metric bug (entity regex
expecting one word) initially reported 0/20 twice; fixed and re-run —
the first two probe outputs are diagnostic trash, the artifact is the
priced end-state.

Phase 3 (minimal paired reader run, one run, 50q, three arms):
`controlled_20261025_182640` —

| arm | acc% | contradiction |
|---|---:|---:|
| AM runtime (off) | 40.0 | 0/5 |
| AM runtime conflict | 42.0 | 1/5 |
| AM runtime conflict dt | 42.0 | 1/5 (the SAME question) |

Paired conflict vs dt: +0.0pt, 21 both-pass, 0 exclusive either way.
Provenance distinguishability is measured complete and the honest
result is **NULL**: real, distinct, deterministic dates did not move the
product reader.  Adopted without further changes (standing rule).

Cumulative M15.1+M15.2 conclusion: the reader converts exactly ONE
contradiction question (contradiction-00, straight off the NOTE) and
nothing about pair reach, co-presence, or provenance distinctness
changes that.  The remaining explain-candidate is now the only one left
from the audit's list: **the ≈1100 tokens of query-unrelated recency
fill surrounding the annotation** (the benchmark compile presents the
note inside a compact raw-turn register).  Next 1-change-1-measurement
(M15.3 candidate): shrink/condition the unrelated fill when a conflict
part is reserved, flag-gated, priced at 0 LLM first.  Bench 4/5 and
product 1/5 stay different-path results.  `conflict_surfacing` and
`provenance_days` stay default-off.

Tests: 2 new unit tests (occurred_at passthrough; provenance knob +
naming guard), 29 unit green.  Artifacts:
`provenance_dt_pricing.json`; run `controlled_20261025_182640`
(citable), plus the two diagnostic probe prints (non-citable, metric
bug before fix).

### 4.30 M15.3 fill-mass ablation: measured, and the result is INVERTED

Phase 1 (0-LLM, `probe_fill_mass.py`, baseline artifact
`fill_mass_baseline.json`).  The unrelated recency fill is **92%** of a
~1270-token contradiction context (~1168 tokens, ~57 lines, newest
first); the reserved conflict part is ~48 tokens and lands at ~93%
position, at the tail of the wall — not "near the start", which is what
one would design for attention.

Phase 2 (the ONE change).  `conflict_fill_ablation` (default off,
frozen-guarded): when the annotation fires, the current-state fill is
skipped entirely — the context is the conflict part only.  0-LLM
comparability `price_fill_ablation.py`
(`fill_ablation_pricing.json`): the reserved part (NOTE + both source
lines + provenance) is **byte-identical** to the control arm 20/20;
co-presence 20/20; and all **180 non-fired contexts are byte-identical**
— the ablation cannot touch anything beyond contradiction questions,
by construction and by measurement.  Fired contexts drop
~1270 → 92–95 tokens.

Phase 3 (minimal paired reader run, 50q, three arms,
`controlled_20261025_190445`):

| arm | acc% | contradiction |
|---|---:|---:|
| AM runtime (off) | 40.0 | 0/5 |
| AM runtime conflict (fill present) | 42.0 | 1/5 |
| AM runtime conflict bare (fill removed) | **40.0** | **0/5** |

Paired conflict vs bare: the fill removal **LOST the one conversion**
(contradiction-00 answered "I don't know" with the note+evidence alone)
and regressed nothing else honestly measured.  Stated plainly: the fill
was not burying the annotation — it was a **necessary condition** for
the single conversion the product reader makes.

**Cumulative M15.1–M15.3 measurement story:** three independent,
0-LLM-priced, builder-chained changes (pair evidence co-present 20/20;
provenance distinct 20/20; noise fill removed entirely) leave the
product reader converting exactly the same single question — and the
removal experiment shows the conversion depends on the surrounding
fill's PRESENCE, not on the annotation's information content.  The
hypothesis axis "the reader is information-starved or buried" is now
measured against; what remains is a reader-behavior axis: with the same
literal note text, the reader converts when ~1150 tokens of unrelated
memory lines accompany it, and refuses when the annotation stands
almost alone.  The next experiments are therefore OUTSIDE the builder
composition: (a) reader-facing instruction/register (one prompt-word
change at a time), or (b) pivot to the bench side (residual-8 triage,
reflection abstraction experiment), where the reader demonstrably
converts 4/5 under a different composition.  The `bare` knob is
measured NEGATIVE and stays default-off (recorded in §5: do not
enable).

Tests: 1 new unit test (ablation: fired-context structure, non-fired
byte-identity, frozen off-arm), 30 unit green.  Artifacts:
`fill_mass_baseline.json`, `fill_ablation_pricing.json`; run
`controlled_20261025_190445` (citable).

## 5. What is already measured as not working (do not re-spend)

| direction | result | mission |
|---|---|---|
| more candidates | top-50 found +1 gold | M5 |
| more context | net 0, +21% tokens | M6 |
| PPR over evidence graph | worse than the current retriever | spike |
| ordering rules | every rule rescues the same 6 | M17 |
| prompt elaboration (`count`/`evidence`/`attribute`) | 3 consecutive failures | commits `6ec219f` etc. |
| abstention prompt clause | 0 gain / 2 worse (n=40) | `6ec219f` |
| learned listwise re-ranker (`qwen2.5:7b`, local Ollama) | 10 -> 9 correct on 18 dev cases, context -11% | 2026-10-28, `--rerank ollama` |

The last row used n=40 and `p=0.48`; it is a **weak** negative, and it tested a
*prompt* intervention. It does not test a deterministic decision.

The re-ranker row is a **weak** negative too, and for a different reason: it is a
*learned* model, which is what makes it worth recording next to the rule-based
rows, but n=18 with 1 fixed / 2 broken cannot separate it from noise. What it
does establish: (a) a learned relevance model over the same 45-candidate pool
does not by itself fix attribution — one break was `lc_conv-43_178`, where the
reader answered John's team for Tim's question after re-ranking had refused
correctly before; (b) re-ranking shrinks the selected context by ~11%, which is a
real mechanism if a future model orders better. Both arms were re-run and
reproduced identically (temperature 0, seed 42), so the -1 is a deterministic
effect on these 18 cases rather than sampling noise — it is still 18 cases, and
the single unparseable batch fell on `lc_conv-43_3`, which failed in both arms
and did not move, so no break is attributable to a parse failure. The
cross-encoder backend (the instrument that would test this properly) is
implemented and unit-tested but **unmeasured** — the development machine could
not reach PyPI for torch.

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
| M3 | answer-provenance refusal gate (0 LLM) | net dev delta > 0 | **DONE: +4.33pt dev / +3.87pt full** — shipped, opt-in (4.9) |
| M4 | construct the list answer from evidence (union engine) | precision >= 55% on fully-available items | +25-42 cat1 questions answerable in-context |
| M5 | Reader-side support verifier (one extra call) | cats1-4 loss < 0.5 pt | **REJECTED at scale: +0.45pt, 77 won / 113 lost** (4.11) |
| M6 | date binding (temporal 59.9% -> 66%+) | zero date regressions | **REJECTED by its ceiling: 23% vs a 61% pipeline** (4.15) |
| M7 | single-value extraction (single-hop 52% -> 58%+) | precision >= 55% | open |
| M8 | committer A/B (is the deterministic fallback worth it?) | paired official F1 | **DONE: reader +10.31pt on the 312 claims; committer now opt-in** (4.17) |
| M9 | holdout validation, once | frozen method | report honestly |
| M10 | R7 chain behind the arena player + hybrid union | arena rows comparable | **DIAGNOSTIC: R7 46.5%, hybrid 54.0% vs lexical 53.0%; abstention fixed by compile (2 -> 19/20); union ceiling 65.5%, realized 54.0%; contradiction + reflection walls remain** (4.21) |
| M11 | hybrid fill from the runtime's own recall stack | paired E2E >= raw fill | **REJECTED: 44.0% vs 54.0%, raw 5 exclusive wins / runtime 0 -- reachability is not conversion; decoy pollution** (4.22) |
| M12 | two-sided probe metric (gold-reach x forbidden-presence) | metric must predict E2E refusal splits | **LIMIT RECORDED: forb-presence does NOT separate refusal conversions; the differentiator is line shape (raw turn vs fact-formatted semantic content). Metric kept as a reported dimension; E2E stays the arbiter** (4.23) |
| M13 | deterministic conflict surfacing (player-side, priced first) | contradiction conversions without collateral losses | **DELIVERED: new arena leader 60.0% full-200 (+6pt paired), contradiction 0/20 -> 12/20, 12 wins / 0 losses; iter-1 wording failure recorded; residual 8 = reader refusal/either-or phrasing; mechanism not yet in the shipped runtime** (4.24) |
| M14 | Runtime Port & Parity | mechanism in the shipped runtime, frozen semantics intact by default | **DELIVERED: single-source module in artificial_memory.recall.conflict_surface; player = thin adapter; RuntimeConfig.conflict_surfacing default-off; provenance drift caught & pinned by test; leakage boundary now AST-enforced; 24 unit + 109 regression green; flag-on product E2E parity + residual-8 triage stay open** (4.25) |
| M15 | flag-on product E2E transfer | same conversions on the product read path | **PARTIAL: 3 pricing iterations (budget race / one-sided ingest / 100-fetch window -- each a real research-vs-product gap); priced 20/20 & 0/180 on the product path; E2E 50q paired 40.0 -> 42.0, contradiction 0/5 -> 1/5, zero losses; residual = product context composition, named as the next frontier; flag stays default-off** (4.27) |
| M15.1 | product composition audit | measure where pair evidence drops; fix the bottleneck with one change | **DELIVERED (structural): baseline reproduced; bottleneck = recency window + recall dedupe (recall parts = 0; pair appearance a run lottery); ONE change (note carries source assertion lines) lifted co-presence 15/20 -> 20/20, 0 collateral, ctx tokens not increased; minimal paired E2E unchanged (40.0 -> 42.0, same 1/5) => pair-reach proven NOT the binding constraint; reader-side explain-candidates named (provenance distinctness, noise mass, register); 2 pre-existing test failures triaged (one fixed, one recorded OPEN)** (4.28) |
| M15.2 | provenance distinguishability | real distinct time points on the product read path | **NULL (measured complete): occurred_at passthrough + provenance_days knob priced 0-LLM (labels 20/20 distinct, firing/co-presence/tokens unchanged); 3-arm reader run conflict vs dt +0.0pt, same single conversion => distinct data-backed dates do NOT move the reader; remaining explain-candidate = unrelated recency fill mass around the note (M15.3 candidate); knobs default-off** (4.29) |
| M15.3 | fill-mass ablation | query-unrelated recency fill measured, then removed | **INVERTED (measured complete): fill = 92% of a ~1270-token contradiction context, note at 93% position; ablation priced 0-LLM (reserved part byte-identical 20/20, 180/180 non-fired byte-identical, fired ctx 1270 -> 95 tokens); reader run 50q: conflict 42.0 (1/5) -> bare 40.0 (0/5) -- fill removal LOST the single conversion; fill presence is a necessary condition, not a burying mass； hypothesis axis measured against; next: reader register axis or bench pivot； bare knob stays OFF (do-not-enable)** (4.30) |

**Current standing** (pinned official LoCoMo F1, 1,986 questions): **54.94%** with
the answer-provenance gate and the committer off, against 49.91% for the
protocol-fixed baseline and 37.41% as originally measured; the v0.3.0 artifact
scores 51.89% on the same protocol. On the controlled arena the shipped runtime
went 11.5% -> 53.0% with the lexical channel, against Simple RAG 51.0%.

"World-class" is still **not** claimed. The honest summary is: two measured engine
changes (+4.33 dev / +3.87 full for the gate, +1.16 full for removing the
committer), one product-side recall fix (+41.5pt on the arena, +16.7pt
out-of-sample), and five directions closed with negative evidence.

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
