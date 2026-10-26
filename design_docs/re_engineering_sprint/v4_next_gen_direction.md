# V4 Next-Generation Research Direction — Evidence-First Memory Control

Status: **research direction, not an implementation.** M14 was deliberately NOT created.
This document converts the M5–M13 NO-GO into a next-generation design brief, and in doing
so it found that **three of the four proposed layers already exist in this repository and
have never been on any measured path.** That finding changes the plan's first step.

---

## 1. The thesis

> From memory that is *retrieved*, to memory that *constructs evidence*.

The current architecture answers by: find relevant memories → pick candidates → let a
Reader answer. The next generation must make the *conditions for answering* explicit and
verifiable: what evidence justifies the answer, what is needed to reject the competing
answers, and how much confirmation is enough before answering at all.

Critically this is **not** "prompt the model to think harder." It is explicit data
structures plus verifiable processing.

---

## 2. What M5–M13 actually established, and what it therefore implies

| Established | Design implication |
|---|---|
| Adding candidates does not help (top-50: +1 case) | Retrieval recall is not the bottleneck. Stop investing there. |
| Adding context does not help (full source: net 0, +21% tokens) | Context volume is not the bottleneck. |
| A competing event is visible ~as often when correct (87.5%) as when wrong (82.1%) | **Visibility is not the discriminator.** The gap is in *comparison*, not in *acquisition*. |
| Position / lexical / speaker / predicate features do not separate, and the gaps point the wrong way | Surface features are exhausted as an explanation class. |
| Everything measured was observational; nothing was ever intervened | **Causality is entirely untested.** |

The 87.5% figure is the single most load-bearing fact. It says the Reader already sees
both the right answer and its rival. So the research target is not "get the right memory
in front of the Reader" — it is "decide between two memories that are already both in
front of it."

That is precisely a *comparison* problem, which is what motivates the contrastive layer.

---

## 3. Four layers, and the pre-existing-code problem

The proposed architecture:

1. **Question Contract** — the question as verification conditions, not search keywords.
2. **Contrastive Evidence Engine** — compare candidate vs rival on evidence.
3. **Evidence Sufficiency Gate** — answer, retrieve more, or abstain.
4. **Answer + Audit Trail** — answer plus grounds, uncertainties, and what was not
   confirmable.

**Three of these already exist here.** This is the most important finding in this
document, and it was not known when the direction was proposed.

| Proposed layer | Already exists as | State |
|---|---|---|
| ③ Evidence Sufficiency Gate | `src/artificial_memory/recall/evidence_sufficiency_gate.py` (118 lines) | **Never imported by `DeterministicMemoryEngine`** |
| ② Contrastive comparison | `src/artificial_memory/recall/answer_verifier.py` (553 lines, `check_subject_binding`, `verify`) | Only 3 files import it; not on the measured path |
| Adaptive verification budget | `src/artificial_memory/recall/adaptive_search.py` (117 lines) | **Zero references from `scripts/benchmarks/`** |
| ④ Audit / commit gate | `src/artificial_memory/skills/evidence_commit_gate.py` (198 lines) | **Zero importers — completely unwired** |
| (also) Evidence Binding | `src/artificial_memory/skills/evidence_binding.py` (211 lines) | Unwired, kept as a deliberate experiment |

Verification: `DeterministicMemoryEngine` contains **0** references to
`evidence_sufficiency`, `adaptive_search`, `state_reconstructor` or `answer_verifier`.
The path that M5–M13 measured is `DeterministicMemoryEngine` + frozen Reader, so **none
of these layers was ever measured.**

There is a sharper detail. The existing ESG docstring documents almost exactly the
motivating example proposed for this direction:

> Query: "Where did Caroline move from 4 years ago?" → top candidate "I moved from my
> home country 4 years ago." → INSUFFICIENT (generic hypernym 'home country' with no
> concrete country) → trigger resolution 'home country' → 'Sweden'.

That is the same shape as the pottery/photography exclusion example. So the idea is not
merely present in outline — a prior phase of this project already implemented a
specialised version of it.

**Consequence for the plan: the next step is not to build these layers. It is to
determine why they are unwired, and whether wiring the existing ones onto the measured
path changes anything.** Building a parallel implementation would produce a second
unwired component, which is how the current state arose.

---

## 4. Stage 1 — causal diagnosis first (gate before any build)

M5–M13 never intervened. Before building anything, run the three interventions on frozen
contexts and the frozen Reader:

| Intervention | Question it answers |
|---|---|
| Remove one competing utterance | Does co-presence cause the wrong choice? |
| Swap the order of the two utterances | Does order causally affect selection? (M12 got −29pp correlationally, wrong direction) |
| Paraphrase, meaning preserved | Is selection surface-form dependent? |

Requirements fixed in advance: control conditions, question set, scorer, repeat count.
Measure both the output delta and the correctness delta. **The 553-question holdout is
not spent here.**

An honest limit: a response changing after an intervention does not prove the cause was
simple competition — removal changes several things at once. Each intervention needs its
control.

---

## 5. Stage 2 — only if Stage 1 justifies it

Three arms, matched on model, context, token budget, scorer, and data split:

- **Baseline** — current path.
- **Contrastive** — add candidate-vs-rival evidence comparison.
- **Evidence-Gated** — contrastive plus additional retrieval and abstention decisions.

**And first: check overlap with the existing unwired modules** (§3), since wiring an
existing `answer_verifier` is a different amount of work from writing a new one.

Measure beyond accuracy:

| Dimension | What to report |
|---|---|
| Answer quality | accuracy, per-question-type, accuracy-after-abstention |
| Evidence quality | grounds recall, superfluous evidence, reliance on wrong grounds |
| Reliability | handling of contradictions and updates, confidence–accuracy alignment |
| Efficiency | tokens, latency, extra model calls |

"Accuracy went up" is not a result. What improved, under which conditions, at what cost,
is a result.

---

## 6. Stage 3 — generalisation on unused holdout

Design and parameters frozen, then the 553 holdout questions. If it fails, **do not
retune on the holdout and re-evaluate the same data** — prepare the next independent set.
Report conversation-level variance, not just point estimates, so that a method working in
one dialogue is not mistaken for a general one.

---

## 7. Where to compete, and the honest caveat

Benchmark landscape, verified externally (not from memory):

- **LongMemEval** (Wu et al., ICLR 2025, arXiv:2410.10813): 500 questions, five
  abilities — information extraction, multi-session reasoning, temporal reasoning,
  knowledge updates, **abstention**. `LongMemEval-S` ~115k tokens; `LongMemEval-M` ~1.5M
  tokens.
- **LongMemEval-V2** (Wu et al., 2026, arXiv:2605.12493, ICML 2026): agent context,
  100–498 sessions, **25M–115M tokens**, 451 questions, and it scores **query latency**
  alongside accuracy.

So the earlier claim of "115M tokens" is correct — and note it is V2's ceiling, not
LongMemEval's.

Three capability levels, which are *different targets*, not a difficulty ranking:

1. Recall what was said (LoCoMo).
2. Manage changing knowledge — old vs new, corrections, exceptions.
3. Reuse experience for future decisions — procedures, failure conditions, evidence
   selection.

**The caveat, which must not be dropped:** public scores vary with model, data split,
scorer and latency budget. Comparing numbers published under different conditions and
declaring a winner is unsafe. So the goal is not a single-benchmark maximum but:

> Find the needed memory, distinguish competing memories, answer correctly with stated
> grounds, and avoid answering when the evidence is insufficient — at low compute cost,
> and in a way that extends beyond conversation to long-term agent experience.

---

## 8. The direction, fixed as a research plan

1. **Freeze the current results.** M5–M13 numbers, artefacts, constraints, unresolved
   problems are the baseline. The NO-GO is not overturned.
2. **Causal selection diagnosis** on frozen contexts and the frozen Reader.
3. **Study evidence-constructing methods** — question conditions, candidate contrast,
   sufficiency, and answer grounds, each separable. Prototype first.
4. **Prove generalisation** on independent evaluation; holdout only after the method is
   frozen.

Step 2's result may reorder this. If the interventions show selection is not steerable by
context surgery, then the contrastive layer is targeting the wrong stage and the plan
should change rather than proceed. That is not decided here.

The goal is not a new mechanism. It is a reproducible improvement **in the conditions
where the current approach fails** — and only once it holds across multiple evaluation
conditions does it get promoted to an architecture.

---

## 9. What this document does not claim

- The four layers are a **design hypothesis**. Nothing here demonstrates they reduce the
  Reader's selection errors.
- The overlap finding in §3 means "build the four layers" was partly wrong as stated;
  the unwired modules must be understood before anything is written.
- No world-class claim is made or implied. Differentiation from existing work still
  needs investigation before implementation.
- Stage 1's causal probes may well show that none of this matters, in which case the
  honest move is to record that and look elsewhere — the same discipline that produced
  the M5–M13 NO-GO.
