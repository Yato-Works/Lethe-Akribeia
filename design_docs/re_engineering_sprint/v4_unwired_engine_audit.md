# V4 Stage-1 Audit — Why the existing engines are unwired

Status: **audit only.** No production change, no wiring, no intervention. This is steps
① and ② of the agreed sequence: establish *why* the modules are unwired, and what
contract each guarantees, before building any measurement path.

---

## 0. A correction to the previous document

`v4_next_gen_direction.md` stated that `answer_verifier.py` had "only 3 files import it."
**That was wrong, and it understated the module's importance.** The count of 3 came from a
grep restricted to `src/`. The real number across `src/` and `scripts/` is **25 files**,
including `research/benchmarks/failure_taxonomy_v2.py`,
`research/benchmarks/external/longmemeval_adapter.py`, and `scripts/run_frozen_longmemeval.py`.

Corrected here rather than left to propagate. The next_gen document's central claim — that
these engines were never on the measured path — still holds, but for different reasons per
engine, and those reasons matter for what to do next.

---

## 1. Per-engine finding

### 1.1 `recall/evidence_sufficiency_gate.py` (118 lines) — unwired, and structurally so

**Contract.**

```
SufficiencyDecision(is_sufficient: bool, unresolved_phrase: str|None, expected_category: str|None)

EvidenceSufficiencyGate.check_sufficiency(query: str, top_units: Sequence[ApexMemoryUnit]) -> SufficiencyDecision
EvidenceSufficiencyGate.evaluate(query: str, text: str) -> SufficiencyDecision
```

`evaluate(query, text)` takes a **plain string** and needs no IR. `check_sufficiency` takes
`Sequence[ApexMemoryUnit]`, i.e. Memory IR 2.0 objects (`ir: StructuredIR`, `role`,
`event_timestamp`, `supported_by`, `confidence`).

**Reason unwired — two separate causes, not one:**

1. **Type mismatch.** `check_sufficiency` requires `ApexMemoryUnit`. The LoCoMo/frozen
   path deals in `Turn` objects (`dia_id`, `header_date`, `speaker`, `text`). There is no
   adapter between them, so the IR-shaped entry point cannot be called from the measured
   path at all. The `evaluate(query, text)` entry point *can* — it is string-in,
   string-out — and is the one a measurement harness should use.
2. **The logic is a hardcoded vocabulary table, not a general mechanism.** `REFERENT_PATTERNS`
   is six hand-written (generic-referent, concrete-resolution, category) triples:

| Generic | Concrete resolution | Category |
|---|---|---|
| `home country` | sweden/canada/germany/japan/… | country |
| `my dog`/`pet` | bailey/oscar/retriever/poodle/… | pet_breed |
| `university`/`college` | stanford/harvard/mit/ucla/… | institution |
| `partner`/`boyfriend`/… | `[A-Z][a-z]+` | partner_name |
| `instrument`/`playing` | guitar/violin/clarinet/piano/… | instrument |
| `destress`/`headspace` | running/pottery/painting/therapy/… | destress_activity |

That covers **six categories**. It is English-only, corpus-specific, and enumerates
answers by name.

**Honest consequence:** ESG is a *prototype demonstrating the shape of a sufficiency
gate*, not a general one. Wiring it as-is onto LoCoMo would not measure "does an evidence
sufficiency gate help" — it would measure "do these six hand-written patterns help," which
is precisely the per-case special-casing the M5–M13 constraints forbid. Its value is as a
**design reference**, and the measurement harness should use `evaluate()` only to
establish a baseline for what a gate-shaped component does, with the vocabulary limitation
stated.

Note also that `check_sufficiency` reads only `top_units[:2]` — a two-unit window.

### 1.2 `recall/answer_verifier.py` (553 lines) — wired into other paths, just not the V4 one

**Contract.**

```
VerificationResult(is_verified: bool, verified_answer: str, hallucination_detected: bool, notes: str|None)

AnswerVerifier(subject_binding: bool = True)
  .verify(question, predicted_answer, context, propositions: Sequence[UnifiedProposition],
          integrity_abstention_recommended: bool = False) -> VerificationResult
  .check_subject_binding(question, ans, context) -> VerificationResult | None
```

**This is a materially different situation from ESG, and the most important finding of this
audit.** It is not an unwired prototype:

- **It is already A/B'd on 1,986 LoCoMo rows.** The source records the numbers inline:
  extending `_OWNER_PRONOUNS` to bare first-person subjects (`i`, `we`) was measured on
  rows where the prior rule was silent — *adversarial +33, single-hop −4, temporal −1,
  net +28 questions* (`scripts/simulate_binding_variants.py`).
- **It carries a measured negative that is deliberately OFF in production.**
  `_SECOND_PERSON_BINDING = False` — second-person binding was measured net-negative
  against the self-assertion rule, so it is disabled and only the offline harness turns it
  on.
- **It has 25 importers**, including `failure_taxonomy_v2.py` and
  `longmemeval_adapter.py` — i.e. the LongMemEval adapter already routes through it.

So the correct statement is not "unwired." It is: **`answer_verifier` is wired into other
evaluation paths (including the LongMemEval adapter), but it was never on the V4
M5–M13 path**, which used `DeterministicMemoryEngine` + the frozen Reader directly.

`verify()` is also strictly *post-hoc*: it runs after an answer exists and can refuse or
rewrite it. That is a **rejection/commit** guard, not a *selection* mechanism — it cannot
choose between two candidates that are both present, which is the exact gap M11 identified
(87.5% competitor visibility). Worth stating precisely, because it bounds what this engine
can contribute.

Its dependency is `Sequence[UnifiedProposition]`, another IR type, so calling `verify()`
from the frozen path also needs an adapter — but `check_subject_binding(question, ans,
context)` is plain-string in/out and is the natural entry point for a harness.

### 1.3 `recall/adaptive_search.py` (117 lines) — unwired, depends on the whole IR stack

```
AdaptiveEvidenceSearcher(max_hops: int = 3).search(...) -> AdaptiveSearchResult
```

Imports `UnifiedProposition`, `EvidenceSufficiencyGate`, `UnifiedPropositionGraph`,
`QueryPlan`. It is the graph-navigation consumer of ESG. Zero references from
`scripts/benchmarks/`. It sits on the proposition-graph IR stack, so it is the furthest
from the measured path of the three.

### 1.4 `skills/evidence_commit_gate.py` (198 lines) — completely unwired, **zero importers**

The only module with literally no importers anywhere. Unaudited contract.

---

## 2. Why they are unwired — the actual pattern

Not "forgotten." The pattern is a **stack split**:

```
Apex IR stack : ApexMemoryUnit / UnifiedProposition / UnifiedPropositionGraph / QueryPlan
                -> evidence_sufficiency_gate, adaptive_search, answer_verifier
Frozen path   : Turn + DeterministicMemoryEngine + frozen Reader
                -> what M5-M13 actually measured
```

The IR-stack engines and the frozen benchmark path were built against **different data
models**, with no adapter between them. Nothing wired them because there was no shared
type to wire through. The 25-importer count for `answer_verifier` is consistent with this:
its importers are other IR-stack and benchmark-adapter code, not the frozen path.

This is a more useful finding than "unwired by accident," because it names the actual
first task: **the missing piece is an adapter, not an algorithm.**

---

## 3. What this changes about the plan

The next_gen direction said the next step was "find out why they are unwired." Now known.
Three consequences:

1. **Do not wire `evidence_sufficiency_gate` as-is and call it a test of the sufficiency-gate
   hypothesis.** Its six hardcoded English patterns would make the result a measurement of
   those patterns — the exact special-casing M5–M13 forbid. If it is wired, it is to
   establish a *baseline for the gate shape*, with the limitation stated up front.

2. **`answer_verifier` is the highest-value wiring target, for a specific and limited
   reason.** It is already A/B'd at +28 net on 1,986 rows, so it is the only one of these
   engines with production-grade evidence behind it. But it is a *post-hoc commit guard*,
   not a selector — so it can address wrong→abstain and unsupported-answer refusal, and it
   **cannot** address the M11 finding (both candidates visible, wrong one chosen). Expecting
   it to fix selection would be a category error.

3. **The adapter is the deliverable, not a new engine.** A `Turn`-to-`UnifiedProposition` /
   `ApexMemoryUnit` bridge is what would let any of these run on the frozen path. That is a
   bounded, testable piece of work, and building it changes nothing in production.

---

## 4. Still unknown, deliberately not guessed

| Unknown | Why it matters |
|---|---|
| `evidence_commit_gate.py` contract | Zero importers, never audited. Cannot be assessed yet. |
| Whether `check_subject_binding` refuses on any of the M11 23 cases | Determines if the +28 net result transfers to the frozen cohort at all. |
| Whether `UnifiedProposition` construction from `Turn` is lossless | If propositions drop the speaker/timestamp, the binding guard loses its input. |
| Whether the +28 net holds outside the adversarial probes it was measured on | The inline comment bounds the measurement to specific row sets. |

None of these is assumed. They are the inputs to step ③, the minimal measurement path.

---

## 5. Sequence, unchanged in order

① **Why unwired** — done, this document.
② **Contracts and existing tests** — largely done for ESG / answer_verifier / adaptive_search;
  `evidence_commit_gate` outstanding.
③ **Minimal measurement path** — production untouched, frozen Reader, existing scorer.
④ **One layer at a time**, then interactions. Never several at once.

The 553-question holdout is still unspent, and stays that way.

## 6. Not claimed

- No engine here has been shown to reduce the Reader's selection errors. None was on the
  M5–M13 path.
- ESG's six patterns are not a general sufficiency mechanism.
- The +28 net figure is from the inline provenance and a specific row set; it has not been
  re-verified here and is not assumed to transfer.
- Wiring any of this is not yet justified. This audit only establishes what would have to
  be true.
