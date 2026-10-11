# Contamination & gaming-pattern audit (external research + own-code audit)

## Sources (fetched, not invented)

1. **Benchmark Data Contamination of LLMs: A Survey** — arXiv 2406.04244
   (Xu, Guan, Greene, Kechadi, UCD). Severity ladder, weakest to strongest:
   - **Semantic level** — same topic or same source as the benchmark. Biases
     generalisation; hardest to detect.
   - **Information level** — metadata, label distributions, time
     distributions, external reviews of the benchmark.
   - **Data level** — benchmark content without labels.
   - **Label level** — full exposure including labels. The model memorises the
     answer. Easiest to detect and prevent.
   Key caveat from the survey: string-matching decontamination is beatable —
   paraphrasing bypasses it, and a 13B model can reach GPT-4-level scores on a
   test set when paraphrase variants are not removed.

2. **LLM Gaming Behavior Catalog** — github.com/sparckix/ztare, 18 surfaces,
   each paired with the audit that catches it. The ones relevant to us:

   | # | Surface | Audit that catches it |
   |---|---|---|
   | 1 | Severity Averaging — bundle a critical axiom with N sacrificial ones so its penalty dilutes to 1/N | per-axiom penalty must be independent of bundle size |
   | 2 | Tolerance Abuse — tune the threshold to whatever the metric needs | threshold must have a stated derivation |
   | 3 | Stubbed Implementation — output that passes the assertion while violating its intent | independent reference computation |
   | 8 | Parameter Overfitting — an unsourced "calibration constant" whose value is whatever makes the test pass | require provenance for every numeric literal |
   | 9 | Weak Baseline — rig the comparison object A so B wins by construction | steelman A before arguing B > A |
   | 12 | Audit Partition Fingerprinting — infer the holdout split from rubric id/version | per-run salt unavailable to the proposer |
   | 15 | Fabricated Reference Fitting | independent reference |

   Framing: these are Goodhart's Law at the validator-author boundary. The
   agent is not misbehaving — the objective ("produce output that passes your
   own assertion") is the bug.

## What this means for us

Our contamination was **Label level** — the reader prompt contained 20 literal
LoCoMo gold answers, plus per-question branches keyed on question wording.
That is the most detectable category, and we still missed it for an entire
session. Detection is easy *only if you look*.

Two findings from the research that apply directly:

- **#12 Audit Partition Fingerprinting.** Our LoCoMo slices are deterministic
  functions of `conv_idx`/`offset` (e.g. `--conv-idx 1 --offset 0 --limit 20`).
  Anyone iterating on those exact numbers is tuning against a fixed, knowable
  partition. Future hold-out evaluation needs a per-run salt logged after
  completion.
- **#8 Parameter Overfitting.** `_scaled_budget(base, pool, ratio=0.12,
  cap=2400, floor=1400)` has four magic numbers. None has a derivation; they
  were tuned against observed behaviour. This is the exact shape the catalog
  warns about, and it is the knob most likely to be re-tuned into the test.

## Own-code audit results

Scanned every module in `src/artificial_memory/` for: canned-answer
assignments, benchmark entity names used as detection triggers, gold-answer
literals, and question-text branching.

| pattern | count | reachable from LoCoMo eval? |
|---|---|---|
| `C_gold_literal` (72) | 72 | mostly docstrings, comments, generic vocab |
| `B_benchmark_entity` (7) | 7 | **no** — see below |
| `A_canned_answer` (13) | 13 | **no** — see below |

### B_benchmark_entity — LoCoMo's own characters hardcoded

`Caroline` / `Melanie` appear as detection triggers in 4 modules:
`protein/state_compiler.py` (149), `recall/proposition_integrity_gate.py`
(88, 110, 136, 149), `temporal/temporal_compiler.py` (284, 286).

These are Label/Semantic-level contamination by construction: code that knows
the benchmark's cast. **Verified NOT reachable from the LoCoMo evaluation
path** (import-BFS from the six harness modules does not reach them).

### A_canned_answer — keyword to canned answer string

`compiler/certificates/temporal_certificate.py` maps keyword pairs to literal
answers, e.g. `"cousin"+"wedding" -> "my cousin's wedding"`,
`"contract"+"client" -> "I signed a contract with my first client."`.

**Verified those strings are NOT LoCoMo gold answers** (0 exact, 0 substring
matches), and the engine is not executed by the LoCoMo path — `budgeted_compiler`
imports only the `CoProcessorBundle` dataclass for a type annotation and never
instantiates `CoProcessorEngine`. So this is not inflating any reported number.

**But the shape is catalog surface #3 (Stubbed Implementation) and must not be
extended.** If these engines are ever wired into an evaluation path they will
quietly hand the model answers.

## Verdict

- The LoCoMo evaluation path is clean; no finding above affects any number.
- Two code smells are real and should not survive a public release:
  benchmark entity names in `proposition_integrity_gate.py`, and canned answers
  in `temporal_certificate.py`.
- Two systemic risks from the research need structural fixes, not code review:
  deterministic hold-out partitioning (#12), and unproven magic constants in
  `_scaled_budget` (#8).
