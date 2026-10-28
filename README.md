<div align="center">

<img src="docs/assets/lethe_hero_banner.jpg" alt="Lethe Akribeia Banner" width="100%" />

# Lethe Akribeia

**An experimental long-term memory runtime for AI agents**

[ **English** | [日本語](README.ja.md) ]

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: experimental](https://img.shields.io/badge/status-experimental-orange.svg)](#status-and-roadmap)
[![Unit tests: 676 passing](https://img.shields.io/badge/unit%20tests-676%20passing-brightgreen.svg)](#reproduction)
[![Coverage: 54%](https://img.shields.io/badge/coverage-54%25-yellow.svg)](#reproduction)

> *“Forgetting is not deletion. It is loss of resolution.”*

A memory system that treats forgetting as progressive resolution loss rather than deletion.
*Deterministic memory compilation · temporal reasoning · evidence provenance · MCP native*
*(Formerly: Artificial Memory / `lethe-akribeia`)*

<br />

<img src="docs/assets/lethe_runtime_demo.gif" alt="Lethe Runtime Terminal Demo" width="92%" />

<br />

</div>

---

## Measurement status — read this before quoting any number

> [!WARNING]
> **This project is not state of the art, and several numbers published in earlier
> releases were re-scored and came out lower.** The paragraphs below say which
> numbers survive, which do not, and what the difference is. Every claim links to
> the artifact or program document that produced it.

**1. LoCoMo, official token F1 (pinned upstream scorer, 1,986 questions).**
The originally published pipeline measures **37.41%** (categories 1–4: **48.24%**).
Two measured changes move it: emitting the benchmark's canonical refusal form
(`not mentioned`) gives **49.91%** — a format fix, not a capability gain — and the
answer-provenance refusal gate plus turning the deterministic committer off gives
**54.94%**. The last figure is a composition of paired deltas (53.77% with the gate,
+1.16pt from removing the committer), not a single stored run. The v0.3.0 artifact
re-scored on the same protocol is **51.89%**. Source:
[`design_docs/v4_world_class/V4_PROGRAM.md`](design_docs/v4_world_class/V4_PROGRAM.md) §2, §4.9, §4.17;
artifacts in [`benchmark_results/official_locomo_score_v4r7_aligned.json`](benchmark_results/official_locomo_score_v4r7_aligned.json),
[`..._v4r7_refusal_form.json`](benchmark_results/official_locomo_score_v4r7_refusal_form.json),
[`..._v030_rules_commit.json`](benchmark_results/official_locomo_score_v030_rules_commit.json).

**2. The binary `65.26%` is not an official F1 and is not comparable to one.**
It comes from an in-house deterministic matcher, and for LoCoMo category 5 the old
harness used `adversarial_answer` as gold. That means 126 category-5 answers were
scored "correct" for matching the distractor; the official scorer gives that
category **0.00%**. The 65.26% figure is categories 1–4 only. Source: V4_PROGRAM §3.1.

**3. LongMemEval numbers (98.20% oracle recall, 83.40% end-to-end) are v0.3.0
artifacts.** They come from one 500-question run with a local 7B reader and an
in-house scorer. Published LongMemEval results in the 95%+ range use different
answerers and LLM judges; no head-to-head under a shared harness exists here.

**4. BEAM `20/20 at 500K` is a needle-in-haystack probe, not a BEAM leaderboard
result.** Published BEAM numbers use LLM judges. This is an in-house binary
matcher over 20 questions at 500K tokens.

**5. The shipped runtime's default configuration scored 11.5% on the controlled
200-question arena.** An IDF-weighted lexical recall channel lifts it to **53.0%**
(Simple RAG 51.0%, Mem0-style 42.5%, Zep-style 35.0%, HippoRAG-style 26.0%) at roughly
3× the read tokens. That channel is now **on by default** (`--no-lexical` or
`LETHE_LEXICAL_RECALL=0` disables it); it was not reachable from the CLI/MCP path when
the arena was measured. Source: V4_PROGRAM §4.12–§4.14.

**6. A holdout leak is recorded, not hidden.** An inverted `--scope` filter caused
one run of the provenance gate to measure only the 499 held-out questions. That
gate's holdout is therefore no longer clean and cannot be cited as validation. The
separate 553-question holdout has not been spent. Source: V4_PROGRAM §4.17.

**7. The tag `v4-r8.1-perfect-score` is a development-set artifact, not a result.**
It records 250/250 on LongMemEval dev-250, produced with 56 deterministic
overrides (22.4% of answers), and it is not part of the V4 measurement log. Dev
saturation is not held-out accuracy and must not be cited.

If you only remember one thing: the honest headline is **54.94% official LoCoMo F1**
with two non-default settings, and **53.0% vs 51.0% for Simple RAG** on one
200-question synthetic arena — not "Apex", not "state of the art", not "100%".

---

## What this repository is

Two Python packages, a CLI, an MCP server, and the benchmark harnesses that produced
the numbers above.

| Path | What it is | Maturity |
|:---|:---|:---|
| `src/artificial_memory/` | The product runtime: SQLite/Postgres store, resolution tiers, MSC context compiler, deterministic co-processors, CLI (`lethe`), MCP server, FastAPI service. | Works; default configuration is conservative (see limitations) |
| `src/artificial_memory_gen/` | The research compiler (R6/R7/V4): query analyzer, closure engine, operators, P2 arena players. | Research code; **not merged** with the product package |
| `scripts/benchmarks/` | Scorers, arena, registry verifier, reproduction entry points. | Required a local Ollama + reader model for most runs |
| `tests/unit/` | 680 tests; the CI gate. | 676 pass, 2 xfail (documented non-capabilities), 2 xpass (rejected experiments that already pass) |

There is no training code and no model weights in this repository. All ingestion is
deterministic; models are used at answer time and, in some research paths, for
retrieval assistance.

---

## How to read our numbers: three layers

High retrieval recall does not imply high end-to-end accuracy, so we report them
separately and never multiply them into one headline.

```
  Layer 1 (Oracle Recall)  ── did the pipeline retrieve/compile the gold evidence turns?
  Layer 2 (Reader on Hits) ── given that evidence, did the Reader answer correctly?
  Layer 3 (End-to-End)     ── final pipeline accuracy (query -> retrieval -> reader -> answer)
```

### v0.3.0 release artifacts (as published at v0.3.0)

These are the numbers that were published with v0.3.0. Read the warning above first:
the binary column is an in-house matcher, and the official F1 for the same LoCoMo
artifact is 51.89%, not 65.26%.

| Benchmark / slice | N | Layer 1: Oracle Recall | Layer 2: Reader on Hits | Layer 3: End-to-End (binary, in-house) |
|:---|:---:|:---:|:---:|:---:|
| LongMemEval (6 capabilities) | 500 | 98.20% (491/500) | 82.08% (403/491) | 83.40% (417/500; baseline 81.60%) |
| BEAM needle-in-haystack **probe** (500K tokens) | 20 | 100% (20/20) | 100% | 100% (in-house binary matcher — not LLM-judged) |
| LoCoMo single-hop (cat 4) | 841 | 83.71% (704/841) | 86.51% (609/704) | 77.76% (654/841) |
| LoCoMo temporal (cat 2) | 321 | 81.62% (262/321) | 56.87% (149/262) | 50.16% (161/321) |
| LoCoMo multi-hop (cat 1) | 282 | 78.37% (221/282) | 60.18% (133/221) | 53.90% (152/282) |
| LoCoMo open-domain (cat 3) | 96 | 57.29% (55/96) | 58.18% (32/55) | 39.58% (38/96) |
| LoCoMo full, non-adversarial (cats 1–4) | 1,540 | 80.65% (1,242/1,540) | 74.32% (923/1,242) | 65.26% (1,005/1,540) binary; **51.89% official F1** |

On the temporal slice, the autonomous committer's own claims were **70.64% precise
(77/109)** on the frozen run and **68.89% (124/180)** on a later offline sweep. It
was subsequently A/B'd and made opt-in, because the reader scored **+10.31pt** higher
on the same 312 claims.

### Current status on the official protocol

Pinned upstream LoCoMo scorer (`task_eval/evaluation.py`, `metric=f1`), 1,986 questions:

| Variant | Adversarial (cat 5) | Categories 1–4 | All 1,986 |
|:---|:---:|:---:|:---:|
| As originally measured | 0.00% | 48.24% | **37.41%** |
| Canonical refusal form (`not mentioned`) | 55.61% | 48.27% | **49.91%** |
| + answer-provenance gate | — | — | **53.77%** |
| + deterministic committer off (current best) | — | — | **54.94%** |

Two caveats that matter for reproducibility: the 54.94% figure is a composition of
paired deltas rather than one stored run, and the refusal-form gain is a format fix.
Neither is a new reasoning capability.

### Controlled arena (200 planted-fact questions)

All players share one frozen answer LLM, one frozen extraction LLM, one embedding
model, a 2,000-token budget and one scorer:

| Player | Accuracy | Read tokens | Notes |
|:---|:---:|:---:|:---|
| **Simple RAG** | **51.0%** | 485k | The baseline that wins on cost |
| Lethe runtime, default config | 11.5% | 431k | Candidate pool is query-independent |
| Lethe runtime + lexical channel | 53.0% | 1,463k | +41.5pt, ~3× tokens |
| Lethe runtime + lexical + hybrid fill | 54.0% | — | +1.0pt paired |
| Lethe conflict surfacing (player-side) | 60.0% | — | +6pt; 12/20 contradictions, 0 losses |
| Mem0-style | 42.5% | — | Reimplementation, same budget |
| Zep-style | 35.0% | — | Reimplementation, same budget |
| HippoRAG-style | 26.0% | — | Reimplementation, same budget |

The arena is synthetic, plants facts, and was run once per player at n=200. The
60.0% conflict-surfacing row is a **player-side** measurement; the same mechanism in
the shipped product path moved 40.0% → 42.0% on a paired 50-question run with one
contradiction conversion out of five. It is not yet a default. Source: V4_PROGRAM
§4.12–§4.21, §4.24–§4.30.

---

## What is measured as *not* working

Keeping this list is the point of the project. Each row was implemented or priced
and then rejected on measurement.

| Direction | Measurement | Verdict |
|:---|:---|:---|
| Deterministic refusal/attribution string gates | −1.11 to −5.34pt | Rejected |
| Reader-side support verifier (one extra call) | +0.45pt at scale, 77 won / 113 lost | Rejected |
| Deterministic date binding | Ceiling 23% vs a 61% pipeline | Rejected by its own ceiling |
| Hybrid fill from the runtime's own recall stack | 44.0% vs 54.0% | Rejected; reachability ≠ conversion |
| Removing unrelated recency fill mass (M15.3) | 42.0% (1/5) → 40.0% (0/5) | Inverted; knob stays off |
| R8.1 LongMemEval dev 250 | 250/250 with 56 overrides | Dev saturation, not citable |

---

## Core mechanics

### Forgetting = loss of resolution, not deletion

The runtime defines six resolution levels; the default pipeline writes the lower
ones and consolidates upward as memory ages and is re-used.

| Level | Name | Content |
|:---:|:---|:---|
| 0 | `RAW` | Verbatim conversation turns |
| 1 | `LIGHT` | Fillers removed, structure preserved |
| 2 | `EPISODE` | What happened, in condensed form |
| 3 | `SEMANTIC` | Key decisions and outcomes |
| 4 | `LONG_TERM` | Highly abstract durable summary |
| 5 | `DEEP_LONG_TERM` | Deep-archive summary for rarely used memories |

Decay is triggered by turn age, access recency/frequency, and a Utility/Token budget
score. Retrieval starts at low resolution and expands only when needed.

### Zero-LLM ingestion and deterministic co-processors

- **Write LLM calls = 0.** Ingestion and indexing are regex, syntactic parsing and a
  token inverted index.
- **Temporal co-processor**: calendar arithmetic, relative expressions resolved
  against message timestamps.
- **Aggregation co-processor**: counting, lists, frequencies.
- **Answer committer**: a strict answer contract. Measured to *cost* points on
  LoCoMo, so it is opt-in, not default.
- **Evidence binding**: a clause-local contract that refuses to attach a value to an
  event it cannot see. It deliberately does not resolve paraphrase — see limitations.

### Context as an intermediate representation (MSC)

Rather than dumping raw text into the prompt, memories compile into a structured
Memory State Context:

```text
[TEMPORAL_ANCHOR] Reference Date: 2023-05-14
[VERIFIED_STATE] User lives in Kyoto (as of 2022-10)
[DETERMINISTIC_TIMELINE]
 - 2023-01-10: Visited Paris
 - 2023-03-04: Switched job to Researcher
[EVIDENCE_CHAIN (Provenance)]
 - [conv_3:turn_12] User: "I am taking the exam next month..."
```

### Vocabulary governance

Every knowledge table (person rosters, nickname maps, synonym expansions, query
patterns, duration anchors) lives in `src/artificial_memory/ontology/` as versioned
YAML with a `_meta.origin` field separating world knowledge (`generic`) from
corpus-derived tuning (`benchmark_tuned`). A CI leak guard
(`tests/unit/hygiene/`) scans runtime code for benchmark fingerprints; comments and
docstrings are exempt, executable code is not, and the shrinking allowlist is itself
tested. This is the mechanism that makes "no benchmark vocabulary in engine logic" a
checked property rather than a claim.

---

## What you can do today

### MCP server

```bash
python -m artificial_memory.mcp
```

Tools: `memory_remember`, `memory_recall`, `memory_expand`, `memory_trace`,
`memory_timeline`, `memory_explain`, `memory_inspect`.

### CLI

```bash
git clone https://github.com/Yato-Works/Lethe-Akribeia.git
cd Lethe-Akribeia
pip install -e ".[vector,llm]"

lethe start "Project/Akribeia"
lethe user "We decided to target 9/30 for the initial release."
lethe assistant "Understood. The release deadline is set to September 30."
lethe end
lethe recall "When is the release scheduled?"     # works in a new process
lethe ask "When is the release scheduled?"        # evidence + compiled context
lethe resume "Project/Akribeia"                   # continue the same topic
lethe timeline
```

`ask` returns what the agent would send to its model: the matching memories, their
provenance, and the compiled MSC context. It does not call an LLM, so it is not a
generated answer. `recall`/`timeline` no longer require an active conversation; they
fall back to the most recent topic unless `--topic` is given.

---

## Known limitations

These are current, measured, and mostly queued for the next milestones.

- **Memory quality on real dialogue is modest.** 54.94% official F1, with the
  single-hop and adversarial categories carrying most of the loss.
- **Recall is tuned on a synthetic arena, not on your token bill.** The lexical
  channel is on by default because it is the measured-better configuration
  (11.5% → 53.0%), but it reads roughly 3× the tokens. Disable it with
  `--no-lexical` or `LETHE_LEXICAL_RECALL=0` when budget matters more.
  Conflict surfacing remains off; its product-path transfer is still partial.
- **No paraphrase resolution.** Evidence binding is clause-local: if the duration
  and the event sit in different comma-clauses, or the question paraphrases the
  event ("creating art" vs "painting"), the engine declines rather than guesses.
  This is covered by two explicit tests (one pass, one strict xfail).
- **Cue detection does not model negation.** "Nothing has been chosen yet" still
  registers as a decision cue, in Japanese and English alike. The markers are
  deterministic word matches, not a parser.
- **Test coverage is 54%.** Many research modules (`artificial_memory_gen`,
  `compiler/eval`) are exercised only by benchmark scripts, not by CI.
- **Type checking is not clean.** `mypy` reports ~1.4k findings, dominated by
  untyped third-party decorators and missing stubs. Ruff is clean and blocking.
- **Two stacks, one repo.** `artificial_memory` and `artificial_memory_gen` are not
  merged; `RuntimeFacade` carries 95 methods (89 public). All 10 `compiler/stages`
  classes are wired into the default pipeline — measured, not dead — so the
  remaining work is the merge itself, not deleting stage modules. Merging is
  planned work, not done work.
- **API defaults are loopback-only, but the service is not a production deployment
  target.** CORS defaults to localhost origins; `LETHE_ALLOW_ORIGINS="*"` is
  accepted only with credentials disabled. `python -m artificial_memory.api.server`
  binds `127.0.0.1`, and `LETHE_ENABLE_AUTH=1` now actually enforces a session or
  API key on every non-public route (before this change the flag protected
  nothing). There is still no TLS, rate limiting, per-user isolation, or WebSocket
  authentication.
- **Benchmark scripts need local models** (Ollama, a reader model, optionally
  embeddings). CI runs unit tests only; it does not reproduce the tables.

---

## Smoke check (one command)

Before and after any pipeline change, run the pinned smoke set. It sends real
LoCoMo questions end to end through the same frozen path as the full benchmark
and shows you exactly what the reader was given:

```bash
# 18 pinned cases, real 7B reader (needs a local Ollama)
python scripts/benchmarks/locomo_smoke.py

# a few cases only
python scripts/benchmarks/locomo_smoke.py -n 4

# no reader at all: 0 LLM calls, still verifies retrieval + context compilation
# run and shows how much evidence each case would receive
python scripts/benchmarks/locomo_smoke.py --no-llm
```

Output lands in `benchmark_results/smoke/<stamp>/`: `report.md` (per-case table,
category rollup, failures in detail, the exact reader input), `results.json`,
`reader_inputs/`, and `run.log`. The case set is pinned in
[`benchmark_config/smoke_manifest.json`](benchmark_config/smoke_manifest.json) and
excludes conv-48, the strict clean holdout. **These numbers are a plumbing
signal, not a benchmark score**; publishable numbers stay in
[`benchmark_results/RESULTS_REGISTRY.md`](benchmark_results/RESULTS_REGISTRY.md).

### Comparing two runs

`--compare` diffs two finished run folders: which cases got fixed, which broke,
the per-case token delta and the mean context change. It reads the artifacts, so
it works on any two runs of the same case set, including one made before a change
you are evaluating.

```bash
python scripts/benchmarks/locomo_smoke.py                                  # baseline
python scripts/benchmarks/locomo_smoke.py --rerank ollama                  # same cases
python scripts/benchmarks/locomo_smoke.py --compare \
    benchmark_results/smoke/<baseline> benchmark_results/smoke/<second-run>
```

### Re-ranking (`--rerank`, off by default)

`--rerank` re-orders the retrieved candidate pool with a learned relevance model
before the budgeted compiler selects evidence:

| backend | model | needs |
|---|---|---|
| `ollama` | `qwen2.5:7b-instruct`, listwise JSON scores | the Ollama server every run already needs |
| `cross-encoder` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | the `[vector]` extra (torch + sentence-transformers) |

Both fail before the first question if their model is absent. A batch whose reply
cannot be parsed keeps its retrieval order and is counted in the manifest's
`rerank_failures`, so a run cannot silently claim it re-ranked anything.

**Measured 2026-10-28 on the 18 pinned dev cases (`ollama` backend): 10 -> 9
correct, 1 fixed / 2 broken, mean context 1410 -> 1258 tokens.** Both arms were
re-run and reproduced identically (temperature 0, seed 42), so the -1 is a real
effect on these 18 cases, not sampling noise — but 18 cases cannot settle the
method, and the smaller context is the only structural effect. One of the two
breaks is the attribution pair itself: `lc_conv-43_178` ("which team did *Tim*
sign with", gold: not mentioned) returned `Minnesota Wolves`, i.e. John's
answer, where the un-reranked run refused. The cross-encoder was never measured
here (no PyPI access on the development machine), so the stronger instrument is
untested, not refuted.

## Reproduction

```bash
# 1. Unit tests (680 collected; 676 pass, 2 xfail, 2 xpass)
pytest tests/unit/

# 2. Coverage (measured 54% on 2026-10-26)
pytest tests/unit/ --cov=artificial_memory --cov=artificial_memory_gen

# 3. Lint
ruff check src/ tests/

# 4. Verify the 21 canonical v0.3.0 metrics and their artifacts
python scripts/benchmarks/verify_registry.py

# 5. Re-score a stored LoCoMo run with the pinned upstream scorer
python scripts/benchmarks/score_locomo_run_json.py \
  --results benchmark_results/locomo1540/locomo_1540_improved2.json

# 6. Three-layer decomposition
python scripts/benchmarks/three_layer_report.py
```

Frozen context caches (`benchmark_results/frozen_ctx/`) and verifier verdicts
(`benchmark_results/support_verifier/`) are generated inputs and are gitignored;
regenerate them with the `--freeze-contexts` / `--out` commands named in
V4_PROGRAM before running the attribution scripts. The two arena runs cited in
V4_PROGRAM §4.21 are committed; 13 other timestamped local runs are not.

The authoritative measurement log, including the negative results, is
[`design_docs/v4_world_class/V4_PROGRAM.md`](design_docs/v4_world_class/V4_PROGRAM.md).
The 21 frozen v0.3.0 metrics are in
[`benchmark_results/RESULTS_REGISTRY.md`](benchmark_results/RESULTS_REGISTRY.md).

---

## Status and roadmap

**Current: v0.3.3, experimental.** The V3 three-layer diagnostic standard and the V4
measurement correction are both done. This branch also closes the product-wiring gap
that the arena exposed:

- the measured recall configuration is now the default (`--lexical/--no-lexical`,
  `LETHE_LEXICAL_RECALL`);
- `lethe ask` and `lethe resume` exist, and `recall`/`timeline` no longer require an
  active conversation;
- decision/reason/timeline cues are bilingual (Japanese and English);
- the embedding model loads lazily instead of on every CLI invocation;
- exact cross-resolution duplicates (same memory id, or the same body after
  whitespace normalization) are removed from the context. Measured on a two-turn
  topic: 20 recall rows across four levels contained 15 repeated ids and 1 repeated
  body; the compiled context is now 4 parts;
- the API is loopback-only by default and `LETHE_ENABLE_AUTH=1` now actually
  enforces a session/API key (before this change the flag protected nothing).

Still queued:

1. Merge the research and product packages; shrink the 95-method facade. (Measured
   note: all 10 `compiler/stages` classes are wired into the default pipeline; no
   dead stage module was found.)
2. Merge paraphrase-level near-duplicates. The dedupe above only removes exact ids
   and exact normalized text; it does not merge a fact re-expressed at
   RAW/EPISODE/SEMANTIC.
3. Deployment hardening for the API (TLS, rate limits, per-user isolation,
   WebSocket authentication).

No accuracy promise is attached to any of these. Each will be accepted or rejected
on the paired measurement described in V4_PROGRAM, not on a demo.

---

## Why release now

I wanted to find out how far a structured, local memory system could go when hardware
and models were small: an RTX 3050 (8 GB VRAM), a 7B q4 reader, nothing else.
This is a working, auditable checkpoint with measurable strengths and measurable
weaknesses. The re-scoring in V4 made the weaknesses visible; that is why the numbers
here are lower than the ones in the v0.3.0 release notes.

I still want to find out how far this architecture can go.

---

## Citation

```bibtex
@software{lethe_akribeia_2026,
  title = {Lethe Akribeia: An Experimental Cognitive Long-Term Memory System for AI},
  author = {Yato-Works},
  year = {2026},
  version = {0.3.3},
  url = {https://github.com/Yato-Works/Lethe-Akribeia}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
