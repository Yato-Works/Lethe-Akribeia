# Lethe Akribeia

> **Forgetting is not deletion.**  
> **It is loss of resolution.**

An experimental long-term memory system for AI that treats forgetting as progressive resolution loss rather than deletion.  
*Deterministic memory compilation · Temporal reasoning · Evidence provenance · MCP native*  
*(Formerly: Artificial Memory / `lethe-akribeia`)*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Experimental v0.2.0](https://img.shields.io/badge/status-experimental%20v0.2.0-orange.svg)](#why-release-now)
[![LoCoMo: 1540 Benchmark](https://img.shields.io/badge/benchmark-LoCoMo%201540-green.svg)](#evaluation-highlights)
[![BEAM: 100K--10M](https://img.shields.io/badge/benchmark-BEAM%20100%25-brightgreen.svg)](#evaluation-highlights)

---

## What is Lethe?

Lethe Akribeia is an open-source cognitive memory runtime designed for persistent AI agents.  
Unlike naive vector RAG or raw context stuffing, Lethe compiles conversational history into structured, resolution-tiered memory states using **zero LLM calls on ingestion (Write LLM = 0)**. It pairs memory storage with deterministic calendar and aggregation co-processors, preserving complete evidentiary provenance back to the exact conversational turn.

```
Conversational Turns
        │ (Zero-LLM Deterministic Ingestion)
        ▼
   [Memory IR] ── Level 0 (Raw) → Level 4 (Anchor)
        │
   [MSC Compiler] + Co-Processors (CHRONOS / Aggregation / Committer)
        │
   [Compiled Context] (Frozen, Auditable)
        │
   [Reader LLM] (1.5B ~ 7B ~ Frontier)
        ▼
   Evidence-Grounded Answer

```

---

## Evaluation Highlights: Three-Layer Architecture

We do not present Lethe Akribeia as a universal SOTA system. Rather, we empirically separate the evaluation of persistent memory into three distinct layers to diagnose where memory succeeds and where language models struggle:

```
  Layer 1 (Oracle Recall)  ── Did Lethe retrieve and compile the necessary gold evidence turns?
  Layer 2 (Reader on Hits) ── Given the evidence in context, did the Reader LLM answer correctly?
  Layer 3 (End-to-End)     ── Final pipeline accuracy (Query → Lethe Retrieval → Reader LLM → Answer)
```

> [!IMPORTANT]
> **Strict Evaluation Protocol & Data Separation**:  
> All benchmarks are evaluated strictly against **official testbeds** and **official evaluation harnesses**. Prompts and hyperparameters were tuned strictly on separate diagnostic splits; the final benchmark sets (LoCoMo 1,540 questions, LongMemEval 500 questions, BEAM official suites) were **held completely separate** to prevent any data leakage or benchmark-specific overfitting.

### Benchmark Summary Table (Three-Layer Decomposition)

| Benchmark / Evaluation Suite | Scope (N) | **Layer 1: Oracle Recall** (Evidence Retrieval) | **Layer 2: Reader on Hits** (7B Reader Accuracy) | **Layer 3: End-to-End** (Final Accuracy) | Key Finding & Architectural Boundary |
|:---|:---:|:---:|:---:|:---:|:---|
| **LongMemEval (All 6 Capabilities)** | 500 Qs | **98.20% (491/500)**<br>*(95% Wilson CI: 96.6%–99.1%)* | **82.08%**<br>*(7B fails on 17.9% of hits)* | **81.60% (408/500)** | Memory retrieval reaches 98.2%; errors are heavily concentrated in Reader false refusals and calendar arithmetic. |
| **BEAM (100K ~ 10M Horizon)** | 48 Qs | **100.0% (48/48)** | **100.0%** | **100.0%** | Deterministic timeline indexing extracts needles across 100K–10M tokens without context degradation. |
| **LoCoMo 1,540 (Single-Hop)** | 841 Qs | **83.71%** | **85.80%** | **77.65%** | Strong direct factual recall; Reader reliably extracts explicit entity facts. |
| **LoCoMo 1,540 (Temporal Cat 2)** | 321 Qs | **81.62%** | **57.25%**<br>*(7B fails on 42.8% of hits)* | **51.09%** | Multi-interval relative dates expose severe 7B reasoning limits (provenance header copying, relative date drift). |
| **LoCoMo 1,540 (Multi-Hop Cat 1)** | 282 Qs | **78.37%** | **55.20%** | **47.87%** | Open research frontier: cross-session graph linking across divergent topics. |
| **LoCoMo 1,540 (Full Non-Adversarial)** | 1,540 Qs | **80.65% (1,242/1,540)** | **72.54%** | **64.68% (996/1,540)** | Overall non-adversarial benchmark with primary local 7B Reader. |

> [!NOTE]
> **Why Separate Oracle Recall from End-to-End?**  
> In LongMemEval (500 questions), Lethe missed the gold evidence in only 9 out of 500 questions (Oracle Recall 98.20%). An error analysis across all 109 incorrect answers revealed that **96.3% had the required evidence present in the context prompt**, but the 7B local Reader failed due to false abstention ("I don't know" despite evidence present, 29 Qs) or arithmetic errors in calendar math (29 Qs).  
> Separating Layer 1 from Layer 2 prevents misattributing Reader reasoning limits to memory retrieval failure.

---

## Model Sensitivity: The Reader Floor (7B vs. 1.5B)

A central finding across 140+ hours of sweeps is that **Lethe's pre-compiled context significantly insulates against Reader scale downgrades**:

| Benchmark Slice | 7B Reader (`qwen2.5:7b-instruct`) | 1.5B Reader (`qwen2.5:1.5b`) | Delta | 7B-only Hits | 1.5B-only Hits | McNemar Test ($p$-value) | Scientific Interpretation |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **LoCoMo 1,540 (Full)** | **64.68% (996/1,540)** | **64.48% (993/1,540)** | +0.20pp | 133 | 131 | **$p = 0.9509$** | No statistically significant difference detected despite 5x parameter drop. |
| **LongMemEval (500 Qs)** | **81.60% (408/500)** | **80.00% (400/500)** | +1.60pp | 39 | 31 | **$p = 0.4030$** | Multi-session reasoning scored identically (84.2% on both models). |

> [!CAUTION]
> **Statistical Rigor on Model Invariance**:  
> A high McNemar $p$-value ($p > 0.05$) does **not** constitute mathematical proof of complete model independence; it indicates that under these frozen benchmark contexts, there is insufficient evidence to reject the null hypothesis of equal performance. Formally establishing equivalence requires two-one-sided tests (TOST) with a pre-defined equivalence margin.  
> Nonetheless, observing that 7B and 1.5B share 863 identical correct answers and 413 identical failures (1,540Q) strongly suggests that accuracy is heavily governed by **the structured quality of the memory context**, rather than raw Reader parameters alone.

### Honest Boundaries & Unverified Frontiers
- **Frontier Reader Scaling (120B / Gemini)**: While our small 10-question probe showed 90% (9/10) with Gemini Flash, asserting that a 120B+ model will achieve 100% on the full 1,540 suite is **an unverified hypothesis** pending compute budget.
- **Autonomous Deterministic Offloading**: To bypass Reader arithmetic and formatting failures, Lethe incorporates zero-LLM deterministic skills (CHRONOS, counting, ontology resolvers). On a hard diagnostic drill of 281 questions where 7B Reader scored 38.79% F1, the deterministic engine committed answers at **100% resolution with 92.87% F1 and 0 regression losses**.
- **Comparison with Contemporaneous Work**: Other systems report strong LongMemEval numbers (e.g., Sibyl Labs 95.6%, OMEGA 95.4%) and LoCoMo numbers under varying evaluation harnesses and judge models. Direct head-to-head evaluation under identical frozen judges remains ongoing.

---

## Why Lethe? (Core Principles)

### 1. Forgetting = Loss of Resolution, Not Deletion
Human memory does not drop files into a recycle bin. Over time, memories decay in resolution:
- **Level 0 (RAW)**: Full verbatim conversation turns.
- **Level 1 (EPISODIC)**: Structured events tagged with speaker, timestamp, and conversational context.
- **Level 2 (CONDENSED)**: Salient points and assertions extracted via syntactic entity-predicate parsing.
- **Level 4 (ANCHOR)**: High-level durable life facts, user profiles, and recurring beliefs.

**Resolution Decay Policy & Triggers**:  
Rather than hard deletion, Lethe decays resolution progressively under three deterministic triggers:
1. **Turn Age Decay (Temporal)**: As conversation advances, older raw turns expire from Level 0 (RAW) and are consolidated into Level 2 (syntactic entity-predicate assertions).
2. **Access Recency & Frequency (Utility)**: Core anchors referenced repeatedly across sessions are reinforced into Level 4 (ANCHOR), while unreferenced peripheral details decay toward minimal representation.
3. **Token Budget Utility Pruning**: When context space is constrained (e.g., prompt budget ceiling of 2,000 tokens), low-utility details are pruned based on a deterministic Utility/Token score. Retrieval starts at minimal token cost, expanding dynamically to higher resolutions only when reasoning ambiguity is detected.

### 2. Zero-LLM Ingestion & Deterministic Co-Processors
Memory ingestion does not rely on non-deterministic, expensive LLM calls:
- **Write LLM Calls = 0**: Ingestion and indexing are purely deterministic (regex, syntactic parsing, token inverted index).
- **CHRONOS Temporal Co-processor**: Computes calendar arithmetic deterministically (resolving relative expressions like "last Tuesday" or "three months ago" against message timestamps).
- **Aggregation Co-processor**: Handles counting, lists, and frequency queries deterministically.
- **Answer Committer**: Enforces a strict answer contract to prevent hallucinated drift.

### 3. Context as an Intermediate Representation (Context IR / MSC)
Rather than dumping raw text into a prompt, Lethe compiles memories into a structured **Memory State Context (MSC)**:
```text
[TEMPORAL_ANCHOR] Reference Date: 2023-05-14
[VERIFIED_STATE] User lives in Kyoto (as of 2022-10)
[DETERMINISTIC_TIMELINE]
 - 2023-01-10: Visited Paris
 - 2023-03-04: Switched job to Researcher
[EVIDENCE_CHAIN (Provenance)]
 - [conv_3:turn_12] User: "I am taking the exam next month..."
```

---

## What Lethe Can Do Today

### AI Agent Integration (MCP Native)

Lethe Akribeia provides a native **Model Context Protocol (MCP)** server for Claude Desktop, Cline, Hermes, and any MCP-compliant agent:

```bash
python -m artificial_memory.mcp
```

Exposed MCP Tools:
- `memory_remember`: Store durable facts and decisions with timestamped provenance
- `memory_recall`: Adaptive-resolution recall with evidence chains
- `memory_expand`: Expand a compressed memory to verbatim resolution
- `memory_trace`: Trace any stored fact back to its exact conversation turn
- `memory_timeline`: Retrieve chronological timeline of an entity or topic
- `memory_explain`: Inspect why specific memories were selected or ranked

### Quickstart CLI

```bash
# Install
git clone https://github.com/Yato-Works/Lethe-Akribeia.git
cd Lethe-Akribeia
pip install -e ".[vector,llm]"
```

> [!NOTE]
> **Package Migration Notice**: For complete backward compatibility with existing workflows and our 82-test suite, internal package imports remain under `artificial_memory` in v0.2.0 (CLI binaries are updated to `lethe`). Full alias migration to `lethe` is scheduled for v0.3.0.

```bash
# Start a session
lethe start "Project/Akribeia"

# Log conversation (0 LLM cost)
lethe user "We decided to target 9/30 for the initial v0.2.0 release."
lethe assistant "Understood. The release deadline is set to September 30."

# Query memories
lethe recall "When is the release scheduled?"

# Inspect timeline & provenance
lethe timeline
```

---

## Hardware & Research Design

This project was built under a **deliberately constrained research budget** using a single local GPU (NVIDIA GeForce RTX 3050 / 8GB VRAM) and local models.

Evaluating 1,540 questions through commercial frontier APIs with massive contexts currently exceeds our budget.  
**Rather than hiding this constraint, Lethe converts it into research design**: by freezing its compiled contexts (`locomo_gold_context_cache.jsonl`), the memory engine and Reader models can be evaluated independently whenever additional compute becomes available.

---

## What Is NOT Finished

To remain scientifically rigorous, here is what is explicitly left as future work:

- [ ] **Full 1,540-question frontier Reader evaluation** (pending compute/API budget)
- [ ] **Systematic scaling laws beyond 7B** (14B vs 32B vs 70B vs Frontier)
- [ ] **Complex overlapping temporal interval resolution**
- [ ] **Distributed multi-agent consensus protocols** (Kubernetes Operator CRDs exist, but distributed consensus is experimental)

---

## Reproduction

All evaluation scripts, adapters, and scoring pipelines are fully reproducible:

```bash
# 1. Run unit tests (82 tests)
pytest tests/unit/

# 2. Run LoCoMo Official Scorer on baseline (Instruct 7B primary Reader)
python scripts/benchmarks/score_locomo_run_json.py --input benchmark_results/locomo1540/locomo_1540_improved2.json

# 3. Inspect Failure Ceiling breakdown
python scripts/benchmarks/failure_ceiling.py

# 4. Run BEAM benchmark (500K scale, evaluated with Coder 7B for strict schema adherence)
python scripts/run_coder7b_beam.py --scale 500K

# 5. Run Model Sensitivity pairing (1.5B vs 7B)
python scripts/benchmarks/model_sensitivity.py --baseline benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json
```

---

## Why Release Now?

I wanted to find out how far a structured, local memory system could go when hardware and models were small.

Lethe Akribeia v0.2.0 is not a finished monument. It is a working, auditable checkpoint.  
It has measurable strengths. It has measurable weaknesses.  
And now, it separates memory retrieval from Reader reasoning so that both can be evaluated systematically.

I still want to find out how far this architecture can go.

---

## Citation

```bibtex
@software{lethe_akribeia_2026,
  title = {Lethe Akribeia: An Experimental Cognitive Long-Term Memory System for AI},
  author = {Yato-Works},
  year = {2026},
  version = {0.2.0},
  url = {https://github.com/Yato-Works/Lethe-Akribeia}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.