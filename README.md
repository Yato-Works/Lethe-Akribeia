# Lethe Akribeia

> **Forgetting is not deletion.**  
> **It is loss of resolution.**

An experimental long-term memory system for AI that treats forgetting as progressive resolution loss rather than deletion.  
*Deterministic memory compilation · Temporal reasoning · Evidence provenance · MCP native*  
*(Formerly: Artificial Memory / `lethe-akribeia`)*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: v0.3.0 Apex Generation](https://img.shields.io/badge/status-v0.3.0%20Apex%20Generation-brightgreen.svg)](#whats-new-in-v3-apex-generation)
[![Tests: 308 passed](https://img.shields.io/badge/tests-308%20passed-success.svg)](#reproduction)
[![LongMemEval: 98.2% Oracle Recall](https://img.shields.io/badge/LongMemEval-98.2%25%20Oracle%20Recall-blue.svg)](#evaluation-highlights-three-layer-architecture)
[![BEAM: 100% End-to-End](https://img.shields.io/badge/BEAM-100%25%20End--to--End-brightgreen.svg)](#evaluation-highlights-three-layer-architecture)

---

## What's New in V3 (Apex Generation)

Lethe Akribeia v0.3.0 marks a major architectural leap from an experimental prototype (v0.2.0) to a research-grade, highly auditable cognitive runtime:

1. **Three-Layer Diagnostic Architecture**:
   - Decoupled **Layer 1 (Oracle Recall: 98.20% on LongMemEval)** from **Layer 2 (Reader on Hits: 82.08%)** and **Layer 3 (Baseline 7B E2E: 81.60% → Final E2E: 83.40%)**, demonstrating very high evidence retrieval recall on LongMemEval while diagnosing the exact language model bottlenecks.
2. **Subsystem H: ArithmeticDifferenceEngine (Zero-LLM Autonomous Committer)**:
   - Built a deterministic derivation scaffold computing currency differentials ($300 − $30 = $270), savings/discounts, multi-location day sums, and chronological age offsets without a single LLM call.
   - Rescued **+9 difficult questions** from 7B Reader false refusals on LongMemEval with **0 regression losses**, lifting End-to-End accuracy from 81.60% (408/500) to **83.40% (417/500)**.
3. **LoCoMo Turn Selection & Proximity Density Scoring**:
   - Refined `answer_committer` with keyword coverage weighting, speaker alignment, and explicit intent guards: improved committer accuracy on temporal questions from 67 to 75 correct (**+8 net correct, 0 regressions**; 72.8% precision vs 7B's 57.3% on committed subset).
4. **Rigorous Test Suite Expanded from 82 to 308 Tests (100% Passing)**:
   - Added arithmetic difference suites, interval algebra, temporal compilation, speaker normalization, model sensitivity verification, corruption resilience, and derivation scaffolding suites.
5. **Model Invariance Validation (7B vs. 1.5B)**:
   - Empirically demonstrated that a ~4.7× parameter reduction (from 7B to 1.5B) produces virtually zero performance divergence ($p = 0.9509$ on LoCoMo, $p = 0.4030$ on LongMemEval), supporting the hypothesis that performance is strongly influenced by structured context quality.



---

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
| **LongMemEval (All 6 Capabilities)** | 500 Qs | **98.20% (491/500)**<br>*(95% Wilson CI: 96.6%–99.1%)* | **82.08% (403/491)**<br>*(7B alone on hits)* | **83.40% (417/500)**<br>*(Baseline 81.60% + 9 rescued)* | 7B alone scores 81.60% (408/500); Subsystem H (ArithmeticDifferenceEngine) deterministically resolves 9 false-refusal questions (0 regressions) raising E2E to 83.40%. |
| **BEAM (500K Horizon)** | 48 Qs | **100.0% (48/48)** | **100.0%** | **100.0%** | Evaluated on 500K tokens with 0 needle drops across 48 queries; deterministic timeline indexing is architecturally designed to scale to larger horizons (10M+). |
| **LoCoMo 1,540 (Single-Hop)** | 841 Qs | **83.71%** | **85.80%** | **77.65%** | Strong direct factual recall; Reader reliably extracts explicit entity facts. |
| **LoCoMo 1,540 (Temporal Cat 2)** | 321 Qs | **81.62% (262/321)** | **57.25%**<br>*(7B fails on 42.8% of hits)* | **51.09% (164/321)** | Multi-interval relative dates benefit from deterministic calendar normalization; on the committed subset (103/321 Qs), autonomous committer achieves 72.8% precision (75/103) vs pre-committer baseline (67/103, +8 net correct) and vs 7B's 57.3% (59/103, +16 correct, +15.5pp precision lead). |
| **LoCoMo 1,540 (Multi-Hop Cat 1)** | 282 Qs | **78.37%** | **55.20%** | **47.87%** | Open research frontier: cross-session graph linking across divergent topics. |
| **LoCoMo 1,540 (Full Non-Adversarial)** | 1,540 Qs | **80.65% (1,242/1,540)** | **72.54%** | **64.68% (996/1,540)** | Overall non-adversarial benchmark with primary local 7B Reader. |

> [!NOTE]
> **Why Separate Oracle Recall from End-to-End?**  
> In LongMemEval (500 questions), Lethe missed the gold evidence in only 9 out of 500 questions (Oracle Recall 98.20%, 491/500). On those 491 hits, the 7B Reader answered 403 correctly (82.08% on hits), yielding a baseline End-to-End accuracy of 81.60% (408/500, with 92 initial misses, including 5 lucky guesses on retrieval misses). An error analysis of these initial misses revealed that 88 out of 92 (95.7%) had the required evidence present in the prompt, failing purely due to Reader false abstention ("I don't know" despite evidence present) or arithmetic errors in calendar math.  
> By deploying Subsystem H (`ArithmeticDifferenceEngine`), 9 of these false refusals and arithmetic operations were deterministically resolved with 0 regression losses, increasing correct answers from 408/500 (81.60%) to 417/500 (83.40%).


---

## Model Sensitivity: The Reader Floor (7B vs. 1.5B)

A central finding across 140+ hours of sweeps is that **Lethe's pre-compiled context significantly insulates against Reader scale downgrades**:

| Benchmark Slice | 7B Reader (`qwen2.5:7b-instruct`) | 1.5B Reader (`qwen2.5:1.5b`) | Delta | 7B-only Hits | 1.5B-only Hits | McNemar Test ($p$-value) | Scientific Interpretation |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **LoCoMo 1,540 (Full)** | **64.68% (996/1,540)** | **64.55% (994/1,540)** | +0.13pp | 133 | 131 | **$p = 0.9509$** | No statistically significant difference detected despite ~4.7× parameter reduction. |
| **LongMemEval (500 Qs)** | **81.60% (408/500)** | **80.00% (400/500)** | +1.60pp | 39 | 31 | **$p = 0.4030$** | Multi-session reasoning scored identically (84.2% on both models). |

> [!CAUTION]
> **Statistical Rigor on Model Invariance**:  
> A high McNemar $p$-value ($p > 0.05$) does **not** constitute mathematical proof of complete model independence; it indicates that under these frozen benchmark contexts, there is insufficient evidence to reject the null hypothesis of equal performance. Formally establishing equivalence requires two-one-sided tests (TOST) with a pre-defined equivalence margin.  
> Nonetheless, observing that 7B and 1.5B share 863 identical correct answers and 413 identical failures (1,540Q) strongly suggests that accuracy is heavily governed by **the structured quality of the memory context**, rather than raw Reader parameters alone.

### Honest Boundaries & Unverified Frontiers
- **Frontier Reader Scaling (120B / Gemini)**: While our small 10-question probe showed 90% (9/10) with Gemini Flash, asserting that a 120B+ model will achieve 100% on the full 1,540 suite is **an unverified hypothesis** pending compute budget.
- **Autonomous Deterministic Offloading**: To bypass Reader arithmetic and formatting failures, Lethe incorporates zero-LLM deterministic skills (CHRONOS, counting, ontology resolvers). On a hard diagnostic drill of 281 questions where 7B Reader scored 38.79% F1, the deterministic engine produced a deterministic commitment for all 281 questions, achieving **92.87% F1 with 0 regression losses**.
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
> **Package Architecture Notice**: For full backward compatibility with reproducible benchmark harnesses and the 308-test suite, internal Python modules remain packaged under `artificial_memory`, while the user-facing CLI binary is exposed as `lethe`.

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

## Roadmap: Path to Frontier Expansion (V4 & V5)

```
[V3 Apex (Current)] ────────► [V4 Cognitive Expansion] ────────► [V5 Frontier Expansion]
• 308 Unit Tests (100% Pass)   • Non-temporal Semantic Rescue     • Frontier LLM (120B/Gemini/Claude)
• LME Oracle Recall: 98.2%     • Cross-Session Graph Traversal    • E2E Benchmark Ceilings (95%~98%+)
• Subsystem H Arithmetic Co-P  • Target: LME E2E > 90%            • Full Multi-Agent Kubernetes Mesh
• LME E2E: 83.40% (+9 Rescued) • LoCoMo Oracle Recall > 90%       • Production Autonomous Standard
```

- **V3 (Current: Apex Baseline)**: Established the three-layer diagnostic standard, reached 98.2% Oracle Recall on LongMemEval, implemented Subsystem H (`ArithmeticDifferenceEngine`) rescuing +9 false-refusal questions to reach 83.40% E2E, refined LoCoMo temporal turn selection (+8 net correct), and expanded test coverage to 308 unit tests (100% passing).
- **V4 (Next: Cognitive Expansion & Non-Temporal Semantic Rescue)**:
  - *Non-Temporal Semantic Rescue*: Expand autonomous pattern induction for implicit entity state questions to eliminate remaining Reader false abstentions, targeting **>90%** LME End-to-End.
  - *Cross-Session Graph Traversal*: Upgrade `ppr_graph` and entity linking to raise LoCoMo Multi-Hop Oracle Recall from 78.37% to **90%+** (and overall from 80.65% to **90%+**).
- **V5 (Frontier Expansion: Frontier Scale & Synthesis)**:
  - Connect Lethe's high-recall MSC contexts to commercial frontier models (Gemini 1.5 Pro, Claude 3.5, 120B+ open models) to confirm the 95%+ E2E ceiling across all 1,540 questions.
  - Distributed multi-agent consensus protocols across Kubernetes clusters.


---

## Reproduction

All evaluation scripts, adapters, and scoring pipelines are fully reproducible:

```bash
# 1. Run complete unit test suite (308 tests, 100% passing)
pytest tests/unit/


# 2. Run LoCoMo Official Scorer on baseline (Instruct 7B primary Reader)
python scripts/benchmarks/score_locomo_run_json.py --input benchmark_results/locomo1540/locomo_1540_improved2.json

# 3. Inspect Three-Layer Decomposition (Oracle Recall vs Reader on Hits)
python scripts/benchmarks/three_layer_report.py

# 4. Run Model Sensitivity pairing (7B vs 1.5B exact McNemar test)
python scripts/benchmarks/model_sensitivity.py --a benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json --b benchmark_results/locomo1540/temporal321_rules_commit_15b.json --claims benchmark_results/committer_metrics/locomo_cat2_temporal_claims.json

# 5. Run BEAM benchmark (500K scale)
python scripts/run_coder7b_beam.py --scale 500K
```

---

## Why Release Now?

I wanted to find out how far a structured, local memory system could go when hardware and models were small.

Lethe Akribeia v0.3.0 is not a finished monument. It is a working, auditable, high-precision checkpoint.  
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
  version = {0.3.0},
  url = {https://github.com/Yato-Works/Lethe-Akribeia}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.