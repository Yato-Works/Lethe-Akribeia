# Lethe Akribeia

> **Forgetting is not deletion.**  
> **It is loss of resolution.**

An experimental long-term memory system for AI that treats forgetting as progressive resolution loss rather than deletion.  
*Deterministic memory compilation · Temporal reasoning · Evidence provenance · MCP native*  
*(Formerly: Artificial Memory)*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Experimental v0.2.0](https://img.shields.io/badge/status-experimental%20v0.2.0-orange.svg)](#why-release-now)
[![LoCoMo: 1540 Benchmark](https://img.shields.io/badge/benchmark-LoCoMo%201540-green.svg)](#evaluation-highlights)
[![BEAM: 100K--10M](https://img.shields.io/badge/benchmark-BEAM%20100%25-brightgreen.svg)](#evaluation-highlights)

---

## What is Lethe?

Lethe Akribeia is an open-source cognitive memory runtime designed for long-term AI persistence.  
Unlike standard vector RAG or naive context stuffing, Lethe compiles conversational history into structured, resolution-tiered memory states using **zero LLM calls on ingestion**, attaches deterministic calendar and aggregation co-processors, and preserves full evidentiary provenance back to the exact source turn.

---

## Evaluation Highlights

We do not present Lethe as a universal SOTA system. However, specific components already perform at the ceiling of current long-context benchmarks, while full pipeline evaluations expose clear architectural insights:

### What is Working vs. What is Unsolved

```text
What is working (Strong Results)
🟢 BEAM Benchmark (500K tokens)      — 100.0% Accuracy across all 10 evaluated probing categories
🟢 BEAM Benchmark (1M & 10M tokens)   — 100.0% Accuracy on evaluated long-horizon probes
🟢 LoCoMo Evidence-Presence Recall   — 81.1% on 1,540 non-adversarial questions (Corrected Oracle)
🟢 Zero-LLM Ingestion Write Path     — 0 LLM calls during memory compilation (deterministic indexing)
🟢 Deterministic Co-Processors        — Calendar arithmetic (CHRONOS) & Counting/aggregation without LLM
🟢 Decoupled Reader Evaluation       — Frozen-context testbed isolating memory retrieval from Reader reasoning

What is not solved (Current Limitations)
🟡 LoCoMo End-to-End QA F1           — 50.9% with deployed local 7B Reader (reasoning bottleneck)
🟡 Temporal Reasoning Accuracy       — Relative date intervals remain challenging for 7B models
🟡 Frontier Reader Probe Scope       — 76.38% F1 observed on a 10-question probe; full 1,540Q pending budget
🟡 Federated Multi-Agent Protocol    — Operator orchestration exists; distributed consistency is ongoing
```

### Benchmark Summary Table

| Benchmark / Evaluation | Result | What it measures |
|:---|:---:|:---|
| **BEAM (500K scale, 10 categories)** | **100.0%** | Probing accuracy across multi-session, contradiction, temporal, event ordering |
| **BEAM (1M & 10M scales)** | **100.0%** | Probing accuracy under extreme context horizons |
| **LoCoMo Evidence Recall** (1,540Q) | **81.1%** | Whether required evidence was successfully retrieved into Lethe's compiled context |
| **LoCoMo Official QA F1** (1,540Q) | **50.9%** | End-to-end answer accuracy using the deployed local 7B Reader |
| **Frozen Context Probe (7B)** | **50.92% F1** | Baseline local 7B Reader on 10 probe questions (64.68% binary hit) |
| **Frozen Context Probe (Frontier)** | **76.38% F1** | Gemini 3.6 Flash on the **exact same frozen context** (90.00% binary hit, **+25.46 pp**) |

> [!IMPORTANT]
> **81.1% evidence recall is not equivalent to 81.1% QA accuracy.**  
> Retrieving the correct evidence into context does not guarantee that a compact 7B Reader can synthesize and reason over it to produce the exact answer string.

---

## The Key Question

When an end-to-end memory benchmark fails, **is the bottleneck the memory retrieval, or the Reader model?**

```
                 SAME FROZEN LETHE CONTEXT
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       Local 7B Reader              Frontier Reader
      (qwen2.5-coder:7b)           (Gemini 3.6 Flash)
              │                             │
          50.92 F1                      76.38 F1
          64.68 Hit                     90.00 Hit
              │                             │
              └──────────────┬──────────────┘
                             ▼
                 +25.46 pp F1 (+25.32 pp Hit)
```

In standard agent pipelines, memory and generation are conflated into a single metric. Lethe decouples them: by freezing the compiled memory context, we can evaluate memory retrieval independently from Reader synthesis.

---

## Frozen Context Probe: Reader Sensitivity

To test whether the 7B Reader was underutilizing Lethe's compiled context, we ran a validation probe across identical contexts and questions:

| Category | Deployed 7B Reader (F1) | Gemini 3.6 Flash (F1) | Delta |
|:---|:---:|:---:|:---:|
| **Multi-hop Reasoning** | 64.7% | **100.0%** | **+35.3 pp** |
| **Temporal Reasoning** | 50.2% | **83.3%** | **+33.1 pp** |
| **Open-domain / Factual** | 39.6% | **100.0%** | **+60.4 pp** |
| **Overall F1** | 50.92% | **76.38%** | **+25.46 pp** |
| **Binary Hit Rate** | 64.68% | **90.00%** | **+25.32 pp** |

> [!NOTE]
> **This is a 10-question validation probe, not a full 1,540-question claim.**  
> However, it provides strong preliminary evidence: when the compiled context is held strictly constant, a frontier-class Reader extracts answers with substantially higher precision (+25.46 pp F1). This indicates that the 7B Reader represents a significant portion of the remaining error surface.

---

## Failure Ceiling: Error Census on 1,540 Questions

Rather than treating errors as an undifferentiated failure score, Lethe provides a diagnostic framework that separates failure causes across all 1,540 non-adversarial questions in LoCoMo:

```
1,540 Total Questions
│
├── 996 (64.7%) Correctly Answered / Hit
├── 134 (8.7%)  Retrieval Failure   → Required evidence was missing from compiled context (Memory limit)
├── 32  (2.1%)  Commitment Failure  → Evidence was present, but Answer Committer rejected/abstained
└── 378 (24.5%) Reasoning Gap       → Evidence was present in context, but Reader failed to synthesize
```

By isolating retrieval failure from Reader reasoning failure, future research can target the actual bottleneck rather than blindly tweaking prompts.

---

## Why Lethe? (Core Principles)

### 1. Forgetting = Loss of Resolution, Not Deletion
Human memory does not drop files into a recycle bin. Over time, memories decay in resolution:
- **Level 0 (RAW)**: Full verbatim conversation turns
- **Level 1 (EPISODIC)**: Structured event records with speaker and tone
- **Level 2 (CONDENSED)**: Salient conversational points and factual assertions
- **Level 3 (FACT/STATE)**: Entity state changes and verified decisions
- **Level 4 (ANCHOR)**: High-level durable life facts and long-term beliefs

Queries start at low token cost and expand resolution dynamically only when ambiguity demands it.

### 2. Zero-LLM Ingestion & Deterministic Co-Processors
Memory ingestion does not rely on non-deterministic LLM summarization:
- **Write LLM Calls = 0**: Conversation turns are ingested and token-indexed deterministically.
- **CHRONOS Temporal Co-processor**: Computes calendar arithmetic deterministically (resolving relative expressions like "last Tuesday" or "three months ago" against conversation timestamps).
- **Aggregation Co-processor**: Handles counting, lists, and frequency queries deterministically.
- **Answer Committer**: Enforces an answer contract to prevent hallucinated drift.

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
git clone https://github.com/Yato-Works/artificial-memory.git
cd artificial-memory
pip install -e ".[vector,llm]"

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

This project was built under a **deliberately constrained research budget** using a single local GPU (RTX 4080 / 16GB VRAM) and a local 7B-class model (`qwen2.5-coder:7b`).

Running 1,540 questions through commercial frontier APIs with long contexts currently exceeds our available research budget.  
**Rather than hiding this constraint, Lethe converts it into research design**: by freezing its compiled contexts (`locomo_gold_context_cache.jsonl`), the memory system and Reader can be evaluated independently whenever additional compute becomes available.

---

## What Is NOT Finished

To remain scientifically rigorous, here is what is explicitly left as future work:

- [ ] **Full 1,540-question frontier Reader evaluation** (pending compute/API budget)
- [ ] **Comprehensive Reader scaling laws** (7B vs 14B vs 32B vs 70B vs Frontier)
- [ ] **Complex overlapping temporal interval resolution**
- [ ] **Full LongMemEval and PersonaMem local benchmark suites**
- [ ] **Distributed multi-agent consensus protocols** (Kubernetes Operator CRDs exist, but distributed consensus is experimental)

---

## Reproduction

All evaluation scripts, adapters, and scoring pipelines are fully reproducible:

```bash
# 1. Run unit tests (82 tests)
pytest tests/unit/

# 2. Run LoCoMo Official Scorer on baseline
python scripts/benchmarks/score_locomo_run_json.py --input benchmark_results/locomo1540/locomo_1540_improved2.json

# 3. Inspect Failure Ceiling breakdown
python scripts/benchmarks/failure_ceiling.py

# 4. Run BEAM benchmark
python scripts/run_coder7b_beam.py --scale 500K

# 5. Run Frontier Reader probe on frozen context (requires GEMINI_API_KEYS)
python scripts/benchmarks/run_frontier_eval.py --limit 10
```

---

## Why Release Now?

I wanted to find out how far a structured, local memory system could go when the hardware and models were small.

Lethe Akribeia v0.2.0 is not a finished monument. It is a working, auditable checkpoint.  
It has measurable strengths. It has measurable weaknesses.  
And now, it separates memory retrieval from Reader reasoning so that both can be improved systematically.

I still want to find out how far this architecture can go.

---

## Citation

```bibtex
@software{lethe_akribeia_2026,
  title = {Lethe Akribeia: An Experimental Cognitive Long-Term Memory System for AI},
  author = {Yato-Works},
  year = {2026},
  version = {0.2.0},
  url = {https://github.com/Yato-Works/artificial-memory}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.