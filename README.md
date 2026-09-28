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

## Evaluation Highlights

We do not present Lethe as a universal SOTA system. Rather, specific components reach benchmark ceilings, while 140+ hours of empirical evaluations reveal clear, actionable architectural boundaries:

> [!IMPORTANT]
> **Strict Evaluation Protocol & Data Separation**:  
> All benchmarks are evaluated strictly against **official testbeds** and **official LLM-as-a-Judge harnesses**. Prompts and hyperparameters were tuned strictly on separate diagnostic splits; the final benchmark sets (LoCoMo 1,540 questions, BEAM official suites) were **held completely separate** to prevent any data leakage or overfitting to specific test cases.

### What is Working vs. What is Unsolved

```text
What is working (Strong Results 🟢)
🟢 BEAM Benchmark (500K tokens)      — 100.0% accuracy across all 10 evaluated probing categories
🟢 BEAM Benchmark (1M & 10M tokens)   — 100.0% accuracy under extreme context horizons
🟢 LoCoMo Evidence-Presence Recall   — 81.1% on 1,540 non-adversarial questions (All-Evidence Oracle)
🟢 Zero-LLM Ingestion Write Path     — 0 LLM calls during memory ingestion (pure deterministic indexing)
🟢 Reader Invariance at Small Scales — 1.5B achieves 64.5% vs 7B at 64.7% (Lethe context absorbs model drop)
🟢 Deterministic Co-Processors        — Calendar arithmetic (CHRONOS) & Counting/aggregation without LLM

What is not solved (Current Limitations 🟡)
🟡 LoCoMo End-to-End QA F1           — 50.9% with deployed 7B Reader (synthesis & reasoning gap)
🟡 Relative Temporal Reasoning       — Multi-interval relative expressions remain challenging for 7B Readers
🟡 Frontier Reader Scale             — Full 1,540-question frontier Reader evaluation is pending compute budget
🟡 Multi-Agent Distributed Consensus — Kubernetes Operator CRDs exist; distributed consensus is experimental
```

### Benchmark Summary Table

| Benchmark / Evaluation | Result | Dataset / Scope | What it measures |
|:---|:---:|:---:|:---|
| **BEAM (500K scale)** | **100.0% (20/20)** | Official 10 categories (all 20 Qs) | Pinpoint extraction, contradiction detection, and event ordering from massive context |
| **BEAM (1M & 10M scales)** | **100.0% (8/8 each scale)** | Extreme probes (1M: 8 Qs, 10M: 8 Qs) | Needle retrieval and state tracking under extreme token budgets |
| **LoCoMo Evidence Recall** | **81.1%** | 1,540 non-adversarial questions | Whether **all** required gold evidence turns were compiled into context (Strict Content Oracle) |
| **LoCoMo Zero-Evidence Failure** | **8.7%** | 134 / 1,540 questions | Complete retrieval failure (no required evidence turns retrieved by the memory engine) |
| **LoCoMo Official QA F1** | **50.9%** | 1,540 questions (7B Reader) | End-to-end question answering using local 7B Reader |
| **LongMemEval (500Q)** | **81.6%** | 500 questions (7B Reader) | Long-term memory evaluation suite accuracy |

> [!NOTE]
> **Why is BEAM 100% while LoCoMo is 50.9%?**  
> BEAM tests long-context needle extraction, contradiction resolution, and event sequencing from structured chats — tasks where Lethe's deterministic timeline extraction and noise filtering excel. (Evaluated with `qwen2.5-coder:7b` for strict schema adherence).  
> In contrast, LoCoMo tests open-domain commonsense synthesis and personality deductions across casual dialogues, placing heavy demands on the Reader's intrinsic reasoning capacity (evaluated with `qwen2.5-7b-instruct` as primary conversational Reader).

---

## Model Sensitivity: The Reader Floor

A core empirical finding of over 140+ hours of benchmark sweeps is that **Lethe's structured context insulates against Reader downgrades**:

| Reader Model | Parameter Scale | LoCoMo 1,540Q Hits (Rate) | LoCoMo Official F1 | LongMemEval 500Q | Behavioral Profile |
|:---|:---:|:---:|:---:|:---:|:---|
| **Qwen 2.5 1.5B** | 1.5B | **993 / 1,540 (64.48%)** | **50.82%** | **80.0%** | Robust: minimal degradation despite 5x parameter drop |
| **Qwen 2.5 7B Instruct** | 7B | **996 / 1,540 (64.68%)** | **50.92%** | **81.6%** | Primary deployed Reader: strong conversational synthesis |
| **Qwen 2.5 7B Coder** | 7B | **996 / 1,540 (64.68%)** | **50.88%** | **81.5%** | Strict formatting adherence; identical hit count under same contract |

*Note on 7B Instruct vs Coder identical scores*: The matching hit count (996/1,540) and near-identical F1 (50.9%) is an empirical convergence result (coincidence) under the identical compiled context and answer-extraction prompts.

### Why 7B? — The Architectural Divide with 120B-Class Systems
Existing agent memory systems (Mem0, LangChain, Zep, etc.) typically assume frontier Readers (GPT-4, Claude 3.5, or 120B+ models). They dump thousands to tens of thousands of tokens of raw conversational history into prompts, relying on brute-force model capacity to filter and reason.  
If you mount a 7B or 1.5B local model to such systems, the massive context window bloats immediately, attention collapses, and the model fails to answer even a single question coherently.  
In contrast, Lethe Akribeia deterministically compiles memories into structured Memory State Contexts (MSC), distilling 100K+ token sessions into just a few hundred tokens. Because of this, **even a 1.5B or 7B model suffers zero context overflow and runs 1,540 benchmark questions continuously for 140 hours without collapsing**. Systems that only function with frontier models vs. systems that remain fully robust on 7B local hardware — this is the distinct arena Lethe defines.

### Context-Dominance Verification (Overlap Analysis)
Across all 1,540 questions, the 7B and 1.5B models **shared 863 identical correct answers** (and 413 identical wrong answers), with only 17.1% flipping outcome (McNemar test p = 0.95). This confirms that answer accuracy is predominantly driven by **the quality of Lethe's pre-compiled context**, not the Reader's intrinsic reasoning capacity.

#### Identical 10-Question Frozen Context Probe
To prevent misleading comparisons between the full 1,540-question local run (64.7%) and a small probe, the table below compares Readers strictly across the **exact same 10 questions** on the **identical frozen context**:

| Reader Model | Parameter Scale | Evaluation Set | Hit Rate | Official QA F1 | Validation Purpose |
|:---|:---:|:---:|:---:|:---:|:---|
| **Qwen 2.5 7B Instruct** | 7B | **Identical 10-Q Probe** | **80.0% (8/10)** | **82.17%** | Local baseline on frozen context |
| **Gemini 3.6 Flash** | Commercial Frontier | **Identical 10-Q Probe** | **90.0% (9/10)** | **76.38%** | Ceiling validation with frontier Reader |

This demonstrates that Lethe's pre-compiled context transfers seamlessly to frontier-grade models (9/10 Qs correct), providing empirical evidence that a substantial portion of the remaining gap on 7B is attributable to Reader reasoning capacity rather than retrieval omission. (Full 1,540-question frontier evaluation remains future work pending compute budget).


---

## Failure Ceiling: Error Anatomy on 1,540 Questions

Rather than treating errors as an undifferentiated failure score, Lethe partitions failure causes across all 1,540 questions in LoCoMo:

```
1,540 Total Questions
│
├── 996 (64.7%) Correctly Answered / Hit
├── 134 (8.7%)  Retrieval Failure   → Zero evidence turns reached the compiled context (Memory limit)
├── 32  (2.1%)  Commitment Failure  → Evidence was present, but Answer Committer rejected or abstained
└── 378 (24.5%) Reasoning Gap       → Evidence was present in context, but Reader failed to synthesize
```

### Clarifying Evidence Recall vs. Retrieval Failure
- **81.1% (1,249 / 1,540 questions)**: **All-Evidence Match** — every single required gold evidence turn was present in Lethe's compiled context.
- **8.7% (134 / 1,540 questions)**: **Zero-Evidence Failure** — the memory engine completely missed the gold evidence.
- **10.2% (157 / 1,540 questions)**: **Partial Retrieval** — some required evidence turns were retrieved, but not all (e.g., in multi-hop questions requiring multiple dates).

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

This project was built under a **deliberately constrained research budget** using a single local GPU (RTX 4080 / 16GB VRAM) and local models.

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