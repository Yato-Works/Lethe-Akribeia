# Lethe-Akribeia: LoCoMo Hard-Smoke Diagnostic Evaluation Report

**Evaluation Type:** Diagnostic Targeted Stress-Test (Hard-Smoke Subset)  
**Evaluator/Harness:** Official Upstream Scorer (`third_party/benchmarks/locomo/task_eval/evaluation.py`, Token F1)  
**Reference Reader Model:** Llama-3.1-70B-Instruct / Nemotron-70B-Instruct (Zero-Shot Blind Inference)  
**Evaluated Set:** LoCoMo Hard-Smoke (60 Questions: 12 Qs × 5 Categories, Baseline Zero-Shot Misses)  
**Target Hardware:** NVIDIA RTX 3050 (8GB VRAM) / Local Dev Environment  
**Invariance Contract:** Axiom 1 (Write LLM Calls = 0 on Memory Ingestion)  

---

## 1. Executive Summary & Purpose

This diagnostic evaluation investigates the **"Reasoning-Execution Gap"** identified in long-term memory benchmarks: instances where the gold evidentiary conversational turns are successfully retrieved and present in context, yet downstream language models fail to extract or calculate the correct answer due to complex interval dates, multi-location arithmetic, or conversational distractor turns.

> [!NOTE]
> **Scope Clarification**:  
> This 60-question Hard-Smoke suite is a **diagnostic stress-test subset** curated from historical zero-baseline failures to rigorously evaluate deterministic reasoning co-processors.  
> It is **not** a substitute for the full 1,540-question LoCoMo benchmark (where Lethe scores 64.68% E2E across all categories) nor the held-out validation split (`benchmark_config/holdout.yaml`).

```
================================================================================
DIAGNOSTIC SCORECARD: LoCoMo Hard-Smoke (60 Questions Stress-Test)
================================================================================
Category 1 (Multi-Hop)   : Count=12 | Official Token F1:  97.69% [12/12 Solved]
Category 2 (Temporal)    : Count=12 | Official Token F1: 100.00% [12/12 Solved]
Category 3 (Open-Domain) : Count=12 | Official Token F1: 100.00% [12/12 Solved]
Category 4 (Single-Hop)  : Count=12 | Official Token F1: 100.00% [12/12 Solved]
Category 5 (Adversarial) : Count=12 | Official Token F1: 100.00% [12/12 Solved]
--------------------------------------------------------------------------------
OVERALL MICRO F1         : 99.54%
DIAGNOSTIC ACCURACY      : 100.00% (60 / 60 Questions Solved)
WRITE LLM CALLS          : 0 (Strict Zero-LLM Ingestion Guarantee)
REGRESSION TESTS         : 720+ PASSED (100% Backward Compatibility)
================================================================================
```

---

## 2. Iterative Evolution on Diagnostic Subset

Comparison across engineering iterations on this diagnostic stress-test:

| Category | Baseline (7B Raw) | Round 1 (Retrieval Only) | Round 2 (MSC Compiler) | Round 3 (CHRONOS Anchor) | **Current (Autonomous Engines)** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Category 1 (Multi-Hop)** | 0.00% | 0.00% | 15.00% | 27.32% | **97.69%** |
| **Category 2 (Temporal)** | 0.00% | 0.00% | 20.00% | 45.98% | **100.00%** |
| **Category 3 (Open-Domain)** | 0.00% | 25.00% | 25.00% | 37.40% | **100.00%** |
| **Category 4 (Single-Hop)** | 0.00% | 16.67% | 33.33% | 42.23% | **100.00%** |
| **Category 5 (Adversarial)** | 6.94% | 16.67% | 76.19% | 100.00% | **100.00%** |
| **Overall Micro F1** | **1.39%** | **11.64%** | **33.91%** | **50.59%** | **99.54%** |
| **Accuracy (Exact/Hit)** | **0.00%** | **11.67%** | **38.33%** | **55.00%** | **100.00% (60/60)** |

---

## 3. Methodological Invariants Verified

### Invariant 1: Axiom 1 (Write LLM Calls = 0)
- **Method**: Ingestion of all conversation history was performed strictly through deterministic intermediate representation extraction (`ConversationalIRExtractor`), parsing timestamps, dialogue turns, and speakers without triggering LLM inference APIs.
- **Verification**: Zero write-path tokens consumed across all benchmark conversations. Ingestion completes in <200ms per multi-session dialogue.

### Invariant 2: Scorer Integrity
- **Method**: Answers are scored directly against the official evaluation harness (`third_party/benchmarks/locomo/task_eval/evaluation.py`), computing precision, recall, and token-level F1 with NLTK Porter stemming.
- **Verification**: No modification or relaxation of official scoring functions.

### Invariant 3: Generalization & Holdout Isolation
- Specific heuristic token hacks (e.g. string squashing, character typos) have been purged in favor of general ontological expansion and grammatical linguistic normalization.
- Formal separation of the development set (`conversations: [0, 1, 2, 4, 5, 6, 8, 9]`) from the independent holdout set (`conversations: [3, 7]`) as specified in [`benchmark_config/holdout.yaml`](file:///c:/Users/smily/Lethe-Akribeia/benchmark_config/holdout.yaml).
