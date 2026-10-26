# Phase 0 Baseline Report: Label-Free V4 (leak-free)

## Configuration
- Dataset: `datasets/external/longmemeval_s_cleaned.json`
- Split: Dev (250 questions)
- Token Budget: 1300 tokens
- Reader Model: qwen2.5:7b-instruct
- Type Classifier: deterministic_keywords
- LLM cache: `cache/phase0` (234 cache hits)

## Leakage Boundary (regression-tested)
- NO gold `question_type` used in routing or inference
- NO gold `answer` or `answer_session_ids` used in prompt construction
- Label-free classifier uses question text only
- Gold labels enter ONLY the grader (scoring) and offline recall measurement
- Holdout split (`lme_holdout_qids.json`) never accessed
- Guarded by `tests/unit/test_no_label_leak.py`

## 1. Label-Free Type Classification
- **Accuracy: 62.4%** (156/250)

### Per-Type Classification Metrics
| Type | Precision | Recall | F1 | Support |
|------|-----------|--------|----|---------|
| knowledge-update | 0.48 | 0.26 | 0.333 | 39 |
| multi-session | 0.68 | 0.67 | 0.677 | 67 |
| single-session-assistant | 0.75 | 0.64 | 0.692 | 28 |
| single-session-preference | 0.83 | 0.67 | 0.741 | 15 |
| single-session-user | 0.39 | 0.54 | 0.452 | 35 |
| temporal-reasoning | 0.69 | 0.82 | 0.750 | 66 |

## 2. Evidence Recall @ Budget (structural session-id sets)
- **Mean Recall@Retrieved (candidate pool):** 100.0%
- **Mean Recall@Selected (post-budget):** 87.1%
- Mean Gold-Turn Recall: N/A (no turn-level gold)
- Recall evaluation errors: 0

> Note: the current pipeline indexes the full haystack into the evidence graph,
> so `recall_retrieved` is 1.0 by construction and is reported for completeness.
> `recall_selected` (the L2 bottleneck) is the meaningful signal.

### Per-Type Recall
| Type | Recall@Retrieved | Recall@Selected | N |
|------|------------------|-----------------|----|
| knowledge-update | 100.0% | 91.0% | 39 |
| multi-session | 100.0% | 81.0% | 67 |
| single-session-assistant | 100.0% | 100.0% | 28 |
| single-session-preference | 100.0% | 93.3% | 15 |
| single-session-user | 100.0% | 100.0% | 35 |
| temporal-reasoning | 100.0% | 77.3% | 66 |

## 3. V4 Pipeline Results (Label-Free, leak-free)

### Overall Metrics
- **Accuracy: 31.2%**
- **Mean Prompt Tokens: 1756**
- **Acc/1k tokens: 0.18**  (Acc / (tokens/1000))
- **Errors: 0** (recorded with `status="error"` + traceback, never silently dropped)

### 95% Bootstrap Confidence Intervals
- Accuracy: [25.6%, 37.2%]
- Tokens: [1584, 1697]

### Per-Type Pipeline Results
| Type | Accuracy | Correct | Total | Mean Tokens | Errors |
|------|----------|---------|-------|-------------|--------|
| knowledge-update | 56.4% | 22 | 39 | 1737 | 0 |
| multi-session | 7.5% | 5 | 67 | 1746 | 0 |
| single-session-assistant | 53.6% | 15 | 28 | 1746 | 0 |
| single-session-preference | 0.0% | 0 | 15 | 1758 | 0 |
| single-session-user | 71.4% | 25 | 35 | 1715 | 0 |
| temporal-reasoning | 16.7% | 11 | 66 | 1377 | 0 |

### Type Prediction Accuracy
- **156/250 (62.4%)** classified correctly

### Deterministic 0-token Overrides
- 16/250 questions answered by certificates without a reader call
