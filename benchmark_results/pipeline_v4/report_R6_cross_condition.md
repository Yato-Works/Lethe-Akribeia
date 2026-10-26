# R6 cross-condition comparison (LongMemEval dev 250)

Conditions: R0, R1, R2, R3, R23, R6
Baseline: R0 (lme_dev250_v4_pipeline.json)

## Headline

| cond | accuracy | correct | mean tokens | mean latency | d_acc vs R0 | d_tok vs R0 |
|---|---|---|---|---|---|---|
| R0 | 73.6% | 184/250 | 1329 | 998ms | +0.0pp | +0 |
| R1 | 70.0% | 175/250 | 1329 | 6250ms | -3.6pp | +0 |
| R2 | 74.0% | 185/250 | 1264 | 948ms | +0.4pp | -65 |
| R3 | 73.2% | 183/250 | 1347 | 1079ms | -0.4pp | +18 |
| R23 | 73.6% | 184/250 | 1279 | 1006ms | +0.0pp | -50 |
| R6 | 75.6% | 189/250 | 1288 | 982ms | +2.0pp | -41 |

## Accuracy by question type

| question_type | n | R0 | R1 | R2 | R3 | R23 | R6 |
|---|---|---|---|---|---|---|---|
| single-session-user | 35 | 97.1% | 97.1% | 94.3% | 97.1% | 94.3% | 97.1% |
| multi-session | 67 | 38.8% | 35.8% | 40.3% | 40.3% | 41.8% | 49.3% |
| single-session-preference | 15 | 86.7% | 80.0% | 86.7% | 86.7% | 86.7% | 86.7% |
| temporal-reasoning | 66 | 77.3% | 74.2% | 78.8% | 74.2% | 75.8% | 77.3% |
| knowledge-update | 39 | 92.3% | 82.1% | 92.3% | 92.3% | 92.3% | 87.2% |
| single-session-assistant | 28 | 85.7% | 85.7% | 85.7% | 85.7% | 85.7% | 85.7% |

## Tokens by question type

| question_type | n | R0 | R1 | R2 | R3 | R23 | R6 |
|---|---|---|---|---|---|---|---|
| single-session-user | 35 | 1208 | 1208 | 1160 | 1212 | 1165 | 1171 |
| multi-session | 67 | 1515 | 1515 | 1393 | 1568 | 1436 | 1440 |
| single-session-preference | 15 | 1110 | 1110 | 1054 | 1110 | 1054 | 1069 |
| temporal-reasoning | 66 | 1240 | 1240 | 1183 | 1241 | 1184 | 1205 |
| knowledge-update | 39 | 1296 | 1296 | 1264 | 1313 | 1280 | 1276 |
| single-session-assistant | 28 | 1409 | 1409 | 1389 | 1409 | 1389 | 1403 |

## R6 feature means (all questions)

| feature | R0 | R1 | R2 | R3 | R23 | R6 |
|---|---|---|---|---|---|---|
| evidence_coverage | 0.458 | 0.458 | 0.458 | 0.458 | 0.458 | 0.461 |
| budget_util | 0.746 | 0.746 | 0.746 | 0.746 | 0.746 | 0.755 |
| recommended_budget | 704.400 | 704.400 | 704.400 | 704.400 | 704.400 | 704.400 |
| top1_anchor_score | 70.949 | 70.949 | 70.949 | 70.949 | 70.949 | 70.920 |
| candidate_count | 91.520 | 91.520 | 91.520 | 91.520 | 91.520 | 118.352 |
| candidate_sessions | 8.116 | 8.116 | 8.116 | 8.116 | 8.116 | 10.472 |
| competing_values_count | 24.240 | 24.240 | 24.240 | 24.240 | 24.240 | 24.240 |
| state_evolution_demand | 0.521 | 0.521 | 0.521 | 0.521 | 0.521 | 0.521 |
| aggregation_demand | 0.285 | 0.285 | 0.285 | 0.285 | 0.285 | 0.285 |
| temporal_demand | 0.172 | 0.172 | 0.172 | 0.172 | 0.172 | 0.172 |
| assistant_recall_demand | 0.171 | 0.171 | 0.171 | 0.171 | 0.171 | 0.171 |
| preference_demand | 0.053 | 0.053 | 0.053 | 0.053 | 0.053 | 0.053 |
| direct_fact_demand | 0.171 | 0.171 | 0.171 | 0.171 | 0.171 | 0.171 |
| requires_state_timeline | 0.724 | 0.724 | 0.724 | 0.724 | 0.724 | 0.724 |
| requires_event_aggregation | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 | 0.300 |
| requires_chronos_diff | 0.176 | 0.176 | 0.176 | 0.176 | 0.176 | 0.176 |
| is_aggregation | 0.216 | 0.216 | 0.216 | 0.216 | 0.216 | 0.216 |

## R6 feature means (correct only)

| feature | R0 | R1 | R2 | R3 | R23 | R6 |
|---|---|---|---|---|---|---|
| evidence_coverage | 0.521 | 0.538 | 0.519 | 0.518 | 0.515 | 0.501 |
| budget_util | 0.831 | 0.857 | 0.831 | 0.825 | 0.826 | 0.812 |
| recommended_budget | 680.707 | 677.429 | 682.703 | 683.060 | 685.054 | 689.683 |
| top1_anchor_score | 75.783 | 74.778 | 75.419 | 75.569 | 75.333 | 72.307 |
| candidate_count | 86.766 | 85.411 | 87.454 | 87.656 | 87.978 | 119.090 |
| candidate_sessions | 7.663 | 7.531 | 7.730 | 7.743 | 7.777 | 10.508 |
| competing_values_count | 24.152 | 24.240 | 24.195 | 24.175 | 24.223 | 24.190 |
| state_evolution_demand | 0.556 | 0.573 | 0.562 | 0.563 | 0.565 | 0.554 |
| aggregation_demand | 0.207 | 0.190 | 0.211 | 0.213 | 0.217 | 0.231 |
| temporal_demand | 0.192 | 0.196 | 0.191 | 0.187 | 0.186 | 0.187 |
| assistant_recall_demand | 0.186 | 0.206 | 0.190 | 0.187 | 0.191 | 0.191 |
| preference_demand | 0.067 | 0.071 | 0.067 | 0.067 | 0.067 | 0.065 |
| direct_fact_demand | 0.180 | 0.180 | 0.179 | 0.180 | 0.178 | 0.175 |
| requires_state_timeline | 0.777 | 0.800 | 0.784 | 0.787 | 0.788 | 0.772 |
| requires_event_aggregation | 0.217 | 0.200 | 0.222 | 0.224 | 0.228 | 0.243 |
| requires_chronos_diff | 0.196 | 0.200 | 0.195 | 0.191 | 0.190 | 0.190 |
| is_aggregation | 0.136 | 0.109 | 0.135 | 0.142 | 0.141 | 0.164 |

## R6 feature means (incorrect only)

| feature | R0 | R1 | R2 | R3 | R23 | R6 |
|---|---|---|---|---|---|---|
| evidence_coverage | 0.284 | 0.274 | 0.287 | 0.297 | 0.300 | 0.335 |
| budget_util | 0.512 | 0.488 | 0.505 | 0.531 | 0.525 | 0.576 |
| recommended_budget | 770.455 | 767.333 | 766.154 | 762.687 | 758.333 | 750.000 |
| top1_anchor_score | 57.473 | 62.014 | 58.227 | 58.331 | 58.726 | 66.623 |
| candidate_count | 104.773 | 105.773 | 103.092 | 102.075 | 101.394 | 116.066 |
| candidate_sessions | 9.379 | 9.480 | 9.215 | 9.134 | 9.061 | 10.361 |
| competing_values_count | 24.485 | 24.240 | 24.369 | 24.418 | 24.288 | 24.393 |
| state_evolution_demand | 0.422 | 0.399 | 0.403 | 0.405 | 0.397 | 0.418 |
| aggregation_demand | 0.504 | 0.507 | 0.497 | 0.482 | 0.475 | 0.452 |
| temporal_demand | 0.119 | 0.118 | 0.121 | 0.132 | 0.134 | 0.129 |
| assistant_recall_demand | 0.130 | 0.089 | 0.117 | 0.128 | 0.115 | 0.109 |
| preference_demand | 0.014 | 0.013 | 0.015 | 0.014 | 0.014 | 0.016 |
| direct_fact_demand | 0.145 | 0.151 | 0.149 | 0.148 | 0.152 | 0.159 |
| requires_state_timeline | 0.576 | 0.547 | 0.554 | 0.552 | 0.545 | 0.574 |
| requires_event_aggregation | 0.530 | 0.533 | 0.523 | 0.507 | 0.500 | 0.475 |
| requires_chronos_diff | 0.121 | 0.120 | 0.123 | 0.134 | 0.136 | 0.131 |
| is_aggregation | 0.439 | 0.467 | 0.446 | 0.418 | 0.424 | 0.377 |

## Feature separation (correct - incorrect)

| feature | R0 | R1 | R2 | R3 | R23 | R6 |
|---|---|---|---|---|---|---|
| evidence_coverage | +0.237 | +0.264 | +0.231 | +0.221 | +0.215 | +0.166 |
| budget_util | +0.319 | +0.369 | +0.326 | +0.294 | +0.300 | +0.236 |
| recommended_budget | -89.748 | -89.905 | -83.451 | -79.626 | -73.279 | -60.317 |
| top1_anchor_score | +18.309 | +12.764 | +17.192 | +17.238 | +16.607 | +5.683 |
| candidate_count | -18.006 | -20.362 | -15.638 | -14.419 | -13.416 | +3.024 |
| candidate_sessions | -1.716 | -1.949 | -1.486 | -1.391 | -1.283 | +0.147 |
| competing_values_count | -0.333 | +0.000 | -0.175 | -0.243 | -0.065 | -0.203 |
| state_evolution_demand | +0.134 | +0.174 | +0.159 | +0.158 | +0.168 | +0.136 |
| aggregation_demand | -0.297 | -0.317 | -0.286 | -0.269 | -0.258 | -0.220 |
| temporal_demand | +0.073 | +0.078 | +0.070 | +0.056 | +0.053 | +0.058 |
| assistant_recall_demand | +0.056 | +0.118 | +0.073 | +0.059 | +0.076 | +0.082 |
| preference_demand | +0.053 | +0.058 | +0.052 | +0.053 | +0.053 | +0.050 |
| direct_fact_demand | +0.035 | +0.029 | +0.030 | +0.032 | +0.027 | +0.016 |
| requires_state_timeline | +0.201 | +0.253 | +0.230 | +0.235 | +0.243 | +0.199 |
| requires_event_aggregation | -0.313 | -0.333 | -0.301 | -0.283 | -0.272 | -0.232 |
| requires_chronos_diff | +0.074 | +0.080 | +0.072 | +0.057 | +0.054 | +0.059 |
| is_aggregation | -0.304 | -0.358 | -0.311 | -0.276 | -0.283 | -0.213 |

## Interpretation

- Condition transforms live at prompt-build time only (R1: evidence-first
  reorder, R2: compact filter, R3: provenance tags, R23: R2+R3). Retrieval,
  planning and budgeted compilation are shared, so pipeline-side R6 features
  are **identical across conditions by construction** (verified: 250/250
  questions identical to R0 excluding `predicted`/`correct`). Read the
  all-questions feature table as the population profile, not a condition effect.
- Features with the largest correct-vs-incorrect separation are candidate
  difficulty signals for an R6 reader/compilation gate (mean over the five
  conditions):
  - `recommended_budget`: -79.388
  - `top1_anchor_score`: +14.632
  - `candidate_count`: -13.136
  - `candidate_sessions`: -1.280
  - `budget_util`: +0.307
- multi-session accuracy range across conditions: 35.8% - 49.3% (the dominant bottleneck; other types stay within their R0 band +/- 10pp)
- R1 mean latency (6.2s) is outlier-driven: median is 900ms (same as R0),
  but 6 items exceed 5s with a max of 307s (transient Ollama stall). The
  R1 transform itself is string reordering only.
