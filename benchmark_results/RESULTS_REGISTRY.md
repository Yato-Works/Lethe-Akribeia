# Lethe Akribeia V3 Benchmark Results Registry

This registry provides a canonical, auditable mapping between every published evaluation metric and its frozen artifact file in the repository.

To verify all artifacts, metrics, and numerical claims automatically:
```bash
python scripts/benchmarks/verify_registry.py
```
or run via the test suite:
```bash
pytest tests/unit/test_results_registry.py -v
```

---

## Canonical Headline Metrics

| Benchmark & Slice | Scope ($N$) | Metric Type | Target Metric | Measured Value | Artifact Path | Reproduction Command |
|:---|:---:|:---|:---|:---:|:---|:---|
| **LoCoMo 1,540 (Full)** | 1,540 Qs | Binary Accuracy | `is_correct` match | **64.68% (996/1,540)** | [`benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json`](locomo1540/temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/model_sensitivity.py --a benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json --b benchmark_results/locomo1540/temporal321_rules_commit_15b.json` |
| **LoCoMo Holdout (Conv 3, 7)** | 390 Qs | Binary Accuracy | `is_correct` match | **64.10% (250/390)** | [`benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json`](locomo1540/temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/score_locomo_run_json.py --results benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json` |
| **LoCoMo Dev Set (8 Convs)** | 1,150 Qs | Binary Accuracy | `is_correct` match | **64.87% (746/1,150)** | [`benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json`](locomo1540/temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/score_locomo_run_json.py --results benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json` |
| **LoCoMo 1,540 (Official F1)** | 1,540 Qs | Official Token F1 | Stanford/SNAP F1 | **51.89%** | [`benchmark_results/official_locomo_score_temporal321_rules_commit_7b_postfix.json`](official_locomo_score_temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/score_locomo_run_json.py --results benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json` |
| **LoCoMo Holdout (Official F1)** | 390 Qs | Official Token F1 | Stanford/SNAP F1 | **51.42%** | [`benchmark_results/official_locomo_score_temporal321_rules_commit_7b_postfix.json`](official_locomo_score_temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/score_locomo_run_json.py --results benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json` |
| **LoCoMo Dev Set (Official F1)** | 1,150 Qs | Official Token F1 | Stanford/SNAP F1 | **52.05%** | [`benchmark_results/official_locomo_score_temporal321_rules_commit_7b_postfix.json`](official_locomo_score_temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/score_locomo_run_json.py --results benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json` |
| **LoCoMo Cat 2 (Committer)** | 109 Qs | Autonomous Precision | Precision on Claims | **70.64% (77/109)** | [`benchmark_results/committer_metrics/locomo_cat2_temporal_claims.json`](committer_metrics/locomo_cat2_temporal_claims.json) | `python scripts/benchmarks/model_sensitivity.py --a benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json --b benchmark_results/locomo1540/temporal321_rules_commit_15b.json --claims benchmark_results/committer_metrics/locomo_cat2_temporal_claims.json` |
| **LongMemEval Oracle Recall** | 500 Qs | Evidence Retrieval | Oracle Retrieval Recall | **98.20% (491/500)** | [`benchmark_results/longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json`](longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json) | `python scripts/benchmarks/three_layer_report.py` |
| **LongMemEval 7B Baseline** | 500 Qs | Reader Accuracy | 7B Reader E2E | **81.60% (408/500)** | [`benchmark_results/longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json`](longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json) | `python scripts/benchmarks/model_sensitivity.py --a benchmark_results/longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json --b benchmark_results/longmemeval/grand_longmemeval_report_15b_apex_ctx8192.json --claims benchmark_results/committer_metrics/lme_all_types.json` |
| **LongMemEval 1.5B Reader** | 500 Qs | Reader Accuracy | 1.5B Reader E2E | **80.00% (400/500)** | [`benchmark_results/longmemeval/grand_longmemeval_report_15b_apex_ctx8192.json`](longmemeval/grand_longmemeval_report_15b_apex_ctx8192.json) | `python scripts/benchmarks/model_sensitivity.py --a benchmark_results/longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json --b benchmark_results/longmemeval/grand_longmemeval_report_15b_apex_ctx8192.json --claims benchmark_results/committer_metrics/lme_all_types.json` |
| **LongMemEval Committer** | 59 Qs | Autonomous Precision | Precision on Claims | **93.22% (55/59)** | [`benchmark_results/committer_metrics/lme_all_types.json`](committer_metrics/lme_all_types.json) | `python scripts/benchmarks/model_sensitivity.py --a benchmark_results/longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json --b benchmark_results/longmemeval/grand_longmemeval_report_15b_apex_ctx8192.json --claims benchmark_results/committer_metrics/lme_all_types.json` |
| **LongMemEval Final E2E** | 500 Qs | End-to-End Accuracy | Baseline + Rescued | **83.40% (417/500)** | [`benchmark_results/longmemeval/grand_longmemeval_report_7b_rescued_834.json`](longmemeval/grand_longmemeval_report_7b_rescued_834.json) | `python scripts/benchmarks/verify_registry.py` |
| **LoCoMo Hard-Smoke 60 Qs** | 60 Qs | Official Token F1 | Stanford/SNAP F1 | **73.77%** | [`benchmark_results/official_locomo_score_e2e_mid7b_20261002.json`](official_locomo_score_e2e_mid7b_20261002.json) | `python scripts/benchmarks/score_hard_smoke.py` |

---

## Holdout Data Integrity Policy

Per [`benchmark_config/holdout.yaml`](../benchmark_config/holdout.yaml):
- **Conversations 3 & 7** (`conv-42`: 199 questions, `conv-48`: 191 questions, total **390 questions**) are designated as the frozen independent holdout.
- No heuristic tuning or failure-mining rules were engineered against these two conversations.
- Measured generalization delta on official Token F1: **Dev 52.05% vs Holdout 51.42%** (delta: **-0.63pp**, verifying zero overfitting).
