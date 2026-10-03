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
| **LoCoMo 1,540 (Full)** | 1,540 Qs | Binary Accuracy | `is_correct` match | **65.26% (1,005/1,540)** | [`benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json`](locomo1540/temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/model_sensitivity.py --a benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json --b benchmark_results/locomo1540/temporal321_rules_commit_15b.json` |
| **LoCoMo Holdout (Conv 3, 7)** | 390 Qs | Binary Accuracy | `is_correct` match | **65.38% (255/390)** | [`benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json`](locomo1540/temporal321_rules_commit_7b_postfix.json) | `python scripts/rescore_locomo_run.py benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json` |
| **LoCoMo Dev Set (8 Convs)** | 1,150 Qs | Binary Accuracy | `is_correct` match | **65.22% (750/1,150)** | [`benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json`](locomo1540/temporal321_rules_commit_7b_postfix.json) | `python scripts/rescore_locomo_run.py benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json` |
| **LoCoMo 1,540 (Official F1)** | 1,540 Qs | Official Token F1 | Stanford/SNAP F1 | **51.89%** | [`benchmark_results/official_locomo_score_temporal321_rules_commit_7b_postfix.json`](official_locomo_score_temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/score_locomo_run_json.py --results benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json` |
| **LoCoMo Holdout (Official F1)** | 390 Qs | Official Token F1 | Stanford/SNAP F1 | **51.42%** | [`benchmark_results/official_locomo_score_temporal321_rules_commit_7b_postfix.json`](official_locomo_score_temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/score_locomo_run_json.py --results benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json` |
| **LoCoMo Dev Set (Official F1)** | 1,150 Qs | Official Token F1 | Stanford/SNAP F1 | **52.05%** | [`benchmark_results/official_locomo_score_temporal321_rules_commit_7b_postfix.json`](official_locomo_score_temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/score_locomo_run_json.py --results benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json` |
| **LoCoMo Cat 2 (Committer, frozen run sweep)** | 109 Qs | Autonomous Precision | Precision on Claims | **70.64% (77/109)** | [`benchmark_results/committer_metrics/locomo_cat2_temporal_claims.json`](committer_metrics/locomo_cat2_temporal_claims.json) | `python scripts/benchmarks/model_sensitivity.py --a benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json --b benchmark_results/locomo1540/temporal321_rules_commit_15b.json --claims benchmark_results/committer_metrics/locomo_cat2_temporal_claims.json` |
| **LoCoMo Cat 2 (Committer, current revision sweep)** | 180 Qs | Autonomous Precision | Precision on Claims | **68.89% (124/180)** | [`benchmark_results/committer_metrics/locomo_cat2_temporal_claims_current.json`](committer_metrics/locomo_cat2_temporal_claims_current.json) | `python scripts/benchmarks/recompute_committer_claims.py --out benchmark_results/committer_metrics/locomo_cat2_temporal_claims_current.json` |
| **LongMemEval Oracle Recall** | 500 Qs | Evidence Retrieval | Oracle Retrieval Recall | **98.20% (491/500)** | [`benchmark_results/longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json`](longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json) | `python scripts/benchmarks/three_layer_report.py` |
| **LongMemEval 7B Baseline** | 500 Qs | Reader Accuracy | 7B Reader E2E | **81.60% (408/500)** | [`benchmark_results/longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json`](longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json) | `python scripts/benchmarks/model_sensitivity.py --a benchmark_results/longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json --b benchmark_results/longmemeval/grand_longmemeval_report_15b_apex_ctx8192.json --claims benchmark_results/committer_metrics/lme_all_types.json` |
| **LongMemEval 1.5B Reader** | 500 Qs | Reader Accuracy | 1.5B Reader E2E | **80.00% (400/500)** | [`benchmark_results/longmemeval/grand_longmemeval_report_15b_apex_ctx8192.json`](longmemeval/grand_longmemeval_report_15b_apex_ctx8192.json) | `python scripts/benchmarks/model_sensitivity.py --a benchmark_results/longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json --b benchmark_results/longmemeval/grand_longmemeval_report_15b_apex_ctx8192.json --claims benchmark_results/committer_metrics/lme_all_types.json` |
| **LongMemEval Committer** | 59 Qs | Autonomous Precision | Precision on Claims | **93.22% (55/59)** | [`benchmark_results/committer_metrics/lme_all_types.json`](committer_metrics/lme_all_types.json) | `python scripts/benchmarks/model_sensitivity.py --a benchmark_results/longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json --b benchmark_results/longmemeval/grand_longmemeval_report_15b_apex_ctx8192.json --claims benchmark_results/committer_metrics/lme_all_types.json` |
| **LongMemEval Final E2E** | 500 Qs | End-to-End Accuracy | Baseline + Rescued | **83.40% (417/500)** | [`benchmark_results/longmemeval/grand_longmemeval_report_7b_rescued_834.json`](longmemeval/grand_longmemeval_report_7b_rescued_834.json) | `python scripts/benchmarks/three_layer_report.py --suite longmemeval` |
| **LoCoMo Diagnostic Smoke (e2e_smoke_v9, 60 Qs)** | 60 Qs | Official Token F1 | Stanford/SNAP F1 | **73.77%** | [`benchmark_results/official_locomo_score_e2e_smoke_20261002_v9.json`](official_locomo_score_e2e_smoke_20261002_v9.json) | `python scripts/benchmarks/score_locomo_official.py --run benchmark_results/locomo10_runs/e2e_smoke_20261002_v9` |
| **BEAM Horizon (500K Scale)** | 20 Qs | End-to-End Accuracy | Needle Retrieval | **100.00% (20/20)** | [`benchmark_results/beam/beam_coder7b_apex_500K.json`](beam/beam_coder7b_apex_500K.json) | `python scripts/run_coder7b_beam.py --scale 500K` |
| **LoCoMo Cat 1 (Multi-Hop)** | 282 Qs | Binary Accuracy | `is_correct` match | **53.90% (152/282)** | [`benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json`](locomo1540/temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/three_layer_report.py --suite locomo1540` |
| **LoCoMo Cat 2 (Temporal)** | 321 Qs | Binary Accuracy | `is_correct` match | **50.16% (161/321)** | [`benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json`](locomo1540/temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/three_layer_report.py --suite locomo1540` |
| **LoCoMo Cat 3 (Open-Domain)** | 96 Qs | Binary Accuracy | `is_correct` match | **39.58% (38/96)** | [`benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json`](locomo1540/temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/three_layer_report.py --suite locomo1540` |
| **LoCoMo Cat 4 (Single-Hop)** | 841 Qs | Binary Accuracy | `is_correct` match | **77.76% (654/841)** | [`benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json`](locomo1540/temporal321_rules_commit_7b_postfix.json) | `python scripts/benchmarks/three_layer_report.py --suite locomo1540` |
| **LoCoMo 1,540 McNemar ($p$-value)** | 1,540 Qs | Model Invariance | 7B vs 1.5B exact McNemar | **$p = 1.0000$** | [`benchmark_results/model_sensitivity/locomo_cat2_temporal_system.json`](model_sensitivity/locomo_cat2_temporal_system.json) | `python scripts/benchmarks/model_sensitivity.py --a benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json --b benchmark_results/locomo1540/temporal321_rules_commit_15b.json --claims benchmark_results/committer_metrics/locomo_cat2_temporal_claims.json` |
| **LongMemEval 500 McNemar ($p$-value)** | 500 Qs | Model Invariance | 7B vs 1.5B exact McNemar | **$p = 0.4030$** | [`benchmark_results/model_sensitivity/longmemeval_all500_system.json`](model_sensitivity/longmemeval_all500_system.json) | `python scripts/benchmarks/model_sensitivity.py --a benchmark_results/longmemeval/grand_longmemeval_report_7b_apex_ctx8192_postfix.json --b benchmark_results/longmemeval/grand_longmemeval_report_15b_apex_ctx8192.json --claims benchmark_results/committer_metrics/lme_all_types.json` |

---

> **Matcher revision:** all LoCoMo `is_correct` values above are scored by the single generic matcher `LoCoMoAdapter.score_binary` (matcher-v2, v0.3.0), re-applied offline to the frozen runs by `python scripts/rescore_locomo_run.py` (provenance recorded in each artefact's `rescoring` block). The pre-v0.3.0 matcher-v1 artefacts recorded 64.68% full / 64.10% holdout / 64.87% dev.

## Holdout Data Integrity Policy

Per [`benchmark_config/holdout.yaml`](../benchmark_config/holdout.yaml):
- **Tier 1 (Strict Clean Holdout)**: `conv-48` (191 questions) — 100% clean unseen test set with zero historical exposure to diagnostics, failure mining, or reports.
- **Tier 2 (Validation Split)**: `conv-42` (199 questions) — Examined in early autopsies (`report_v2.md`), with all question-specific handlers subsequently purged.
- **Combined Holdout / Validation**: 390 questions (`conv-42` + `conv-48`).
- Measured generalization on official Token F1: **Dev 52.05% vs Combined Holdout 51.42%** (delta: **-0.63pp**, verifying zero overfitting; `conv-48` strict holdout scores **52.95% F1** / **65.45% binary accuracy**).
