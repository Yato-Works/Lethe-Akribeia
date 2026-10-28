"""Contracts for the single-command LoCoMo smoke path.

Why these tests exist: the repo spent months accumulating one-off benchmark
scripts, so "run a few real questions and look at what the reader saw" had no
single answer.  `scripts/benchmarks/locomo_smoke.py` is that answer, and these
tests pin the three properties that make it trustworthy:

1. The pinned case set is stable, covers every category, and NEVER touches
   conv-48 (the strict clean holdout).  A smoke run that quietly sampled the
   holdout would destroy the only uncontaminated measurement we have.
2. The manifest format the smoke path writes is the format the canonical
   harness reads (object with `cases`, not just a bare list).
3. The report says it is a plumbing check, surfaces failures, and never
   presents a smoke number as a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from eval_locomo_full import manifest_cases  # noqa: E402
from locomo_smoke import (  # noqa: E402
    build_comparison,
    build_report,
    load_manifest,
    normalize_frozen,
    normalize_results,
)

MANIFEST = REPO / "benchmark_config" / "smoke_manifest.json"
DATASET = REPO / "datasets" / "external" / "locomo10.json"


def _meta(mode: str = "llm") -> dict:
    return {
        "stamp": "20261027_120000",
        "name": "locomo-smoke-v1",
        "manifest": "benchmark_config/smoke_manifest.json",
        "data": "datasets/external/locomo10.json",
        "mode": mode,
        "model": "qwen2.5:7b-instruct",
        "prompt_style": "concise",
        "revision": "abc1234",
        "duration": 12.3,
    }


def _result(qid: str, cat: int, *, ok: bool, pred: str = "x", gold: str = "g",
            q: str = "question?", tokens: int = 1400) -> dict:
    return {
        "question_id": qid,
        "category": cat,
        "question": q,
        "ground_truth": gold,
        "predicted_answer": pred,
        "is_correct": ok,
        "context_tokens": tokens,
        "latency_seconds": 2.0,
    }


class TestPinnedManifest:
    def test_is_stable_and_covers_every_category(self):
        manifest = load_manifest(MANIFEST)
        cases = manifest["cases"]
        assert manifest["name"] == "locomo-smoke-v1"
        assert len(cases) == 18
        keys = [(c["conv_idx"], c["q_idx"]) for c in cases]
        assert len(set(keys)) == len(keys), "duplicate pinned cases"
        assert {c["category"] for c in cases} == {1, 2, 3, 4, 5}

    def test_never_touches_the_clean_holdout(self):
        manifest = load_manifest(MANIFEST)
        assert all(c["conv_idx"] != 7 for c in manifest["cases"])
        excluded = {e["conv_idx"] for e in manifest.get("excluded", [])}
        assert 7 in excluded, "conv-48 must stay documented as excluded"

    def test_cases_exist_in_the_dataset_and_match_it(self):
        cases = load_manifest(MANIFEST)["cases"]
        data = json.loads(DATASET.read_text(encoding="utf-8"))
        for case in cases:
            conv = data[case["conv_idx"]]
            qa = conv["qa"][case["q_idx"]]
            assert conv["sample_id"] == case["conversation"]
            assert int(qa["category"]) == case["category"]
            assert qa["question"] == case["question"]
            gold = "" if qa.get("answer") is None else str(qa["answer"])
            assert gold == case["gold"]


class TestManifestFormat:
    def test_object_manifest_is_accepted(self):
        raw = {"name": "smoke", "cases": [{"conv_idx": 0, "q_idx": 3}]}
        assert manifest_cases(raw) == [{"conv_idx": 0, "q_idx": 3}]

    def test_bare_list_still_works(self):
        assert manifest_cases([{"conv_idx": 1, "q_idx": 2}]) == [{"conv_idx": 1, "q_idx": 2}]

    def test_object_without_cases_is_rejected(self):
        with pytest.raises(SystemExit):
            manifest_cases({"name": "no cases here"})

    def test_non_list_is_rejected(self):
        with pytest.raises(SystemExit):
            manifest_cases(42)


class TestReport:
    def test_says_it_is_not_a_score_and_surfaces_failure_detail(self):
        rows = normalize_results([
            _result("lc_conv-26_0", 2, ok=True, pred="7 May 2023", gold="7 May 2023"),
            _result("lc_conv-26_32", 1, ok=False,
                    pred="pride parade, poetry reading",
                    gold="Pride parade, school speech, support group"),
        ])
        report = build_report(rows, _meta())
        assert "not a benchmark score" in report
        assert "1/2 correct" in report
        assert "## Failures in detail" in report
        assert "lc_conv-26_32" in report
        assert "Pride parade, school speech, support group" in report

    def test_reader_input_is_inlined_only_when_provided(self):
        rows = normalize_results([_result("lc_conv-26_32", 1, ok=False)])
        assert "## Reader input" not in build_report(rows, _meta())
        report = build_report(rows, _meta(), prompts={"lc_conv-26_32": "EVIDENCE LINE"})
        assert "## Reader input - lc_conv-26_32" in report
        assert "EVIDENCE LINE" in report

    def test_crashed_case_is_labelled_error_not_fail(self):
        rows = normalize_results([
            _result("lc_x_1", 4, ok=False, pred="<ERROR: KeyError: 'qa'>"),
        ])
        report = build_report(rows, _meta())
        assert "ERROR" in report
        assert "| 1 | lc_x_1 | 4 single-hop | ERROR" in report

    def test_context_only_report_claims_no_accuracy(self):
        rows = normalize_frozen([
            {"question_id": "lc_conv-26_0", "category": 2,
             "question": "when?", "gold": "7 May 2023",
             "context": "EVIDENCE", "context_tokens": 1400},
        ])
        report = build_report(rows, _meta("contexts"),
                              prompts={"lc_conv-26_0": "EVIDENCE"})
        assert "not a benchmark score" in report
        assert "0 LLM calls" in report
        assert "correct" not in report.lower()
        assert "## Compiled context - lc_conv-26_0" in report


class TestComparison:
    def test_fixed_and_broken_are_listed_separately(self):
        before = {"mode": "llm", "rows": normalize_results([
            _result("c1", 1, ok=False, pred="old wrong"),
            _result("c2", 2, ok=True, pred="right"),
            _result("c3", 4, ok=True, pred="right"),
        ])}
        after = {"mode": "llm", "rows": normalize_results([
            _result("c1", 1, ok=True, pred="now right"),
            _result("c2", 2, ok=False, pred="now wrong"),
            _result("c3", 4, ok=True, pred="right"),
        ])}
        text = build_comparison(before, after, ("before", "after"))
        assert "correct: 2 -> 2 (+0)" in text
        assert "fixed: 1 | broken: 1 | unchanged: 1" in text
        assert "| c1 | 1 multi-hop | FAIL | OK |" in text
        assert "| c2 | 2 temporal | OK | FAIL |" in text
        assert "| c3 |" not in text, "unchanged cases stay out of the change table"
        assert "lead to investigate" in text

    def test_context_only_comparison_reports_token_delta(self):
        before = {"mode": "contexts", "rows": normalize_frozen([
            {"question_id": "c1", "category": 1, "question": "q", "gold": "g",
             "context": "x", "context_tokens": 1000},
        ])}
        after = {"mode": "contexts", "rows": normalize_frozen([
            {"question_id": "c1", "category": 1, "question": "q", "gold": "g",
             "context": "x", "context_tokens": 700},
        ])}
        text = build_comparison(before, after, ("before", "after"))
        assert "| c1 | 1 multi-hop | 1000 | 700 | -300 |" in text
        assert "Mean context tokens: 1000 -> 700 (-300)." in text

