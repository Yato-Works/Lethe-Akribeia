"""Unit tests for the unified matcher-v2 binary scoring path.

Every published LoCoMo ``is_correct`` flag must come from exactly one
implementation - ``LoCoMoAdapter.score_binary`` - so these tests pin:

* the offline re-scoring script delegates to that same implementation,
* matcher-v2 semantics (generic rules only, deterministic stemming, no
  compiler instance needed, refusal-shaped credit for empty ground truth),
* ``apply_rescore`` rewrites artefacts with a *percentage* ``overall_accuracy``
  and keeps provenance stable across repeated runs,
* the frozen headline artefact's stored flags actually equal a matcher-v2
  recomputation (guards against silent matcher drift).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "rescore_locomo_run.py"
RUN = REPO / "benchmark_results" / "locomo1540" / "temporal321_rules_commit_7b_postfix.json"


def load_script():
    spec = importlib.util.spec_from_file_location("_rescore_locomo_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rs = load_script()


def test_rescore_script_scores_with_the_shared_adapter_matcher() -> None:
    """The offline scorer and the live path cannot drift: both call
    ``LoCoMoAdapter.score_binary``."""
    cases = [
        (1, "John and Gina", "John and Gina"),
        (2, "February 24, 2023", "24 February 2023"),
        (3, "Yes", "Yes; it's classical music"),
        (4, "Guggenheim", "the Guggenheim museum"),
        (5, "", "I don't know"),
    ]
    for cat, gt, pred in cases:
        assert rs.score(cat, gt, pred) == LoCoMoAdapter.score_binary(cat, gt, pred)


def test_score_binary_needs_no_instance_and_stems_deterministically() -> None:
    """matcher-v2 stems through the class-level ``WideSlicer`` path, so the
    result does not depend on whether a compiler instance exposed one."""
    assert LoCoMoAdapter.score_binary(4, "cats", "cat") is True
    assert LoCoMoAdapter.score_binary(4, "completely unrelated", "cat") is False


def test_generic_rules_reject_non_matching_answers_without_gt_vocab_branches() -> None:
    """A ground-truth phrase special-cased by matcher-v1 ("two cats and a dog")
    now goes through the generic word-overlap rules like any other string."""
    assert LoCoMoAdapter.score_binary(1, "two cats and a dog", "two cats and a dog") is True
    assert LoCoMoAdapter.score_binary(1, "two cats and a dog", "a fish") is False


def test_empty_ground_truth_credits_only_refusal_shaped_predictions() -> None:
    assert LoCoMoAdapter.score_binary(5, "", "I don't know") is True
    assert LoCoMoAdapter.score_binary(5, "", "The answer is Paris") is False


def _mini_blob() -> dict:
    return {
        "results": [
            {"question_id": "q1", "category": 4, "ground_truth": "Guggenheim",
             "prediction": "Guggenheim", "is_correct": False},
            {"question_id": "q2", "category": 4, "ground_truth": "Eiffel Tower",
             "prediction": "Statue of Liberty", "is_correct": False},
        ],
        "overall_accuracy": 0.0,
        "categories": {"Single-Hop": {"acc": 0.0, "corr": 0, "tot": 2}},
    }


def test_apply_rescore_rewrites_flags_and_stores_percentage_accuracy() -> None:
    blob = _mini_blob()
    flipped = rs.apply_rescore(blob)
    # q1 now matches under matcher-v2; q2 genuinely does not.
    assert flipped == 1
    assert [r["is_correct"] for r in blob["results"]] == [True, False]
    # ``overall_accuracy`` is a percentage (0-100), matching the artefact schema;
    # ``categories[...]["acc"]`` stays a fraction.
    assert blob["overall_accuracy"] == pytest.approx(50.0)
    assert blob["categories"]["Single-Hop"] == {"acc": 0.5, "corr": 1, "tot": 2}
    assert blob["scorer_revision"] == "matcher-v2"


def test_apply_rescore_is_idempotent_and_preserves_provenance() -> None:
    blob = _mini_blob()
    blob["rescoring"] = {"matcher": "LoCoMoAdapter.score_binary",
                         "from_revision": "matcher-v1-original"}
    rs.apply_rescore(blob)
    first = json.dumps(blob, sort_keys=True)
    flipped_again = rs.apply_rescore(blob)
    assert flipped_again == 0
    assert json.dumps(blob, sort_keys=True) == first
    # Re-running must never rewrite the recorded origin of the artefact.
    assert blob["rescoring"]["from_revision"] == "matcher-v1-original"


def test_frozen_headline_artefact_flags_equal_matcher_v2_recomputation() -> None:
    blob = json.loads(RUN.read_text(encoding="utf-8"))
    assert blob["scorer_revision"] == "matcher-v2"
    correct = 0
    for row in blob["results"]:
        expected = LoCoMoAdapter.score_binary(
            int(row["category"]), str(row["ground_truth"]), str(row["prediction"]))
        assert bool(row["is_correct"]) == expected, row["question_id"]
        correct += int(expected)
    n = len(blob["results"])
    assert n == 1540
    assert blob["overall_accuracy"] == pytest.approx(correct / n * 100.0)
    # Headline number: matcher-v2 full-slice accuracy (see RESULTS_REGISTRY.md).
    assert (correct, n) == (1005, 1540)
