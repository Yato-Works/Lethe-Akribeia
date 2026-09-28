"""Unit tests for the LoCoMo scoring contract.

Three policies, one run set.  The tests pin the parts that decide the absolute
number - the official normalisation, the open-domain gold truncation, the
multi-hop partial F1 - and the one part that decides the claim: the paired delta
must be reported under every policy, because "the readers are interchangeable"
turned out to be a property of Lethe's matcher, not of the benchmark.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "benchmarks" / "locomo_scorer.py"


def load_script():
    spec = importlib.util.spec_from_file_location("_locomo_scorer_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ls = load_script()


def test_official_normalisation_drops_case_punctuation_and_articles() -> None:
    assert ls.normalize_official("The Guggenheim, in NYC.") == "guggenheim in nyc"
    assert ls.normalize_official("a dog and the cat") == "dog cat"


def test_official_f1_is_graded_not_binary() -> None:
    """A verbose answer that merely contains the gold does not score 1.0: the
    official metric rewards the complete answer over a padded one."""
    assert ls.f1_official("24 February 2023", "February 24, 2023") == 1.0
    padded = ls.f1_official("in June 2023 with Caroline", "June 2023")
    unrelated = ls.f1_official("the weekend before 15 July", "June 2023")
    assert 0.0 < padded < 1.0
    assert unrelated == 0.0


def test_open_domain_gold_is_truncated_at_the_semicolon() -> None:
    """This is why a bare "Yes." is a correct open-domain answer: the official
    harness cuts the gold at the first semicolon before scoring."""
    assert ls.official_score("Yes.", "Yes; it's classical music", 3) == 1.0
    # Without the category-3 rule the same pair does not match at all.
    assert ls.official_score("Yes.", "Yes; it's classical music", 4) < 0.5


def test_multi_hop_is_scored_as_partial_f1_over_sub_answers() -> None:
    """The official multi-hop rule averages per-sub-answer F1 against the whole
    prediction, so even a complete list is not a 1.0 unless it is bare."""
    gold = ["John", "Gina"]
    both = ls.official_score("John and Gina both started businesses", gold, 1)
    one = ls.official_score("John started a business", gold, 1)
    assert 0.0 < one < both < 1.0
    # Answering one of the two sub-answers scores 0.5, not 0: the official
    # multi-hop metric is graded, which is exactly what a binary matcher cannot say.
    assert ls.official_score("John", gold, 1) == 0.5
    # Even the minimal correct list is 2/3, not 1.0: precision is measured over the
    # whole prediction, so a complete answer is still penalised for its length.
    assert ls.official_score("John Gina", gold, 1) == pytest.approx(2 / 3)


def test_strict_requires_the_gold_to_appear_in_the_prediction() -> None:
    assert ls.strict_score("Caroline went on 24 February 2023",
                           "24 February 2023") == 1.0
    # The substring the other way round, and a re-ordered date, are exactly what
    # the strict policy refuses - it is a lower bound, not the official protocol.
    assert ls.strict_score("February", "February 24, 2023") == 0.0
    assert ls.strict_score("24 February 2023", "February 24, 2023") == 0.0


def test_every_policy_is_reported_for_a_run_so_no_column_is_optional() -> None:
    run = {"model": "x", "path": "x", "rows": {
        "q1": {"correct": True, "category": 4, "prediction": "24 February 2023",
               "ground_truth": "February 24, 2023"},
        "q2": {"correct": False, "category": 3, "prediction": "Yes.",
               "ground_truth": "Yes; it's classical music"},
    }}
    scored = ls.score_run(run)
    assert set(scored["totals"]) == set(ls.POLICIES)
    # q2 is a miss for the binary policies and a hit for the official one, which is
    # the whole reason the columns cannot be collapsed into a single number.
    assert scored["totals"]["lethe_current"]["hits"] == 1
    assert scored["totals"]["official_f1"]["sum"] == 2.0
    assert scored["per_category"]["open-domain"]["official_f1"]["hits"] == 1
