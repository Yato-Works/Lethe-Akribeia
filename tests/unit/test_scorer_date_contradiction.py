"""Regression tests for the date-contradiction guard in the LoCoMo scorer.

Why this test exists: the production scorer credited a WRONG DATE as correct
whenever month and year agreed, because the 60%-token-overlap rule drops 1-2
digit day numbers via its `len(w) > 2` filter:

    pred '10 May, 2023'  gold '27 May, 2023'   ->  was scored CORRECT
    pred '25 June, 2023' gold '24 June, 2023'  ->  was scored CORRECT

Measured on 188 committer claims: 8 of 159 credits were spurious this way (5.0%).
That inflated the reported gain, which is the same failure mode as the
2026-10-10 contamination retraction -- a score that looks better than the engine
is.

These tests pin both directions:
  * a day disagreement is a factual contradiction and must NOT be credited
  * a day AGREEMENT, and prose that carries no day at all, must still be credited
    so the guard does not silently suppress legitimate answers

The guard must run BEFORE the credit rules.  An earlier placement after the
containment checks let 'The week before 13 October 2023' credit against
'On the night of October 30, 2023' through the shared month alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from eval_locomo_full import _date_contradiction, _date_days, score  # noqa: E402


# --------------------------------------------------------------- day disagreement
@pytest.mark.parametrize(
    "pred,gold",
    [
        ("10 May, 2023", "27 May, 2023"),
        ("25 June, 2023", "24 June, 2023"),
        ("The Tuesday before 20 July 2023", "7 May 2023"),
        ("The Friday before 23 July 2023", "21 July 2023"),
        ("The Sunday before 13 June 2023", "June 11, 2023"),
        ("The week before 24 October 2023", "between October 19 and 24, 2023"),
        ("The week before 13 October 2023", "On the night of October 30, 2023"),
    ],
)
def test_wrong_day_is_not_credited(pred: str, gold: str) -> None:
    """A day disagreement is a contradiction, however much month/year agrees."""
    assert not score(pred, gold)


# ------------------------------------------------------------------ day agreement
@pytest.mark.parametrize(
    "pred,gold",
    [
        ("16 March, 2023", "16 March, 2023"),
        ("27 May, 2023", "27 May, 2023"),
        ("24 June, 2023", "24 June, 2023"),
        ("21 July 2023", "21 July 2023"),
        ("7 May 2023", "7 May 2023"),
        ("The Saturday before 25 May 2023", "The sunday before 25 May 2023"),
        ("two weekends before 17 July 2023", "two weekends before 17 July 2023"),
    ],
)
def test_agreeing_day_is_still_credited(pred: str, gold: str) -> None:
    """The guard must not suppress answers whose day actually matches."""
    assert score(pred, gold)


# ----------------------------------------------------- no day named on one side
@pytest.mark.parametrize(
    "pred,gold",
    [
        # gold names no day -- a month-only gold cannot contradict
        ("May, 2023", "May, 2023"),
        ("February, 2023", "February, 2023"),
        # pred names no day, so there is nothing to contradict
        ("The summer of 2022", "2022"),
    ],
)
def test_missing_day_on_one_side_is_not_a_contradiction(pred: str, gold: str) -> None:
    """Only a disagreement between two NAMED days counts."""
    assert not _date_contradiction(pred.lower(), gold.lower())


# ------------------------------------------------- stray numbers are not days
def test_stray_number_in_prose_is_not_a_day() -> None:
    """A number with no adjacent month name must not be read as a day-of-month.

    Without the month-adjacency requirement, 'she has 3 cats' would contribute a
    phantom day '3' and could contradict an unrelated date.
    """
    assert _date_days("she has 3 cats and 2 dogs") == set()


def test_ordinal_suffix_is_normalised() -> None:
    """'23rd July' and '23 July' must yield the same day, or a correct answer
    would be contradicted purely by formatting."""
    assert _date_days("23rd July 2023") == {"23"}
    assert _date_days("23 July 2023") == {"23"}


def test_both_month_orderings_are_understood() -> None:
    """'10 May' and 'May 10' are the same date; the guard must see both."""
    assert _date_days("10 May 2023") == _date_days("May 10, 2023") == {"10"}
