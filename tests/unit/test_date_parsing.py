"""Regression test: absolute date mentions must reach the date-proximity boost.

Before this, _parse_relative_date returned None for every absolute date, and the
LoCoMo harness always passes reference_date=None, so the date-proximity boost in
rank_turns silently never fired for questions like "in April 2022".  The Oracle
Search showed those are exactly the questions evidence alone fixes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmarks"))
from eval_v4_pipeline_dev import V4CandidateRetriever  # noqa: E402

P = V4CandidateRetriever._parse_relative_date

# (query, reference_date, expected target date or None)
CASES = [
    # absolute, day + month + year -- no reference date needed
    ("What movie did Joanna watch on 1 May, 2022?", None, "2022-05-01"),
    ("What happened on 3 August 2023?", None, "2023-08-03"),
    ("Where was John between August 11 and August 15 2023?", None, "2023-08-15"),
    # absolute, month + year -- mid-month is the best available target
    ("What did Nate undergo in April 2022?", None, "2022-04-15"),
    ("Where did Joanna travel to in July 2022?", None, "2022-07-15"),
    # month + day + year word order
    ("What happened in May 1, 2022?", None, "2022-05-01"),
    # relative still works, and still needs a reference date
    ("What did Melanie do 3 weeks ago?", "2023/05/01", "2023-04-10"),
    ("What did Melanie do last week?", "2023/05/01", "2023-04-24"),
    ("What did Melanie do 3 weeks ago?", None, None),      # no reference -> None
    # non-date questions must stay None so the boost does not misfire
    ("What is Caroline's identity?", "2023/05/08", None),
    ("Where did Caroline move from 4 years ago?", "2023/05/08", None),
    ("When did Caroline go to the LGBTQ support group?", "2023/05/08", None),
]


@pytest.mark.parametrize("query,ref,expected", CASES)
def test_date_parsing(query, ref, expected):
    got = P(query, ref)
    if expected is None:
        assert got is None, f"{query!r} parsed to {got}, want None"
    else:
        assert got is not None, f"{query!r} parsed to None, want {expected}"
        assert str(got)[:10] == expected, f"{query!r} -> {got}, want {expected}"


def test_absolute_needs_no_reference_date():
    """The harness always passes reference_date=None.

    If absolute parsing sat behind the reference_date guard it would never run,
    which is the original bug.
    """
    assert P("What happened in April 2022?", None) is not None
    assert P("What happened on 1 May, 2022?", None) is not None
