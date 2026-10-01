"""Unit tests for Allen's Interval Algebra."""

from __future__ import annotations

import datetime
import pytest

from artificial_memory.temporal.interval_algebra import (
    IntervalRelation,
    TimeInterval,
)


def test_invalid_interval_raises_error() -> None:
    with pytest.raises(ValueError):
        TimeInterval(start=datetime.date(2023, 5, 10), end=datetime.date(2023, 5, 1))


def test_before_and_after() -> None:
    i1 = TimeInterval(start=datetime.date(2023, 1, 1), end=datetime.date(2023, 1, 10), label="Trip 1")
    i2 = TimeInterval(start=datetime.date(2023, 2, 1), end=datetime.date(2023, 2, 10), label="Trip 2")

    assert i1.relation_to(i2) == IntervalRelation.BEFORE
    assert i2.relation_to(i1) == IntervalRelation.AFTER
    assert not i1.overlaps_with(i2)
    assert "completed before" in i1.format_derivation(i2)


def test_meets_and_met_by() -> None:
    i1 = TimeInterval(start=datetime.date(2023, 3, 1), end=datetime.date(2023, 3, 15), label="Sprint 1")
    i2 = TimeInterval(start=datetime.date(2023, 3, 15), end=datetime.date(2023, 3, 31), label="Sprint 2")

    assert i1.relation_to(i2) == IntervalRelation.MEETS
    assert i2.relation_to(i1) == IntervalRelation.MET_BY
    assert "concluded exactly as" in i1.format_derivation(i2)


def test_overlaps_and_during() -> None:
    # Overlap
    i1 = TimeInterval(start=datetime.date(2023, 4, 1), end=datetime.date(2023, 4, 20), label="Exhibition")
    i2 = TimeInterval(start=datetime.date(2023, 4, 10), end=datetime.date(2023, 4, 30), label="Festival")
    assert i1.relation_to(i2) == IntervalRelation.OVERLAPS
    assert i2.relation_to(i1) == IntervalRelation.OVERLAPPED_BY
    assert i1.overlaps_with(i2)

    # During and Contains
    vacation = TimeInterval(start=datetime.date(2023, 5, 1), end=datetime.date(2023, 5, 31), label="May Holiday")
    concert = TimeInterval(start=datetime.date(2023, 5, 10), end=datetime.date(2023, 5, 12), label="Music Fest")
    assert concert.relation_to(vacation) == IntervalRelation.DURING
    assert vacation.relation_to(concert) == IntervalRelation.CONTAINS
    assert "entirely within" in concert.format_derivation(vacation)


def test_datetime_precision() -> None:
    i1 = TimeInterval(
        start=datetime.datetime(2023, 6, 1, 10, 0),
        end=datetime.datetime(2023, 6, 1, 12, 0),
        label="Morning Keynote",
    )
    i2 = TimeInterval(
        start=datetime.datetime(2023, 6, 1, 12, 0),
        end=datetime.datetime(2023, 6, 1, 13, 0),
        label="Lunch",
    )
    assert i1.relation_to(i2) == IntervalRelation.MEETS
    assert i2.relation_to(i1) == IntervalRelation.MET_BY
