"""Unit tests for time-of-day precision in TemporalResolver."""

from __future__ import annotations

import datetime

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.recall.temporal_resolver import (
    TemporalResolver,
    format_temporal_point,
    parse_datetime_or_date,
    to_comparable_datetime,
)


def test_parse_datetime_or_date_extracts_hh_mm() -> None:
    # 24-hour format in parentheses
    res1 = parse_datetime_or_date("2023/05/30 (Tue) 17:27")
    assert isinstance(res1, datetime.datetime)
    assert res1.year == 2023 and res1.month == 5 and res1.day == 30
    assert res1.hour == 17 and res1.minute == 27

    # 12-hour am/pm format in LoCoMo header
    res2 = parse_datetime_or_date("1:56 pm on 8 May, 2023")
    assert isinstance(res2, datetime.datetime)
    assert res2.year == 2023 and res2.month == 5 and res2.day == 8
    assert res2.hour == 13 and res2.minute == 56

    # Date-only format
    res3 = parse_datetime_or_date("2023-01-15")
    assert isinstance(res3, datetime.date)
    assert not isinstance(res3, datetime.datetime)
    assert res3 == datetime.date(2023, 1, 15)


def test_to_comparable_datetime() -> None:
    d = datetime.date(2023, 3, 15)
    dt1 = datetime.datetime(2023, 3, 15, 10, 0)
    dt2 = datetime.datetime(2023, 3, 15, 14, 30)

    cd = to_comparable_datetime(d)
    assert isinstance(cd, datetime.datetime)
    assert cd < dt1 < dt2


def test_format_temporal_point() -> None:
    dt = datetime.datetime(2023, 3, 15, 14, 30)
    assert format_temporal_point(dt) == "2023-03-15 14:30"

    d = datetime.date(2023, 3, 15)
    assert format_temporal_point(d) == "2023-03-15"


def test_temporal_resolver_resolves_same_day_with_time() -> None:
    resolver = TemporalResolver()
    records = [
        StructuredIR(
            entity="Galaxy",
            property="device",
            value="Samsung Galaxy S22 phone",
            time_scope="2023-03-15 10:00",
            raw_content="[s1 on 2023/03/15 (Wed) 10:00] user: I just unboxed the Samsung Galaxy S22 phone!",
        ),
        StructuredIR(
            entity="Dell",
            property="device",
            value="Dell XPS 13 laptop",
            time_scope="2023-03-15 14:30",
            raw_content="[s2 on 2023/03/15 (Wed) 14:30] user: Later that day I received the Dell XPS 13 laptop.",
        ),
    ]
    query = "Which device did I receive first, the Samsung Galaxy S22 or the Dell XPS 13?"
    grounding = resolver.resolve(query, records)

    assert grounding is not None
    assert grounding.calculation_type == "ordering"
    assert "samsung galaxy s22" in grounding.grounding_text
    assert "occurred on 2023-03-15 10:00" in grounding.grounding_text
    assert "occurred on 2023-03-15 14:30" in grounding.grounding_text
    assert "The event that happened first is 'samsung galaxy s22'." in grounding.grounding_text
