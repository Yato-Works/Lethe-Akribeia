"""Unit tests for Temporal Algebraic AST, evaluation, and compound normalization (P3).

Tests:
1. DatePointNode and OffsetNode evaluation
2. OrdinalWeekdayNode evaluation (e.g. 3rd Monday of June 2023)
3. IntervalSpanNode evaluation
4. parse_compound_temporal_expression from natural language
5. End-to-end integration via TemporalNormalizer (rule 19)
"""

from __future__ import annotations

import datetime
from artificial_memory.context.temporal_normalizer import TemporalNormalizer
from artificial_memory.temporal.interval_algebra import (
    DatePointNode,
    IntervalSpanNode,
    OffsetNode,
    OrdinalWeekdayNode,
    TimeInterval,
    parse_compound_temporal_expression,
    resolve_ast,
)


def test_offset_node_day_and_week() -> None:
    base = DatePointNode(datetime.date(2023, 5, 10))
    # +2 weeks
    node_w = OffsetNode(base=base, amount=2, unit="weeks", direction=1)
    res_w = resolve_ast(node_w)
    assert res_w == datetime.date(2023, 5, 24)

    # -3 days
    node_d = OffsetNode(base=base, amount=3, unit="days", direction=-1)
    res_d = resolve_ast(node_d)
    assert res_d == datetime.date(2023, 5, 7)


def test_offset_node_month_and_year() -> None:
    base = DatePointNode(datetime.date(2023, 1, 31))
    # +1 month (capped to 28 Feb in 2023 non-leap year)
    node_m = OffsetNode(base=base, amount=1, unit="months", direction=1)
    res_m = resolve_ast(node_m)
    assert res_m == datetime.date(2023, 2, 28)

    # +2 years
    node_y = OffsetNode(base=base, amount=2, unit="years", direction=1)
    res_y = resolve_ast(node_y)
    assert res_y == datetime.date(2025, 1, 31)


def test_ordinal_weekday_node() -> None:
    # 3rd Monday of June 2023:
    # June 2023 mondays: June 5 (1st), June 12 (2nd), June 19 (3rd), June 26 (4th)
    node = OrdinalWeekdayNode(ordinal=3, weekday=0, month=6, year=2023)
    res = resolve_ast(node)
    assert res == datetime.date(2023, 6, 19)

    # Last Friday of June 2023: June 30
    node_last = OrdinalWeekdayNode(ordinal=-1, weekday=4, month=6, year=2023)
    res_last = resolve_ast(node_last)
    assert res_last == datetime.date(2023, 6, 30)


def test_interval_span_node() -> None:
    start = DatePointNode(datetime.date(2023, 1, 1))
    end = OffsetNode(base=start, amount=14, unit="days", direction=1)
    span = IntervalSpanNode(start=start, end=end, label="Two Week Sprint")
    res = resolve_ast(span)
    assert isinstance(res, TimeInterval)
    assert res.start == datetime.date(2023, 1, 1)
    assert res.end == datetime.date(2023, 1, 15)
    assert res.duration_days == 14


def test_parse_compound_temporal_expression() -> None:
    ref = datetime.date(2023, 5, 1)

    # Offset from concrete date
    ast1 = parse_compound_temporal_expression("2 weeks after 25 December 2022", ref)
    assert isinstance(ast1, OffsetNode)
    res1 = resolve_ast(ast1)
    assert res1 == datetime.date(2023, 1, 8)

    # Ordinal weekday
    ast2 = parse_compound_temporal_expression("the third Monday of June 2023", ref)
    assert isinstance(ast2, OrdinalWeekdayNode)
    res2 = resolve_ast(ast2)
    assert res2 == datetime.date(2023, 6, 19)


def test_temporal_normalizer_compound_integration() -> None:
    normalizer = TemporalNormalizer()
    text = "We scheduled the meeting for 2 weeks after 25 December 2022 in Berlin."
    normalized = normalizer.normalize(text, "10 May 2023")
    assert "2 weeks after 25 December 2022 (8 January 2023)" in normalized

    text_ord = "Let us meet on the third Monday of June 2023 to review."
    norm_ord = normalizer.normalize(text_ord, "10 May 2023")
    assert "the third Monday of June 2023 (19 June 2023)" in norm_ord


def test_temporal_certificate_compound_resolution() -> None:
    from artificial_memory.compiler.certificates.temporal_certificate import TemporalCertificateEngine
    from artificial_memory.core.ir.structured import StructuredIR

    engine = TemporalCertificateEngine()
    records = [
        StructuredIR(
            entity="Joanna",
            property="first_watch",
            value="3 years ago",
            time_scope="2022-04-10",
            source="Joanna",
            raw_content="[S5 on 2022-04-10] Joanna: I first watched Eternal Sunshine of the Spotless Mind 3 years ago.",
        ),
        StructuredIR(
            entity="Nate",
            property="won_money",
            value="the week before",
            time_scope="2022-10-06",
            source="Nate",
            raw_content="[S12 on 2022-10-06] Nate: I won a lot of money in a video game tournament the week before 6 October 2022.",
        ),
    ]

    # Test 1: "When did Joanna first watch..."
    cert1 = engine.generate("When did Joanna first watch Eternal Sunshine of the Spotless Mind?", records, "2022-05-01")
    assert cert1.has_certificate
    assert cert1.query_type == "compound"
    assert cert1.direct_answer_candidate == "2019"
    assert "2019" in cert1.formatted_certificate

    # Test 2: "When did Nate win money in the tournament?"
    cert2 = engine.generate("When did Nate win a lot of money in a video game tournament?", records, "2022-10-10")
    assert cert2.has_certificate
    assert cert2.query_type == "compound"
    assert "September 2022" in cert2.direct_answer_candidate


def test_temporal_certificate_allen_pairwise_ordering() -> None:
    from artificial_memory.compiler.certificates.temporal_certificate import TemporalCertificateEngine
    from artificial_memory.core.ir.structured import StructuredIR

    engine = TemporalCertificateEngine()
    records = [
        StructuredIR(
            entity="Joanna",
            property="visited",
            value="Rome",
            time_scope="2021-06-15",
            source="Joanna",
            raw_content="[S1 on 2021-06-15] Joanna: I visited Rome.",
        ),
        StructuredIR(
            entity="Joanna",
            property="visited",
            value="Kyoto",
            time_scope="2022-09-20",
            source="Joanna",
            raw_content="[S2 on 2022-09-20] Joanna: I visited Kyoto.",
        ),
    ]

    cert = engine.generate("Which city did Joanna visit first, Rome or Kyoto?", records, "2023-01-01")
    assert cert.has_certificate
    assert cert.query_type == "ordering"
    assert cert.direct_answer_candidate == "Rome"
    assert cert.interval_relation == "before"
    assert "completed before" in cert.allen_derivation
