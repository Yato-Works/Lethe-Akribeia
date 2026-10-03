"""Unit tests for the generalized SessionFuser cross-session aggregation engine."""

from artificial_memory.protein.session_fuser import SessionFuser


def test_is_aggregation_query():
    fuser = SessionFuser()
    assert fuser.is_aggregation_query("How much did I spend in total on electronics?") is True
    assert fuser.is_aggregation_query("How many days did I spend in Kyoto?") is True
    assert fuser.is_aggregation_query("What was the combined cost of the flights?") is True
    assert fuser.is_aggregation_query("What time did I arrive at the hotel?") is False
    assert fuser.is_aggregation_query("Who was the speaker at the keynote?") is False
    assert fuser.is_aggregation_query("Where did I leave my passport?") is False


def test_is_multi_session_reasoning_query():
    fuser = SessionFuser()
    assert fuser.is_multi_session_reasoning_query("What is the difference in price between the laptop and the tablet?") is True
    assert fuser.is_multi_session_reasoning_query("How much faster did I run the 10K compared to last year?") is True
    assert fuser.is_multi_session_reasoning_query("What percentage discount did I get on the hotel?") is True
    assert fuser.is_multi_session_reasoning_query("What is the capital of France?") is False


def test_determine_unit():
    fuser = SessionFuser()
    assert fuser.determine_unit("How much did I spend on groceries?") == "$"
    assert fuser.determine_unit("What was the total cost of the hotel?") == "$"
    assert fuser.determine_unit("How many days did I take off work?") == "days"
    assert fuser.determine_unit("How many hours of sleep did I get?") == "hours"
    assert fuser.determine_unit("How many weeks did the project take?") == "weeks"
    assert fuser.determine_unit("How many months did I live in Berlin?") == "months"
    assert fuser.determine_unit("How many years have I lived here?") == "years"
    assert fuser.determine_unit("How many books did I finish reading this month?") == "books"
    assert fuser.determine_unit("How many classes did I attend?") == "classes"


def test_fuse_financial_multi_session():
    fuser = SessionFuser()
    records = (
        "[s1] user: I bought a mechanical keyboard for $120.\n"
        "[s2] user: I ordered a 4K monitor for $350.\n"
        "[s3] user: Picked up an ergonomic mouse for $80.\n"
    )
    res = fuser.fuse("How much did I spend in total on computer accessories?", records)
    assert res.is_aggregation_query is True
    assert res.unit == "$"
    assert res.total_value == 550.0
    assert len(res.found_snippets) == 3
    assert "[GLOBAL STATE AGGREGATION" in res.certificate
    assert "$550" in res.certificate


def test_fuse_temporal_days_multi_session():
    fuser = SessionFuser()
    records = (
        "[s1] user: I spent 4 days hiking in Yosemite.\n"
        "[s2] user: Later I spent 3 days exploring Lake Tahoe.\n"
    )
    res = fuser.fuse("How many days did I spend hiking and exploring outdoors?", records)
    assert res.is_aggregation_query is True
    assert res.unit == "days"
    assert res.total_value == 7.0
    assert len(res.found_snippets) == 2
    assert "7 days" in res.certificate


def test_fuse_delivery_days():
    fuser = SessionFuser()
    records = (
        "[s1] user: I ordered the new lens on 04/10.\n"
        "[s2] user: The camera lens arrived on 04/15.\n"
    )
    res = fuser.fuse("How many days did it take for the camera lens to arrive after I ordered it?", records)
    assert res.is_aggregation_query is True
    assert res.unit == "days"
    assert res.total_value == 5.0
    assert len(res.found_snippets) == 2
    assert "5 days" in res.certificate


def test_fuse_distinct_entity_counts():
    fuser = SessionFuser()
    records = (
        "[s1] user: I attended the machine learning workshop on Saturday.\n"
        "[s2] user: I attended the cloud architecture workshop next week.\n"
    )
    res = fuser.fuse("How many workshops did I attend in total?", records)
    assert res.is_aggregation_query is True
    assert res.unit == "workshops"
    assert res.total_value == 2.0
    assert len(res.found_snippets) == 2
