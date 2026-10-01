"""Unit tests for Bi-temporal modeling in StructuredIR."""

from __future__ import annotations

from artificial_memory.core.ir.structured import IRRelation, IRStatus, StructuredIR


def test_bi_temporal_defaults_and_validity() -> None:
    ir = StructuredIR(
        entity="Caroline",
        property="city",
        value="New York",
        valid_from="2021-05-01",
        valid_until="2023-04-30",
        assertion_time="2021-05-02 10:00",
    )
    # Valid within interval
    assert ir.is_valid_at("2022-01-01")
    # Invalid before
    assert not ir.is_valid_at("2020-12-31")
    # Invalid after
    assert not ir.is_valid_at("2023-05-01")


def test_supersede_updates_valid_until_and_status() -> None:
    ir = StructuredIR(
        entity="Project",
        property="database",
        value="PostgreSQL",
        valid_from="2022-01-01",
    )
    ir.supersede("ClickHouse", superseded_at="2023-06-01")

    assert ir.status == IRStatus.SUPERSEDED
    assert ir.relation == IRRelation.MIGRATED
    assert ir.old_value == "PostgreSQL"
    assert ir.value == "ClickHouse"
    assert ir.valid_until == "2023-06-01"

    line = ir.format_context_line()
    assert "previously PostgreSQL until 2023-06-01" in line
    assert "currently ClickHouse" in line


def test_to_dict_includes_bi_temporal_fields() -> None:
    ir = StructuredIR(
        entity="Alice",
        property="role",
        value="Engineer",
        valid_from="2020-01-01",
        valid_until="2024-01-01",
        assertion_time="2020-01-05 09:00",
    )
    d = ir.to_dict()
    assert d["valid_from"] == "2020-01-01"
    assert d["valid_until"] == "2024-01-01"
    assert d["assertion_time"] == "2020-01-05 09:00"
