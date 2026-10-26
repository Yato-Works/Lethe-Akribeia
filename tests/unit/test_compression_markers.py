"""English/Japanese parity for deterministic compression and memory extraction.

The compressor used to key its decision/reason tables on Japanese words only, so
an English conversation produced an empty decision list ("**決定事項:**" never
appeared, and no decision lines were extracted).  These tests pin the bilingual
behaviour so that regression is visible.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.compression.markers import (
    detect_language,
    has_decision,
    has_important,
    has_reason,
)
from artificial_memory.core.models import Conversation, Message, MessageRole
from artificial_memory.memory.compiler import IncrementalMemoryCompiler


def test_detect_language_english_and_japanese() -> None:
    assert detect_language("We decided to ship on Friday.") == "en"
    assert detect_language("金曜日にリリースすると決定した。") == "ja"
    assert detect_language("12345 !!!") == "en"  # no letters -> default


def test_english_decision_markers() -> None:
    assert has_decision("We decided to target September 30.")
    assert has_decision("Let's adopt option B.")
    assert has_decision("The team agreed on the scope.")
    assert has_decision("Ship it, final answer.")
    # Word boundaries: a negated/derived word is not a decision.
    assert not has_decision("The question is still undecided.")
    # Known limit: negation is not modelled, so "chosen" still fires.
    assert has_decision("Nothing has been chosen yet.")


def test_japanese_decision_markers_unchanged() -> None:
    assert has_decision("9月30日にリリースすることを決めた。")
    assert has_decision("案Bを採用する。")
    assert not has_decision("まだ何も決まっていない。")


def test_reason_markers_in_both_languages() -> None:
    assert has_reason("Because the beta needs two weeks of soak time.")
    assert has_reason("The reason is future extensibility.")
    assert has_reason("なぜなら検証に2週間必要だからだ。")
    assert not has_reason("The cat sat on the mat.")


def test_important_markers_in_both_languages() -> None:
    assert has_important("This is a critical blocker for the release.")
    assert has_important("予算は1回30ドルで、リスクは小さい。")


def test_semantic_compression_english_uses_english_labels() -> None:
    content = (
        "We decided to target September 30 for the release. "
        "Because the beta needs two weeks of soak time. "
        "The budget is 30 dollars per run."
    )
    compressed, meta = RuleBasedCompressor().compress_semantic(content, {})

    assert "**Decisions:**" in compressed
    assert "**Reasons:**" in compressed
    assert "**Key points:**" in compressed
    assert "決定事項" not in compressed
    assert meta["decisions_found"] == 1
    assert meta["reasons_found"] == 1


def test_semantic_compression_japanese_is_unchanged() -> None:
    content = (
        "リリースは9月30日に決定した。\n"
        "なぜならベータ版に2週間の検証が必要だからだ。\n"
        "予算は1回30ドル。\n"
    )
    compressed, meta = RuleBasedCompressor().compress_semantic(content, {})

    assert "**決定事項:**" in compressed
    assert "**理由:**" in compressed
    assert meta["decisions_found"] == 1
    assert meta["reasons_found"] == 1


def test_light_compression_removes_english_disfluencies() -> None:
    compressed, _ = RuleBasedCompressor().compress_light(
        "Um, we, uh, decided to ship. You know, it is fine.", {}
    )
    lowered = compressed.lower()
    assert "um," not in lowered
    assert " uh," not in lowered
    assert "you know" not in lowered
    assert "decided to ship" in lowered


def _compiler() -> IncrementalMemoryCompiler:
    return IncrementalMemoryCompiler(
        MagicMock(), RuleBasedCompressor(), MagicMock()
    )


def _conversation() -> Conversation:
    return Conversation(
        topic_id=1, title="Release", started_at=datetime(2026, 10, 26, 9, 0)
    )


def _message(role: MessageRole, content: str, seq: int) -> Message:
    return Message(
        conversation_id=1, role=role, content=content, sequence_num=seq
    )


def test_timeline_memory_english_labels() -> None:
    conv = _conversation()
    messages = [
        _message(MessageRole.USER, "When should we release?", 1),
        _message(MessageRole.ASSISTANT, "We decided to ship on Friday.", 2),
    ]

    mem = _compiler()._create_timeline_memory(conv, messages)

    assert mem is not None
    assert "Decision: We decided to ship on Friday." in mem.content
    assert "Topic: When should we release?" in mem.content


def test_timeline_memory_japanese_labels_unchanged() -> None:
    conv = _conversation()
    messages = [
        _message(MessageRole.USER, "いつリリースしますか？", 1),
        _message(MessageRole.ASSISTANT, "金曜日にリリースすると決定した。", 2),
    ]

    mem = _compiler()._create_timeline_memory(conv, messages)

    assert mem is not None
    assert "決定: 金曜日にリリースすると決定した。" in mem.content
    assert "トピック: いつリリースしますか？" in mem.content
