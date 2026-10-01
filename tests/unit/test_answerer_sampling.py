"""Unit tests for category/question-type bounded answer sampling."""

from __future__ import annotations

from typing import Any

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer


class DummyRecordingAnswerer:
    """Mock answerer that records max_tokens and stop attributes while preserving strict 2-arg signature."""

    def __init__(self) -> None:
        self.max_tokens: int | None = None
        self.stop: list[str] | None = None
        self.recorded_tokens_during_call: int | None = None
        self.recorded_stop_during_call: list[str] | None = None

    def answer(self, question_text: str, context: str) -> Any:
        # Record what attributes were set during execution
        self.recorded_tokens_during_call = self.max_tokens
        self.recorded_stop_during_call = self.stop

        class DummyResponse:
            text = "42"

        return DummyResponse()


class PureFunctionAnswererWithoutAttributes:
    """Mock answerer without max_tokens/stop attributes."""

    def answer(self, question_text: str, context: str) -> Any:
        class DummyResponse:
            text = "Pure Answer"

        return DummyResponse()


def test_locomo_adapter_call_answerer_bounds() -> None:
    adapter = LoCoMoAdapter(dataset_path="datasets/external/locomo10.json")
    answerer = DummyRecordingAnswerer()

    # Category 2: temporal -> max_tokens=48
    res = adapter._call_answerer(answerer, "When was that?", "Prompt text", category=2)
    assert res.text == "42"
    assert answerer.recorded_tokens_during_call == 48
    assert "\n\nUser:" in (answerer.recorded_stop_during_call or [])
    # Reverts after call
    assert answerer.max_tokens is None

    # Category 1: multi-hop -> max_tokens=80
    res = adapter._call_answerer(answerer, "Who did what?", "Prompt text", category=1)
    assert answerer.recorded_tokens_during_call == 80


def test_locomo_adapter_call_answerer_fallback() -> None:
    adapter = LoCoMoAdapter(dataset_path="datasets/external/locomo10.json")
    pure = PureFunctionAnswererWithoutAttributes()

    # Should gracefully execute even if answerer doesn't have max_tokens attribute
    res = adapter._call_answerer(pure, "When was that?", "Prompt text", category=2)
    assert res.text == "Pure Answer"


def test_longmemeval_adapter_call_answerer_bounds() -> None:
    adapter = LongMemEvalAdapter()
    answerer = DummyRecordingAnswerer()

    # Question type: single-session-preference -> max_tokens=160
    res = adapter._call_answerer(
        answerer,
        "What do I like?",
        "Prompt",
        question_type="single-session-preference",
    )
    assert res.text == "42"
    assert answerer.recorded_tokens_during_call == 160

    # Question type: temporal-reasoning -> max_tokens=64
    res = adapter._call_answerer(
        answerer,
        "How long ago?",
        "Prompt",
        question_type="temporal-reasoning",
    )
    assert answerer.recorded_tokens_during_call == 64


def test_longmemeval_adapter_call_answerer_fallback() -> None:
    adapter = LongMemEvalAdapter()
    pure = PureFunctionAnswererWithoutAttributes()

    res = adapter._call_answerer(
        pure,
        "How long ago?",
        "Prompt",
        question_type="temporal-reasoning",
    )
    assert res.text == "Pure Answer"


def test_ollama_answerer_sampling_attribute_behavior() -> None:
    """Verify that OllamaAnswerer exposes max_tokens/stop attributes without breaking API signature."""
    import inspect

    answerer = OllamaAnswerer()
    # Invariant: strict signature is strictly 2 arguments (plus self on class)!
    params = list(inspect.signature(OllamaAnswerer.answer).parameters)
    assert params == ["self", "question_text", "context"]

    # Verify attributes exist and can be set
    assert hasattr(answerer, "max_tokens")
    assert hasattr(answerer, "stop")
    answerer.max_tokens = 42
    answerer.stop = ["\n\n"]
    assert answerer.max_tokens == 42
    assert answerer.stop == ["\n\n"]
