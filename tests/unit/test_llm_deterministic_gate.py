"""Unit tests for the deterministic co-processor pre-pass in LLMManager.

The gate lives in ``artificial_memory.llm.manager``: a question-shaped
message with assembled context may be answered directly by the
DeterministicMemoryEngine without calling the LLM provider.
"""
from __future__ import annotations

import asyncio

import pytest

from artificial_memory.core.models import TokenUsage
from artificial_memory.llm.manager import LLMManager, _is_question_like

CONTEXT = (
    "[D4:5 on 27 June, 2023] Caroline: I have two dogs, Rex and Bella.\n"
    "[D4:6 on 27 June, 2023] Jake: Nice, any cats?\n"
    "[D4:7 on 27 June, 2023] Caroline: No cats, just the dogs."
)
COUNT_QUESTION = "How many dogs does Caroline have?"
COUNT_ANSWER = "two, 2"


class FakeTopic:
    id = 42


class FakeStore:
    """Only the members the deterministic path needs."""

    def __init__(self):
        self.token_usages: list[TokenUsage] = []

    def get_project_by_name(self, name):
        return {"id": 1}

    def create_project(self, project):
        project.id = 1
        return project

    def get_topic_by_path(self, path):
        return FakeTopic() if path in ("Work",) else None

    def create_topic(self, topic):
        topic.id = 42
        return topic

    def get_memories(self, **kwargs):
        return []

    def log_token_usage(self, usage):
        self.token_usages.append(usage)
        return usage


class FakeContextBuilder:
    def build_context(self, query, topic_id, max_tokens=4000, current_memories=None):
        return CONTEXT

    def build_system_prompt(self, topic):
        return f"system[{topic}]"


class FakeProvider:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages, temperature=0.7, max_tokens=4000, stream=False):
        self.calls += 1
        return "LLM-FALLBACK"

    async def generate_stream(self, messages, temperature=0.7, max_tokens=4000):
        self.calls += 1
        yield "LLM-FALLBACK"

    def count_tokens(self, text):
        return len(text) // 4

    @property
    def model(self):
        return "fake"


def make_manager(enable_deterministic: bool = True) -> tuple[LLMManager, FakeProvider]:
    store = FakeStore()
    manager = LLMManager(
        store, recall_engine=None, context_builder=FakeContextBuilder(),
        enable_deterministic=enable_deterministic,
    )
    provider = FakeProvider()
    manager.add_provider("fake", provider)
    manager.default_provider = "fake"
    return manager, provider


def test_question_like_gate():
    assert _is_question_like("How many dogs does she have?")
    assert _is_question_like("who called the police?")
    assert _is_question_like("What did they eat?")
    assert not _is_question_like("hey, what is up")
    assert not _is_question_like("please summarize this for me")


def test_chat_commits_deterministic_answer_without_llm():
    manager, provider = make_manager()
    result = asyncio.run(
        manager.chat(COUNT_QUESTION, topic_path="Work")
    )
    assert result["deterministic"] is True
    assert result["response"] == COUNT_ANSWER
    assert result["deterministic_source"]
    assert provider.calls == 0
    assert result["model"] == "deterministic-coprocessor"
    assert result["total_tokens"] > 0


def test_chat_falls_back_to_llm_for_non_question():
    manager, provider = make_manager()
    result = asyncio.run(manager.chat("hello there", topic_path="Work"))
    assert result["response"] == "LLM-FALLBACK"
    assert result.get("deterministic", False) is False
    assert provider.calls == 1


def test_chat_deterministic_disabled_routes_to_llm():
    manager, provider = make_manager(enable_deterministic=False)
    result = asyncio.run(manager.chat(COUNT_QUESTION, topic_path="Work"))
    assert result["response"] == "LLM-FALLBACK"
    assert provider.calls == 1


def test_chat_stream_commits_deterministic_answer():
    manager, provider = make_manager()

    async def consume():
        chunks = []
        async for chunk in manager.chat_stream(COUNT_QUESTION, topic_path="Work"):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(consume())
    assert chunks == [COUNT_ANSWER]
    assert provider.calls == 0


def test_deterministic_logs_token_usage():
    manager, _ = make_manager()
    asyncio.run(manager.chat(COUNT_QUESTION, topic_path="Work"))
    store = manager.store
    assert len(store.token_usages) == 1
    usage = store.token_usages[0]
    assert usage.operation == "chat-deterministic"
    assert usage.model == "deterministic-coprocessor"
    assert usage.output_tokens > 0
    assert usage.total_tokens == usage.output_tokens


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
