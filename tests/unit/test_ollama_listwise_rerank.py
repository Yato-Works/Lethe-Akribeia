"""Contract tests for the Ollama listwise re-ranker (no server required).

The backend exists so the reranking hypothesis can be measured on a machine that
cannot install torch. What must hold regardless of the model:

1. a reply is only used when it yields exactly one score per passage;
2. anything else keeps the retrieval order and is counted as a failure, so a run
   cannot silently claim it re-ranked;
3. scores land on the 0-100 band the selector consumes, and inputs are not
   mutated;
4. the prompt shows the model the speaker and the date, not just the text.
"""

from __future__ import annotations

import json
import re

from artificial_memory_gen.compiler.rerank.ollama_listwise import (
    OllamaListwiseReranker,
)


def _turns(n: int = 3):
    return [
        ("session_1", "10:37 am on 27 June, 2023", "Caroline", f"caroline turn {i}")
        for i in range(n)
    ]


def _scores(n: int = 3):
    return {
        ("session_1", f"caroline turn {i}"): 90.0 - i for i in range(n)
    }


def _reply(payload: object) -> str:
    return json.dumps(payload)


class TestParse:
    def test_object_with_scores(self):
        assert OllamaListwiseReranker.parse_scores('{"scores": [0, 5, 10]}', 3) == [
            0.0, 5.0, 10.0]

    def test_bare_array(self):
        assert OllamaListwiseReranker.parse_scores("[1, 2]", 2) == [1.0, 2.0]

    def test_line_format_fallback(self):
        raw = "1: 4\n2: 8\n3: 2"
        assert OllamaListwiseReranker.parse_scores(raw, 3) == [4.0, 8.0, 2.0]

    def test_wrong_count_is_a_failure_not_a_guess(self):
        assert OllamaListwiseReranker.parse_scores('{"scores": [1, 2]}', 3) is None

    def test_garbage_is_a_failure(self):
        assert OllamaListwiseReranker.parse_scores("I think passage 1 is best", 1) is None


class TestReRank:
    def test_reorders_and_scales(self):
        reranker = OllamaListwiseReranker(
            chat_fn=lambda messages, max_tokens: _reply({"scores": [0, 9, 3]}))
        turns, scores = reranker.rerank_turns("q", _turns(), _scores())

        assert [t[3] for t in turns] == [
            "caroline turn 1", "caroline turn 2", "caroline turn 0"]
        assert scores[("session_1", "caroline turn 1")] == 100.0
        assert scores[("session_1", "caroline turn 0")] == 0.0
        assert max(scores.values()) == 100.0
        assert reranker.failures == 0

    def test_unparseable_batch_keeps_retrieval_order_and_is_counted(self):
        reranker = OllamaListwiseReranker(
            chat_fn=lambda messages, max_tokens: "no scores here")
        turns_before = _turns()
        scores_before = _scores()
        turns, scores = reranker.rerank_turns("q", turns_before, scores_before)

        assert [t[3] for t in turns] == [t[3] for t in turns_before]
        assert scores == scores_before
        assert reranker.failures == 1

    def test_batches_are_scored_separately_and_merged(self):
        calls = []

        def chat(messages, max_tokens):
            calls.append(messages)
            numbered = len(re.findall(r"^\d+\. ", messages[-1]["content"], re.M))
            return _reply({"scores": [10] + [0] * (numbered - 1)})

        reranker = OllamaListwiseReranker(chat_fn=chat, batch_size=2)
        turns, _scores_out = reranker.rerank_turns("q", _turns(5), _scores(5))

        assert len(calls) == 3, "5 turns at batch_size=2 is three calls"
        # Equal relevance inside a batch keeps its order; batches concatenate.
        assert [t[3] for t in turns] == [
            "caroline turn 0", "caroline turn 2", "caroline turn 4",
            "caroline turn 1", "caroline turn 3"]
        assert reranker.failures == 0

    def test_scores_all_equal_stay_at_midpoint(self):
        reranker = OllamaListwiseReranker(
            chat_fn=lambda messages, max_tokens: _reply({"scores": [7, 7, 7]}))
        turns, scores = reranker.rerank_turns("q", _turns(), _scores())
        assert [scores[("session_1", t[3])] for t in turns] == [50.0, 50.0, 50.0]

    def test_inputs_are_not_mutated(self):
        turns_in = _turns()
        scores_in = _scores()
        turns_copy, scores_copy = list(turns_in), dict(scores_in)

        OllamaListwiseReranker(
            chat_fn=lambda messages, max_tokens: _reply({"scores": [0, 9, 3]})
        ).rerank_turns("q", turns_in, scores_in)

        assert turns_in == turns_copy
        assert scores_in == scores_copy

    def test_prompt_carries_speaker_and_date(self):
        seen = {}

        def chat(messages, max_tokens):
            seen["messages"] = messages
            return _reply({"scores": [1, 1, 1]})

        OllamaListwiseReranker(chat_fn=chat).rerank_turns("q", _turns(), _scores())
        user = seen["messages"][-1]["content"]
        assert "Caroline" in user
        assert "27 June, 2023" in user
        assert "1." in user and "3." in user
