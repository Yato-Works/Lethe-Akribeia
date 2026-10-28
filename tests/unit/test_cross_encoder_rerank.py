"""Contract tests for cross-encoder re-ranking.

These run without torch or a model download: the scorer is injected, so the tests
pin the behaviour the pipeline depends on (ordering, score band, stability,
tail handling, laziness) rather than the model's weights.
"""

from __future__ import annotations

import builtins

import pytest

from artificial_memory_gen.compiler.rerank.cross_encoder import (
    CrossEncoderReranker,
    CrossEncoderUnavailable,
    passage_text,
)


class _StubScorer:
    """Mimics ``sentence_transformers.CrossEncoder.predict``."""

    def __init__(self, logits: list[float]) -> None:
        self.logits = logits
        self.seen: list[list[tuple[str, str]]] = []

    def predict(self, pairs, batch_size: int = 32, show_progress_bar: bool = False):
        self.seen.append(list(pairs))
        return self.logits


class TestPassage:
    def test_keeps_speaker_and_date(self):
        passage = passage_text(
            ("session_1", "10:37 am on 27 June, 2023", "Caroline", "I went to a parade."))
        assert "Caroline" in passage
        assert "27 June, 2023" in passage
        assert "parade" in passage

    def test_empty_metadata_does_not_leave_a_dangling_colon(self):
        assert passage_text(("s1", "", "", "I went to a parade.")) == "I went to a parade."


class TestReRank:
    def _turns(self):
        return [
            ("session_1", "d1", "A", "pottery class chatter"),
            ("session_1", "d1", "B", "I went to the pride parade"),
            ("session_2", "d2", "A", "a stained glass window"),
        ]

    def _scores(self):
        return {
            ("session_1", "pottery class chatter"): 80.0,
            ("session_1", "I went to the pride parade"): 10.0,
            ("session_2", "a stained glass window"): 5.0,
        }

    def test_reorders_by_cross_encoder_and_rescales_to_selector_band(self):
        reranker = CrossEncoderReranker(scorer=_StubScorer([0.0, 5.0, 0.0]))
        turns, scores = reranker.rerank_turns("q", self._turns(), self._scores())

        assert turns[0][3] == "I went to the pride parade"
        assert scores[("session_1", "I went to the pride parade")] == 100.0
        assert scores[("session_1", "pottery class chatter")] == 0.0
        assert max(scores.values()) <= 100.0
        assert min(scores.values()) >= 0.0

    def test_equal_relevance_keeps_retrieval_order(self):
        reranker = CrossEncoderReranker(scorer=_StubScorer([0.0, 0.0, 0.0]))
        turns, scores = reranker.rerank_turns("q", self._turns(), self._scores())

        assert [t[3] for t in turns] == [t[3] for t in self._turns()]
        assert set(scores.values()) >= {50.0}

    def test_top_k_scores_only_the_head_and_leaves_the_tail_alone(self):
        stub = _StubScorer([0.0, 5.0])
        reranker = CrossEncoderReranker(scorer=stub)
        turns, scores = reranker.rerank_turns("q", self._turns(), self._scores(), top_k=2)

        assert len(stub.seen[0]) == 2, "top_k must bound the scored pairs"
        assert turns[-1][3] == "a stained glass window"
        assert scores[("session_2", "a stained glass window")] == 5.0

    def test_inputs_are_not_mutated(self):
        turns_in = self._turns()
        scores_in = self._scores()
        turns_copy = list(turns_in)
        scores_copy = dict(scores_in)

        CrossEncoderReranker(scorer=_StubScorer([0.0, 5.0, 0.0])).rerank_turns(
            "q", turns_in, scores_in)

        assert turns_in == turns_copy
        assert scores_in == scores_copy

    def test_empty_pool_is_a_no_op(self):
        reranker = CrossEncoderReranker(scorer=_StubScorer([]))
        turns, scores = reranker.rerank_turns("q", [], {})
        assert turns == []
        assert scores == {}


class TestDependency:
    def test_construction_is_lazy(self):
        # No import happens until a score is requested, so the product path and
        # `--rerank off` never pay for torch.
        CrossEncoderReranker()

    def test_missing_stack_raises_an_actionable_error(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("sentence_transformers"):
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(CrossEncoderUnavailable, match=r"\[vector\]"):
            CrossEncoderReranker().score_pairs("q", ["a passage"])
