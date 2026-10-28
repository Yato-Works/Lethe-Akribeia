"""Cross-encoder re-ranking of retrieved candidate turns.

Why this exists: retrieval here is lexical + entity scored, with a large
hand-tuned expansion table.  It reliably surfaces the right *region* of a
conversation, but the order inside the 45-turn candidate pool is not a relevance
estimate -- so the token-budgeted compiler can spend its budget on fluent,
on-topic noise while the one turn that answers the question sits below the cut.

A cross-encoder scores each (question, passage) pair jointly.  The default model
is trained on general web QA pairs, so it carries no LoCoMo vocabulary and no
per-question rules: the same weights run on any corpus.  That is the point -- it
is a general relevance model, not another tuning pass.

The dependency is optional (``sentence-transformers``, part of the ``[vector]``
extra) and imported lazily, so the product path never pays for it and the
benchmark path keeps working with ``--rerank off`` (the default).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: (session_id, session_date, speaker, content) -- the pipeline's turn tuple.
Turn = tuple[str, str, str, str]


class CrossEncoderUnavailable(RuntimeError):
    """The optional cross-encoder stack is not installed."""


def passage_text(turn: Turn) -> str:
    """Render one turn as the passage the cross-encoder scores.

    Speaker and session date are kept: the failures this is meant to address are
    silence about *who* said something and *when*, and a passage stripped of
    attribution cannot represent either.
    """
    _sid, date, speaker, content = turn
    prefix = " ".join(part for part in (date, speaker) if part)
    return f"{prefix}: {content}" if prefix else content


class CrossEncoderReranker:
    """Order a candidate pool by (question, passage) relevance.

    ``scorer`` is injectable for tests only; production always loads a
    ``sentence_transformers.CrossEncoder`` whose ``predict`` takes
    ``[(query, passage), ...]`` and returns one logit per pair.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        max_length: int = 512,
        batch_size: int = 32,
        device: str | None = None,
        scorer: Callable[..., Sequence[float]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self._scorer = scorer
        self._model: Callable[..., Sequence[float]] | None = None

    # ------------------------------------------------------------------ model
    def _load(self) -> Callable[..., Sequence[float]]:
        if self._model is not None:
            return self._model
        if self._scorer is not None:
            self._model = self._scorer
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise CrossEncoderUnavailable(
                "cross-encoder re-ranking needs the optional vector stack: "
                'pip install -e ".[vector]"  (sentence-transformers + torch)'
            ) from exc
        self._model = CrossEncoder(
            self.model_name, max_length=self.max_length, device=self.device)
        return self._model

    def score_pairs(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance of each passage to the query, squashed to 0..1."""
        if not passages:
            return []
        model = self._load()
        predictions = model.predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [_sigmoid(float(value)) for value in predictions]

    def ensure_loaded(self) -> None:
        """Load the model now.

        A benchmark run is long; discovering a missing optional dependency at
        question 1 of 1,986 -- or worse, having the per-question error handler
        swallow it into an empty artifact -- is not acceptable.  Callers that are
        about to start a run ask for the model up front.
        """
        self._load()

    # ------------------------------------------------------------------ rerank
    def rerank_turns(
        self,
        query: str,
        turns: Sequence[Turn],
        scores: dict[tuple[str, str], float],
        *,
        top_k: int = 0,
    ) -> tuple[list[Turn], dict[tuple[str, str], float]]:
        """Re-order ``turns`` by cross-encoder relevance and rewrite ``scores``.

        Only the first ``top_k`` turns are scored (``0`` = the whole pool); the
        rest keep their retrieval order and scores.  Scores are rescaled
        min-max onto the 0-100 band the selector already consumes, so a
        confidently-relevant turn is not outranked merely because the lexical
        scale happened to be wider.

        Returns new objects; the caller's lists/dicts are not mutated.
        """
        ordered_input = list(turns)
        if not ordered_input:
            return ordered_input, dict(scores)
        head = ordered_input[:top_k] if top_k > 0 else ordered_input
        tail = ordered_input[len(head):]

        relevance = self.score_pairs(query, [passage_text(t) for t in head])
        lo, hi = min(relevance), max(relevance)
        if hi - lo < 1e-9:
            scaled = [50.0 for _ in relevance]
        else:
            scaled = [100.0 * (value - lo) / (hi - lo) for value in relevance]

        merged = dict(scores)
        for turn, value in zip(head, scaled):
            merged[(turn[0], turn[3])] = value

        # Stable sort: equal relevance keeps the retrieval order.
        reordered = sorted(head, key=lambda t: merged[(t[0], t[3])], reverse=True)
        return reordered + tail, merged


def _sigmoid(value: float) -> float:
    # ms-marco cross-encoders emit an unbounded logit; the 0..1 form is what the
    # rest of this module documents.  Clamp before exp() so extreme logits on a
    # pathological pair cannot raise OverflowError.
    clamped = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-clamped))
