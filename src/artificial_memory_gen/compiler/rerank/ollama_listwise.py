"""Listwise LLM re-ranking through the local Ollama server.

Why a second backend exists: the cross-encoder is the better instrument, but it
needs `sentence-transformers` + torch from PyPI. On a machine that cannot reach
PyPI (the case this was developed on), that leaves the reranking hypothesis
untested. Ollama is already a hard dependency of every benchmark run here, so a
listwise relevance pass costs one extra call per question and zero installs.

It is still a *general* relevance model: the prompt states the task, the model
brings its own pretraining, and no LoCoMo vocabulary or per-question rule is
involved. It is not a cross-encoder, and the run manifest says which backend
produced a number.

Design notes:
- one call per batch of candidates, JSON output, temperature 0, seed 42;
- a parse failure is never fatal: the batch keeps its retrieval order, and the
  failure is counted so a run cannot silently claim it reranked;
- the prompt sees speaker + session date, because the losses this targets are
  attribution and time.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from artificial_memory_gen.compiler.rerank.cross_encoder import Turn, passage_text

DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_BASE_URL = "http://localhost:11434"

_SYSTEM = (
    "You score how useful each evidence passage is for answering a question. "
    "You never answer the question. Reply with JSON only."
)

_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Passages:\n{passages}\n\n"
    "Score every passage from 0 to 10 for how directly it contains information "
    "that answers the question. 0 = unrelated, 10 = contains the answer.\n"
    'Reply with JSON exactly like: {{"scores": [{zeros}]}}'
)


class OllamaListwiseReranker:
    """Re-order candidate turns with a local Ollama chat model.

    ``chat_fn`` is injectable for tests. It receives the message list and the
    batch size and must return the model's raw reply text.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 180.0,
        batch_size: int = 25,
        max_passage_words: int = 70,
        chat_fn: Callable[[list[dict[str, str]], int], str] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_passage_words = max_passage_words
        self._chat_fn = chat_fn
        self.failures = 0
        self.requests = 0

    # ------------------------------------------------------------------ ollama
    def _chat(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        self.requests += 1
        if self._chat_fn is not None:
            return self._chat_fn(messages, max_tokens)
        import httpx

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "seed": 42,
                "num_predict": max_tokens,
            },
        }
        response = httpx.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return str(response.json().get("message", {}).get("content", ""))

    # ---------------------------------------------------------------- scoring
    def _prompt(self, question: str, passages: Sequence[str]) -> list[dict[str, str]]:
        numbered = "\n".join(
            f"{i}. {passage}" for i, passage in enumerate(passages, 1))
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER_TEMPLATE.format(
                question=question, passages=numbered,
                zeros=", ".join("0" for _ in passages))},
        ]

    @staticmethod
    def parse_scores(raw: str, expected: int) -> list[float] | None:
        """Extract ``expected`` scores from a model reply, or None.

        Accepts ``{"scores": [...]}``, a bare array, and the degraded
        ``1: 7`` line format some models fall back to. Anything that does not
        yield exactly one number per passage is a parse failure, not a
        best-effort guess.
        """
        payload: Any = None
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            payload = None

        values: list[Any] | None = None
        if isinstance(payload, dict):
            for key in ("scores", "relevance", "ratings"):
                if isinstance(payload.get(key), list):
                    values = payload[key]
                    break
        elif isinstance(payload, list):
            values = payload

        if values is None:
            line_pairs = re.findall(r"^\s*(\d+)\s*[:.)-]\s*(-?\d+(?:\.\d+)?)\s*$",
                                    raw, re.MULTILINE)
            if line_pairs and len(line_pairs) == expected:
                ordered = sorted(line_pairs, key=lambda p: int(p[0]))
                values = [score for _idx, score in ordered]

        if values is None or len(values) != expected:
            return None
        try:
            return [float(value) for value in values]
        except (TypeError, ValueError):
            return None

    def token_scores(self, question: str, passages: Sequence[str]) -> list[float] | None:
        """0..1 relevance for each passage, or None when the reply is unusable."""
        reply = self._chat(
            self._prompt(question, passages),
            max_tokens=min(16 * len(passages) + 32, 1024),
        )
        raw = self.parse_scores(reply, len(passages))
        if raw is None:
            self.failures += 1
            return None
        return [max(0.0, min(1.0, value / 10.0)) for value in raw]

    # ------------------------------------------------------------------ rerank
    def rerank_turns(
        self,
        query: str,
        turns: Sequence[Turn],
        scores: dict[tuple[str, str], float],
        *,
        top_k: int = 0,
    ) -> tuple[list[Turn], dict[tuple[str, str], float]]:
        """Re-order ``turns`` by model-judged relevance; scale to 0-100.

        Same contract as ``CrossEncoderReranker.rerank_turns``: the tail beyond
        ``top_k`` is untouched, inputs are not mutated, and a batch whose reply
        cannot be parsed keeps its retrieval order.
        """
        ordered_input = list(turns)
        if not ordered_input:
            return ordered_input, dict(scores)
        head = ordered_input[:top_k] if top_k > 0 else ordered_input
        tail = ordered_input[len(head):]

        passages = [
            " ".join(passage_text(turn).split()[: self.max_passage_words])
            for turn in head
        ]
        relevance: list[float | None] = []
        for start in range(0, len(passages), self.batch_size):
            chunk = passages[start:start + self.batch_size]
            batch_scores = self.token_scores(query, chunk)
            relevance.extend(
                batch_scores if batch_scores is not None else [None] * len(chunk))

        merged = dict(scores)
        known = [value for value in relevance if value is not None]
        lo = min(known) if known else 0.0
        hi = max(known) if known else 0.0

        scored: list[tuple[Turn, float]] = []
        unscored: list[Turn] = []
        for turn, value in zip(head, relevance):
            key = (turn[0], turn[3])
            if value is None:
                # An unparseable batch keeps its retrieval score and position.
                unscored.append(turn)
                continue
            scaled = 50.0 if hi - lo < 1e-9 else 100.0 * (value - lo) / (hi - lo)
            merged[key] = scaled
            scored.append((turn, scaled))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        reordered = [turn for turn, _ in scored] + unscored
        return reordered + tail, merged


def model_available(model: str, base_url: str = DEFAULT_BASE_URL,
                    timeout: float = 5.0) -> str | None:
    """Return an error string when ``model`` is not served, or None."""
    import httpx

    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - the message is the point
        return (f"Ollama is not reachable at {base_url} ({exc}).\n"
                "  start it with:  ollama serve")
    names = {m.get("name") for m in response.json().get("models", [])
             if isinstance(m, dict)}
    if model not in names:
        found = ", ".join(sorted(n for n in names if n)) or "none"
        return (f"Ollama is running but {model!r} is not installed "
                f"(found: {found}).\n  pull it with:  ollama pull {model}")
    return None
