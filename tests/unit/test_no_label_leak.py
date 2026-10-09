"""Regression tests: the V4 inference path must be invariant to gold labels.

These tests lock the fundamental design rule of the label-free baseline:
gold ``question_type`` / ``answer`` / ``answer_session_ids`` may be consumed by
the *grader* and the offline *recall measurement*, but never by routing,
retrieval, prompt construction, or answer generation.

Two complementary guards are used:
1. A static AST scan of the inference class, which fails if a gold-label
   identifier or dict key appears anywhere in the inference methods.
2. A behavioural invariance test: two items that differ ONLY in their gold
   fields must produce byte-identical prompts and predictions.
3. A signature guard on the frozen reader, which cannot accept a question object.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

import artificial_memory.compiler.eval.phase0_runner as phase0
from artificial_memory.research.benchmarks.llm import OllamaAnswerer


# Gold-label tokens that must never appear in the inference path.
# Note: bare "answer" is intentionally excluded because the frozen reader
# legitimately exposes an ``answer()`` method that the pipeline calls.
FORBIDDEN_TOKENS = {
    "question_type",
    "answer_session_ids",
    "ground_truth",
    "qtype",
    "is_abs",
}

INFERENCE_METHODS = (
    "run",
    "_classify",
    "_turns_to_records_and_ir",
    "_build_bundles",
    "_get_anchor_scores",
    "_answer_cached",
    "_compile",
    "_build_soft_prompt",
)


def _collect_forbidden_hits(source: str) -> list[str]:
    """Return forbidden identifiers used by ``source`` (docstrings excluded).

    Only exact identifier names, attribute names, keyword-argument names, and
    string constants equal to a forbidden token are flagged. Prose docstrings
    mentioning the labels (e.g. "NEVER uses: question_type") are intentionally
    not flagged, because they are single long strings, not the token itself.
    """
    tree = ast.parse(textwrap.dedent(source))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_TOKENS:
            hits.append(f"Name:{node.id}")
        elif isinstance(node, ast.arg) and node.arg in FORBIDDEN_TOKENS:
            hits.append(f"arg:{node.arg}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_TOKENS:
            hits.append(f"attr:{node.attr}")
        elif isinstance(node, ast.keyword) and node.arg in FORBIDDEN_TOKENS:
            hits.append(f"kwarg:{node.arg}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in FORBIDDEN_TOKENS:
            hits.append(f"str:{node.value}")
    return hits


def test_inference_methods_reference_no_gold_labels():
    """Static guard: no gold-label token may appear in the inference methods."""
    cls = phase0.LabelFreeV4Pipeline
    violations: dict[str, list[str]] = {}
    for method_name in INFERENCE_METHODS:
        method = getattr(cls, method_name)
        hits = _collect_forbidden_hits(inspect.getsource(method))
        if hits:
            violations[method_name] = hits
    assert not violations, f"gold-label leakage in inference methods: {violations}"


def test_run_signature_accepts_only_query_and_haystack():
    """The inference entry point must not accept gold fields."""
    params = set(inspect.signature(phase0.LabelFreeV4Pipeline.run).parameters)
    assert params == {
        "self", "question", "sessions", "session_dates", "session_ids", "reference_date",
    }


def test_frozen_reader_signature_cannot_receive_gold():
    """The leak boundary is enforced by signature: (question_text, context) only."""
    params = list(inspect.signature(OllamaAnswerer.answer).parameters)
    assert params == ["self", "question_text", "context"]


# ---------------------------------------------------------------------------
# Behavioural invariance
# ---------------------------------------------------------------------------

MINI_SESSIONS = [[
    {"role": "user", "content": "My favorite color is teal."},
    {"role": "assistant", "content": "Teal is a calm color."},
]]
MINI_DATES = ["2024-01-01"]
MINI_IDS = ["s1"]


class _StubReader:
    """Records the exact prompt, returns a fixed answer (no LLM call)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def answer(self, question_text: str, context: str):
        self.prompts.append(context)
        return SimpleNamespace(
            text="teal", prompt_tokens=123, completion_tokens=2, latency_ms=1.0,
        )


def _make_pipeline(tmp_path, monkeypatch):
    pipeline = phase0.LabelFreeV4Pipeline(
        token_budget=1300,
        cache_dir=tmp_path / "cache",
        use_llm_cache=False,
    )
    # Deterministic, dependency-free routing for the test.
    monkeypatch.setattr(pipeline, "_classify", lambda q: "single-session-user")
    stub = _StubReader()
    monkeypatch.setattr(pipeline.reader, "answer", stub.answer)
    return pipeline, stub


@pytest.mark.parametrize(
    ("gold_type", "gold_answer", "gold_sessions"),
    [
        ("temporal-reasoning", "32 weeks", ["s9"]),
        ("multi-session", "42", ["s7", "s8"]),
    ],
)
def test_output_is_invariant_to_gold_labels(tmp_path, monkeypatch, gold_type, gold_answer, gold_sessions):
    """Two items differing ONLY in gold fields must run identically."""
    pipeline, stub = _make_pipeline(tmp_path, monkeypatch)

    # A deliberately operator-agnostic question: it maps to no deterministic
    # operator (no quantity to sum, no list to index, no state to supersede),
    # so both runs go through the LLM reader and we can assert the reader
    # prompts are byte-identical — the core no-leak guarantee.
    base = {"question": "How are you doing today?", "question_date": "2024-01-02"}
    item_a = {**base, "question_type": "single-session-user", "answer": "teal",
              "answer_session_ids": ["s1"]}
    item_b = {**base, "question_type": gold_type, "answer": gold_answer,
              "answer_session_ids": gold_sessions}

    def run_item(item):
        return pipeline.run(
            question=item["question"],
            sessions=MINI_SESSIONS,
            session_dates=MINI_DATES,
            session_ids=MINI_IDS,
            reference_date=item["question_date"],
        )

    out_a = run_item(item_a)
    out_b = run_item(item_b)

    assert out_a["predicted_answer"] == out_b["predicted_answer"]
    assert out_a["context_tokens"] == out_b["context_tokens"]
    assert out_a["selected_session_ids"] == out_b["selected_session_ids"]
    # The exact reader prompts must be identical across both runs.
    assert len(stub.prompts) == 2
    assert stub.prompts[0] == stub.prompts[1]
