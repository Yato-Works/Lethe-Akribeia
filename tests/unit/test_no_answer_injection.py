"""Guard against re-introducing memorised (hand-written) answer keys.

``LoCoMoAdapter`` used to carry two dicts whose *values were the expected
answers*, matched against the question text by substring and injected into the
reader prompt as ``[DIRECTOR GUIDANCE: ...]`` / ``[COMMONSENSE GUIDANCE: ...]``.
That is answer injection, not retrieval or reasoning, and it makes any LoCoMo
score unreproducible on a hidden set.  Both tables were removed; these tests
fail if that class of shortcut is ever added back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "datasets" / "external" / "locomo10.json"

#: fingerprints of the removed injection mechanism
INJECTION_MARKERS = (
    "[DIRECTOR GUIDANCE",
    "[COMMONSENSE GUIDANCE",
    "(Answer:",
    "OPEN_DOMAIN_GUIDELINES",
    "SINGLE_HOP_DIRECTIVES",
    "allow_question_hints",
)

#: real LoCoMo question ids whose wording used to hit an answer key,
#: with the ground truth that used to be handed to the reader.
REGRESSION_QUESTIONS = {
    "conv-26-qa-110": "A cup with a dog face on it.",
    "conv-26-qa-050": "Liberal",
}


class _StubAnswer:
    def __init__(self) -> None:
        self.text = "stub answer"
        self.latency_ms = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0


class _RecordingAnswerer:
    """Stands in for OllamaAnswerer and records every prompt it is handed."""

    model = "stub"
    num_ctx = None
    answer_adapter = None

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def answer(self, question: str, prompt: str) -> _StubAnswer:
        self.prompts.append(prompt)
        return _StubAnswer()


def test_memorised_answer_tables_are_gone() -> None:
    for attr in ("OPEN_DOMAIN_GUIDELINES", "SINGLE_HOP_DIRECTIVES"):
        assert not hasattr(LoCoMoAdapter, attr), f"{attr} was reintroduced"


def test_question_hint_flag_is_gone() -> None:
    adapter = LoCoMoAdapter()
    assert not hasattr(adapter, "allow_question_hints")


def test_benchmark_adapters_carry_no_injection_fingerprint() -> None:
    pkg = REPO / "src" / "artificial_memory" / "research" / "benchmarks" / "external"
    offenders: list[str] = []
    for path in sorted(pkg.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for marker in INJECTION_MARKERS:
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
    assert not offenders, "answer-injection fingerprints found: " + ", ".join(offenders)


@pytest.mark.skipif(not DATASET.exists(), reason="LoCoMo dataset not present")
@pytest.mark.parametrize("qid,leaked_answer", sorted(REGRESSION_QUESTIONS.items()))
def test_prompt_never_contains_the_former_answer_key(qid: str, leaked_answer: str) -> None:
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    conv_idx = next(i for i, c in enumerate(raw) if c.get("sample_id") == "conv-26")

    adapter = LoCoMoAdapter()
    _, questions, ir_records = adapter.load_conversation(conv_idx)
    question = next(q for q in questions if q.question_id == qid)

    answerer = _RecordingAnswerer()
    adapter.evaluate_question(question, [], ir_records, answerer)

    assert answerer.prompts, "the reader was never called"
    for prompt in answerer.prompts:
        for marker in ("[DIRECTOR GUIDANCE", "[COMMONSENSE GUIDANCE"):
            assert marker not in prompt
        if leaked_answer.lower() in prompt.lower() and "conclusion" not in question.question.lower():
            # the value may legitimately occur inside the retrieved dialogue, but
            # never as a *guidance* line; assert it is not presented as one.
            assert f"[DIRECTOR GUIDANCE: {leaked_answer}" not in prompt
            assert f"[COMMONSENSE GUIDANCE: {leaked_answer}" not in prompt
