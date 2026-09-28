"""Unit tests for the oracle false-negative audit (Step 3a.5).

The audit exists because the retrieval oracle is an *id* test over a context whose
turns the compiler numbers in its own order, so a question can be flagged "no
evidence" while the gold turn is sitting in the context under a different label.
These tests pin the two rules that decide a verdict - the evidence is located by
content and date, and the flag is what gets called wrong, not the reader.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "benchmarks" / "oracle_fn_audit.py"


def load_script():
    spec = importlib.util.spec_from_file_location("_oracle_fn_audit_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


of = load_script()

TURNS = [
    {"id": "D1:3", "date": "1:56 pm on 8 May, 2023",
     "text": "Melanie: I visited the botanical garden last Saturday with Caroline."},
    {"id": "D1:4", "date": "1:56 pm on 8 May, 2023",
     "text": "Caroline: The garden was closed, we ended up at the cafe instead."},
]
GOLD = "8 May 2023"


def test_evidence_is_found_by_content_even_when_the_id_is_absent() -> None:
    """The label is not the test: the turn's words and its session date are."""
    context = ("[D2:7 on 1:56 pm on 8 May, 2023] Melanie: I visited the botanical "
               "garden last Saturday with Caroline. Anything else?")
    present, matched = of.evidence_present(context, TURNS, ["D1:3"])
    assert present is True and matched == "D1:3"


def test_evidence_from_another_session_does_not_count() -> None:
    """Same words, wrong date: a coincidence elsewhere in the conversation is not
    the gold evidence turn."""
    context = "[D9:1 on 3:00 pm on 2 February, 2024] Melanie: I visited the botanical garden last Saturday."
    assert of.evidence_present(context, TURNS, ["D1:3"])[0] is False


def test_a_disputed_question_is_called_wrong_when_the_evidence_is_present() -> None:
    context = ("[D7:2 on 1:56 pm on 8 May, 2023] Melanie: I visited the botanical "
               "garden last Saturday with Caroline.")
    record = of.audit_one(
        "conv-1-qa-000",
        {"a": {"prediction": "last Saturday"}, "b": {"prediction": "8 May 2023"}},
        {"question": "when?", "answer": GOLD, "evidence_ids": ["D1:3"],
         "category": 2, "turns": TURNS},
        context)
    assert record["verdict"] == "ORACLE_FALSE_NEGATIVE"
    assert record["evidence_text_present"] is True
    assert record["evidence_matched_id"] == "D1:3"


def test_a_reader_that_answers_without_the_evidence_is_a_true_retrieval_failure() -> None:
    """The other reader getting it right does not make the evidence present."""
    record = of.audit_one(
        "conv-1-qa-001",
        {"a": {"prediction": "I don't know"}, "b": {"prediction": "8 May 2023"}},
        {"question": "when?", "answer": GOLD, "evidence_ids": ["D1:3"],
         "category": 2, "turns": TURNS},
        "[D3:1 on 4 July, 2023] Caroline: Let's meet next week.")
    assert record["verdict"] == "TRUE_RETRIEVAL_FAILURE"
    assert record["evidence_text_present"] is False


def test_the_audit_only_picks_questions_the_flag_and_the_reader_disagree_on() -> None:
    rows = [
        {"question_id": "q1", "is_correct": False, "oracle_recall": False, "answer_source": "reader"},
        {"question_id": "q2", "is_correct": False, "oracle_recall": True, "answer_source": "reader"},
        {"question_id": "q3", "is_correct": True, "oracle_recall": False, "answer_source": "reader"},
        {"question_id": "q4", "is_correct": False, "oracle_recall": False, "answer_source": "committed: t"},
    ]
    other = [{**row, "is_correct": True} for row in rows]
    tmp = Path(tempfile.mkdtemp(prefix="ofn-"))
    a_path, b_path = tmp / "a.json", tmp / "b.json"
    a_path.write_text(json.dumps({"model": "big", "results": rows}), encoding="utf-8")
    b_path.write_text(json.dumps({"model": "small", "results": other}), encoding="utf-8")
    disputed = of.load_runs(str(a_path), str(b_path))
    assert set(disputed) == {"q1"}
