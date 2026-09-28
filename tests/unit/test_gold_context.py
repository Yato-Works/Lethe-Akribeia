"""Unit tests for the gold-context ceiling experiment.

The experiment is only worth running if its input is exactly the set the failure
ceiling called *unresolved* - wrong for the reference reader, not committed, and
the evidence demonstrably present.  These tests pin that predicate, the gold
context rendering, and the four-way split, so a drift in any of them shows up
here rather than as a plausible-looking ceiling.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GOLD = REPO / "scripts" / "benchmarks" / "gold_context_cache.py"
CEILING = REPO / "scripts" / "benchmarks" / "gold_context_ceiling.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gc = load("_gold_context_under_test", GOLD)
ce = load("_gold_ceiling_under_test", CEILING)

TURNS = [
    {"id": "D1:1", "date": "1:56 pm on 8 May, 2023", "text": "Caroline: Hi Melanie."},
    {"id": "D1:3", "date": "1:56 pm on 8 May, 2023",
     "text": "Melanie: I visited the botanical garden last Saturday."},
    {"id": "D2:0", "date": "1:14 pm on 25 May, 2023", "text": "Caroline: See you soon."},
]


def test_the_gold_context_is_the_evidence_and_nothing_else() -> None:
    context = gc.gold_context({"evidence_ids": ["D1:3"], "turns": TURNS})
    assert context == "[D1:3 on 1:56 pm on 8 May, 2023] Melanie: I visited the botanical garden last Saturday."
    assert "Hi Melanie" not in context and "See you soon" not in context


def test_a_multi_evidence_question_keeps_every_evidence_turn_in_order() -> None:
    context = gc.gold_context({"evidence_ids": ["D1:3", "D2:0"], "turns": TURNS})
    assert context.index("botanical") < context.index("See you soon")


def _write_run(tmp: Path, rows: list[dict]) -> str:
    path = tmp / "run.json"
    path.write_text(json.dumps({"model": "big", "results": rows}), encoding="utf-8")
    return str(path)


def test_unresolved_means_wrong_uncommitted_and_evidence_present() -> None:
    """The three conditions the census used, spelled out so the input cannot drift."""
    tmp = Path(tempfile.mkdtemp(prefix="goldctx-"))
    rows = [
        {"question_id": "q1", "is_correct": False, "answer_source": "reader"},   # in
        {"question_id": "q2", "is_correct": True, "answer_source": "reader"},    # right
        {"question_id": "q3", "is_correct": False, "answer_source": "committed: t"},  # claimed
        {"question_id": "q4", "is_correct": False, "answer_source": "reader"},   # no evidence
    ]
    run_file = _write_run(tmp, rows)
    dataset = {
        "q1": {"evidence_ids": ["D1:3"], "turns": TURNS},
        "q2": {"evidence_ids": ["D1:3"], "turns": TURNS},
        "q3": {"evidence_ids": ["D1:3"], "turns": TURNS},
        "q4": {"evidence_ids": ["D2:0"], "turns": TURNS},
    }
    contexts = {
        "q1": "[D9:1 on 1:56 pm on 8 May, 2023] Caroline: I visited the botanical garden last Saturday.",
        "q2": "[D9:1 on 1:56 pm on 8 May, 2023] Caroline: I visited the botanical garden last Saturday.",
        "q3": "[D9:1 on 1:56 pm on 8 May, 2023] Caroline: I visited the botanical garden last Saturday.",
        "q4": "[D9:1 on 1:56 pm on 8 May, 2023] Caroline: Nothing relevant here.",
    }
    assert gc.unresolved_qids(run_file, dataset, contexts) == ["q1"]


def test_the_four_cells_are_the_four_outcomes() -> None:
    def run(model: str, verdicts: dict[str, bool]) -> dict:
        return {"model": model, "rows": {
            q: {"correct": ok, "category": 2, "ground_truth": "x", "prediction": "x"}
            for q, ok in verdicts.items()}}

    rows = ce.classify(run("big", {"q1": True, "q2": True, "q3": False, "q4": False}),
                       run("small", {"q1": True, "q2": False, "q3": True, "q4": False}))
    assert {r["qid"]: r["cell"] for r in rows} == {
        "q1": "context_limited", "q2": "reader_capability",
        "q3": "small_reader_advantage", "q4": "semantic",
    }
