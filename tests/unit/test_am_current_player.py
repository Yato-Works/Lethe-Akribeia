"""Unit tests for the AM current player (R7 chain behind the arena interface).

Deterministic unit mode (no LLM): answers are the frozen abstention text.
Checks cover protocol compliance: the 2,000-token arena contract, honest
cost records (0 write LLM, one read per answer), determinism and metadata
provenance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from artificial_memory.research.benchmarks.arena import build_dataset
from artificial_memory.research.benchmarks.llm import ABSTENTION_TEXT

# The player is measurement machinery in scripts/, not shipped package code.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from am_current_player import (  # noqa: E402
    AMCurrentPlayer,
    compile_r7_context,
)


@pytest.fixture(scope="module")
def dataset():
    return build_dataset()


@pytest.fixture(scope="module")
def player(dataset):
    p = AMCurrentPlayer({"context_budget": 2000}, answerer=None)
    p.ingest(list(dataset.scenarios))
    return p


def test_ingest_is_write_llm_free(player):
    assert player._all_turns, "the shared history must become pipeline turns"
    assert player._fact_index is not None
    costs = player.finalize()
    assert costs.write_llm_calls == 0, "the R7 chain is deterministic up to the reader"
    assert costs.write_latency_ms >= 0


def test_answer_honours_arena_budget(player, dataset):
    question = dataset.questions[0]
    answer = player._answer_with_budget(question, player.context_budget)
    assert answer.player_name == "AM current"
    assert answer.context_tokens_used <= 2000, "the arena contract is absolute"
    assert answer.text == ABSTENTION_TEXT  # unit mode: no answerer


def test_budget_enforcement_intercepts_overflow(player, dataset):
    question = dataset.questions[0]
    player.reset_costs()
    player.answer(question, player.context_budget)
    # ControlledPlayer.answer recorded a budget violation whenever the
    # compiled context exceeded the shared contract -- it must never happen.
    assert player.costs.budget_violations == 0, (
        "compile_r7_context must deliver contexts within the shared budget")


def test_metadata_carries_provenance(player, dataset):
    question = dataset.questions[1]
    answer = player._answer_with_budget(question, player.context_budget)
    assert answer.metadata["plan_mode"]
    assert answer.metadata["token_budget"] > 0
    assert answer.metadata["web_ctx_tokens"] <= 2000


def test_read_costs_recorded(player, dataset):
    player.reset_costs()
    player._answer_with_budget(dataset.questions[0], player.context_budget)
    costs = player.finalize()
    assert costs.retrieval_calls >= 1
    assert costs.answer_llm_calls >= 1  # unit fallback counts as the read call site
    assert costs.context_tokens_used <= 2000


def test_chain_is_deterministic(player, dataset):
    question = dataset.by_category(dataset.questions[0].category)[0]
    p = player._pipeline
    ctx_a, meta_a = compile_r7_context(
        p, player._all_turns, question.question, fact_index=player._fact_index)
    ctx_b, meta_b = compile_r7_context(
        p, player._all_turns, question.question, fact_index=player._fact_index)
    assert ctx_a == ctx_b, "the R7 chain must be deterministic"
    assert meta_a["token_budget"] == meta_b["token_budget"]


def test_scoring_plan_modes_dont_hit_missing_scenarios(player, dataset):
    # A question per category must compile without raising (coverage of the
    # planner's distinct modes).
    compiled = 0
    for question in dataset.questions[:10]:
        answer = player._answer_with_budget(question, player.context_budget)
        assert answer.context_tokens_used <= 2000
        compiled += 1
    assert compiled == 10


def test_hybrid_player_fills_remainder_with_lexical_lines(dataset):
    from am_current_player import AMHybridPlayer

    hybrid = AMHybridPlayer({"context_budget": 2000}, answerer=None)
    hybrid.ingest(list(dataset.scenarios))
    question = dataset.questions[0]
    answer = hybrid._answer_with_budget(question, hybrid.context_budget)
    assert hybrid.name == "AM hybrid"
    assert answer.metadata["lexical_lines"] > 0, "the remainder must be filled"
    assert answer.context_tokens_used <= 2000
    assert answer.metadata["hybrid_tokens"] == answer.context_tokens_used
    # Other players' context token records all use the shared estimator.
    assert answer.metadata["hybrid_tokens"] == answer.context_tokens_used


def test_hybrid_budget_a_respected(dataset):
    from am_current_player import AMHybridPlayer, compile_r7_context

    hybrid = AMHybridPlayer({"context_budget": 2000}, answerer=None)
    hybrid.ingest(list(dataset.scenarios))
    question = dataset.questions[0]
    ctx, meta = compile_r7_context(
        hybrid._pipeline, hybrid._all_turns, question.question,
        arena_budget=hybrid.hybrid_budget_a, fact_index=hybrid._fact_index)
    assert len(ctx) // 4 <= hybrid.hybrid_budget_a
    full_answer = hybrid._answer_with_budget(question, hybrid.context_budget)
    assert full_answer.context_tokens_used > hybrid.hybrid_budget_a, (
        "the lexical fill must add to the reduced compile budget")


def test_hybrid_runtime_fill_uses_the_lexical_stack(dataset):
    from am_current_player import AMHybridPlayer

    hybrid = AMHybridPlayer({"context_budget": 2000, "hybrid_fill": "runtime"},
                            answerer=None)
    hybrid.ingest(list(dataset.scenarios))
    assert hybrid._lex_runtime is not None, "ingest must build the lexical runtime"
    question = dataset.questions[0]
    answer = hybrid._answer_with_budget(question, hybrid.context_budget)
    assert answer.metadata["fill_source"] == "runtime"
    assert answer.metadata["lexical_lines"] > 0
    assert answer.context_tokens_used <= 2000
    assert answer.metadata["hybrid_tokens"] == answer.context_tokens_used


def test_hybrid_raw_fill_still_default_name(dataset):
    from am_current_player import AMHybridPlayer

    raw = AMHybridPlayer({"context_budget": 2000}, answerer=None)
    raw.ingest(list(dataset.scenarios))
    assert raw._lex_runtime is None, "raw fill must not build the runtime"
    answer = raw._answer_with_budget(dataset.questions[2], raw.context_budget)
    assert answer.metadata["fill_source"] == "raw"
    assert answer.context_tokens_used <= 2000


def test_conflict_detector_captures_the_planted_pairs(dataset):
    from am_current_player import detect_conflicts

    conflicts = detect_conflicts(_player(dataset)._all_turns)
    keys = [k for k in conflicts if k[0] != "(none)"]
    assert len(keys) == 20, "exactly the twenty planted contradiction keys"
    planted = 0
    for q in dataset.questions:
        if q.category.value != "contradiction_handling":
            continue
        qlow = q.question.lower()
        vals = {str(g) for g in q.ground_truth[0]}
        have = {v for k in keys if k[0] in qlow
                for v, _ in conflicts[k]}
        if vals and vals.issubset(have):
            planted += 1
    assert planted == 20, "both planted values must sit under one key"


def _player(dataset):
    from am_current_player import AMCurrentPlayer

    p = AMCurrentPlayer({"context_budget": 2000}, answerer=None)
    p.ingest(list(dataset.scenarios))
    return p


def test_conflict_annotation_fires_only_on_contradiction(dataset):
    from am_current_player import AMConflictPlayer, conflict_annotations, detect_conflicts

    conflicts = detect_conflicts(_player(dataset)._all_turns)
    by_cat: dict[str, int] = {}
    for q in dataset.questions:
        fired = len(conflict_annotations(q.question, conflicts))
        by_cat[q.category.value] = by_cat.get(q.category.value, 0) + fired
    assert by_cat.get("contradiction_handling") == 20
    other = {c: n for c, n in by_cat.items() if c != "contradiction_handling"}
    assert not any(n for n in other.values()), f"no cross-category fire: {other}"


def test_conflict_player_annotates_within_budget(dataset):
    from am_current_player import AMConflictPlayer

    cp = AMConflictPlayer({"context_budget": 2000}, answerer=None)
    assert cp.name == "AM conflict"
    cp.ingest(list(dataset.scenarios))
    q = next(q for q in dataset.questions
             if q.category.value == "contradiction_handling")
    answer = cp._answer_with_budget(q, cp.context_budget)
    assert answer.metadata.get("conflict_lines") == 1
    assert answer.context_tokens_used <= 2000
    # the bare hybrid must stay inert: no conflict key was read for it
    from am_current_player import AMHybridPlayer

    hyb = AMHybridPlayer({"context_budget": 2000}, answerer=None)
    hyb.ingest(list(dataset.scenarios))
    bare = hyb._answer_with_budget(q, hyb.context_budget)
    assert "conflict_lines" not in bare.metadata


# ---- M14 Phase 1: freeze the M13 result -----------------------------------
# The M13 mechanism is scored against ground_truth group 2, whose vocabulary
# (CONFLICT_MARKERS in the scorer-side arena module) must never enter player
# code.  These tests make that boundary mechanical, so a later edit that
# imports the scorer's marker list makes the suite fail loudly.


def test_conflict_player_module_never_imports_scorer_vocabulary():
    import ast as _ast
    from pathlib import Path as _Path

    module = _Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" \
        / "am_current_player.py"
    tree = _ast.parse(module.read_text(encoding="utf-8"))
    root = _ast.parse(
        (_Path(__file__).resolve().parents[2] / "src"
         / "artificial_memory" / "research" / "benchmarks" / "arena.py")
        .read_text(encoding="utf-8"))
    scorer_names = {
        node.targets[0].id for node in root.body
        if isinstance(node, _ast.Assign)
        and isinstance(node.targets[0], _ast.Name)
    }
    imported = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ImportFrom) and node.module and \
                "arena" in node.module:
            imported |= {a.name for a in node.names}
        elif isinstance(node, _ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
    leaked = imported & scorer_names
    assert not leaked, (
        f"player must not import the scorer's vocabulary: {leaked}")


def test_annotation_text_conveys_both_on_record_and_neither_confirmed(dataset):
    from am_current_player import AMConflictPlayer, conflict_annotations, detect_conflicts

    conflicts = detect_conflicts(_player(dataset)._all_turns)
    q = next(q for q in dataset.questions
             if q.category.value == "contradiction_handling")
    (note,) = conflict_annotations(q.question, conflicts)
    assert "conflicting reports" in note          # surfaced, not hidden
    assert "report both values" in note           # plan-21 semantics
    assert "(day " in note, "the M13-measured provenance surface must hold"
    # the reader must not be steered into the iter-1 refusal phrasing
    assert "which is correct is unverified" not in note
