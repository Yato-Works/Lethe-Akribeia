"""Subject-admission policy in the budgeted compiler (opt-in, synthetic corpus).

Two defects are pinned here, both measured on LoCoMo list questions before the
fix:

1. `_bundle_names_subject` checked only the turn TEXT.  A first-person turn
   ("I hiked to the forest") does not contain its own speaker's name, so the
   enumeration escape hatch was blind to exactly the turns it existed for.  The
   SPEAKER label now counts.
2. Admission was checked by the first gate only.  The very next gate
   ("no new query words, no anchor score -> skip") dropped the same bundle, so
   the escape hatch was dead code: enabling it moved multi-hop list item recall
   by exactly zero.

The policy is OFF by default; these tests also assert that the default path is
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory_gen.compiler.budgeted_compiler import (  # noqa: E402
    BudgetedCognitiveCompiler,
)
from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder  # noqa: E402


def _compile(question: str, turns, admit=None, budget=500):
    graph = GraphBuilder().build(turns, [])
    bundles = graph.form_bundles([], max_bundle_tokens=150)
    compiler = BudgetedCognitiveCompiler()
    return compiler.compile(
        question=question,
        bundles=bundles,
        token_budget=budget,
        anchor_scores={},
        admit_subject_names=admit,
    )


#: The asked person's turn shares no vocabulary with the question.
TURNS = [
    ("s1", "1 May 2023", "Alice", "We hiked for hours and the forest was beautiful."),
    ("s2", "2 May 2023", "Bob", "The weather was lovely that day."),
]
QUESTION = "Where has Alice camped?"


class TestSpeakerIsTreatedAsSubject:
    def test_first_person_turn_carries_no_name_in_its_text(self):
        assert "alice" not in TURNS[0][3].lower()

    def test_bundle_naming_subject_uses_the_speaker_label(self):
        graph = GraphBuilder().build(TURNS, [])
        bundles = graph.form_bundles([], max_bundle_tokens=150)
        compiler = BudgetedCognitiveCompiler()
        alice = [b for b in bundles
                 if any(n.speaker.lower() == "alice" for n in b.member_nodes)]
        assert alice, "expected a bundle containing Alice's turn"
        assert compiler._bundle_names_subject(alice[0], {"alice"}) is True

    def test_unrelated_subject_is_not_matched(self):
        graph = GraphBuilder().build(TURNS, [])
        bundles = graph.form_bundles([], max_bundle_tokens=150)
        compiler = BudgetedCognitiveCompiler()
        bob = [b for b in bundles
               if any(n.speaker.lower() == "bob" for n in b.member_nodes)]
        assert compiler._bundle_names_subject(bob[0], {"alice"}) is False


class TestAdmissionSurvivesBothGates:
    def test_default_path_does_not_admit(self):
        result = _compile(QUESTION, TURNS)
        assert "forest" not in result.context_text.lower()
        assert result.total_tokens == 0

    def test_admission_includes_the_subject_turn(self):
        result = _compile(QUESTION, TURNS, admit={"Alice"})
        assert "forest" in result.context_text.lower()
        assert result.total_tokens > 0

    def test_admission_does_not_pull_the_other_speaker(self):
        result = _compile(QUESTION, TURNS, admit={"Alice"})
        assert "weather" not in result.context_text.lower()

    def test_admission_is_case_insensitive(self):
        result = _compile(QUESTION, TURNS, admit={"alice"})
        assert "forest" in result.context_text.lower()
