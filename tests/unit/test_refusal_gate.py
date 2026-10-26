"""Answer-provenance refusal gate (synthetic corpus, no benchmark vocabulary).

The gate refuses an answer whose own wording only ever appears in another
speaker's turn.  These tests pin the four boundaries that matter, because each
one was a measured failure mode in the question-keyed gates that preceded it:

  - an answer that DOES come from the asked person is never refused,
  - a date-shaped answer is never refused (a date legitimately appears in both
    speakers' turns; enforcing provenance there cost 7.8pt on temporal),
  - a question naming nobody, or naming both speakers, is never refused,
  - an answer whose wording is not in the context at all is never refused.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.skills.refusal_gate import (  # noqa: E402
    AnswerProvenanceGate,
    is_temporal_answer,
)

CONTEXT = "\n".join([
    "[SPEAKER ATTRIBUTION & OWNERSHIP CERTIFICATE]",
    "- Verified Actor: alice",
    "[/SPEAKER ATTRIBUTION & OWNERSHIP CERTIFICATE]",
    "",
    "[session_1 on 1 May 2023] Alice: I ran the charity race last Saturday.",
    "[session_1 on 1 May 2023] Bob: I realized that self care is really important.",
    "[session_2 on 2 May 2023] Alice: That sounds great, Bob.",
])

GATE = AnswerProvenanceGate()


class TestProvenance:
    def test_answer_from_the_other_speaker_is_refused(self):
        decision = GATE.decide("What did Alice realize?", CONTEXT,
                               "self care is really important")
        assert decision.refuse is True
        assert decision.source_speaker == "Bob"
        assert decision.subject == "Alice"

    def test_answer_from_the_asked_person_is_kept(self):
        decision = GATE.decide("What did Alice do?", CONTEXT, "ran the charity race")
        assert decision.refuse is False

    def test_reason_names_the_provenance(self):
        decision = GATE.decide("What did Alice realize?", CONTEXT,
                               "self care is really important")
        assert "Bob" in decision.reason and "Alice" in decision.reason


class TestTemporalAnswersAreOutOfScope:
    def test_date_answer_is_never_refused(self):
        decision = GATE.decide("When did Alice run?", CONTEXT, "7 May 2023")
        assert decision.refuse is False

    def test_relative_date_answer_is_never_refused(self):
        decision = GATE.decide("When did Alice run?", CONTEXT,
                               "the Saturday before 1 May 2023")
        assert decision.refuse is False

    def test_temporal_detection(self):
        assert is_temporal_answer("2023") is True
        assert is_temporal_answer("last friday") is True
        assert is_temporal_answer("pottery") is False


class TestSafeByDefault:
    def test_question_naming_nobody_is_kept(self):
        assert GATE.decide("What happened at the race?", CONTEXT,
                           "self care is really important").refuse is False

    def test_question_naming_both_speakers_is_kept(self):
        assert GATE.decide("What did Alice and Bob discuss?", CONTEXT,
                           "self care is really important").refuse is False

    def test_untraceable_answer_is_kept(self):
        assert GATE.decide("What did Alice realize?", CONTEXT,
                           "quantum entanglement").refuse is False

    def test_existing_refusal_is_kept(self):
        assert GATE.decide("What did Alice realize?", CONTEXT, "I don't know.").refuse is False

    def test_empty_answer_is_kept(self):
        assert GATE.decide("What did Alice realize?", CONTEXT, "").refuse is False

    def test_empty_context_is_kept(self):
        assert GATE.decide("What did Alice realize?", "", "self care is really important").refuse is False


ROLE_CONTEXT = "\n".join([
    "[s1 on 2023/05/30] user: I keep a bag of coffee beans in the second drawer.",
    "[s1 on 2023/05/30] assistant: You keep a bag of coffee beans in the second drawer.",
])


class TestRoleModeIsOptIn:
    """Agent-memory corpora label speakers by role, not name.

    Role mode exists and is measured, but it is NOT the default: on LongMemEval it
    refuses 2 correct answers (-1.56pt) because the assistant legitimately restates
    the user's facts, so "the wording comes from the other role" is not evidence of
    misattribution there.
    """

    def test_default_ignores_role_labelled_contexts(self):
        assert GATE.decide("Where do I keep the coffee beans?", ROLE_CONTEXT,
                           "the second drawer").refuse is False

    def test_role_mode_attributes_a_first_person_question_to_the_user(self):
        gate = AnswerProvenanceGate(allow_roles=True)
        decision = gate.decide("Where do I keep the coffee beans?", ROLE_CONTEXT,
                               "coffee beans in the second drawer")
        # The wording exists in the user's own turn, so it is kept.
        assert decision.refuse is False

    def test_role_mode_refuses_an_answer_only_the_assistant_states(self):
        context = "\n".join([
            "[s1 on 2023/05/30] user: Hello again.",
            "[s1 on 2023/05/30] assistant: Your favourite colour is turquoise.",
        ])
        gate = AnswerProvenanceGate(allow_roles=True)
        assert gate.decide("What is my favourite colour?", context,
                           "favourite colour is turquoise").refuse is True


class TestThresholdIsHonoured:
    """The threshold must be a real parameter.

    Measured defect: a sweep of `--thresholds` produced five identical rows because
    the gate read the class constant and ignored the instance setting.
    """

    #: The best-matching line carries 3 of the 4 answer terms (0.75).
    CONTEXT = "\n".join([
        "[s1 on 1 May 2023] Alice: I said hello.",
        "[s1 on 1 May 2023] Bob: painting pottery evenings.",
    ])
    ANSWER = "painting pottery evenings together"

    def test_low_threshold_fires_high_threshold_does_not(self):
        low = AnswerProvenanceGate(threshold=0.5).decide(
            "What did Alice do?", self.CONTEXT, self.ANSWER)
        high = AnswerProvenanceGate(threshold=0.9).decide(
            "What did Alice do?", self.CONTEXT, self.ANSWER)
        assert low.refuse is True
        assert high.refuse is False

    def test_default_threshold_is_the_documented_value(self):
        assert AnswerProvenanceGate().threshold == AnswerProvenanceGate.THRESHOLD
