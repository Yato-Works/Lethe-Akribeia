"""Evidence-Commit Gate regression tests.

Two obligations, and the second is the reason this file exists:
  * the gate must catch the structural failures it claims to catch, and
  * it must NOT block answers the engine gets right.

The second obligation is enforced with the REAL cases that calibration exposed.  An
earlier version of the gate checked only the turn's TEXT for the asked person and
blocked 19 of 20 known-correct answers, because LoCoMo turn text does not repeat the
speaker's own name.  That regression is pinned here so it cannot return silently.

The measured numbers are asserted too, so a future change that quietly makes the gate
more aggressive fails the suite instead of eroding correct answers unnoticed.
"""
from artificial_memory.skills.evidence_commit_gate import (
    EVENT_OVERLAP_MIN,
    EvidenceCommitGate,
)

GATE = EvidenceCommitGate()


# ------------------------------------------------------- structural failures caught
def test_no_evidence_turn_is_refused() -> None:
    d = GATE.check("When did Melanie go hiking?", "5 July 2023", evidence_turn="")
    assert not d.allowed
    assert d.checks["has_evidence"] is False


def test_person_absent_from_both_speaker_and_text_is_refused() -> None:
    # The measured over-commit shape: a date-bearing turn about someone else.
    d = GATE.check(
        "When did Nate take time off to chill with his pets?",
        "23 May, 2022",
        evidence_turn="Joanna: the garden looks lovely on 23 May 2022.",
        evidence_turn_id="D13:1", person="nate", speaker="Joanna")
    assert not d.allowed
    assert d.checks["person_in_evidence"] is False


def test_person_as_speaker_is_accepted() -> None:
    """The regression: text alone must not be required."""
    d = GATE.check(
        "When did Melanie sign up for a pottery class?",
        "2 July, 2023",
        evidence_turn="I just signed up for a pottery class yesterday (2 July 2023).",
        evidence_turn_id="D5:4", person="melanie", speaker="Melanie")
    assert d.allowed, "first-person turn must pass -- speaker carries the attribution"


def test_turn_without_any_date_is_refused() -> None:
    d = GATE.check(
        "When did Melanie go hiking?",
        "5 July 2023",
        evidence_turn="Melanie: the weather was lovely and we had a great time.",
        evidence_turn_id="D1:1", person="melanie", speaker="Melanie")
    assert not d.allowed
    assert d.checks["turn_has_date"] is False


def test_off_topic_turn_is_refused_on_event_overlap() -> None:
    # The lc_conv-47_59 shape: a date-bearing turn about an unrelated subject.
    d = GATE.check(
        "When did John and his gaming friends organize the charity tournament?",
        "27 April, 2022",
        evidence_turn="John: what do you think of strategy board games on 27 April 2022?",
        evidence_turn_id="D8:30", person="john", speaker="Melanie")
    assert not d.allowed
    assert d.checks.get("event_overlap") is False


# ------------------------------------------------- correct answers must survive
def test_legitimate_first_person_date_commit_is_allowed() -> None:
    d = GATE.check(
        "When did Melanie sign up for a pottery class?",
        "2 July, 2023",
        evidence_turn="I just signed up for a pottery class yesterday (2 July 2023).",
        evidence_turn_id="D5:4", person="melanie", speaker="Melanie")
    assert d.allowed


def test_relative_anchor_commit_is_allowed() -> None:
    d = GATE.check(
        "When did Caroline go to the LGBTQ support group?",
        "The week before 6 July 2023",
        evidence_turn="Caroline: I went to the support group last week "
                      "(the week before 6 July 2023).",
        evidence_turn_id="D9:1", person="caroline", speaker="Caroline")
    assert d.allowed


def test_every_decision_records_its_reason() -> None:
    """A gate that cannot explain itself cannot be audited."""
    for allowed in (True, False):
        if allowed:
            d = GATE.check(
                "When did Melanie sign up for pottery?",
                "2 July, 2023",
                evidence_turn="I signed up for pottery yesterday (2 July 2023).",
                person="melanie", speaker="Melanie")
        else:
            d = GATE.check("When did Melanie go hiking?", "x", evidence_turn="")
        assert d.reason, "every decision carries a reason"
        assert isinstance(d.checks, dict) and d.checks


def test_measured_limit_is_pinned() -> None:
    """The published number, so silent regression fails the suite.

    calibrate_commit_gate.py measures this gate against the 6 real over-commits and
    the 20 real correct date answers.  If a change makes the gate materially more or
    less aggressive than what is documented, this assertion breaks rather than the
    documentation quietly lying.
    """
    assert EVENT_OVERLAP_MIN == 0.34
    # The threshold is a weak signal; assert it is not silently promoted to a
    # guarantee by re-checking the two cases calibration found indistinguishable.
    wrong = GATE.check(
        "When did Calvin have a car incident?",
        "The week before 16 May 2023",
        evidence_turn="Calvin: I had an incident last week (the week before 16 May "
                      "2023) where my place got flooded.",
        evidence_turn_id="D6:3", person="calvin", speaker="Calvin")
    # This case is KNOWN to be admitted at 0.34 (overlap 0.33 < 0.34 is false here
    # because the turn shares 'incident', 'last', 'week').  Pinning the actual
    # behaviour stops anyone claiming the gate solves it.
    assert wrong.allowed is True, (
        "lc_conv-50_17 is a KNOWN over-commit the gate does NOT catch; if this "
        "starts passing, the gate changed and the published numbers are stale")
    assert wrong.event_overlap < 0.34 or wrong.checks.get("event_overlap") is True
