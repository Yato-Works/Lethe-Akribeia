"""The retrieval oracle is a content test, not a label test.

The old implementation asked whether the gold evidence *id* appeared in the
compiled context.  The context compiler numbers the turns it emits in its own
group order, so those ids are not the dataset's ids: across LoCoMo 1,540 the flag
was wrong on 22% of questions in both directions while the aggregate barely moved.
These tests pin the replacement, and the one property that makes it trustworthy -
it must not be fooled by the same words appearing in a different session.
"""

from artificial_memory.research.benchmarks.external.locomo_adapter import (
    LoCoMoTurn,
    _evidence_present,
)


def turn(dia_id: str, text: str, date: str) -> LoCoMoTurn:
    return LoCoMoTurn(dia_id=dia_id, speaker="Caroline", text=text, session_num=1,
                      session_date=date, raw_content=text)


def test_evidence_is_found_by_content_when_the_id_label_is_absent() -> None:
    """The turn is in the context under a different label - which is the normal case."""
    context = ("[D2:7 on 1:56 pm on 8 May, 2023] Caroline: I visited the botanical "
               "garden last Saturday with Melanie.")
    present, by_id = _evidence_present(
        context, [turn("D1:3", "I visited the botanical garden last Saturday with Melanie.",
                       "1:56 pm on 8 May, 2023")], ["D1:3"])
    assert present is True
    assert by_id is False  # the label test would have said "no evidence"


def test_the_same_words_in_another_session_do_not_count() -> None:
    """A coincidence elsewhere in the conversation is not the gold evidence turn."""
    context = "[D9:1 on 3:00 pm on 2 February, 2024] Caroline: I visited the botanical garden."
    present, _ = _evidence_present(
        context, [turn("D1:3", "I visited the botanical garden last Saturday.",
                       "1:56 pm on 8 May, 2023")], ["D1:3"])
    assert present is False


def test_condensed_text_still_counts() -> None:
    """The compiler rewrites and truncates unit bodies, so the test is word overlap:
    a turn whose tail the compiler dropped (here one word in five) still counts."""
    context = "[D2:7 on 1:56 pm on 8 May, 2023] Caroline: I visited the botanical garden."
    present, _ = _evidence_present(
        context,
        [turn("D1:3", "I visited the botanical garden yesterday", "1:56 pm on 8 May, 2023")],
        ["D1:3"])
    assert present is True


def test_a_different_wording_of_the_same_fact_does_not_count_as_the_turn() -> None:
    """Overlap is deliberately not semantic: a paraphrase is not the evidence turn."""
    context = "[D2:7 on 1:56 pm on 8 May, 2023] Caroline: We went to a botanical garden."
    present, _ = _evidence_present(
        context, [turn("D1:3", "I visited the botanical garden last Saturday with Melanie.",
                       "1:56 pm on 8 May, 2023")], ["D1:3"])
    assert present is False


def test_a_question_without_evidence_ids_is_not_a_retrieval_failure() -> None:
    """The caller handles that case; here an unrelated context must not match."""
    present, _ = _evidence_present("[D1:1 on 1 May, 2023] Caroline: Hi.", [], [])
    assert present is False
