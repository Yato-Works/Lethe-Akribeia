"""An explicit in-text date beats a relative phrase resolved from the header.

REJECTED EXPERIMENT -- these tests are xfail, not deleted.

The branch they describe was implemented, measured, and REJECTED: FIXED 3 / BROKE 33
/ net -30 on 987 frozen questions under identical contexts and scorer (see
design_docs/re_engineering_sprint/experiments/REJECTED_explicit_date_branch.md).

The diagnosis overturned the hypothesis these tests encode.  They assumed the engine
was missing the capability -- that 'yesterday (24 August 2023)' was being read as the
header date.  In fact the engine ALREADY answered lc_conv-26_58 correctly, and the
branch shadowed a working code path and then failed to replace it: all 33 BROKE cases
have the new engine ABSTAINING, none produces a wrong date.

So these are not specs for a missing feature.  They are the record of a rejected
change, kept xfail (non-strict) so that if a future fix genuinely adds the capability
the tests turn XPASS and the rejection gets re-examined rather than silently rotting.

A correct re-implementation must additionally: tie the in-text date to the ASKED
EVENT, and not fire at all when the question's gold is not a date.  The rejected
branch had neither guard, which is why 320 of its 357 changed answers were wrong->wrong.
"""
import pytest

from artificial_memory_gen.skills.deterministic_memory_engine import DeterministicMemoryEngine


@pytest.mark.xfail(
    strict=False,
    reason="REJECTED experiment: the engine already answered these correctly; the "
           "branch that added this precedence netted -30 (FIXED 3 / BROKE 33). "
           "See experiments/REJECTED_explicit_date_branch.md.",
)
def test_explicit_parenthesised_date_beats_yesterday_from_header() -> None:
    # Fixture mirrors the real corpus turn D14:4, where the question's event
    # terms ("plate", "pottery", "class") must all appear in the window or
    # _date_event_matches filters the turn out before any date branch runs.
    context = (
        "[D14:4 on 1:33 pm on 25 August, 2023] Melanie: Yeah, I made that "
        "plate in pottery class yesterday (24 August 2023). I love it.\n"
    )
    answer = DeterministicMemoryEngine.resolve(
        "When did Melanie make a plate in pottery class?", context)
    assert answer.used
    # The engine's date format carries a comma before the year.
    assert "24 August, 2023" in answer.answer


@pytest.mark.xfail(
    strict=False,
    reason="REJECTED experiment -- see the module docstring and "
           "experiments/REJECTED_explicit_date_branch.md.",
)
def test_explicit_inline_date_beats_last_weekday_from_header() -> None:
    # Fixture mirrors corpus turn D16:6.  The plural 'networking events' in the
    # question does not lemma-match a singular 'networking event' in the turn, so
    # the fixture uses the plural to pass _date_event_matches -- a pre-existing
    # lemma limitation, unrelated to this fix.
    context = (
        "[D16:6 on 2:15 pm on 21 June, 2023] Jon: That's awesome, Gina! "
        "Yesterday (20 June 2023) I visited networking events for my store.\n"
    )
    answer = DeterministicMemoryEngine.resolve(
        "When did Jon visit networking events for his store?", context)
    assert answer.used
    assert "20 June, 2023" in answer.answer


def test_relative_resolution_still_works_without_an_explicit_date() -> None:
    """The fix must not disable the branch it precedes."""
    context = (
        "[D1:2 on 4:00 pm on 15 July, 2023] Caroline: I went to the LGBTQ "
        "support group last Friday and it was wonderful.\n"
    )
    answer = DeterministicMemoryEngine.resolve(
        "When did Caroline go to the LGBTQ support group?", context)
    assert answer.used
    assert "Friday" in answer.answer


def test_bare_relative_date_still_resolves_from_header() -> None:
    context = (
        "[D2:1 on 9:00 am on 16 August, 2023] Melanie: I made the plate "
        "yesterday in pottery class.\n"
    )
    answer = DeterministicMemoryEngine.resolve(
        "When did Melanie make a plate in pottery class?", context)
    assert answer.used
    assert "15 August, 2023" in answer.answer
