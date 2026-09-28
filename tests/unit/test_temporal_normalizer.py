"""Unit tests for TemporalNormalizer's relative-date expansion.

These cover the three rules that were missing entirely (`this week`,
`next week`, `last <season>`) and the documented output shape: the original
expression is kept and the resolved date is appended in parentheses, so the
reader can copy a fully-qualified answer without doing arithmetic.
"""

from artificial_memory.context.temporal_normalizer import TemporalNormalizer

REF = "1:51 pm on 15 July, 2023"


def make() -> TemporalNormalizer:
    return TemporalNormalizer()


def test_this_week_resolves_to_week_of_reference():
    out = make().normalize("I joined a mentorship program this week.", REF)
    assert "this week (the week of 15 July 2023)" in out


def test_next_week_resolves_to_week_after_reference():
    out = make().normalize("The show opens next week.", REF)
    assert "next week (the week after 15 July 2023)" in out


def test_last_season_resolves_to_previous_year():
    out = make().normalize("We went last summer and it was great.", REF)
    assert "last summer (the summer of 2022)" in out


def test_last_season_is_case_insensitive_and_leaves_other_text():
    out = make().normalize("Last Winter was cold, but spring is here.", REF)
    assert "last Winter (the Winter of 2022)" in out
    assert "spring is here" in out


def test_existing_weekday_rule_still_expands():
    out = make().normalize("Last Fri we took the kids out.", REF)
    assert "last Friday (the Friday before 15 July 2023)" in out




def test_unparseable_reference_returns_text_untouched():
    out = make().normalize("I went last week.", "")
    assert out == "I went last week."


# ---------------------------------------------------------------- new rules
# Rules 17/18 + the season rewrite: the reader cannot do calendar arithmetic, so
# the resolved value must exist in the context in the form the official scorer
# expects (LoCoMo answers year questions with the year, not with "10 years ago").

def test_years_ago_resolves_to_the_year():
    out = make().normalize("John got his dog Max 10 years ago.", REF)
    assert "10 years ago (2013)" in out


def test_years_ago_accepts_spelled_numbers():
    out = make().normalize("We moved three years ago.", REF)
    assert "three years ago (2020)" in out


def test_months_ago_resolves_to_month_and_year():
    out = make().normalize("She moved 3 months ago.", REF)
    assert "3 months ago (April 2023)" in out


def test_last_season_year_depends_on_the_reference_date():
    """A 'last <season>' phrase means the most recent season that has *ended*.

    From 15 July 2023, summer 2023 is still running so "last summer" is 2022,
    but from December the same summer has ended and it becomes 2023.  The v1 rule
    wrapped into the previous year unconditionally, which was wrong for every
    reference date inside or after the season.
    """
    n = make()
    assert "last summer (the summer of 2022)" in n.normalize("We went last summer.", REF)
    assert "last summer (the summer of 2023)" in n.normalize(
        "We went last summer.", "15 December 2023"
    )
    # winter spans a year boundary, so it is labelled by the year it started in
    assert "last winter (the winter of 2022)" in n.normalize("It was last winter.", REF)
    assert "last winter (the winter of 2021)" in n.normalize(
        "It was last winter.", "20 January 2023"
    )


def test_year_ago_rule_can_be_selected_per_rule():
    """The new rules obey ``enabled_rules`` like every other rule."""
    n = make()
    text = "It happened 5 years ago and 2 months ago."
    assert n.normalize(text, REF, enabled_rules={"days_ago"}) == text
    assert "5 years ago (2018)" in n.normalize(text, REF, enabled_rules={"years_ago"})
    assert "2 months ago (May 2023)" in n.normalize(text, REF, enabled_rules={"months_ago"})

