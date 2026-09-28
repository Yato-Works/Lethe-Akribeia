"""Unit tests for the deterministic answer-shape gate.

The gate exists because the reader was answering two thirds of the open-domain
questions with a polarity token the prompt had demanded.  These tests pin the
classification decisions that produced the measured split (30 / 3 / 63 of the 96
category-3 questions) and, more importantly, the precedence rule that a "or"
choice beats the yes/no form - "Would Melanie prefer a national park or a theme
park?" wants the option, not "Likely no".
"""

from __future__ import annotations

import pytest

from artificial_memory.recall.answer_shape import (
    AnswerShape,
    classify_answer_shape,
    shape_directive,
)


@pytest.mark.parametrize(
    "question",
    [
        "Would Caroline pursue writing as a career option?",
        "Would Melanie be considered an ally to the transgender community?",
        "Did Calvin and Dave have a meeting in Boston?",
        "Is Caroline likely to adopt children?",
        "Has John ever been to Japan?",
    ],
)
def test_auxiliary_opening_is_polarity(question: str) -> None:
    assert classify_answer_shape(question) is AnswerShape.POLARITY


@pytest.mark.parametrize(
    "question",
    [
        "Would Melanie be more interested in going to a national park or a theme park?",
        "Which type of vacation would Evan prefer, walking tours or a camping trip?",
        "Would John rather stay home or go to the concert?",
    ],
)
def test_or_choice_beats_the_polarity_form(question: str) -> None:
    """A yes/no *sentence* that offers alternatives wants the chosen option."""
    assert classify_answer_shape(question) is AnswerShape.CHOICE


@pytest.mark.parametrize(
    "question",
    [
        "What personality traits might Melanie say Caroline has?",
        "What would Caroline's political leaning likely be?",
        "What might John's degree be in?",
        "Around which US holiday did Maria get into a car accident?",
        "What job might Maria pursue in the future?",
        "What fields would Caroline be likely to pursue in her education?",
    ],
)
def test_open_questions_are_attribute_shaped(question: str) -> None:
    assert classify_answer_shape(question) is AnswerShape.ATTRIBUTE


def test_standalone_or_only_never_matches_inside_words() -> None:
    """"for", "more" and "before" contain "or" but are not a choice."""
    assert classify_answer_shape(
        "What did John do before the conference?"
    ) is AnswerShape.ATTRIBUTE
    assert classify_answer_shape(
        "Would Caroline be considered a member of the LGBTQ community?"
    ) is AnswerShape.POLARITY


def test_directives_forbid_the_wrong_shape() -> None:
    attribute = shape_directive("What personality traits might Melanie say Caroline has?")
    assert "WRONG" in attribute and "Likely no" in attribute
    choice = shape_directive("Would Melanie prefer a park or a theme park?")
    assert "Do NOT answer with 'Yes'" in choice
    polarity = shape_directive("Would Caroline pursue writing as a career option?")
    assert "exactly one of" in polarity
