"""Regression tests for the attribute SHAPE gate.

Two obligations:
  * it must refuse the 14 measured fragment/verb-phrase values, and
  * it must NOT refuse a value that contains the right answer.

The second obligation is pinned because the first version of this gate broke it.  It
listed 'thing'/'stuff' as activity words, so it refused
'favourite thing is Hawaiian pizza...' -- a value that CONTAINED the gold.  The
refusal handed the question to a worse branch and turned a correct answer into a
wrong one.  Measured as 3 BROKE.  A gate that refuses a value containing the right
answer is not a gate, and these tests exist so that cannot come back quietly.
"""
from artificial_memory_gen.skills.autonomous_engines.entity_attribute_resolver import (
    EntityAttributeResolver as R,
)


# ------------------------------------------------- the measured failures, refused
def test_deictic_fragments_are_refused() -> None:
    for v in ("this series", "the smell", "how's", "pizza too",
              "also cheese pizza", "the same way with my team"):
        assert R._answer_is_a_noun_phrase(v) is False, v


def test_activity_words_are_refused() -> None:
    for v in ("cooking", "running my own studio", "getting immersed in the feeling",
              "creating a serene space with", "switch"):
        assert R._answer_is_a_noun_phrase(v) is False, v


def test_category_word_is_refused_for_a_concrete_head() -> None:
    """'action' is not a movie; 'fantasy' is not a video game."""
    assert R._answer_is_a_noun_phrase("action", head="movie") is False
    assert R._answer_is_a_noun_phrase("fantasy", head="game") is False
    assert R._answer_is_a_noun_phrase("action", head="trilogy") is False


def test_category_word_is_ALLOWED_for_a_category_head() -> None:
    """The same words are legitimate when the question asks for a genre."""
    assert R._answer_is_a_noun_phrase("action", head="genre") is True
    assert R._answer_is_a_noun_phrase("fantasy", head="style") is True


# ------------------------------- values containing the right answer must survive
def test_value_containing_the_gold_survives() -> None:
    """The regression that caused 3 BROKE.  Pinned so it cannot return."""
    for v in ("recipe is Chicken Pot Pie. It's",
              "thing is Hawaiian pizza. This combination of sweet",
              "dishes is lasagna! Comfort food can be a"):
        assert R._answer_is_a_noun_phrase(v) is True, v


def test_real_titles_survive() -> None:
    for v, head in (("Lord of the Rings", "trilogy"),
                    ("Xenoblade Chronicles", "game"),
                    ("Chicken Pot Pie", "recipe"),
                    ("Monster Hunter: World", "game"),
                    ("It takes two", "game")):
        assert R._answer_is_a_noun_phrase(v, head=head) is True, v


def test_plain_noun_survives() -> None:
    assert R._answer_is_a_noun_phrase("lasagna", head="dishes") is True
    assert R._answer_is_a_noun_phrase("Photography", head="hobby") is True


def test_empty_is_refused() -> None:
    assert R._answer_is_a_noun_phrase("") is False


def test_head_noun_extraction() -> None:
    assert R._question_head_noun("What is Nate's favorite movie trilogy?") == "trilogy"
    assert R._question_head_noun("What is one of Jolene's favorite dishes?") == "dishes"
    assert R._question_head_noun("What hobby did Calvin take up recently?") is not None
