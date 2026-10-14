"""Unit tests for the evaluation scorer in scripts/benchmarks/eval_locomo_full.py.

Guarantees:
1. Q47-type: Multi-item lists with permuted order are credited as correct,
   while incomplete subsets (e.g. 1 item of 4) are strictly rejected.
2. Q30-type: Hedged/explained Yes/No answers (e.g. "Likely no, she does not...")
   are correctly matched against concise "No" / "Yes", while stance-reversals
   ("Yes" vs "No") are strictly rejected.
3. Preserves existing strictness for single-entity containment and numeric equality.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from eval_locomo_full import _has_content_overlap, _month_contradiction, score


class TestListOrderEquivalence:
    """Q47: Sets with permuted element order should be credited."""

    def test_q47_exact_permuted_mentors_family_friends(self):
        gold = "Her mentors, family, and friends"
        pred = "friends, family and mentors"
        assert score(pred, gold) is True

    def test_permuted_camping_pottery_painting(self):
        gold = "pottery, camping, painting"
        pred = "camping, pottery, painting"
        assert score(pred, gold) is True

    def test_list_incomplete_subset_is_rejected(self):
        """Incomplete enumeration (e.g. 1 or 2 items out of 4) must NOT be credited."""
        gold = "pottery, camping, painting, swimming"
        assert score("painting", gold) is False
        assert score("painting, pottery", gold) is False

    def test_list_completely_wrong_is_rejected(self):
        gold = "pottery, camping, painting"
        pred = "hiking, running, chess"
        assert score(pred, gold) is False


class TestHedgedYesNoMatching:
    """Q30: Hedged or explained Yes/No answers."""

    def test_q30_no_against_explained_likely_no(self):
        gold = "Likely no, she does not refer to herself as part of it"
        pred = "No"
        assert score(pred, gold) is True

    def test_yes_against_explained_yes(self):
        gold = "Yes, since she collects classic children's books"
        pred = "Yes"
        assert score(pred, gold) is True

    def test_stance_reversal_is_rejected(self):
        """Opposite stance must never be credited."""
        gold_neg = "Likely no, she does not refer to herself as part of it"
        assert score("Yes", gold_neg) is False

        gold_pos = "Yes, since she collects classic children's books"
        assert score("No", gold_pos) is False


    def test_conflicting_months_are_rejected(self):
        assert score("March 5 2023", "5 May 2023") is False
        assert score("15 December 2023", "August 15, 2023") is False
        assert score("26 August 2023", "November 2023") is False
        assert score("12 November 2023", "September 2023") is False

    # The month guard must not eat the credits it was never meant to touch.
    def test_agreed_months_still_pass(self):
        assert score("2 September 2022", "September 2, 2022") is True
        assert score("11 September 2023", "September 11, 2023") is True
        assert score("8 May, 2023", "May 8, 2023") is True

    def test_guard_is_silent_when_one_side_names_no_month(self):
        # The guard must not claim a contradiction it has no basis for.
        assert _month_contradiction("August 2023", "in the summer") is False
        assert _month_contradiction("June", "June") is False

    def test_multi_month_prediction_matching_gold_is_not_a_contradiction(self):
        assert score("I visited in June and again in May", "May") is True


    def test_article_only_credit_is_rejected(self):
        # The content-overlap requirement: a shared article is not an answer.
        assert score("the", "The Alchemist") is False
        assert score("the book", "The Alchemist") is False
        assert score("The Name of the Wind", "The Alchemist") is False
        assert score("The headliner", "The Fireworks") is False

    def test_short_label_credits_survive(self):
        """The set most at risk: a careless stopword list would veto these."""
        assert score("She is lactose intolerant", "lactose intolerance") is True
        assert score("a letter", "a handwritten letter") is True
        assert score(
            "Chicken dish with garlic, lemon, and herbs", "Roasted Chicken") is True
        assert score("She moved from Sweden", "Sweden") is True

    def test_content_free_gold_is_not_vetoed(self):
        # A gold with no content words at all must not be blocked by this guard.
        assert _has_content_overlap("the", "the") is True


class TestBaselineSanity:
    """Standard containment sanity checks."""

    def test_exact_match(self):
        assert score("Sweden", "Sweden") is True

    def test_pred_contains_gold(self):
        assert score("She moved from Sweden", "Sweden") is True

    def test_unrelated_rejected(self):
        assert score("France", "Sweden") is False

    def test_wrong_count_word_is_rejected(self):
        # This was an xfail recording a KNOWN scorer defect: the token-overlap
        # rule credited a wrong count word because 'times' carried the overlap
        # and the differing numeral was never compared.  The NUMERAL
        # CONTRADICTION guard fixed it, so this is now a hard assertion.
        #
        # Measured on 46 stored runs / 17,604 pairs: 382 conflicting-numeral
        # pairs, 32 credited correct (8.4%).
        assert score("four times, 4", "three times") is False

    def test_conflicting_quantity_nouns_are_rejected(self):
        assert score("two dogs", "three dogs") is False
        assert score("5 dogs", "2 dogs") is False
        assert score("ten times", "two times") is False
        assert score("2 weeks", "three weeks") is False
        assert score("0 years", "three years") is False

    def test_conflicting_years_are_rejected(self):
        assert score("22 August, 2023", "August 2022") is False
        assert score("8 October 2023", "October 2022") is False

    # The guard must not eat the credits it was never meant to touch.
    def test_agreed_quantities_still_pass(self):
        assert score("two dogs", "two dogs") is True
        assert score("three times", "three times") is True
        assert score("2 dogs", "two dogs") is True
        assert score("3 dogs", "three dogs") is True
        assert score("4 months", "four months") is True

    def test_guard_is_silent_when_gold_names_no_quantity(self):
        assert score("two dogs", "she has dogs") is True
        assert score("5 cats", "cats") is True

    def test_multi_quantity_prediction_matching_gold_is_not_a_contradiction(self):
        # gold's value is present among the prediction's quantities.
        assert score("two dogs and three cats", "three cats") is True

    def test_shared_year_prediction_may_be_more_precise(self):
        assert score("22 August, 2023", "August 2023") is True
