"""Category-5 (adversarial) protocol contract for the LoCoMo harness.

Guarantees under test:

1. `load_questions` NEVER uses `adversarial_answer` as gold.  LoCoMo category 5
   stores no `answer`; the distractor is what the pinned official harness scores
   zero for emitting (third_party/.../evaluation.py:217-221).  Using it as gold
   inverted the category and credited the wrong option.
2. `score_question` implements the official category-5 contract: a refusal in an
   accepted form is correct, anything else is not.
3. Categories 1-4 are untouched by the protocol switch.
4. `is_refusal` recognises both the canonical forms and the legacy forms, so the
   normalisation is lossless.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from eval_locomo_full import (  # noqa: E402
    canonical_refusal,
    is_refusal,
    load_questions,
    score_question,
)


def _conv(category: int, **qa) -> dict:
    base = {
        "question": "What did Caroline realize after her charity race?",
        "evidence": ["D2:3"],
        "category": category,
    }
    base.update(qa)
    return {"sample_id": "conv-26", "qa": [base]}


class TestAdversarialGoldIsNeverTheDistractor:
    def test_adversarial_answer_is_not_used_as_gold(self):
        conv = _conv(5, adversarial_answer="self-care is important")
        q = load_questions(conv)[0]
        assert q["answer"] == ""
        assert q["adversarial_answer"] == "self-care is important"

    def test_distractor_kept_for_audit_only(self):
        """The distractor stays readable, but it is a separate field."""
        conv = _conv(5, adversarial_answer="researching adoption agencies")
        q = load_questions(conv)[0]
        assert q["answer"] != q["adversarial_answer"]

    def test_normal_question_still_carries_its_answer(self):
        conv = _conv(4, answer="pottery")
        q = load_questions(conv)[0]
        assert q["answer"] == "pottery"


class TestCategory5Contract:
    def test_canonical_refusals_are_correct(self):
        q = {"category": 5, "answer": ""}
        assert score_question("not mentioned", q) is True
        assert score_question("No information available.", q) is True

    def test_legacy_refusal_is_correct_after_normalisation(self):
        """The model's own wording is accepted by is_refusal, not by the scorer."""
        q = {"category": 5, "answer": ""}
        assert score_question("I don't know.", q) is False

    def test_emitting_the_distractor_is_wrong(self):
        q = {"category": 5, "answer": "", "adversarial_answer": "self-care is important"}
        assert score_question("self-care is important", q) is False

    def test_any_answer_is_wrong_on_category_5(self):
        q = {"category": 5, "answer": ""}
        for pred in ("LGBTQ+ folks", "figurines", "because of their inclusivity"):
            assert score_question(pred, q) is False


class TestCategoriesOneToFourUnchanged:
    def test_single_value_match(self):
        assert score_question("pottery", {"category": 4, "answer": "pottery"}) is True

    def test_list_permutation(self):
        q = {"category": 1, "answer": "pottery, camping, painting"}
        assert score_question("camping, pottery, painting", q) is True

    def test_incomplete_list_rejected(self):
        q = {"category": 1, "answer": "pottery, camping, painting, swimming"}
        assert score_question("painting", q) is False

    def test_refusal_scoring_against_a_real_gold_is_still_wrong(self):
        assert score_question("not mentioned", {"category": 4, "answer": "pottery"}) is False


class TestRefusalRecognition:
    def test_legacy_forms(self):
        for text in ("I don't know.", "none", "not mentioned", ""):
            assert is_refusal(text) is True

    def test_canonical_forms_inside_sentences(self):
        assert is_refusal("This is not mentioned in the conversation.") is True
        assert is_refusal("There is no information available.") is True

    def test_real_answers_are_not_refusals(self):
        assert is_refusal("pottery") is False
        assert is_refusal("self-care is important") is False

    def test_canonical_refusal_is_accepted_by_the_scorer(self):
        assert score_question(canonical_refusal(), {"category": 5, "answer": ""}) is True
