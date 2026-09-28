"""Unit tests for AM Apex Phase QUAD-CONQUEST."""

from artificial_memory.core.ir.proposition import UnifiedProposition
from artificial_memory.core.ir.structured import IRStatus, StructuredIR
from artificial_memory.protein.temporal_supersession_protein import TemporalSupersessionProtein
from artificial_memory.recall.answer_verifier import AnswerVerifier
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.steroid.wide_slicer import WideSlicer


def test_wide_slicer_stemming():
    slicer = WideSlicer(per_channel_budget=10)
    assert slicer._stem("documentaries") == "documentary"
    assert slicer._stem("recommendations") == "recommend"
    assert slicer._stem("married") == "marry"
    assert slicer._stem("baking") == "bake"
    assert slicer._stem("ran") == "run"
    assert slicer._stem("bought") == "buy"

    # Test stemming recall
    records = [
        StructuredIR(
            entity="Melanie",
            property="activity",
            value="watching documentaries",
            raw_content="Melanie enjoys watching documentaries on weekends.",
        )
    ]
    res = slicer.slice("Any documentary recommendations?", records)
    assert len(res.candidate_records) == 1


def test_adversarial_integrity_verifier():
    verifier = AnswerVerifier()

    # The integrity override needs propositions to reason about: it asks "is the
    # proposition's subject present in the context at all?".  The LoCoMo adapter
    # passes an empty list (and therefore never triggers the override - see
    # AnswerVerifier.verify), so these tests state the proposition explicitly.
    bowl = UnifiedProposition(
        id="p-bowl", subject="Caroline", predicate="made", object="the bowl"
    )
    realization = UnifiedProposition(
        id="p-selfcare", subject="Caroline", predicate="realized", object="self-care matters"
    )

    # Boolean question entity-swap denial
    res_bool = verifier.verify(
        question="Did Caroline make the black and white bowl in the photo?",
        predicted_answer="Yes, she made it in her pottery class.",
        context="[D5:8] Melanie: I made this bowl in my class.",
        propositions=[bowl],
        integrity_abstention_recommended=True,
    )
    assert res_bool.verified_answer == "No"
    assert res_bool.hallucination_detected is True

    # Open-domain unmentioned detail abstention
    res_open = verifier.verify(
        question="What did Caroline realize after her charity race?",
        predicted_answer="Caroline realized that self-care is really important.",
        context="[D2:3] Melanie: I'm starting to realize that self-care is really important.",
        propositions=[realization],
        integrity_abstention_recommended=True,
    )
    assert "None" in res_open.verified_answer
    assert res_open.hallucination_detected is True


def test_integrity_override_is_skipped_without_propositions():
    """No propositions means the override cannot judge evidence presence.

    Regression guard: the LoCoMo path always passes ``propositions=[]``, and an
    unconditional override there turned correct answers into abstentions (the
    strict spurious-refusal count in AM_APEX_STATUS.md 0f).  With no propositions
    the verifier must leave the reader's answer alone.
    """
    verifier = AnswerVerifier()
    res = verifier.verify(
        question="Did Caroline make the black and white bowl in the photo?",
        predicted_answer="Yes, she made it in her pottery class.",
        context="[D5:8] Melanie: I made this bowl in my class.",
        propositions=[],
        integrity_abstention_recommended=True,
    )
    assert res.verified_answer == "Yes, she made it in her pottery class."
    assert res.hallucination_detected is False


def test_temporal_supersession_expanded_properties():
    protein = TemporalSupersessionProtein()
    records = [
        StructuredIR(
            entity="user",
            property="hours",
            value="5-6 hours",
            raw_content="[2023-06-11] user: I've spent around 5-6 hours on my sculpture.",
            time_scope="2023-06-11",
        ),
        StructuredIR(
            entity="user",
            property="hours",
            value="10-12 hours",
            raw_content="[2023-06-17] user: I've already spent 10-12 hours on my sculpture.",
            time_scope="2023-06-17",
        ),
    ]
    active, groups = protein.resolve("How many hours have I spent on my sculpture?", records)
    assert len(groups) == 1
    assert groups[0].current_record.value == "10-12 hours"
    assert groups[0].superseded_records[0].value == "5-6 hours"
    assert groups[0].superseded_records[0].status == IRStatus.SUPERSEDED
    assert groups[0].current_record.status == IRStatus.ACTIVE


def test_longmemeval_preference_matching():
    rubric = (
        "The user would prefer baking suggestions that take into account their previous "
        "success with the lemon poppyseed cake, such as variations of that recipe."
    )
    # Good tailored answer mentioning poppyseed
    ans_good = "Since you had great success with your lemon poppyseed cake, I recommend a lavender lemon pound cake!"
    assert LongMemEvalAdapter._preference_answer_matches(rubric, ans_good) is True

    # Generic answer not mentioning key preference
    ans_generic = "You could bake chocolate brownies or classic vanilla cupcakes."
    assert LongMemEvalAdapter._preference_answer_matches(rubric, ans_generic) is False

    # Abstention answer
    ans_abstain = "I don't know."
    assert LongMemEvalAdapter._preference_answer_matches(rubric, ans_abstain) is False
