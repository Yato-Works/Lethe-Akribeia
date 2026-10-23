from artificial_memory_gen.skills.deterministic_memory_engine import DeterministicMemoryEngine


def test_console_attribute_is_bound_to_question_subject_and_turn() -> None:
    ctx = (
        "[D1:1 on 1 January 2024] Alice: I play on the Gamecube.\n"
        "[D1:2 on 1 January 2024] Bob: I play on the PlayStation."
    )
    answer = DeterministicMemoryEngine.resolve("What console does Alice play on?", ctx)
    assert answer.used
    assert answer.answer == "gamecube"
    assert answer.evidence_turn == "D1:1"


def test_console_attribute_abstains_when_target_is_not_in_evidence() -> None:
    ctx = "[D1:1 on 1 January 2024] Bob: I play on the PlayStation."
    answer = DeterministicMemoryEngine.resolve("What console does Alice play on?", ctx)
    assert not answer.used


def test_generic_attribute_exposes_evidence_turn_id() -> None:
    ctx = "[D1:1 on 1 January 2024] Alice: My hobby is pottery."
    answer = DeterministicMemoryEngine.resolve("What is Alice's hobby?", ctx)
    assert answer.used
    assert "pottery" in answer.answer.lower()
    assert answer.evidence_turn == "D1:1"


def test_generic_action_probe_does_not_match_inside_doing() -> None:
    ctx = "[D1:1 on 1 January 2024] Melanie: You're doing something amazing with your family."
    answer = DeterministicMemoryEngine.resolve("What does Melanie do with her family?", ctx)
    assert not answer.used


def test_generic_action_probe_rejects_underspecified_takeaway_relation() -> None:
    ctx = "[D1:1 on 1 January 2024] Caroline: Take care of yourself after reading that book."
    answer = DeterministicMemoryEngine.resolve(
        'What did Caroline take away from the book "Becoming Nicole"?', ctx
    )
    assert not answer.used


def test_generic_attribute_head_requires_an_explicit_value_relation() -> None:
    ctx = "[D1:1 on 1 January 2024] Evan: I have lots of hobbies, recently to take my mind off it."
    answer = DeterministicMemoryEngine.resolve("What kind of hobbies does Evan pursue?", ctx)
    assert not answer.used


def test_generic_verb_requires_question_object_anchor_in_evidence_turn() -> None:
    ctx = (
        "[D1:1 on 1 January 2024] Gina: This design reminds me of the grit it takes to stand out.\n"
        "[D1:2 on 1 January 2024] Gina: I designed the space for my store."
    )
    answer = DeterministicMemoryEngine.resolve("What did Gina design for her store?", ctx)
    assert answer.used
    assert answer.evidence_turn == "D1:2"


def test_preference_is_not_automatically_a_hobby() -> None:
    ctx = "[D1:1 on 1 January 2024] Evan: I love ginger snaps."
    answer = DeterministicMemoryEngine.resolve("What kind of hobbies does Evan pursue?", ctx)
    assert not answer.used

