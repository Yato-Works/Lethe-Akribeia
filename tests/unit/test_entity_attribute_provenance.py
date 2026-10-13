from artificial_memory.skills.deterministic_memory_engine import DeterministicMemoryEngine


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

