from artificial_memory.skills.deterministic_memory_engine import DeterministicMemoryEngine


def test_pet_count_accepts_concrete_animal_as_owned_total_across_turns() -> None:
    context = (
        "[D1:2 on 9:00 am on 20 March, 2022] James: My two dogs are my best pals.\n"
        "[D19:12 on 9:00 am on 10 August, 2022] James: I took my three dogs to the beach.\n"
    )

    answer = DeterministicMemoryEngine.resolve("How many pets does James have?", context)

    assert answer.used
    assert answer.answer == "three, 3"
    assert answer.evidence_turn == "D19:12"


def test_pet_count_does_not_import_another_speakers_report() -> None:
    context = (
        "[D1:2 on 9:00 am on 20 March, 2022] John: James has three dogs, and I have one cat.\n"
    )

    answer = DeterministicMemoryEngine.resolve("How many pets does James have?", context)

    assert not answer.used


def test_pet_count_abstains_when_evidence_only_says_another_pet_was_adopted() -> None:
    context = (
        "[D1:2 on 9:00 am on 20 March, 2022] James: I already have two dogs.\n"
        "[D2:3 on 9:00 am on 21 March, 2022] James: I adopted another pup today.\n"
    )

    answer = DeterministicMemoryEngine.resolve("How many pets does James have?", context)

    # The latest event does not state the new total, so the engine cannot add
    # it to the old total without evidence that the animals coexist.
    assert not answer.used


def test_time_bounded_pet_count_does_not_reuse_a_later_total() -> None:
    context = (
        "[D1:2 on 9:00 am on 20 March, 2022] James: My two dogs are my best pals.\n"
        "[D19:12 on 9:00 am on 10 August, 2022] James: I took my three dogs to the beach.\n"
    )

    answer = DeterministicMemoryEngine.resolve(
        "How many pets did James have, as of April 2022?", context
    )

    assert not answer.used


def test_unrelated_explicit_frequency_does_not_answer_a_different_event() -> None:
    context = (
        "[D1:1 on 1 January 2024] Audrey: I meet other dog owners once a week for tips.\n"
        "[D1:2 on 1 January 2024] Andrew: I hope we plan to hike together soon."
    )
    answer = DeterministicMemoryEngine.resolve(
        "How many times did Audrey and Andrew plan to hike together?", context
    )
    assert not answer.used


def test_repeated_planned_event_mentions_are_not_counted_as_occurrences() -> None:
    context = (
        "[D1:1 on 1 January 2024] Audrey: We plan to hike together next month.\n"
        "[D2:1 on 1 February 2024] Andrew: Can't wait for our hike; let's plan it soon.\n"
        "[D3:1 on 1 March 2024] Audrey: I hope we plan another hike together."
    )
    answer = DeterministicMemoryEngine.resolve(
        "How many times did Audrey and Andrew plan to hike together?", context
    )
    assert not answer.used


def test_counting_abstention_is_not_stolen_by_generic_attribute_routing() -> None:
    context = (
        "[D1:1 on 1 January 2024] Audrey: We plan to hike together next month.\n"
        "[D2:1 on 1 February 2024] Audrey: Make sure I give each of them a tag."
    )
    answer = DeterministicMemoryEngine.resolve(
        "How many times did Audrey and Andrew plan to hike together?", context
    )
    assert not answer.used
