from artificial_memory.skills.deterministic_memory_engine import DeterministicMemoryEngine


def test_explicit_cause_is_extracted_from_target_speaker_turn() -> None:
    context = (
        "[D2:12 on 1:14 pm on 25 May, 2023] Caroline: "
        "I chose them 'cause they help LGBTQ+ folks with adoption. "
        "Their inclusivity and support really spoke to me.\n"
    )

    answer = DeterministicMemoryEngine.resolve(
        "Why did Caroline choose the adoption agency?", context
    )

    assert answer.used
    assert answer.answer == "they help LGBTQ+ folks with adoption"
    assert answer.evidence_turn == "D2:12"


def test_explicit_cause_matches_event_topic_in_another_conversation_shape() -> None:
    context = (
        "[D1:4 on 4:04 pm on 20 January, 2023] Jon: "
        "I'm starting a dance studio 'cause I'm passionate about dancing "
        "and it'd be great to share it with others.\n"
    )

    answer = DeterministicMemoryEngine.resolve(
        "Why did Jon decide to start his dance studio?", context
    )

    assert answer.used
    assert answer.answer == "I'm passionate about dancing and it'd be great to share it with others"
    assert answer.evidence_turn == "D1:4"


def test_explicit_cause_is_not_attributed_to_another_person() -> None:
    context = (
        "[D2:12 on 1:14 pm on 25 May, 2023] Caroline: "
        "I chose them 'cause they help LGBTQ+ folks with adoption.\n"
        "[D2:13 on 1:14 pm on 25 May, 2023] Melanie: "
        "I am still researching different agencies.\n"
    )

    answer = DeterministicMemoryEngine.resolve(
        "Why did Melanie choose the adoption agency?", context
    )

    assert not answer.used


def test_implicit_after_event_realization_answers_why_with_source_turn() -> None:
    context = (
        "[D7:4 on 25 February, 2023] John: After my last run, I saw the impact I could make "
        "in the community through politics. It's rewarding to work towards positive changes.\n"
    )

    answer = DeterministicMemoryEngine.resolve(
        "Why did John decide to run for office again?", context
    )

    assert answer.used
    assert answer.answer == "the impact I could make in the community through politics"
    assert answer.evidence_turn == "D7:4"


def test_implicit_purpose_is_bound_to_the_queried_action() -> None:
    context = (
        "[D8:4 on 9 June, 2023] Dave: I've been exploring some parks on the weekends to relax - "
        "it's so peaceful being surrounded by nature.\n"
        "[D8:8 on 9 June, 2023] Dave: Hiking clears your mind and brings a sense of calm.\n"
    )

    answer = DeterministicMemoryEngine.resolve("Why does Dave regularly visit parks?", context)

    assert answer.used
    assert answer.answer == "It relaxes and feels peaceful"
    assert answer.evidence_turn == "D8:4"


def test_nearby_same_speaker_intention_bridges_the_action_and_its_reason() -> None:
    context = (
        "[D12:1 on 5 May, 2023] John: Since last time, I've been blogging about politics and government.\n"
        "[D12:3 on 5 May, 2023] John: I'm researching policies and writing about my ideas. "
        "Hoping to raise awareness and start conversations to create positive change.\n"
    )

    answer = DeterministicMemoryEngine.resolve(
        "Why did John start blogging about politics and policies?", context
    )

    assert answer.used
    assert answer.answer == "raise awareness and start conversations to create positive change"
    assert answer.evidence_turn == "D12:1, D12:3"


def test_nearby_intention_does_not_cross_speakers() -> None:
    context = (
        "[D12:1 on 5 May, 2023] John: Since last time, I've been blogging about politics and government.\n"
        "[D12:3 on 5 May, 2023] Maria: Hoping to raise awareness and start conversations about politics.\n"
    )

    answer = DeterministicMemoryEngine.resolve(
        "Why did John start blogging about politics?", context
    )

    assert not answer.used


def test_implicit_cause_does_not_use_reason_inside_another_speakers_reply_quote() -> None:
    context = (
        '[D8:2 on 3 April, 2023] (In reply to Jon: "I had to shut down my bank account '
        'because I needed it for my business.") Gina: Oh no, Jon! Sorry to hear that.\n'
    )

    answer = DeterministicMemoryEngine.resolve("Why did Gina shut down her bank account?", context)

    assert not answer.used


def test_why_resolver_does_not_treat_a_travel_destination_as_a_purpose() -> None:
    context = (
        "[D23:1 on 30 August, 2023] Jolene: Yesterday my partner and I got back from a trip to Rio; "
        "we checked out some cool yoga classes.\n"
    )

    answer = DeterministicMemoryEngine.resolve(
        "Why did Jolene sometimes put off doing yoga?", context
    )

    assert not answer.used


def test_implicit_reason_does_not_answer_a_multi_person_question_from_one_speaker() -> None:
    context = (
        "[D6:15 on 1 February, 2022] John: I want to make a difference away from the court, "
        "like through charity or inspiring people.\n"
    )

    answer = DeterministicMemoryEngine.resolve(
        "Why do Tim and John find LeBron inspiring?", context
    )

    assert not answer.used


def test_inspiration_is_bound_to_the_nearby_requested_art_event() -> None:
    context = (
        "[D1:14 on 1 January 2024] Caroline: I have a painting for the art show.\n"
        "[D1:15 on 1 January 2024] Melanie: What inspired you?\n"
        "[D1:16 on 1 January 2024] Caroline: I painted this after I visited an LGBTQ center. "
        "I wanted to capture everyone's unity and strength.\n"
        "[D7:3 on 7 January 2024] Caroline: I also painted a sunset last week."
    )

    answer = DeterministicMemoryEngine.resolve(
        "What inspired Caroline's painting for the art show?", context
    )

    assert answer.used
    assert "visited an LGBTQ center" in answer.answer
    assert "unity and strength" in answer.answer
    assert "D1:16" in answer.evidence_turn
    assert "D7:3" not in answer.evidence_turn


def test_inspiration_does_not_borrow_a_different_qualifier_from_same_session() -> None:
    context = (
        "[D11:10 on 6 October 2023] Evan: The sunset painting was inspired by a vacation a few years back.\n"
        "[D11:11 on 6 October 2023] Evan: A friend gave me a painting and it inspired me to start."
    )
    answer = DeterministicMemoryEngine.resolve(
        "What did Evan start painting due to being inspired by a friend's gift?", context
    )
    assert not answer.used
