from artificial_memory_gen.skills.deterministic_memory_engine import DeterministicMemoryEngine


def test_pet_names_are_unioned_across_target_speaker_turns() -> None:
    context = (
        "[D7:18 on 4:33 pm on 12 July, 2023] Melanie: Luna and Oliver! They are sweet.\n"
        "[D13:4 on 3:31 pm on 23 August, 2023] Melanie: We got another cat named Bailey too.\n"
        "[D13:3 on 3:31 pm on 23 August, 2023] Caroline: Oscar is my guinea pig."
    )
    answer = DeterministicMemoryEngine.resolve("What are Melanie's pets' names?", context)
    assert answer.used
    assert answer.answer == "Luna, Oliver, Bailey"
    assert answer.evidence_turn == "D7:18, D13:4"


def test_pet_union_does_not_import_names_from_other_speakers() -> None:
    context = (
        "[D1:1 on 1 May, 2023] Melanie: My pets are Luna and Oliver.\n"
        "[D1:2 on 1 May, 2023] Caroline: My pets are Oscar and Bailey."
    )
    answer = DeterministicMemoryEngine.resolve("What are Melanie's pets' names?", context)
    assert answer.used
    assert answer.answer == "Luna, Oliver"


def test_pet_names_are_extracted_from_their_names_are_list() -> None:
    context = (
        "[D1:7 on 1:10 pm on 27 March, 2023] Audrey: I've had them for 3 years! "
        "Their names are Pepper, Precious and Panda. I can't live without them.\n"
        "[D2:2 on 1:10 pm on 27 March, 2023] Andrew: Pepper is adjusting well."
    )
    answer = DeterministicMemoryEngine.resolve("What are the names of Audrey's dogs?", context)
    assert answer.used
    assert answer.answer == "Pepper, Precious, Panda"
    assert answer.evidence_turn == "D1:7"


def test_incomplete_pet_name_set_abstains_when_evidence_states_larger_count() -> None:
    context = (
        "[D1:7 on 1:10 pm on 27 March, 2023] Audrey: Their names are Pepper, Precious and Panda.\n"
        "[D15:1 on 9:58 pm on 16 August, 2023] Audrey: I got a tattoo of my four dogs."
    )
    answer = DeterministicMemoryEngine.resolve("What are the names of Audrey's dogs?", context)
    assert not answer.used


def test_relation_aligned_lowercase_list_unions_across_turns_with_provenance() -> None:
    context = (
        "[D4:14 on 4 February, 2023] Deborah: We were trying out different dance poses, "
        "like Warrior II, which builds strength and boosts focus.\n"
        "[D14:3 on 26 June, 2023] Deborah: By the way, I tried a new pose - "
        "Dancer Pose (Natarajasana).\n"
        "[D14:15 on 26 June, 2023] Deborah: This is also a new yoga pose that we tried. "
        "It is a tree pose."
    )
    answer = DeterministicMemoryEngine.resolve("What new yoga poses did Deborah try?", context)
    assert answer.used
    assert answer.answer == "Warrior II, Dancer Pose (Natarajasana), tree pose"
    assert answer.evidence_turn == "D4:14, D14:3, D14:15"


def test_relation_list_does_not_match_generic_have_or_other_speakers() -> None:
    context = (
        "[D1:1 on 1 May, 2023] Andrew: I have many special memories with my dog.\n"
        "[D1:2 on 1 May, 2023] Caroline: I have memories of long walks and exploring paths."
    )
    answer = DeterministicMemoryEngine.resolve(
        "What special memories does Andrew have with his childhood dog?", context
    )
    assert not answer.used


def test_which_quoted_titles_require_target_speaker_and_keep_evidence_ids() -> None:
    context = (
        '[D1:1 on 1 May, 2023] Alice: I recommended "The Left Hand of Darkness" to Sam.\n'
        '[D2:1 on 2 May, 2023] Alice: I recommended "Kindred" to Sam.\n'
        '[D3:1 on 3 May, 2023] Sam: I recommended "Dune" to Alice.'
    )
    answer = DeterministicMemoryEngine.resolve(
        "Which two books did Alice recommend to Sam?", context
    )
    assert answer.used
    assert answer.answer == "The Left Hand of Darkness, Kindred"
    assert answer.evidence_turn == "D1:1, D2:1"


def test_open_ended_quoted_recommendation_list_abstains_without_completeness_evidence() -> None:
    context = (
        '[D1:1 on 1 May, 2023] John: Definitely! My two favorites are "A" and "B". '
        'If SF is your thing, check out "C".\n'
        '[D1:2 on 1 May, 2023] James: Thanks for the recommendations!'
    )
    answer = DeterministicMemoryEngine.resolve(
        "Which books has John recommended to James?", context
    )
    assert not answer.used


def test_reply_link_can_supply_relation_without_claiming_parent_words_as_answerer_facts() -> None:
    context = (
        '[D1:1 on 1 May, 2023] James: Which two books would you recommend?\n'
        '[D1:2 on 1 May, 2023] (In reply to James: "Which two books would you recommend?") '
        'Alice: My suggestions are "Dune" and "Kindred".\n'
        '[D2:1 on 2 May, 2023] James: Which two books would you recommend?\n'
        '[D2:2 on 2 May, 2023] (In reply to James: "Which two books would you recommend?") '
        'Alice: I would choose "Kindred" and "Dune".'
    )
    answer = DeterministicMemoryEngine.resolve(
        "Which two books did Alice recommend to James?", context
    )
    assert answer.used
    assert answer.answer == "Dune, Kindred"
    assert answer.evidence_turn == "D1:2, D1:1, D2:1, D2:2"

