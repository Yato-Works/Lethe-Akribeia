from artificial_memory.skills.deterministic_memory_engine import DeterministicMemoryEngine


def test_explicit_event_date_is_committed_from_matching_turn() -> None:
    ctx = (
        "[D6:6 on 2:35 pm on 16 March, 2023] Gina: My online clothes store is open!\n"
        "[D14:8 on 9:38 pm on 16 June, 2023] Gina: I opened an online clothing store and it's been great."
    )
    ans = DeterministicMemoryEngine.resolve("When did Gina open her online clothing store?", ctx)
    assert ans.used
    assert ans.answer == "16 March, 2023"
    assert ans.evidence_turn == "D6:6"


def test_car_ownership_date_is_not_mistaken_for_a_car_accident() -> None:
    ctx = (
        "[D2:1 on 6:10 pm on 22 December, 2022] Maria: "
        "I donated my old car to a homeless shelter yesterday (21 December 2022)."
    )
    ans = DeterministicMemoryEngine.resolve("When did Maria get in a car accident?", ctx)
    assert not ans.used


def test_related_hike_and_roadtrip_must_be_bound_to_the_same_local_event() -> None:
    ctx = (
        "[D4:8 on 10:37 am on 27 June, 2023] Melanie: "
        "We went on a hike in the mountains.\n"
        "[D18:1 on 6:55 pm on 20 October, 2023] Melanie: "
        "That roadtrip this past weekend was insane."
    )
    ans = DeterministicMemoryEngine.resolve(
        "When did Melanie go on a hike after the roadtrip?", ctx
    )
    assert not ans.used


def test_relative_day_can_bind_event_description_in_adjacent_turn() -> None:
    ctx = (
        "[D21:3 on 8:00 pm on 3 July, 2023] Maria: Something not so great happened yesterday.\n"
        "[D21:4 on 8:01 pm on 3 July, 2023] John: What happened?\n"
        "[D21:5 on 8:02 pm on 3 July, 2023] Maria: A car ran a red light and hit us."
    )
    ans = DeterministicMemoryEngine.resolve("When did Maria get in a car accident?", ctx)
    assert ans.used
    assert ans.answer == "2 July, 2023"
    assert ans.evidence_turn == "D21:3"


def test_broad_season_does_not_switch_to_another_turns_precise_anchor() -> None:
    ctx = (
        "[D18:2 on 2:34 pm on 10 July, 2022] Maria: Your camping trip sounds awesome.\n"
        "[D30:6 on 12:10 am on 11 August, 2023] John: Max and I had a blast on our camping trip last summer."
    )
    ans = DeterministicMemoryEngine.resolve("When did John go on a camping trip with Max?", ctx)
    assert ans.used
    assert ans.answer == "last summer"
    assert ans.evidence_turn == "D30:6"

