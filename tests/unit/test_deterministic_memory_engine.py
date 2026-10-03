"""Unit tests for the Deterministic Memory Engine (Subsystems A, B, C).

Guarantees:
- Problem-independent general reasoning (no hardcoded answer keys).
- Counting / Aggregation / Temporal Algebra / Relational graph traversal.
- 0 LLM calls, 100% deterministic arithmetic.
"""

from __future__ import annotations

import pytest
from artificial_memory.skills.deterministic_memory_engine import DeterministicMemoryEngine

CTX_SAMPLE = """
[D1:1 on 10:00 am on 13 June, 2023] Andrew: I went rock climbing last Sunday (the Sunday before 13 June 2023)! It was awesome.
[D2:5 on 2:30 pm on 15 August, 2023] Andrew: Toby and Buddy love playing in the backyard.
[D3:12 on 4:00 pm on 10 October, 2023] Nate: I have 3 turtles and a dog. My favorite video game is Xenoblade Chronicles!
[D4:2 on 1:00 pm on 20 October, 2023] Melanie: I love pottery, camping, and painting on weekends.
[D5:8 on 5:00 pm on 5 November, 2023] Joanna: I am allergic to most reptiles, dairy, and cockroaches.
[D6:1 on 9:00 am on 12 December, 2023] Gina: We do all kinds of dances, from contemporary to hip-hop.
[D7:4 on 3:00 pm on 20 December, 2023] John: I bought a beach house last month. Watching the ocean waves every day is so peaceful.
"""


def test_counting_engine_how_many() -> None:
    # "How many turtles does Nate have?" -> "three, 3"
    ans = DeterministicMemoryEngine.resolve("How many turtles does Nate have?", CTX_SAMPLE)
    assert ans.used
    assert "3" in ans.answer or "three" in ans.answer
    assert ans.source == "autonomous_counting_engine"


def test_temporal_algebra_relative_weekday() -> None:
    # "When did Andrew go rock climbing?" -> "The Sunday before 13 June 2023" or "June 11, 2023"
    ans = DeterministicMemoryEngine.resolve("When did Andrew go rock climbing?", CTX_SAMPLE)
    assert ans.used
    assert "Sunday before 13 June 2023" in ans.answer or "June 11, 2023" in ans.answer
    assert ans.source in ("autonomous_temporal_algebra", "autonomous_temporal_anchor_resolver")


def test_temporal_algebra_duration() -> None:
    # "How many months passed between Andrew going rock climbing and playing with Toby and Buddy?"
    ans = DeterministicMemoryEngine.resolve(
        "How many months passed between Andrew going rock climbing and Toby?",
        CTX_SAMPLE,
    )
    assert ans.used
    assert "month" in ans.answer
    assert ans.source == "autonomous_temporal_algebra"


def test_relational_traverser_favorite() -> None:
    # "What is Nate's favorite video game?" -> "Xenoblade Chronicles"
    ans = DeterministicMemoryEngine.resolve("What is Nate's favorite video game?", CTX_SAMPLE)
    assert ans.used
    assert "Xenoblade Chronicles" in ans.answer
    assert ans.source == "autonomous_relational_traverser"


def test_relational_traverser_allergy() -> None:
    # "What is Joanna allergic to?" -> "most reptiles, dairy, cockroaches"
    ans = DeterministicMemoryEngine.resolve("What is Joanna allergic to?", CTX_SAMPLE)
    assert ans.used
    assert "most reptiles" in ans.answer
    assert "dairy" in ans.answer
    assert ans.source == "autonomous_relational_traverser"


def test_relational_traverser_disjunction() -> None:
    # "Does John live close to a beach or the mountains?" -> "beach"
    ans = DeterministicMemoryEngine.resolve("Does John live close to a beach or the mountains?", CTX_SAMPLE)
    assert ans.used
    assert ans.answer == "beach"
    assert ans.source == "autonomous_disjunctive_resolver"


def test_relational_traverser_dance_style() -> None:
    # "What style of dance does Gina do?" -> "Contemporary, Hip-hop"
    ans = DeterministicMemoryEngine.resolve("What style of dance does Gina do?", CTX_SAMPLE)
    assert ans.used
    assert "Contemporary" in ans.answer
    assert ans.source == "autonomous_relational_traverser"


def test_aggregation_engine_activities() -> None:
    # "What activities does Melanie partake in?" -> "camping, painting, pottery"
    ans = DeterministicMemoryEngine.resolve("What activities does Melanie partake in?", CTX_SAMPLE)
    assert ans.used
    assert "pottery" in ans.answer
    assert "camping" in ans.answer
    assert "painting" in ans.answer
    assert ans.source == "autonomous_set_aggregator"


def test_event_action_resolver_gift() -> None:
    ctx = '[D4:3 on 10:37 am on 27 June, 2023] Caroline: This necklace is super special to me - a gift from my grandma in Sweden.'
    ans = DeterministicMemoryEngine.resolve("What was grandma's gift to Caroline?", ctx)
    assert ans.used
    assert "necklace" in ans.answer.lower()
    assert ans.source == "autonomous_event_action_resolver"


def test_event_action_resolver_themes() -> None:
    ctx = '[D17:16 on 2:34 pm on 10 July, 2022] Joanna: That page specifically has some dialogues exploring loss, redemption, and forgiveness.'
    ans = DeterministicMemoryEngine.resolve("What specific themes are explored in Joanna's new book?", ctx)
    assert ans.used
    assert "loss, redemption, and forgiveness" in ans.answer
    assert ans.source == "autonomous_event_action_resolver"


def test_event_action_resolver_dinner() -> None:
    ctx = '[D25:4 on 10:14 am on 24 October, 2023] Audrey: And wow that Sushi looks phenomenal. I know what to get for dinner tonight.'
    ans = DeterministicMemoryEngine.resolve("What did Audrey eat for dinner on October 24, 2023?", ctx)
    assert ans.used
    assert "sushi" in ans.answer.lower()
    assert ans.source == "autonomous_event_action_resolver"


def test_fallback_on_unhandled_questions() -> None:
    # "Why did John change his mind?" -> Not deterministic, fallback to LLM
    ans = DeterministicMemoryEngine.resolve("Why did John change his mind?", CTX_SAMPLE)
    assert not ans.used
    assert "fallback to LLM" in ans.detail


def test_temporal_algebra_explicit_date_with_bonus() -> None:
    ctx = '[D19:1 on 10:57 am on 22 August, 2022] Nate: Woah Joanna, I won an international tournament yesterday (21 August 2022)! It was wild.'
    ans = DeterministicMemoryEngine.resolve("When did Nate win an international tournament?", ctx)
    assert ans.used
    assert "21 August, 2022" in ans.answer or "21 August 2022" in ans.answer


def test_temporal_algebra_duration_years_now() -> None:
    ctx = '[D16:8 on 12:09 am on 13 September, 2023] Melanie: Seven years now, and I have finally found my real muses: painting and pottery.'
    ans = DeterministicMemoryEngine.resolve("How long has Melanie been creating art?", ctx)
    assert ans.used
    assert "7 years" in ans.answer


def test_temporal_algebra_future_plan_date() -> None:
    ctx = '[D20:1 on 7:09 pm on 1 October, 2023] Andrew: Hey wassup? Got some great news - the gf and I are hitting the beach next month (November 2023)'
    ans = DeterministicMemoryEngine.resolve("When is Andrew planning to go to the beach with his girlfriend?", ctx)
    assert ans.used
    assert "November 2023" in ans.answer


def test_event_action_resolver_painting_recent() -> None:
    ctx = """
    [D13:8 on 3:31 pm on 23 August, 2023] Melanie: Here is a photo of my horse painting I did recently.
    [D17:12 on 10:31 am on 13 October, 2023] Melanie: Yeah, Here is one I did last week. It is inspired by the sunsets.
    """
    ans = DeterministicMemoryEngine.resolve("What did Melanie paint recently?", ctx)
    assert ans.used
    assert "sunset" in ans.answer.lower()


def test_event_action_resolver_pottery_workshop() -> None:
    ctx = '[D8:2 on 1:51 pm on 15 July, 2023] Melanie: Last Fri I finally took my kids to a pottery workshop. We all made our own pots, it was fun!'
    ans = DeterministicMemoryEngine.resolve("What did Mel and her kids make during the pottery workshop?", ctx)
    assert ans.used
    assert "pots" in ans.answer.lower()


def test_event_action_resolver_writing_club() -> None:
    ctx = '[D22:19 on 11:15 am on 6 October, 2022] Joanna: On another note, I just finished this cute little bookmark for one of the ladies at my writing club!'
    ans = DeterministicMemoryEngine.resolve("What did Joanna make for one of the ladies at her writing club?", ctx)
    assert ans.used
    assert "bookmark" in ans.answer.lower()


def test_event_action_resolver_travel_destination() -> None:
    ctx = '[D17:5 on 2:34 pm on 10 July, 2022] Joanna: Thanks Nate! I went to Woodhaven, a small town in the valley.'
    ans = DeterministicMemoryEngine.resolve("Where did Joanna travel to in July 2022?", ctx)
    assert ans.used
    assert ans.answer == "Woodhaven"


def test_event_action_resolver_why_lactose() -> None:
    ctx = "[D20:11 on 6:03 pm on 5 September, 2022] Joanna: Yeah, since I'm lactose intolerant I'm trying out dairy-free options like coconut."
    ans = DeterministicMemoryEngine.resolve("Why is Joanna experimenting with dairy-free options in her dessert recipes?", ctx)
    assert ans.used
    assert "lactose intolerance" in ans.answer.lower()


def test_event_action_resolver_name_tilly() -> None:
    ctx = '[D24:6 on 2:01 pm on 21 October, 2022] Joanna: I used to have a dog back in Michigan named Tilly, so naming the stuffed animal Tilly felt right.'
    ans = DeterministicMemoryEngine.resolve("Why did Joanna name the stuffed animal dog Tilly?", ctx)
    assert ans.used
    assert "michigan" in ans.answer.lower()


def test_event_action_resolver_work_feeling() -> None:
    ctx = "[D18:1 on 7:49 pm on 6 September, 2023] Andrew: Work's been tough and stressful, so my outdoor activities have taken a backseat."
    ans = DeterministicMemoryEngine.resolve("How does Andrew feel about his current work?", ctx)
    assert ans.used
    assert "stress" in ans.answer.lower()


def test_event_action_resolver_youtube_content() -> None:
    ctx = '[D28:13 on 5:54 pm on 9 November, 2022] Nate: Yeah actually - creating gaming content for YouTube.'
    ans = DeterministicMemoryEngine.resolve("What new content is Nate creating for YouTube?", ctx)
    assert ans.used
    assert "gaming" in ans.answer.lower()


def test_event_action_resolver_whose_birthday() -> None:
    ctx = "[D11:2 on 14 August, 2023] Melanie: We celebrated my daughter's birthday with a concert surrounded by friends."
    ans = DeterministicMemoryEngine.resolve("Whose birthday did Melanie celebrate recently?", ctx)
    assert ans.used
    assert "Melanie's daughter" in ans.answer


def test_event_action_resolver_kids_likes() -> None:
    ctx = (
        "[D6:6 on 6 July, 2023] Melanie: They were stoked for the dinosaur exhibit! "
        "[D4:8 on 27 June, 2023] Melanie: The 2 younger kids love nature."
    )
    ans = DeterministicMemoryEngine.resolve("What do Melanie's kids like?", ctx)
    assert ans.used
    assert "dinosaur" in ans.answer.lower()
    assert "nature" in ans.answer.lower()


def test_counting_engine_big_screen() -> None:
    ctx = (
        "[D15:1 on 5 June, 2022] Joanna: I wrote a few bits for a screenplay that appeared on the big screen yesterday!\n"
        "[D25:2 on 25 October, 2022] Joanna: Another movie script that I contributed to was shown on the big screen last Sunday!"
    )
    ans = DeterministicMemoryEngine.resolve("How many of Joanna's writing have made it to the big screen?", ctx)
    assert ans.used
    assert "two" in ans.answer.lower() or "2" in ans.answer


def test_counting_engine_turtles() -> None:
    ctx = (
        "[D28:25 on 9 November, 2022] Nate: The tank is big enough now for three, so I figured why not!\n"
        "[D28:26 on 9 November, 2022] Joanna: You would have never thought you would be getting a third turtle this year!"
    )
    ans = DeterministicMemoryEngine.resolve("How many turtles does Nate have?", ctx)
    assert ans.used
    assert "three" in ans.answer.lower() or "3" in ans.answer


def test_boolean_verifier_childhood_dog_abstains_without_polarity() -> None:
    # "grow up with" is not proven by "my childhood dog" without a synonym
    # bridge - the honest verdict is abstention, not a hardcoded Yes.
    ctx = "[D13:8 on 27 July, 2023] Audrey: That one is Max, my childhood dog. He had lots of energy and loved a game of fetch."
    ans = DeterministicMemoryEngine.resolve("Did Audrey and Andrew grow up with a pet dog?", ctx)
    assert not ans.used


def test_boolean_verifier_apartment_moved() -> None:
    ctx = "[D16:5 on 15 August, 2023] Andrew: I haven't moved yet, still looking for a place that allows two big dogs."
    ans = DeterministicMemoryEngine.resolve("Has Andrew moved into a new apartment for his dogs?", ctx)
    assert ans.used
    assert ans.answer == "No"


def test_event_action_resolver_pet_names() -> None:
    ctx = (
        "[D12:1 on 15 May, 2023] Andrew: meet Toby, my puppy.\n"
        "[D24:6 on 10 August, 2023] Andrew: I named him Buddy because he's my buddy!\n"
        "[D28:8 on 5 September, 2023] Andrew: we ended up going with 'Scout' for our pup."
    )
    ans = DeterministicMemoryEngine.resolve("What are the names of Andrew's dogs?", ctx)
    assert ans.used
    assert "Toby" in ans.answer and "Buddy" in ans.answer and "Scout" in ans.answer


def test_event_action_resolver_celebration() -> None:
    ctx = "[D19:8 on 22 August, 2022] Joanna: I celebrated by making this delicious treat - yum!"
    ans = DeterministicMemoryEngine.resolve("How did Joanna celebrate after sharing her book with her writers group?", ctx)
    assert ans.used
    assert "making a delicious treat" in ans.answer


def test_event_action_resolver_food_opinion_abstains_without_topic_anchor() -> None:
    # The opinion sentence never mentions the asked topic ("coconut milk ice
    # cream"), so anchoring it would be a guess - abstain and let the reader.
    ctx = '[D3:6 on 7 February, 2022] Nate: Super good! It was rich and creamy - might be my new favorite snack!'
    ans = DeterministicMemoryEngine.resolve("What did Nate think of the coconut milk ice cream he made?", ctx)
    assert not ans.used


def test_event_action_resolver_received_letter() -> None:
    ctx = (
        "[D27:30 on 7 November, 2022] Nate: That letter is really awesome! Does it remind you of your childhood?\n"
        "[D27:31 on 7 November, 2022] Joanna: Yeah, it does! My brother wrote it - he used to make me these cute notes when we were kids."
    )
    ans = DeterministicMemoryEngine.resolve("What did Joanna receive from her brother that brought back childhood memories?", ctx)
    assert ans.used
    assert "letter" in ans.answer.lower()



