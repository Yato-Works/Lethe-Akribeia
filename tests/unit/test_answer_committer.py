"""Unit tests for the deterministic answer committer (AM decides, reader renders).

The committer exists because the memory runtime already holds the answer: on
LoCoMo-1,540 with ``qwen2.5:7b-instruct`` the gold turn is retrieved 86.6% of
the time for temporal questions while the reader scores only 42.7%.  These tests
pin the three properties the committer must keep to be safe as a *replacement*
for a reader call:

* it prefers the resolved annotation the runtime produced over the provenance
  header date (the exact mistake a copy-style reader makes);
* it abstains instead of guessing when the winning turn carries no dated content;
* it is deterministic - same input, same output, no LLM, no randomness.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import artificial_memory.skills.answer_committer as committer
from artificial_memory.skills.answer_committer import (
    extract_certificate_answer,
    extract_temporal_answer,
    parse_turns,
)

REPO = Path(__file__).resolve().parents[2]

CTX = "\n".join([
    '[D1:14 on 1:56 pm on 8 May, 2023] (In reply to Caroline: "Is this yours?") '
    "Melanie: Yeah, I painted that lake sunrise last year (2022)! It is special to me.",
    "[D2:1 on 1:14 pm on 25 May, 2023] "
    "Melanie: I ran a charity race for mental health last Saturday "
    "(the Saturday before 25 May 2023).",
    "[D3:2 on 3:00 pm on 2 July, 2023] Caroline: I adopted a dog and he has been "
    "with us for 4 years.",
    "[D4:3 on 4:44 pm on 26 July, 2023] Caroline: We talked about the project plan "
    "and nothing else that afternoon.",
])


def test_parse_turns_recovers_header_speaker_and_payload() -> None:
    turns = parse_turns(CTX)
    assert len(turns) == 4
    assert turns[0].header_date == "8 May, 2023"
    assert turns[0].speaker == "Melanie"
    # The (In reply to ...) quote is stripped so the other speaker's words never
    # count as this turn's own content.
    assert turns[0].text.startswith("Yeah, I painted")
    assert "Is this yours?" not in turns[0].text


def test_commits_the_resolved_annotation_not_the_session_header() -> None:
    """The runtime's own annotation wins over the provenance header date.

    Turn 2's header is the session date (25 May 2023) while the answer lives in
    the annotation the normalizer appended.  Committing the header reproduces
    exactly the reader bug the post-mortem measured (45 of 184 wrong temporal
    answers quoted a header date).
    """
    out = extract_temporal_answer("When did Melanie run the charity race?", CTX)
    assert out.used, out.detail
    assert out.answer == "the Saturday before 25 May 2023"
    assert "25 May 2023" in out.answer  # the anchor date is preserved


def test_duration_question_commits_the_duration_not_a_date() -> None:
    """"How long ..." asks for an elapsed time, so a date span is the wrong shape."""
    out = extract_temporal_answer("How long has the dog been with them?", CTX)
    assert out.used, out.detail
    assert out.answer == "4 years"


def test_abstains_when_the_matching_turn_carries_no_date() -> None:
    """No dated content in the evidence turn -> return control to the reader.

    The *reason* is allowed to vary (low score vs no span); the contract is that
    nothing is invented and the reader path stays in charge.
    """
    out = extract_temporal_answer("What did Caroline talk about that afternoon?", CTX)
    assert not out.used
    assert out.answer == ""
    assert out.evidence_turn == ""


def test_abstains_when_no_turn_shares_the_key_term() -> None:
    out = extract_temporal_answer("When did Bob win the yacht regatta in Monaco?", CTX)
    assert not out.used
    assert out.evidence_turn == ""


def test_short_context_still_scores() -> None:
    """Regression for the IDF floor.

    In a two-turn context every token has ``df == n_turns``, so a raw IDF would
    be 0 everywhere and the committer would never fire (a silent no-op).
    """
    short = "\n".join([
        "[D1:1 on 1:00 pm on 3 March, 2023] Alice: I signed the lease on the "
        "flat last Friday (the Friday before 3 March 2023).",
        "[D2:1 on 1:00 pm on 4 March, 2023] Bob: The weather was lovely that day.",
    ])
    out = extract_temporal_answer("When did Alice sign the lease?", short)
    assert out.used, out.detail
    assert out.answer == "the Friday before 3 March 2023"


def test_is_deterministic() -> None:
    question = "When did Melanie run the charity race?"
    first = extract_temporal_answer(question, CTX)
    second = extract_temporal_answer(question, CTX)
    assert (first.used, first.answer) == (second.used, second.answer)
    assert first.confidence == second.confidence


# ------------------------------------------------------- LongMemEval format
# Same committer, second context dialect: "[answer_280352e9 on 2023/05/30
# (Tue) 17:27] user: ..." - a role instead of a name, and an ISO-ish timestamp.

LME_CTX = "\n".join([
    "[answer_280352e9 on 2023/05/30 (Tue) 17:27] user: I graduated with a degree in "
    "Business Administration, which has definitely shaped how I approach problems.",
    "[evidence_9f2c on 2023/06/12 (Mon) 09:12] assistant: Congrats on the new role!",
    "[evidence_1a44 on 2023/07/03 (Mon) 20:41] user: I started the new job two weeks ago "
    "(the week before 3 July 2023) and it has been busy since.",
])


def test_parse_turns_handles_the_longmemeval_dialect() -> None:
    turns = parse_turns(LME_CTX)
    assert len(turns) == 3
    assert turns[0].dia_id == "answer_280352e9"
    assert turns[0].header_date == "2023/05/30 (Tue) 17:27"
    assert turns[0].speaker == "user"
    assert turns[0].text.startswith("I graduated with a degree")
    assert turns[1].speaker == "assistant"


def test_commits_an_answer_from_the_longmemeval_dialect() -> None:
    out = extract_temporal_answer("When did I start the new job?", LME_CTX)
    assert out.used, out.detail
    assert "3 July 2023" in out.answer


# -------------------------------------------------------- temporal certificates
# LongMemEval contexts open with a certificate the runtime has already computed
# ("Exactly 7 days passed"), so the model was being asked to redo arithmetic the
# memory system had done.  These tests pin the three committable forms and, just
# as importantly, the forms that must be left to the reader.

CALC_CTX = (
    "[Temporal Calculation: Event 1 ('visit to the museum of modern art') occurred on "
    "2023-01-08. Event 2 ('ancient civilizations' exhibit) occurred on 2023-01-15. "
    "Exactly 7 days passed between these events.]\n"
    "[answer_1 on 2023/01/15 (Sun) 00:27] user: Sounds great!"
)
AGO_CTX = (
    "[Temporal Calculation: Reference date is 2023-04-01. The event ('meet up with my "
    "aunt') occurred on 2023-03-04. Exactly 4 weeks passed (approx. 4 weeks ago).]\n"
    "[answer_1 on 2023/03/04 (Sat) 10:00] user: Great to see you!"
)
ORDERING_CTX = (
    "[Temporal Ordering: In chronological order from first to last: First, i helped my "
    "friend prepare the nursery, then i helped my cousin, and lastly i ordered a case.]\n"
    "[answer_1 on 2023/03/01 (Wed) 10:00] user: Lots of news!"
)
# Ordering certificates.  The single-winner form is committable *only* when the
# events sit on different dates: the certificate keeps the date and drops the
# time of day, so a same-day pair is a coin flip the reader usually wins.
DISTINCT_CTX = (
    "[Temporal Ordering: 'cousin's wedding' occurred on 2023-06-15. 'michael's "
    "engagement party' occurred on 2023-05-06. The event that happened first is "
    "'michael's engagement party'.]\n"
    "[answer_1 on 2023/05/06 (Sat) 10:00] user: Congratulations!"
)
SAME_DAY_CTX = (
    "[Temporal Ordering: 'samsung galaxy s22' occurred on 2023-03-15. 'dell xps 13' "
    "occurred on 2023-03-15. The event that happened first is 'dell xps 13'.]\n"
    "[answer_1 on 2023/03/15 (Wed) 10:00] user: Both arrived today."
)
UNKNOWN_WINNER_CTX = (
    "[Temporal Ordering: 'a' occurred on 2023-01-01. 'b' occurred on 2023-02-02. The "
    "event that happened first is 'c'.]\n"
    "[answer_1 on 2023/01/01 (Sun) 10:00] user: Notes."
)
NUMBERED_CTX = (
    "[Chronological Order of Museums:\n1. On 2022-12-19: [c2c249ea on 2022/12/19 "
    "(Mon) 19:53] user: I've been trying to get into a consistent routine.\n2. On "
    "2022-12-22: [331197d5 on 2022/12/22 (Thu) 15:57] user: Thinking of a new case.]\n"
    "[answer_1 on 2023/01/01 (Sun) 10:00] user: Museum recap."
)


SAME_DAY_WITH_TIME_CTX = (
    "[Temporal Ordering: 'samsung galaxy s22' occurred on 2023-03-15 10:00. 'dell xps 13' "
    "occurred on 2023-03-15 14:30. The event that happened first is 'samsung galaxy s22'.]\n"
    "[answer_1 on 2023/03/15 (Wed) 10:00] user: Galaxy arrived in the morning."
)


def test_commits_a_winner_when_the_dates_differ() -> None:
    out = extract_certificate_answer(
        "Which event happened first, the wedding or the engagement party?",
        DISTINCT_CTX,
    )
    assert out.used, out.detail
    assert out.answer == "michael's engagement party"


def test_same_day_with_time_resolves_winner() -> None:
    """Both events are on 2023-03-15, but hours/minutes resolve the order."""
    out = extract_certificate_answer(
        "Which device did I get first, the Galaxy S22 or the Dell XPS?",
        SAME_DAY_WITH_TIME_CTX,
    )
    assert out.used, out.detail
    assert out.answer == "samsung galaxy s22"


def test_same_day_tie_is_left_to_the_reader() -> None:
    """Both events are 2023-03-15 without times, so the certificate cannot order them."""
    out = extract_certificate_answer(
        "Which device did I get first, the Galaxy S22 or the Dell XPS?", SAME_DAY_CTX
    )
    assert not out.used
    assert "tie" in out.detail or "no committable" in out.detail


def test_winner_outside_the_dated_events_is_left_to_the_reader() -> None:
    out = extract_certificate_answer("Which happened first?", UNKNOWN_WINNER_CTX)
    assert not out.used


def test_numbered_ordering_dump_is_left_to_the_reader() -> None:
    """The numbered entries are not reliably the events the question asks about."""
    out = extract_certificate_answer(
        "What is the order of the museums I visited?", NUMBERED_CTX
    )
    assert not out.used


def test_absent_certificate_abstains() -> None:
    out = extract_certificate_answer("How many days passed?", LME_CTX)
    assert not out.used
    assert out.answer == ""


def test_commits_the_interval_the_runtime_already_computed() -> None:
    out = extract_certificate_answer(
        "How many days passed between the museum and the exhibit?", CALC_CTX
    )
    assert out.used, out.detail
    assert out.answer == "7 days"
    assert out.source == "temporal_certificate"


def test_commits_a_weeks_ago_interval() -> None:
    out = extract_certificate_answer("How many weeks ago did I meet my aunt?", AGO_CTX)
    assert out.used, out.detail
    assert out.answer == "4 weeks"


def test_commits_an_ordered_enumeration() -> None:
    out = extract_certificate_answer("Which three events happened in order?", ORDERING_CTX)
    assert out.used, out.detail
    assert out.answer.lower().startswith("first, i helped my friend")


# ---------------------------------------------------------------- determinism
# The module promises "same input, same output" (no LLM, no network, no
# randomness).  Set iteration order is *not* part of that promise: it changes with
# PYTHONHASHSEED, and two tie-breaks used to read it.  The same command therefore
# scored 77, 78, 79 or 80 of the 321 LoCoMo temporal questions depending on the
# process, which made the scorecard's `commit_correct` a random variable.  These
# tests pin both tie-breaks and then the end-to-end property.

#: ``cathedral`` and ``observatory`` appear in exactly one turn each, so their IDF
#: weights are identical - only the tie-break decides which turn may answer.  The
#: seven shared words keep the winning turn's score above ``MIN_TURN_SCORE`` so the
#: choice is observable as an answer ("4 April" vs "5 April") rather than as an
#: abstention both ways.
TIE_CTX = "\n".join([
    "[D1:4 on 11:00 am on 6 April, 2023] Alice: The morning visit with the guide, "
    "the tickets, the photos, the queue and the lunch was the cathedral, on 4 April 2023.",
    "[D2:5 on 11:00 am on 6 April, 2023] Alice: The morning visit with the guide, "
    "the tickets, the photos, the queue and the lunch was the observatory, on 5 April 2023.",
])
TIE_QUESTION = ("The morning visit with the guide, the tickets, the photos, the queue and "
                "the lunch: was it the cathedral or the observatory?")


def test_key_terms_resolve_weight_ties_by_name() -> None:
    turn_list = parse_turns(TIE_CTX)
    weights = committer._token_idf(turn_list)
    df = committer._document_frequency(turn_list)
    q_words = committer.content_words(TIE_QUESTION) - {"knew", "know", "happen", "happened"}

    keys = committer._key_terms(q_words, weights, df, len(turn_list))

    assert weights["cathedral"] == weights["observatory"]
    assert keys == ["cathedral", "observatory"]


def test_anchor_position_ignores_the_order_of_the_matched_words() -> None:
    """The anchor must be a property of the text, not of set iteration order."""
    text = "we met the observatory guide, then the cathedral choir"
    weights = {"cathedral": 0.9, "observatory": 0.9}  # a tie, so only the order can decide
    forward = committer._anchor_position(text, ["cathedral", "observatory"], weights)
    backward = committer._anchor_position(text, ["observatory", "cathedral"], weights)
    assert forward == backward == text.index("cathedral")


def test_a_tied_question_answers_identically_under_any_hash_seed() -> None:
    """End-to-end: four processes, four hash seeds, one answer."""
    script = (
        "from artificial_memory.skills.answer_committer import extract_temporal_answer\n"
        f"out = extract_temporal_answer({TIE_QUESTION!r}, {TIE_CTX!r})\n"
        "print(out.used, out.answer, out.confidence)\n"
    )
    answers = set()
    for seed in ("0", "1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=env,
                              capture_output=True, text=True, check=True)
        answers.add(proc.stdout.strip())

    assert answers == {"True 4 April 2023 0.91"}

