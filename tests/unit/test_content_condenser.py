"""Unit tests for the deterministic content condenser.

The invariant that matters most: the ``[D7:22 ...]`` provenance label and the
speaker name must survive condensation, because the Layer-1 oracle predicate is
a substring test for that label inside ``context_text``.
"""

from artificial_memory.context.content_condenser import (
    CondenseOptions,
    compact_header,
    condense_turn,
    drop_redundant_quotes,
)

HEADER = "[D7:22 on 8:56 pm on 20 July, 2023] "
DROP = CondenseOptions(
    strip_greetings=False,
    strip_signoffs=False,
    strip_fillers=False,
    condense_agreements=False,
    quote_mode="keep",
)


def test_greeting_stripped_from_opening():
    out = condense_turn(HEADER + "Caroline: Hey Mel! A lot has happened since we last chatted.")
    assert out.startswith(HEADER)
    assert "Hey Mel!" not in out
    assert "happened since we last chatted" in out


def test_signoff_stripped_from_closing():
    out = condense_turn(HEADER + "Caroline: I joined a new group. Have a great day!")
    assert out.endswith("group.")
    assert "Have a great day" not in out


def test_conversational_filler_stripped():
    out = condense_turn(HEADER + "Caroline: Well, I did join a new group last week.")
    assert out.startswith(HEADER)
    assert not out.split("] ", 1)[1].startswith("Well")


def test_header_and_speaker_survive_every_stripper():
    out = condense_turn(
        HEADER + 'Caroline: Hey Mel! To be honest, I joined a group. Take care!',
        CondenseOptions(quote_mode="drop"),
    )
    assert out.startswith(HEADER)
    assert "Caroline:" in out
    assert "joined a group" in out


def test_agreement_only_utterance_is_shortened_not_deleted():
    out = condense_turn(
        HEADER + "Melanie: Yeah, absolutely! That sounds really awesome and I totally agree with you on that."
    )
    assert out.startswith(HEADER)
    assert "That sounds really awesome" not in out
    assert len(out.split()) < 12


def test_quote_mode_keep_is_verbatim():
    text = HEADER + '(In reply to Melanie: "Hey Caroline!") Caroline: Hey Mel!'
    assert condense_turn(text, DROP) == text


def test_quote_mode_drop_removes_quote_keeps_body():
    text = HEADER + '(In reply to Melanie: "Hey Caroline!") Caroline: I joined a new group.'
    out = condense_turn(text, CondenseOptions(quote_mode="drop"))
    assert "(In reply to" not in out
    assert "Caroline: I joined a new group." in out
    assert out.startswith(HEADER)


def test_quote_mode_provenance_keeps_speaker_link():
    text = HEADER + '(In reply to Melanie: "Hey Caroline!") Caroline: I joined a new group.'
    out = condense_turn(text, CondenseOptions(quote_mode="provenance"))
    assert "(reply to Melanie)" in out
    assert "Hey Caroline!" not in out
    assert out.startswith(HEADER)


def test_emptied_body_keeps_provenance_header():
    out = condense_turn(HEADER + "Hey!", CondenseOptions(quote_mode="drop"))
    assert out == HEADER.rstrip()
    assert "D7:22" in out


def test_emptied_body_without_header_is_dropped():
    assert condense_turn("Hey!", CondenseOptions(quote_mode="drop")) == ""


def test_default_options_do_not_change_text_when_nothing_matches():
    text = HEADER + "Caroline: I joined the Connected LGBTQ Activists group last Tuesday."
    assert condense_turn(text, DROP) == text


# --- Option 2: zero-loss compression ---------------------------------------

def test_compact_header_keeps_label_date_and_time():
    out = compact_header("[D10:3 on 8:56 pm on 20 July, 2023] Caroline: Hi")
    assert out.startswith("[D10:3 20 July, 2023 20:56]")
    assert "D10:3" in out
    assert "Caroline: Hi" in out
    # the date must stay in the natural form the ground truths use
    assert "2023-07-20" not in out


def test_compact_header_handles_am_midnight_and_day_padding():
    assert compact_header("[D4:11 on 10:37 am on 27 June, 2023] x").startswith(
        "[D4:11 27 June, 2023 10:37]"
    )
    assert compact_header("[D16:9 on 12:09 am on 13 September, 2023] x").startswith(
        "[D16:9 13 September, 2023 00:09]"
    )
    assert compact_header("[D1:1 on 12:00 pm on 1 May, 2023] x").startswith(
        "[D1:1 1 May, 2023 12:00]"
    )


def test_compact_header_is_noop_on_unrecognised_header():
    odd = "[unexpected-provenance] x"
    assert compact_header(odd) == odd
    assert compact_header("[D1:1 on 8:56 pm on 20 Foobar, 2023] x").startswith("[D1:1 ")


def test_compact_header_is_off_by_default():
    assert condense_turn(HEADER + "Caroline: Hi", DROP).startswith(
        "[D7:22 on 8:56 pm on 20 July, 2023]"
    )


LONG_QUOTE = "I went to the pride parade last weekend with all of my friends"
A = f"[D1:1 on 9:00 am on 1 May, 2023] Melanie: {LONG_QUOTE} and it was amazing."
B = (
    '[D1:2 on 9:01 am on 1 May, 2023] (In reply to Melanie: "'
    f'{LONG_QUOTE} and it was amazing.") Caroline: That sounds wonderful!'
)
C = (
    '[D2:1 on 9:00 am on 2 May, 2023] (In reply to Caroline: '
    '"I adopted a brand new dog named Buster from the shelter yesterday.") '
    "Melanie: Congratulations on the new dog!"
)


def test_redundant_mode_is_a_noop_at_line_level():
    assert condense_turn(B, CondenseOptions(quote_mode="redundant")) == B


def test_drop_redundant_quotes_removes_payload_present_elsewhere():
    out = drop_redundant_quotes([A, B])
    assert "(In reply to" not in out[1]
    assert "That sounds wonderful!" in out[1]
    assert out[0] == A


def test_drop_redundant_quotes_keeps_payload_found_nowhere_else():
    out = drop_redundant_quotes([A, C])
    assert "(In reply to" in out[1]
    assert "Buster" in out[1]


def test_drop_redundant_quotes_keeps_short_quotes_even_if_repeated():
    short_body = "[D3:1 on 9:00 am on 3 May, 2023] Melanie: Yeah, exactly right."
    short_quote = (
        '[D3:2 on 9:01 am on 3 May, 2023] (In reply to Melanie: "Yeah, exactly") '
        "Caroline: Thanks!"
    )
    out = drop_redundant_quotes([short_body, short_quote])
    assert "(In reply to" in out[1]
