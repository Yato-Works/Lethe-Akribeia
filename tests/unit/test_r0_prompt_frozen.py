"""R0 must stay byte-identical to the frozen V4 baseline prompt.

The Reader x Memory Engine experiments change ONE variable at a time.
``build_prompt(..., cond="R0")`` is the frozen reference: the golden strings
below are copied verbatim from the original inline prompt construction of
``eval_v4_pipeline_dev.py``.  If any edit changes an R0 prompt byte, this
test fails before a single benchmark token is spent.

R1 may only PERMUTE evidence blocks: same bytes, same multiset of blocks,
preamble untouched.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "eval_v4_pipeline_dev",
    str(REPO / "scripts" / "benchmarks" / "eval_v4_pipeline_dev.py"),
)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

Q_DATE = "2023/06/15"
CTX = "[CTX]"
Q = "What did I buy?"


class _Plan:
    def __init__(self, preference_demand: float) -> None:
        self.preference_demand = preference_demand


# ---- golden R0 prompts (verbatim copies of the frozen baseline) -----------

GOLDEN_PREFERENCE = (
    "[INSTRUCTION: PERSONALIZED RECOMMENDATIONS]\n"
    "You are a helpful assistant providing personalized recommendations based on the user's past conversations.\n"
    "RULES:\n"
    "1. The user previously discussed this topic in their past conversations. You MUST identify the specific past experiences, pets, past recipes, venues, favorite artists, or background they mentioned.\n"
    "2. Explicitly build upon or incorporate their specific past mentions into your suggestions (e.g. mention their pet by name, build upon a past recipe they made, reference venues/artists from their past experiences, or account for their equipment/classes).\n"
    "3. DO NOT give generic tourist or generic hobby advice. Anchor directly in their personal history from the evidence.\n"
    "4. Never say 'I don't know'.\n\n"
    f"Question date: {Q_DATE}\n\n"
    f"{CTX}"
)

GOLDEN_AGGREGATION = (
    "[INSTRUCTION: EXHAUSTIVE AGGREGATION & COMPARISON]\n"
    "A memory system has retrieved ALL available evidence across multiple sessions.\n"
    "You must count, sum, or compute the answer by examining EVERY piece of evidence.\n"
    "RULES:\n"
    "1. First, locate and list ALL individual amounts, items, or instances mentioned across all sessions (including multiple items or costs mentioned within the SAME session).\n"
    "2. If the question asks 'how many' or for a total amount: add up ALL amounts or count ALL items across all sessions (e.g. '10', '$75', '$450').\n"
    "3. If the question asks for a difference, comparison, or margin (e.g. 'difference in price/cost', 'how much faster/slower', 'how many years older/younger', 'how much more/less compared to'): find the two relevant values and calculate the difference (A - B = difference), NOT just the single amount or the sum!\n"
    "4. If multiple costs or items are mentioned in the evidence (e.g. vet visit $50 and medication $25), add them together: $50 + $25 = $75.\n"
    "5. DO NOT stop at the first amount or first session. Check EVERY detail in the evidence.\n"
    "6. NEVER say 'I don't know'. Compute the answer directly from the evidence available.\n\n"
    f"Question date: {Q_DATE}\n\n"
    f"{CTX}\n\n"
    "Now, enumerate the relevant values and state the final computed answer directly:"
)

GOLDEN_EVIDENCE_FIRST = (
    "[INSTRUCTION: EVIDENCE-FIRST ANSWERING]\n"
    "A memory system has already retrieved the verified evidence below for this question.\n"
    "The answer IS present in this evidence.\n"
    "RULES:\n"
    "1. Locate the specific fact the question asks about and state it directly.\n"
    "2. Reply with the final value only: a few words or one short sentence.\n"
    "3. NEVER reply \"I don't know\", \"not enough information\", \"not mentioned\", or any other refusal.\n"
    "4. For coupon redemptions, transactions, or shopping: name the merchant or store (e.g. Target, Amazon), NOT an email inbox.\n"
    "5. If the question asks for current / new / updated (job, address, preference, status), provide the latest one. If the question asks for previous / earlier / former, provide the previous one from before the transition.\n"
    "6. For questions asking 'Did I...', 'Was I...', 'Have I...': answer 'Yes' or 'No' directly.\n"
    "7. For questions asking 'How much more / less / faster / slower compared to...': calculate and state the difference (e.g. A - B = difference), NOT just the raw single amount.\n"
    "8. Use only the evidence; do not invent details that are not in it.\n\n"
    f"Question date: {Q_DATE}\n\n"
    f"{CTX}"
)


def test_r0_preference_prompt_frozen() -> None:
    out = ev.build_prompt("R0", _Plan(0.5), False, CTX, Q, Q_DATE)
    assert out == GOLDEN_PREFERENCE


def test_r0_aggregation_prompt_frozen() -> None:
    out = ev.build_prompt("R0", _Plan(0.0), True, CTX, Q, Q_DATE)
    assert out == GOLDEN_AGGREGATION


def test_r0_evidence_first_prompt_frozen() -> None:
    out = ev.build_prompt("R0", _Plan(0.0), False, CTX, Q, Q_DATE)
    assert out == GOLDEN_EVIDENCE_FIRST


def test_r0_uses_compiled_context_verbatim() -> None:
    """R0 must never reorder, strip, or rewrite the compiled context."""
    ctx = "HEAD\n\n--- Event 1 ---\nalpha\n\n--- Event 2 ---\nbeta"
    out = ev.build_prompt("R0", _Plan(0.0), True, ctx, "beta question", Q_DATE)
    assert ctx in out


# ---- R1 invariants --------------------------------------------------------


def test_r1_preserves_content_multiset_and_head() -> None:
    """R1 may only permute blocks: identical bytes, preamble untouched."""
    blocks = [
        "[EXHAUSTIVE MULTI-SESSION EVIDENCE TIMELINE]",
        "--- Event 1 (2023/01/01) ---\n  User: I bought a guitar",
        "--- Event 2 (2023/02/01) ---\n  User: banana bread recipe",
        "--- Event 3 (2023/03/01) ---\n  User: unrelated chatter",
    ]
    text = "\n\n".join(blocks)
    out = ev.reorder_evidence_first(text, "banana bread")
    assert out.split("\n\n")[0] == blocks[0], "preamble must stay first"
    assert Counter(out.split("\n\n")) == Counter(blocks), "content must be preserved"
    # the banana block must be promoted to the top of the evidence section
    assert out.split("\n\n")[1] == blocks[2]


def test_r1_is_noop_without_lexical_signal() -> None:
    text = "HEAD\n\n--- Event 1 ---\nfoo\n\n--- Event 2 ---\nbar"
    assert ev.reorder_evidence_first(text, "zzzz unmatched") == text


def test_r1_is_noop_for_single_block_contexts() -> None:
    assert ev.reorder_evidence_first("[CTX only]", "anything") == "[CTX only]"


def test_r1_prompt_differs_only_in_context_order() -> None:
    ctx = (
        "[VERIFIED CONVERSATION EVIDENCE]\n\n"
        "[S1 on 2023/01/01 (Sun) 09:00] User: I went hiking near Red Rocks\n\n"
        "[S2 on 2023/02/01 (Wed) 09:00] User: I love banana smoothies"
    )
    r0 = ev.build_prompt("R0", _Plan(0.0), False, ctx, "banana smoothie", Q_DATE)
    r1 = ev.build_prompt("R1", _Plan(0.0), False, ctx, "banana smoothie", Q_DATE)
    prefix_r0 = r0.split("[VERIFIED")[0]
    prefix_r1 = r1.split("[VERIFIED")[0]
    assert prefix_r0 == prefix_r1, "instruction block must be identical"
    body_r0 = "[VERIFIED" + r0.split("[VERIFIED", 1)[1]
    body_r1 = "[VERIFIED" + r1.split("[VERIFIED", 1)[1]
    assert Counter(body_r0.split("\n\n")) == Counter(body_r1.split("\n\n"))
    assert re.search(r"banana", body_r1.split("\n\n")[1]), "relevant evidence moved up"


def test_unsupported_condition_is_rejected() -> None:
    assert "R0" in ev.SUPPORTED_CONDS and "R1" in ev.SUPPORTED_CONDS
    assert "R2" in ev.SUPPORTED_CONDS and "R3" in ev.SUPPORTED_CONDS
    assert "R23" in ev.SUPPORTED_CONDS
    assert "R4" not in ev.SUPPORTED_CONDS, "unimplemented conditions must not silently run"


# ---- R2 invariants --------------------------------------------------------


def test_r2_drops_zero_overlap_blocks_and_keeps_head_and_order() -> None:
    blocks = [
        "[VERIFIED CONVERSATION EVIDENCE]",
        "[S1 on 2023/01/01] User: I went banana shopping",
        "[S2 on 2023/02/01] User: unrelated chatter about weather",
        "[S3 on 2023/03/01] User: bought banana bread supplies",
        "[S4 on 2023/04/01] User: more banana stories with friends",
    ]
    text = "\n\n".join(blocks)
    out = ev.filter_evidence(text, "banana")
    kept = out.split("\n\n")
    assert kept[0] == blocks[0], "head must be kept"
    assert blocks[1] in kept and blocks[3] in kept and blocks[4] in kept
    assert blocks[2] not in kept, "zero-overlap block dropped"
    assert kept == [blocks[0], blocks[1], blocks[3], blocks[4]], "R0 order preserved"


def test_r2_safety_floor_never_starves_small_contexts() -> None:
    """< MIN_EVIDENCE_BLOCKS relevant -> earliest drops are reinstated."""
    blocks = [
        "HEAD",
        "[S1 on 2023/01/01] User: I went banana shopping",
        "[S2 on 2023/02/01] User: unrelated chatter about weather",
        "[S3 on 2023/03/01] User: gardening notes for the week",
    ]
    text = "\n\n".join(blocks)
    # only 1 relevant evidence block < floor 3 -> nothing may be starved
    assert ev.filter_evidence(text, "banana") == text


def test_r2_is_noop_when_no_block_is_relevant() -> None:
    text = "HEAD\n\n--- Event 1 ---\nfoo\n\n--- Event 2 ---\nbar"
    assert ev.filter_evidence(text, "zzzz unmatched") == text


def test_r2_is_noop_when_every_block_is_relevant() -> None:
    text = "HEAD\n\n--- Event 1 ---\nbanana one\n\n--- Event 2 ---\nbanana two"
    assert ev.filter_evidence(text, "banana") == text


def test_r2_prompt_is_a_subset_of_r0_content() -> None:
    ctx = (
        "[VERIFIED CONVERSATION EVIDENCE]\n\n"
        "[S1 on 2023/01/01] User: I went hiking near Red Rocks\n\n"
        "[S2 on 2023/02/01] User: I love banana smoothies\n\n"
        "[S3 on 2023/03/01] User: more banana bread talk with friends\n\n"
        "[S4 on 2023/04/01] User: unrelated weather chatter about rain"
    )
    r0 = ev.build_prompt("R0", _Plan(0.0), False, ctx, "banana smoothie", Q_DATE)
    r2 = ev.build_prompt("R2", _Plan(0.0), False, ctx, "banana smoothie", Q_DATE)
    assert r2 != r0
    # every block of R2 must appear verbatim inside R0 (content-reducing only)
    assert Counter(r0.split("\n\n")) >= Counter(r2.split("\n\n"))
    assert "banana smoothies" in r2
    # the zero-overlap tail block is dropped (floor = 3, relevant = 2 -> +1 kept)
    assert "unrelated weather chatter" not in r2
