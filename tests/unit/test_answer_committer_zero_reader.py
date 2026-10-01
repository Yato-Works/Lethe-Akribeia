"""Unit tests for expanded zero-reader committer skills (P2).

Tests:
1. commit_derivation_scaffold (COUNT / SUM / AGGREGATION bypass)
2. commit_single_hop_fact (Occupation, Residence, Preferences, Kinship/Pet Names)
3. commit_answer unified dispatcher
4. Preservation of abstention on ambiguous context
"""

from __future__ import annotations

import pytest
from artificial_memory.skills.answer_committer import (
    commit_answer,
    commit_derivation_scaffold,
    commit_single_hop_fact,
)


def test_commit_derivation_scaffold_count() -> None:
    context = (
        "[DerivationScaffold: COUNT=3, TARGET=cat]\n"
        "[D1:1 on 10:00 am on 5 May 2023] Alice: I have three rescue cats at home."
    )
    q = "How many cats does Alice have?"
    ans = commit_derivation_scaffold(q, context)
    assert ans.used is True
    assert ans.answer == "3"
    assert ans.source == "derivation_scaffold"
    assert ans.confidence >= 0.90


def test_commit_derivation_scaffold_sum() -> None:
    context = (
        "[DerivationScaffold: SUM=150.0, TARGET=expense]\n"
        "[D1:2 on 11:00 am on 6 May 2023] Bob: I spent $50 on lunch and $100 on books."
    )
    q = "What was the total number of expense Bob incurred?"
    ans = commit_derivation_scaffold(q, context)
    assert ans.used is True
    assert ans.answer == "150"
    assert ans.source == "derivation_scaffold"


def test_commit_derivation_scaffold_abstains_on_unrelated_query() -> None:
    context = "[DerivationScaffold: COUNT=3, TARGET=cat]\nAlice: I love my cats."
    q = "Where does Alice go to school?"
    ans = commit_derivation_scaffold(q, context)
    assert ans.used is False


def test_commit_single_hop_fact_occupation() -> None:
    context = (
        "[D1:1 on 1:00 pm on 10 June 2023] Caroline: I work as an architect in Chicago.\n"
        "[D1:2 on 1:05 pm on 10 June 2023] Melanie: That sounds amazing!"
    )
    q = "What is Caroline's occupation?"
    ans = commit_single_hop_fact(q, context)
    assert ans.used is True
    assert "architect" in ans.answer.lower()
    assert ans.source == "single_hop_fact"


def test_commit_single_hop_fact_residence() -> None:
    context = (
        "[D2:1 on 2:00 pm on 12 June 2023] Melanie: I live in Seattle now.\n"
        "[D2:2 on 2:05 pm on 12 June 2023] Caroline: I love the Pacific Northwest."
    )
    q = "Where does Melanie live?"
    ans = commit_single_hop_fact(q, context)
    assert ans.used is True
    assert "Seattle" in ans.answer


def test_commit_single_hop_fact_favorite() -> None:
    context = (
        "[D3:1 on 3:00 pm on 15 June 2023] Caroline: My favorite food is sushi, hands down."
    )
    q = "What is Caroline's favorite food?"
    ans = commit_single_hop_fact(q, context)
    assert ans.used is True
    assert "sushi" in ans.answer.lower()


def test_commit_single_hop_fact_msc_triplet() -> None:
    context = (
        "[MSC Fact: (David, OCCUPATION, marine biologist)]\n"
        "[D4:1 on 4:00 pm on 20 June 2023] David: I study ocean reefs."
    )
    q = "What is David's profession?"
    ans = commit_single_hop_fact(q, context)
    assert ans.used is True
    assert ans.answer == "marine biologist"


def test_commit_answer_dispatcher_priority() -> None:
    # 1. Temporal certificate wins when present
    cert_ctx = (
        "[Temporal Calculation: Event 1 occurred on 2023-01-08. Exactly 14 days passed.]\n"
        "[DerivationScaffold: COUNT=2, TARGET=event]\n"
        "[D1:1 on 1 Jan 2023] User: Went on a trip."
    )
    ans1 = commit_answer("How much time passed?", cert_ctx)
    assert ans1.used is True
    assert ans1.source == "temporal_certificate"
    assert ans1.answer == "14 days"

    # 2. Derivation scaffold wins on count question without temporal cert
    scaff_ctx = (
        "[DerivationScaffold: COUNT=5, TARGET=books]\n"
        "[D2:1 on 2 Feb 2023] Alice: Read 5 books."
    )
    ans2 = commit_answer("How many books did Alice read?", scaff_ctx)
    assert ans2.used is True
    assert ans2.source == "derivation_scaffold"
    assert ans2.answer == "5"

    # 3. Single hop fact wins on occupation query
    fact_ctx = (
        "[D3:1 on 3 Mar 2023] Bob: I work as a doctor at St. Jude."
    )
    ans3 = commit_answer("What is Bob's job?", fact_ctx)
    assert ans3.used is True
    assert ans3.source == "single_hop_fact"
    assert "doctor" in ans3.answer.lower()
