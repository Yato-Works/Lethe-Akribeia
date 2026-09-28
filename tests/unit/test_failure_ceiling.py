"""Unit tests for the failure-ceiling census (Step 3).

The tool's whole value is that it refuses to guess.  These tests pin the two
properties that make it trustworthy: the buckets **partition** the failures
exactly, and the questions whose cause cannot be told apart from stored evidence
stay in one `unresolved` bucket instead of being allocated to a hypothesis.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "benchmarks" / "failure_ceiling.py"


def load_script():
    spec = importlib.util.spec_from_file_location("_failure_ceiling_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fc = load_script()


def row(qid: str, correct: bool, *, oracle: bool = True, committed: bool = False,
        category: int = 2, ground_truth: str = "7 May 2023") -> dict:
    entry = {"question_id": qid, "is_correct": correct, "oracle_recall": oracle,
             "category": category, "ground_truth": ground_truth}
    entry["answer_source"] = "committed: turn#0" if committed else "reader"
    return entry


def run(rows: list[dict], model: str = "big") -> dict:
    data = {"model": model, "results": rows}
    path = Path(tempfile.mkdtemp(prefix="fc-")) / "run.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return fc.load_run(str(path))


def test_every_failure_lands_in_exactly_one_bucket() -> None:
    a = run([row("q1", True), row("q2", False), row("q3", False, oracle=False),
             row("q4", False, committed=True)])
    analysis = fc.classify(a)
    buckets = analysis["buckets"]
    assert sum(buckets.values()) == analysis["wrong"] == 3
    assert buckets["retrieval"] == 1
    assert buckets["commit"] == 1
    assert buckets["unresolved_reader_insensitive"] == 1
    assert analysis["correct"] == 1 and analysis["n"] == 4


def test_a_commit_without_evidence_is_a_commit_failure_not_a_retrieval_one() -> None:
    """A committer claim is an explicit promise to answer with no reader: if it
    is wrong, the runtime is at fault even when the evidence was never there."""
    a = run([row("q1", False, oracle=False, committed=True)])
    analysis = fc.classify(a)
    assert analysis["buckets"]["commit"] == 1
    assert analysis["buckets"]["retrieval"] == 0
    assert analysis["signals"]["commit_without_evidence"] == 1
    report = fc.render_report("synthetic", a, None, analysis)
    assert "over-claims" in report


def test_the_two_readers_decide_only_reader_dependent_questions() -> None:
    a = run([row("q1", True), row("q2", False), row("q3", False), row("q4", True)])
    b = run([row("q1", True), row("q2", True), row("q3", False), row("q4", False)], model="small")
    analysis = fc.classify(a, b)
    sig = analysis["signals"]
    assert sig["both_right"] == 1 and sig["reference_only"] == 1
    assert sig["other_reader_only"] == 1 and sig["neither_reader"] == 1
    assert sig["reader_determined"] == 2
    # The four cells account for every question exactly once.
    assert sig["both_right"] + sig["reference_only"] + sig["other_reader_only"] \
        + sig["neither_reader"] == analysis["n"]
    # And the second reader resolving a question is what makes it reader-sensitive.
    assert analysis["buckets"]["unresolved_reader_sensitive"] == 1
    assert analysis["buckets"]["unresolved_reader_insensitive"] == 1


def test_unresolved_is_reported_as_unresolved_and_never_allocated() -> None:
    """The point of the tool: no artefact separates B, C and E, so the report must
    say so instead of handing the mass to a hypothesis."""
    a = run([row("q1", False), row("q2", False)])
    report = fc.render_report("synthetic", a, None, fc.classify(a))
    assert "Unresolved, reader-insensitive" in report
    assert "indistinguishable from stored evidence" in report
    for hypothesis in ("context noise (B)", "reasoning failure (C)"):
        assert hypothesis in report  # named, but only as what it cannot separate
    assert "| Unresolved, reader-insensitive | 2" in report


def test_per_group_counts_add_up_to_the_overall_buckets() -> None:
    a = run([row("q1", False, category=1), row("q2", True, category=1),
             row("q3", False, category=2), row("q4", False, oracle=False, category=2)])
    analysis = fc.classify(a)
    per_group = {g["name"]: g["buckets"] for g in analysis["per_group"]}
    assert set(per_group) == {"multi-hop", "temporal"}
    # Each bucket partitions across the groups: the groups sum to the totals.
    for key, value in analysis["buckets"].items():
        assert sum(buckets[key] for buckets in per_group.values()) == value, key
    assert sum(sum(b.values()) for b in per_group.values()) == analysis["wrong"]


def test_the_ground_truth_signal_is_read_from_the_compiled_context() -> None:
    """`gt_verbatim` is the one stored signal that narrows B: the fact is either
    in the context verbatim or only in another wording."""
    a = run([row("q1", False, ground_truth="7 May 2023"),
             row("q2", False, ground_truth="Redmond, Washington")])
    cache = {"q1": {"context": "Caroline went on 7 May 2023 to the support group."},
             "q2": {"context": "The trip was somewhere in the Pacific Northwest."}}
    analysis = fc.classify(a, cache=cache)
    assert analysis["signals"]["gt_verbatim_in_context"] == 1
    # A retrieval failure is not counted here: the question never reached the reader.
    a2 = run([row("q3", False, oracle=False, ground_truth="7 May 2023")])
    assert fc.classify(a2, cache={"q3": {"context": "7 May 2023"}})["signals"][
        "gt_verbatim_in_context"] == 0
