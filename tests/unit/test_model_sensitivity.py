"""Unit tests for the model-sensitivity pairing (§7 of the Apex scorecard).

These pin the contract of the *analysis* layer, not the readers: two runs of one
slice that differ only in ``--model`` are joined question by question, every
question lands in exactly one of four buckets, and the report must refuse to
pretend a pairing is clean when retrieval drifted underneath it.  The scorecard
half of the feature is covered here too, because §7 has to obey the §6 rule:
raw counts in, every ratio re-derived, "NOT MEASURED" when there is no
measurement.

Both files are scripts rather than library modules, so they are loaded by path.
"""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
import tempfile
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "benchmarks"


def load_script(name: str, path: Path):
    """Import a standalone script as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ms = load_script("_model_sensitivity_under_test", SCRIPTS / "model_sensitivity.py")
sc = load_script("_apex_scorecard_section7_under_test", SCRIPTS / "apex_scorecard.py")


@contextmanager
def temp_dir():
    """Scratch directory for synthetic runs (same rationale as the §6 tests)."""
    path = Path(tempfile.mkdtemp(prefix="model-sensitivity-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def row(qid: str, correct: bool, *, tokens: int = 100, oracle: bool = True,
        category: int = 2, question_type: str | None = None) -> dict:
    """One result entry in the shape the runners write."""
    entry = {"question_id": qid, "is_correct": correct, "tokens": tokens,
             "oracle_recall": oracle, "prediction": "x"}
    if question_type is None:
        entry["category"] = category
    else:
        entry["question_type"] = question_type
    return entry


def write_run(path: Path, rows: list[dict], model: str) -> str:
    path.write_text(json.dumps({"model": model, "n_questions": len(rows), "results": rows}),
                    encoding="utf-8")
    return str(path)


def pair(tmp: Path, a_rows: list[dict], b_rows: list[dict], claims: dict | None = None) -> dict:
    """Pair two synthetic runs written into ``tmp``."""
    a = ms.load_run(write_run(tmp / "a.json", a_rows, "big-reader"))
    b = ms.load_run(write_run(tmp / "b.json", b_rows, "small-reader"))
    return ms.pair_runs(a, b, claims)


def test_every_question_lands_in_exactly_one_bucket() -> None:
    with temp_dir() as tmp:
        analysis = pair(
            tmp,
            [row("q1", True), row("q2", True), row("q3", False), row("q4", False)],
            [row("q1", True), row("q2", False), row("q3", True), row("q4", False)],
        )
    counts = analysis["counts"]
    assert counts == {"n": 4, "both_correct": 1, "a_only": 1, "b_only": 1, "both_wrong": 1}
    assert ms.accuracy_a(counts) == 0.5
    assert ms.accuracy_b(counts) == 0.5
    assert ms.delta_pp(counts) == 0.0
    # Without a claims file there is no zone split to report, and no invariant.
    assert analysis["zones"] == {}
    assert analysis["controls"]["deterministic_zone_identical"] is None


def test_mcnemar_is_the_exact_binomial_tail() -> None:
    # 8 vs 2 discordant questions: 2 * (C(10,0) + C(10,1) + C(10,2)) / 2^10.
    assert ms.mcnemar_exact(8, 2) == 2 * (1 + 10 + 45) / 1024
    assert ms.mcnemar_exact(0, 0) == 1.0
    assert ms.mcnemar_exact(5, 5) == 1.0  # perfectly even split: no evidence either way
    assert ms.mcnemar_exact(20, 0) < 0.0001
    # Symmetric in its arguments: the test is two-sided.
    assert ms.mcnemar_exact(9, 1) == ms.mcnemar_exact(1, 9)


def test_the_scorecard_and_the_tool_agree_on_the_p_value() -> None:
    """§7 re-derives the p-value rather than trusting the artefact; pin the copies."""
    for a_only, b_only in [(8, 2), (3, 7), (0, 0), (12, 1), (4, 4), (30, 12)]:
        assert sc.mcnemar_exact(a_only, b_only) == ms.mcnemar_exact(a_only, b_only)


def test_retrieval_drift_is_detected_in_both_counters() -> None:
    """Token counts and oracle recall are runtime-side, so they must match exactly."""
    with temp_dir() as tmp:
        identical = pair(tmp,
                         [row("q1", True), row("q2", False)],
                         [row("q1", True), row("q2", True)])
        drifted = pair(tmp,
                       [row("q1", True, tokens=100), row("q2", False, tokens=100, oracle=True)],
                       [row("q1", True, tokens=100), row("q2", True, tokens=180, oracle=False)])
    assert identical["controls"]["token_mismatch"]["n"] == 0
    assert identical["controls"]["oracle_mismatch"]["n"] == 0
    assert drifted["controls"]["token_mismatch"] == {"n": 1, "sample": ["q2"]}
    assert drifted["controls"]["oracle_mismatch"] == {"n": 1, "sample": ["q2"]}
    section = ms.render_report("synthetic", {"model": "big"}, {"model": "small"}, drifted)
    assert "!! 1 retrieval token counts differ: q2" in section


def test_zones_split_the_committer_questions_from_the_reader_ones() -> None:
    """The delta has to be locatable: code answers q1, the reader answers q2."""
    with temp_dir() as tmp:
        a_rows = [row("q1", True), row("q2", True)]
        b_rows = [row("q1", True), row("q2", False)]
        for rows in (a_rows, b_rows):
            rows[0]["answer_source"] = "committed: certificate"
        analysis = pair(tmp, a_rows, b_rows, claims={"q1": True, "q2": False})
    deterministic = analysis["zones"]["deterministic"]
    llm_zone = analysis["zones"]["llm"]
    assert deterministic["n"] == 1 and deterministic["a_only"] == 0
    assert llm_zone["n"] == 1 and llm_zone["a_only"] == 1
    # The whole delta lives in the zone the reader owns.
    assert ms.delta_pp(deterministic) == 0.0
    assert ms.delta_pp(llm_zone) == 100.0
    # Only a pairing where *both* arms ran the committer can assert the invariant.
    assert analysis["zone_mode"] == "system"
    assert analysis["controls"]["deterministic_zone_identical"] is True


def test_a_moving_deterministic_zone_is_a_bug_not_a_result() -> None:
    with temp_dir() as tmp:
        a_rows = [row("q1", True), row("q2", True)]
        b_rows = [row("q1", False), row("q2", True)]
        for rows in (a_rows, b_rows):
            rows[0]["answer_source"] = "committed: certificate"
        analysis = pair(tmp, a_rows, b_rows, claims={"q1": True, "q2": False})
    assert analysis["zone_mode"] == "system"
    assert analysis["controls"]["deterministic_zone_identical"] is False
    section = ms.render_report("synthetic", {"model": "big"}, {"model": "small"}, analysis)
    assert "Deterministic zone MOVED between readers" in section
    assert "determinism bug" in section


def test_movement_in_a_counterfactual_zone_is_reader_behaviour_not_a_bug() -> None:
    """Neither arm committed: the reader answered those questions, so a flip there
    is the reader being a reader - calling it a determinism bug would be wrong."""
    with temp_dir() as tmp:
        analysis = pair(
            tmp,
            [row("q1", True), row("q2", True)],
            [row("q1", False), row("q2", True)],
            claims={"q1": True, "q2": False},
        )
    assert analysis["zone_mode"] == "counterfactual"
    assert analysis["controls"]["deterministic_zone_identical"] is None
    section = ms.render_report("synthetic", {"model": "big"}, {"model": "small"}, analysis)
    assert "determinism bug" not in section
    assert "Zone mode: counterfactual" in section


def test_a_question_outside_the_claims_file_counts_as_reader_answered() -> None:
    """Only the committer may skip the reader, so an unknown question is not free."""
    with temp_dir() as tmp:
        analysis = pair(
            tmp,
            [row("q1", True), row("q2", True)],
            [row("q1", True), row("q2", True)],
            claims={"q1": True},
        )
    assert analysis["zones"]["deterministic"]["n"] == 1
    assert analysis["zones"]["llm"]["n"] == 1
    assert analysis["controls"]["claims_missing"] == {"n": 1, "sample": ["q2"]}


def test_runs_with_different_question_sets_are_reported_not_padded() -> None:
    with temp_dir() as tmp:
        analysis = pair(
            tmp,
            [row("q1", True), row("q2", True), row("q3", True)],
            [row("q1", True), row("q2", True)],
        )
    assert analysis["shared"] == 2 and analysis["counts"]["n"] == 2
    assert analysis["controls"]["only_in_a"] == {"n": 1, "sample": ["q3"]}
    assert analysis["controls"]["only_in_b"]["n"] == 0


def test_groups_follow_the_longmemeval_question_type() -> None:
    with temp_dir() as tmp:
        analysis = pair(
            tmp,
            [row("q1", True, question_type="temporal-reasoning"),
             row("q2", True, question_type="multi-session")],
            [row("q1", False, question_type="temporal-reasoning"),
             row("q2", True, question_type="multi-session")],
        )
    groups = {g["name"]: g["counts"] for g in analysis["groups"]}
    assert set(groups) == {"temporal-reasoning", "multi-session"}
    assert groups["temporal-reasoning"]["a_only"] == 1
    assert groups["multi-session"]["a_only"] == 0


def test_main_writes_a_counts_only_artefact_and_survives_a_missing_claims_file() -> None:
    with temp_dir() as tmp:
        a_path = write_run(tmp / "a.json", [row("q1", True), row("q2", False)], "big-reader")
        b_path = write_run(tmp / "b.json", [row("q1", False), row("q2", False)], "small-reader")
        out = tmp / "artefact.json"
        argv = sys.argv
        sys.argv = ["model_sensitivity.py", "--a", a_path, "--b", b_path, "--label", "synthetic",
                    "--claims", str(tmp / "missing.json"), "--out", str(out)]
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                code = ms.main()
        finally:
            sys.argv = argv
        printed = buffer.getvalue()
        payload = json.loads(out.read_text(encoding="utf-8"))

    assert code == 0
    assert "!! missing claims file" in printed and "synthetic" in printed
    # Counts only: no percentage is stored, so a later aggregator must re-derive it.
    assert set(payload["counts"]) == {"n", "both_correct", "a_only", "b_only", "both_wrong"}
    assert payload["counts"] == {"n": 2, "both_correct": 0, "a_only": 1, "b_only": 0,
                                 "both_wrong": 1}
    assert payload["zones"] == {} and payload["claims"] is None
    assert payload["a"]["model"] == "big-reader"


def test_the_offline_cache_is_checked_against_what_the_runs_retrieved() -> None:
    """A claims file produced from a cache replay has to agree with the runs' oracle."""
    with temp_dir() as tmp:
        a = ms.load_run(write_run(tmp / "a.json", [row("q1", True), row("q2", True)], "big"))
        b = ms.load_run(write_run(tmp / "b.json", [row("q1", True), row("q2", True)], "small"))
        drifted = ms.pair_runs(a, b, {"q1": True, "q2": False}, {"q1": True, "q2": False})
        agreeing = ms.pair_runs(a, b, {"q1": True}, {"q1": True})

    assert drifted["controls"]["claims_oracle_compared"] == 2
    assert drifted["controls"]["claims_oracle_mismatch"] == {"n": 1, "sample": ["q2"]}
    section = ms.render_report("synthetic", {"model": "big"}, {"model": "small"}, drifted)
    assert "disagrees with what the runs retrieved on 1 of 2 questions" in section
    # With nothing to compare the control stays quiet instead of inventing a verdict.
    assert agreeing["controls"]["claims_oracle_compared"] == 1
    assert agreeing["controls"]["claims_oracle_mismatch"]["n"] == 0
    assert "agrees with the runs' own oracle recall" in ms.render_report(
        "synthetic", {"model": "big"}, {"model": "small"}, agreeing)


def test_a_reader_only_pairing_labels_the_zone_counterfactual() -> None:
    """Reader-only arms: the split is the deployed system's, the accuracy is not."""
    with temp_dir() as tmp:
        analysis = pair(tmp, [row("q1", True)], [row("q1", True)], claims={"q1": True})
    assert analysis["zone_mode"] == "counterfactual"
    assert analysis["zones_committed_in_run"] == 0
    section = ms.render_report("synthetic", {"model": "big"}, {"model": "small"}, analysis)
    assert "Zone mode: counterfactual" in section


def test_a_run_that_recorded_its_own_committer_names_the_system_zone() -> None:
    """When the arm really skipped the reader, the zone belongs to the system."""
    big = [row("q1", True), row("q2", False)]
    small = [row("q1", True), row("q2", False)]
    for rows in (big, small):
        rows[0]["answer_source"] = "committed: certificate"
    with temp_dir() as tmp:
        analysis = pair(tmp, big, small)
    assert analysis["claims_source"] == "run:answer_source (a)"
    assert analysis["zone_mode"] == "system"
    assert analysis["zones"]["deterministic"]["n"] == 1
    assert analysis["controls"]["answer_source_compared"] is True


def test_arms_that_disagree_about_who_answered_are_flagged() -> None:
    """The committer never calls the model, so it must claim the same questions."""
    with temp_dir() as tmp:
        big = [row("q1", True), row("q2", True)]
        small = [row("q1", True), row("q2", True)]
        big[0]["answer_source"] = "committed: certificate"
        big[1]["answer_source"] = "committed: certificate"
        small[0]["answer_source"] = "committed: certificate"
        small[1]["answer_source"] = "reader"
        analysis = pair(tmp, big, small)
    assert analysis["controls"]["answer_source_compared"] is True
    assert analysis["controls"]["answer_source_mismatch"] == {"n": 1, "sample": ["q2"]}
    section = ms.render_report("synthetic", {"model": "big"}, {"model": "small"}, analysis)
    assert "questions answered by a different engine in the two arms: q2" in section


def test_one_committing_arm_is_mixed_not_a_reader_swap() -> None:
    """Only an arm pair that *both* ran the committer can be compared engine-wise;
    one committing arm against one reader arm is the committer's own A/B."""
    with temp_dir() as tmp:
        big = [row("q1", True)]
        big[0]["answer_source"] = "committed: certificate"
        analysis = pair(tmp, big, [row("q1", True)])
    assert analysis["zone_mode"] == "mixed"
    assert analysis["controls"]["answer_source_compared"] is False
    assert analysis["controls"]["answer_source_mismatch"]["n"] == 0
    section = ms.render_report("synthetic", {"model": "big"}, {"model": "small"}, analysis)
    assert "Zone mode: mixed" in section


def test_a_moved_anchor_is_not_a_changed_engine() -> None:
    """Both arms committed the same claim at a different turn/anchor: the *decision*
    held, only the extraction point moved - a different control, not the same one."""
    with temp_dir() as tmp:
        big = [row("q1", True), row("q2", True)]
        small = [row("q1", True), row("q2", True)]
        big[0]["answer_source"] = "committed: turn#0 score=1.00 anchor=5"
        small[0]["answer_source"] = "committed: turn#0 score=1.00 anchor=9"
        analysis = pair(tmp, big, small)
    assert analysis["controls"]["answer_source_mismatch"]["n"] == 0
    assert analysis["controls"]["committed_detail_mismatch"] == {"n": 1, "sample": ["q1"]}
    section = ms.render_report("synthetic", {"model": "big"}, {"model": "small"}, analysis)
    assert "questions committed by both arms at a different anchor: q1" in section
    assert "answered by a different engine" not in section


def test_a_superset_arm_is_an_exclusion_a_non_nested_set_is_a_warning() -> None:
    """Running an extra slice beside this one costs nothing; losing questions does."""
    with temp_dir() as tmp:
        nested = pair(tmp, [row("q1", True), row("q2", True)],
                      [row("q1", True), row("q2", True), row("q3", True)])
        torn = pair(tmp, [row("q1", True), row("q2", True)],
                    [row("q1", True), row("q3", True)])
    assert nested["controls"]["only_in_a"]["n"] == 0
    assert nested["controls"]["only_in_b"] == {"n": 1, "sample": ["q3"]}
    nested_report = ms.render_report("synthetic", {"model": "big"}, {"model": "small"}, nested)
    assert "B is a superset of A (+1 questions outside this slice)" in nested_report
    assert "!! 1 questions missing from A" not in nested_report
    torn_report = ms.render_report("synthetic", {"model": "big"}, {"model": "small"}, torn)
    assert "the arms are not nested" in torn_report


def test_restrict_both_reader_keeps_only_pure_reader_questions() -> None:
    """The repeat control: drop every question the committer touched, so what is
    left measures the reader against itself (or against its smaller sibling)."""
    with temp_dir() as tmp:
        big = [row("q1", True), row("q2", True)]
        small = [row("q1", False), row("q2", True)]
        for rows in (big, small):
            rows[1]["answer_source"] = "committed: turn#0 score=1.0 anchor=5"
        a = ms.load_run(write_run(tmp / "a.json", big, "big"))
        b = ms.load_run(write_run(tmp / "b.json", small, "small"))
        restricted = ms.pair_runs(a, b, {"q1": False, "q2": True},
                                  restrict_both_reader=True)
        full = ms.pair_runs(a, b, {"q1": False, "q2": True})
    assert restricted["counts"]["n"] == 1 and restricted["counts"]["a_only"] == 1
    assert full["counts"]["n"] == 2
    # Nothing commits after the filter, so the zone cannot claim code answered it.
    assert restricted["zone_mode"] == "counterfactual"


def test_a_missing_run_artefact_is_an_error_not_an_empty_measurement() -> None:
    with temp_dir() as tmp:
        real = write_run(tmp / "a.json", [row("q1", True)], "big")
        argv = sys.argv
        sys.argv = ["model_sensitivity.py", "--a", real, "--b", str(tmp / "absent.json")]
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                code = ms.main()
        finally:
            sys.argv = argv
    assert code == 2
    assert "!! missing run artefact" in buffer.getvalue()
