"""Unit tests for §7 of the Apex scorecard (model sensitivity).

Companion to ``test_scorecard_committer.py``: §6 asks how much of a benchmark the
runtime can decide without an LLM, §7 asks whether the rest of the score survives
a weaker reader.  The reporting contract is the same as §6 - the artefacts store
raw 4-way counts, the scorecard re-derives every ratio from them, pools slices by
adding counts instead of averaging percentages, and says "NOT MEASURED" rather
than rendering an empty section.

The scorecard is a script, not a library module, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCORECARD = REPO / "scripts" / "benchmarks" / "apex_scorecard.py"
SENSITIVITY_DIR = REPO / "benchmark_results" / "model_sensitivity"


def load_scorecard():
    """Import ``scripts/benchmarks/apex_scorecard.py`` as a module."""
    spec = importlib.util.spec_from_file_location("_apex_scorecard_sensitivity_under_test",
                                                  SCORECARD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = load_scorecard()


@contextmanager
def temp_artefact_dir():
    """Scratch directory for synthetic artefacts (same rationale as the §6 tests)."""
    path = Path(tempfile.mkdtemp(prefix="model-sensitivity-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def clean_controls(**extra) -> dict:
    """The controls block a drift-free pairing writes."""
    controls = {
        "only_in_a": {"n": 0, "sample": []},
        "only_in_b": {"n": 0, "sample": []},
        "token_mismatch": {"n": 0, "sample": []},
        "oracle_mismatch": {"n": 0, "sample": []},
        "claims_missing": {"n": 0, "sample": []},
        "deterministic_zone_identical": None,
    }
    controls.update(extra)
    return controls


def artefact(counts: dict, **extra) -> dict:
    """One ``model_sensitivity/*.json`` payload in the shape the tool writes."""
    base = {
        "label": "synthetic",
        "a": {"run": "benchmark_results/a.json", "model": "qwen2.5:7b-instruct"},
        "b": {"run": "benchmark_results/b.json", "model": "qwen2.5:1.5b"},
        "claims": None,
        "shared": counts.get("n", 0),
        "counts": counts,
        "zones": {},
        "groups": [],
        "controls": clean_controls(),
        "mcnemar": {},
    }
    base.update(extra)
    return base


def write(directory: Path, name: str, payload: dict) -> None:
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def test_pairing_metrics_are_derived_from_counts() -> None:
    m = sc.sensitivity_metrics({"n": 321, "both_correct": 100, "a_only": 40, "b_only": 10,
                                "both_wrong": 171})
    assert m["n"] == 321 and m["discordant"] == 50
    assert m["accuracy_a"] == 140 / 321
    assert m["accuracy_b"] == 110 / 321
    assert m["delta_pp"] == 30 / 321 * 100
    assert m["bigger_reader_share"] == 40 / 50
    assert m["p_exact"] == sc.mcnemar_exact(40, 10)


def test_metrics_survive_a_slice_with_no_disagreement() -> None:
    """Two readers that always agree: no delta, no p-value, and no division by zero."""
    m = sc.sensitivity_metrics({"n": 50, "both_correct": 40, "a_only": 0, "b_only": 0,
                                "both_wrong": 10})
    assert m["delta_pp"] == 0.0
    assert m["bigger_reader_share"] is None
    assert m["p_exact"] is None
    empty = sc.sensitivity_metrics({"n": 0})
    assert empty["accuracy_a"] is None and empty["delta_pp"] is None
    assert sc.pct_or(empty["accuracy_a"]) == "n/a"


def test_section_7_pools_counts_instead_of_averaging_percentages() -> None:
    """A 90pp slice and a 1pp slice must not average to 45.5pp."""
    with temp_artefact_dir() as tmp:
        write(tmp, "small.json", artefact(
            {"n": 10, "both_correct": 0, "a_only": 9, "b_only": 0, "both_wrong": 1},
            label="small slice"))
        write(tmp, "large.json", artefact(
            {"n": 100, "both_correct": 0, "a_only": 1, "b_only": 0, "both_wrong": 99},
            label="large slice"))
        payload = sc.collect_model_sensitivity(tmp)

    assert payload["status"] == "MEASURED"
    # Sorted by file name, so the report is deterministic for a fixed directory.
    assert [s["label"] for s in payload["slices"]] == ["large slice", "small slice"]
    total = payload["total"]
    assert total["n"] == 110 and total["a_only"] == 10
    assert round(total["delta_pp"], 4) == round(10 / 110 * 100, 4)  # not (90 + 1) / 2
    assert round(total["accuracy_a"], 4) == round(10 / 110, 4)
    assert total["bigger_reader_share"] == 1.0


def test_section_7_prints_the_zone_invariant_when_it_holds() -> None:
    """The deterministic zone contributes exactly 0, so the delta is locatable."""
    with temp_artefact_dir() as tmp:
        write(tmp, "paired.json", artefact(
            {"n": 35, "both_correct": 20, "a_only": 5, "b_only": 1, "both_wrong": 9},
            label="LoCoMo 1,540 - category 2 (temporal)",
            claims="benchmark_results/committer_metrics/synthetic_claims.json",
            zone_mode="system",
            controls=clean_controls(deterministic_zone_identical=True),
            zones={
                "deterministic": {"n": 20, "both_correct": 19, "a_only": 0, "b_only": 0,
                                  "both_wrong": 1},
                "llm": {"n": 15, "both_correct": 1, "a_only": 5, "b_only": 1, "both_wrong": 8},
            },
            groups=[{"name": "2 (temporal)",
                     "counts": {"n": 35, "both_correct": 20, "a_only": 5, "b_only": 1,
                                "both_wrong": 9}},
                    {"name": "1 (multi-hop)",
                     "counts": {"n": 0, "both_correct": 0, "a_only": 0, "b_only": 0,
                                "both_wrong": 0}}]))
        section = "\n".join(sc.render_sensitivity_section(sc.collect_model_sensitivity(tmp)))

    assert "### 7.1 Where the delta lives: the deterministic zone vs the LLM zone" in section
    assert "| Deterministic zone (0 LLM calls) | 20 |" in section
    assert "contributes exactly 0 to every delta above" in section
    # The pooled total is 35 questions, so the zone counts add up to it.
    assert "carried by the 15 questions the reader still answers (5 won only by A" in section
    assert "+11.43pp" in section  # (5 - 1) / 35, over the whole slice
    assert "**Pooled (all slices)**" in section
    assert "A = `qwen2.5:7b-instruct` vs B = `qwen2.5:1.5b`" in section


def test_section_7_flags_drift_and_a_moving_deterministic_zone() -> None:
    """A pairing that retrieved different context, or moved code-only questions, warns."""
    with temp_artefact_dir() as tmp:
        write(tmp, "drifted.json", artefact(
            {"n": 20, "both_correct": 15, "a_only": 4, "b_only": 1, "both_wrong": 0},
            label="drifted",
            controls=clean_controls(token_mismatch={"n": 3, "sample": ["q1"]},
                                    deterministic_zone_identical=False)))
        payload = sc.collect_model_sensitivity(tmp)
        section = "\n".join(sc.render_sensitivity_section(payload))

    assert payload["slices"][0]["controls"]["retrieval_drift"] == 3
    assert "!! Retrieval drifted in 1 slice(s)" in section
    assert "!! The deterministic zone moved in 1 slice(s)" in section
    assert "so read it with care" in section


def test_missing_artefact_directory_reports_not_measured() -> None:
    with temp_artefact_dir() as tmp:
        payload = sc.collect_model_sensitivity(tmp / "does_not_exist")
    assert payload["status"] == "NOT MEASURED"
    assert payload["slices"] == [] and payload["total"] is None
    section = "\n".join(sc.render_sensitivity_section(payload))
    assert "## 7. Model sensitivity (same slice, smaller reader)" in section
    assert "NOT MEASURED" in section


def test_a_corrupt_artefact_does_not_take_the_section_down() -> None:
    with temp_artefact_dir() as tmp:
        write(tmp, "good.json", artefact(
            {"n": 10, "both_correct": 5, "a_only": 2, "b_only": 1, "both_wrong": 2}, label="good"))
        (tmp / "broken.json").write_text("{not json", encoding="utf-8")
        write(tmp, "notes.json", {"label": "no counts here"})
        (tmp / "readme.txt").write_text("ignored", encoding="utf-8")
        payload = sc.collect_model_sensitivity(tmp)

    assert payload["status"] == "MEASURED"
    assert [s["label"] for s in payload["slices"]] == ["good"]
    assert payload["total"]["n"] == 10


def test_section_7_names_the_counterfactual_zone() -> None:
    """Reader-only arms must not be described as if code had answered."""
    with temp_artefact_dir() as tmp:
        write(tmp, "paired.json", artefact(
            {"n": 20, "both_correct": 14, "a_only": 3, "b_only": 1, "both_wrong": 2},
            label="LoCoMo temporal (reader-only arms)",
            zone_mode="counterfactual",
            zones={"deterministic": {"n": 6, "both_correct": 4, "a_only": 2, "b_only": 0,
                                     "both_wrong": 0},
                   "llm": {"n": 14, "both_correct": 10, "a_only": 1, "b_only": 1, "both_wrong": 2}}))
        section = "\n".join(sc.render_sensitivity_section(sc.collect_model_sensitivity(tmp)))

    assert "Neither arm skipped the reader in `LoCoMo temporal (reader-only arms)`" in section
    assert "operational model-sensitivity number for those slices is the LLM-zone row" in section
    # The held-out row is accounted for arithmetically rather than folded into a
    # "every point of the delta" claim it would contradict (its delta is not 0).
    assert "6 in counterfactual deterministic rows" in section
    # The "invariant, contributes exactly 0" sentence needs a real system arm.
    assert "contributes exactly 0" not in section


def test_one_pairing_per_cohort_enters_the_pooled_row() -> None:
    """Two pairings of the same question set must not double the pooled counts."""
    with temp_artefact_dir() as tmp:
        write(tmp, "readers.json", artefact(
            {"n": 10, "both_correct": 4, "a_only": 4, "b_only": 1, "both_wrong": 1},
            label="readers-only", cohort="locomo_temporal", zone_mode="counterfactual",
            zones={"deterministic": {"n": 4, "both_correct": 2, "a_only": 2, "b_only": 0,
                                     "both_wrong": 0},
                   "llm": {"n": 6, "both_correct": 2, "a_only": 2, "b_only": 1,
                           "both_wrong": 1}}))
        write(tmp, "system.json", artefact(
            {"n": 10, "both_correct": 5, "a_only": 3, "b_only": 1, "both_wrong": 1},
            label="deployed system", cohort="locomo_temporal", zone_mode="system",
            zones={"deterministic": {"n": 4, "both_correct": 4, "a_only": 0, "b_only": 0,
                                     "both_wrong": 0},
                   "llm": {"n": 6, "both_correct": 1, "a_only": 3, "b_only": 1,
                           "both_wrong": 1}}))
        write(tmp, "other.json", artefact(
            {"n": 5, "both_correct": 1, "a_only": 2, "b_only": 1, "both_wrong": 1},
            label="longmemeval", cohort="longmemeval_all500", zone_mode="counterfactual",
            zones={"deterministic": {"n": 1, "both_correct": 0, "a_only": 0, "b_only": 1,
                                     "both_wrong": 0},
                   "llm": {"n": 4, "both_correct": 1, "a_only": 2, "b_only": 0,
                           "both_wrong": 1}}))
        payload = sc.collect_model_sensitivity(tmp)
        section = "\n".join(sc.render_sensitivity_section(payload))

    # Every pairing is still displayed, but the cohort is counted once.
    assert [s["label"] for s in payload["slices"]] == ["longmemeval", "readers-only",
                                                       "deployed system"]
    assert payload["total"]["n"] == 15  # 10 + 5, not 25
    assert {s["label"]: s["pooled"] for s in payload["slices"]} == {
        "longmemeval": True, "deployed system": True, "readers-only": False,
    }
    # The deterministic row only carries the pairing where code answered both arms.
    assert payload["zones"]["deterministic"]["n"] == 4
    assert payload["zones"]["llm"]["n"] == 10
    assert "1 in counterfactual deterministic rows" in section
    assert "The rows above account for all" not in section


def test_section_7_names_repeat_controls_and_opposing_directions() -> None:
    """A same-model pairing is a noise floor, not a reader swap; a pooled delta
    that nets a win against a loss must say so instead of reading as stability."""
    with temp_artefact_dir() as tmp:
        write(tmp, "locomo.json", artefact(
            {"n": 20, "both_correct": 10, "a_only": 6, "b_only": 1, "both_wrong": 3},
            label="locomo system", cohort="locomo"))
        write(tmp, "lme.json", artefact(
            {"n": 20, "both_correct": 10, "a_only": 1, "b_only": 6, "both_wrong": 3},
            label="longmemeval", cohort="lme"))
        write(tmp, "repeat.json", artefact(
            {"n": 20, "both_correct": 18, "a_only": 0, "b_only": 0, "both_wrong": 2},
            label="repeat control", cohort="repeat",
            a={"run": "benchmark_results/a.json", "model": "qwen2.5:7b-instruct"},
            b={"run": "benchmark_results/b.json", "model": "qwen2.5:7b-instruct"}))
        section = "\n".join(sc.render_sensitivity_section(sc.collect_model_sensitivity(tmp)))

    assert "A = B = `qwen2.5:7b-instruct` (repeat control)" in section
    assert "run-to-run noise floor" in section
    assert "A = `qwen2.5:7b-instruct` vs B = `qwen2.5:1.5b`" in section
    assert "opposite directions" in section


def test_render_markdown_carries_section_7_after_section_6() -> None:
    """Even with nothing measured, the scorecard has to name both gaps."""
    md = sc.render_markdown([], [], None, None)
    assert "## 6. Deterministic memory engine (answer committer)" in md
    assert "## 7. Model sensitivity (same slice, smaller reader)" in md
    assert md.index("## 6.") < md.index("## 7.")
    assert SENSITIVITY_DIR.name in md
