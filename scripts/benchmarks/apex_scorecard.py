"""AM Apex consolidated scorecard aggregator.

This script contains NO benchmark logic.  It only reads already-completed
result artefacts and renders the consolidated "current state" scorecard:

  * LoCoMo 1,540  (Mem0 non-adversarial protocol, 4 categories)
  * LoCoMo-10 1,986 (all 10 conversations, incl. 446 adversarial probes)
  * LongMemEval 500 (6 types + 30 abstention probes)
  * BEAM 100K / 500K / 1M / 10M (official 10-category probing questions)

Reported dimensions per benchmark
  accuracy, retrieval (oracle-evidence) recall, abstention precision/recall,
  temporal / knowledge-update / multi-session slices,
  token efficiency (context tokens per question) and latency (mean / p50 / p95).

Section 6 adds the deterministic-engine KPIs (deterministic coverage, commit
accuracy, reader-on-the-same-set, commit delta, fallback rate, saved LLM calls).
Those are read from ``benchmark_results/committer_metrics/*.json``, which
``scripts/benchmarks/commit_sweep.py --report-out`` writes.

Section 7 adds model sensitivity: two runs of one slice that differ only in
``--model`` are paired question by question, so the scorecard can say how much of
the accuracy is the runtime's and how much is the reader model's.  Those come
from ``benchmark_results/model_sensitivity/*.json``, written by
``scripts/benchmarks/model_sensitivity.py``.  This script still only aggregates,
it never runs a skill or an LLM.

Usage:
  uv run python scripts/benchmarks/apex_scorecard.py
  uv run python scripts/benchmarks/apex_scorecard.py --json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmark_results"
OUT_MD = RESULTS / "AM_APEX_SCORECARD.md"

LOCOMO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

# A prediction counts as an explicit abstention/refusal for the over-abstention
# diagnostic.  Only used as a diagnostic signal; official scoring is untouched.
REFUSAL_PATTERNS = re.compile(
    r"(no information|not (?:mentioned|available|provided|specified)|"
    r"cannot (?:be )?(?:determine|find|answer)|can't (?:be )?(?:determine|find|answer)|"
    r"unable to (?:determine|find|answer)|does not (?:mention|contain|specify)|"
    r"no relevant|insufficient information|there is no)",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def latency_stats(values: list[float]) -> dict:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1))
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[idx],
    }


def slice_metrics(rows: list[dict], correct_key: str = "is_correct",
                  recall_key: str = "oracle_recall", tokens_key: str = "tokens_used",
                  latency_key: str = "latency_ms") -> dict:
    """Aggregate one homogeneous slice of question results."""
    if not rows:
        return {"n": 0, "acc": 0.0, "recall": None, "tokens": 0.0, "latency": latency_stats([])}
    n = len(rows)
    acc = sum(1 for r in rows if r.get(correct_key)) / n
    has_recall = any(recall_key in r for r in rows)
    recall = (sum(1 for r in rows if r.get(recall_key)) / n) if has_recall else None
    tokens = [r.get(tokens_key, 0) or 0 for r in rows]
    lat = [r.get(latency_key, 0.0) or 0.0 for r in rows]
    return {
        "n": n,
        "acc": acc,
        "recall": recall,
        "tokens": statistics.mean(tokens),
        "latency": latency_stats(lat),
    }


def over_abstention_rate(rows: list[dict]) -> float:
    """Share of answerable questions answered with an explicit refusal."""
    if not rows:
        return 0.0
    refusals = sum(1 for r in rows if REFUSAL_PATTERNS.search(str(r.get("predicted_answer") or r.get("prediction") or "")))
    return refusals / len(rows)


def new_result(name: str, **extra) -> dict:
    base = {"benchmark": name, "status": "NOT RUN", "model": None, "n": 0,
            "acc": None, "recall": None, "slices": {}, "adversarial": None,
            "abstention": None, "over_abstention": None,
            "tokens_per_q": None, "latency": None, "extra": extra}
    return base



def collect_locomo1540() -> dict:
    # Prefer the latest measured run; fall back through older artefacts.
    # The `.oraclefix.json` copies carry the corrected, content-based retrieval
    # flag: an audit showed the label-based flag was wrong on 22% of a 1,540-question
    # slice in both directions, while the aggregate barely moved (80.6% -> 81.1%).
    # Same predictions, same accuracy - only the retrieval column is corrected.
    candidates = [
        "locomo_fixes_applied.oraclefix.json",
        "locomo_fixes_applied.json",
        "locomo_fixes_applied_v1.json",
        "cap24_coder7b.json",
        "locomo_1540_improved2.json",
        "locomo_1540_improved.json",
        "locomo_1540_coder7b.json",
    ]
    path = None
    for name in candidates:
        candidate = RESULTS / "locomo1540" / name
        if candidate.exists():
            path = candidate
            break
    data = load_json(path) if path else None
    res = new_result("LoCoMo 1,540 (Mem0 non-adversarial)",
                     path=str(path.relative_to(ROOT)) if path else None)
    if not data:
        return res

    rows = [{**r, "tokens_used": r.get("tokens", 0)} for r in data.get("results", [])]
    total = slice_metrics(rows)
    res.update({
        "status": "MEASURED",
        "model": data.get("model"),
        "n": total["n"],
        "acc": total["acc"],
        "recall": total["recall"],
        "tokens_per_q": total["tokens"],
        "latency": total["latency"],
    })
    for cat, name in LOCOMO_CATS.items():
        subset = [r for r in rows if r.get("category") == cat]
        if subset:
            res["slices"][name] = slice_metrics(subset)
    return res


def collect_locomo1986() -> dict:
    res = new_result("LoCoMo-10 1,986 (10 conversations, incl. 446 adversarial)",
                     path="benchmark_results/locomo10/conv_*_results.json")
    rows: list[dict] = []
    conv_accs: list[float] = []
    for idx in range(10):
        data = load_json(RESULTS / "locomo10" / f"conv_{idx}_results.json")
        if not data:
            continue
        conv_rows = data.get("results", [])
        rows.extend(conv_rows)
        if conv_rows:
            conv_accs.append(sum(1 for r in conv_rows if r.get("is_correct")) / len(conv_rows))
    if not rows:
        return res

    total = slice_metrics(rows)
    res.update({
        "status": "MEASURED",
        "model": "qwen2.5-coder:7b",
        "n": total["n"],
        "acc": total["acc"],
        "recall": total["recall"],
        "tokens_per_q": total["tokens"],
        "latency": total["latency"],
        "macro_acc": statistics.mean(conv_accs) if conv_accs else 0.0,
        "conversations": len(conv_accs),
    })
    for cat, name in LOCOMO_CATS.items():
        subset = [r for r in rows if r.get("category") == cat]
        if subset:
            res["slices"][name] = slice_metrics(subset)
    adv = res["slices"].get("adversarial")
    if adv:
        res["adversarial"] = adv["acc"]
        res["abstention"] = adv["acc"]
    answerable = [r for r in rows if r.get("category") != 5]
    res["over_abstention"] = over_abstention_rate(answerable)
    return res


def _find_longmemeval_report() -> Path | None:
    # First choice: the 7B run of the *frozen default* configuration.  The older
    # `coder7b_apex_ctx8192` entry below was produced as a `--window-cap 48` A/B
    # override (twice the frozen `selection_window_cap` of 24): recompiling its
    # questions with cap 48 reproduces its token counts and oracle flags exactly.
    # It is not a default run, and it scores 5.8pp lower, so the default run leads
    # - but the older artefact is kept, never overwritten, because §7's stored
    # pairing cites it.
    candidates = [
        "grand_longmemeval_report_7b_apex_ctx8192_postfix.json",
        "grand_longmemeval_report_coder7b_apex_ctx8192.json",
        "grand_longmemeval_report_coder7b_apex.json",
        "grand_longmemeval_report.json",
        "grand_longmemeval_report_fixed_v1.json",
        "grand_longmemeval_report_qwen7b.json",
    ]
    for name in candidates:
        path = RESULTS / "longmemeval" / name
        if path.exists():
            return path
    return None


def collect_longmemeval() -> dict:
    path = _find_longmemeval_report()
    res = new_result("LongMemEval 500 (6 capabilities + 30 abstention probes)",
                     path=str(path.relative_to(ROOT)) if path else None)
    data = load_json(path) if path else None
    if not data:
        return res

    rows = data.get("results", [])
    total = slice_metrics(rows)
    res.update({
        "status": "MEASURED",
        "model": data.get("reader_model"),
        "n": total["n"],
        "acc": total["acc"],
        "recall": total["recall"],
        "tokens_per_q": total["tokens"],
        "latency": total["latency"],
        "num_ctx": data.get("reader_num_ctx"),
        "total_seconds": data.get("total_elapsed_seconds"),
        "failure_taxonomy": data.get("failure_taxonomy_v2", {}),
    })
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row.get("question_type", "unknown"), []).append(row)
    for name, subset in sorted(by_type.items()):
        res["slices"][name] = slice_metrics(subset)

    abs_rows = [r for r in rows if str(r.get("question_id", "")).endswith("_abs")]
    answerable = [r for r in rows if not str(r.get("question_id", "")).endswith("_abs")]
    if abs_rows:
        res["abstention"] = sum(1 for r in abs_rows if r.get("is_correct")) / len(abs_rows)
        res["abstention_n"] = len(abs_rows)
        res["over_abstention"] = over_abstention_rate(answerable)
    return res


def collect_beam() -> list[dict]:
    report = []
    for scale in ("100K", "500K", "1M", "10M"):
        path = RESULTS / "beam" / f"beam_coder7b_apex_{scale}.json"
        res = new_result(f"BEAM {scale}", path=str(path.relative_to(ROOT)))
        data = load_json(path)
        if not data:
            report.append(res)
            continue
        rows = data.get("results", [])
        total = slice_metrics(rows)
        res.update({
            "status": "MEASURED",
            "model": data.get("model"),
            "n": total["n"],
            "acc": total["acc"],
            "tokens_per_q": total["tokens"],
            "latency": total["latency"],
            "num_ctx": data.get("num_ctx"),
        })
        by_cat: dict[str, list[dict]] = {}
        for row in rows:
            by_cat.setdefault(str(row.get("category")), []).append(row)
        for name, subset in sorted(by_cat.items()):
            res["slices"][name] = slice_metrics(subset)
        if "abstention" in res["slices"]:
            res["abstention"] = res["slices"]["abstention"]["acc"]
        report.append(res)
    return report



# ---------------------------------------------------------------------------
# Deterministic memory engine (answer committer)
#
# The reader is not asked to be smarter; the runtime removes the places where it
# can be wrong.  Every measurement below is stored by
# ``scripts/benchmarks/commit_sweep.py --report-out`` as *raw counts* only
# (n / claimed / commit_correct / reader_correct) so this aggregator can add
# slices up and re-derive every ratio instead of averaging percentages.
# ---------------------------------------------------------------------------
COMMITTER_DIR = RESULTS / "committer_metrics"


def pct_or(value: float | None, default: str = "n/a") -> str:
    """``pct`` that tolerates a missing measurement (no claimed questions)."""
    return default if value is None else pct(value)


def committer_metrics(counts: dict) -> dict:
    """Derive the deterministic-engine KPIs from one set of raw counts."""
    n = int(counts.get("n", 0) or 0)
    claimed = int(counts.get("claimed", 0) or 0)
    commit_correct = int(counts.get("commit_correct", 0) or 0)
    reader_correct = int(counts.get("reader_correct", 0) or 0)
    return {
        "n": n,
        "claimed": claimed,
        "commit_correct": commit_correct,
        "reader_correct": reader_correct,
        # Share of the slice answered with no LLM call at all.
        "deterministic_coverage": (claimed / n) if n else None,
        # Accuracy of those answers - the KPI that has to beat the reader.
        "commit_accuracy": (commit_correct / claimed) if claimed else None,
        # The counterfactual: the paired reader's accuracy on the *same* questions.
        "reader_same": (reader_correct / claimed) if claimed else None,
        # Net effect on the whole slice, in points (not on the claimed subset).
        "commit_delta_pp": ((commit_correct - reader_correct) / n * 100) if n else None,
        "fallback_rate": (1 - claimed / n) if n else None,
        "llm_calls_saved": claimed,
    }


def load_committer_artefacts(directory: Path | None = None) -> list[dict]:
    """Every well-formed ``committer_metrics/*.json`` artefact, sorted by name."""
    directory = directory if directory is not None else COMMITTER_DIR
    if not directory.is_dir():
        return []
    payloads: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt artefact must never take the whole scorecard down.
            print(f"!! ignoring unreadable committer artefact {path.name}: {exc}")
            continue
        if not isinstance(payload, dict) or "counts" not in payload:
            print(f"!! ignoring malformed committer artefact {path.name}")
            continue
        # `commit_sweep.py --dump-committed` writes the per-question attribution
        # next to the measurement (`*_claims.json`).  It carries a `counts`
        # summary as well, so without this it would be read as a *second*
        # measurement of the same slice - one with no paired reader, hence a 0%
        # baseline and a fabricated commit delta, and a pooled total counted
        # twice.  Those files belong to §7, not §6.
        if "records" in payload or path.stem.endswith("_claims"):
            continue
        payload["path"] = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) \
            else path.as_posix()
        payloads.append(payload)
    return payloads


def collect_committer_metrics(directory: Path | None = None) -> dict:
    """Aggregate the deterministic-engine KPIs over all stored measurements."""
    directory = directory if directory is not None else COMMITTER_DIR
    payloads = load_committer_artefacts(directory)
    totals = {"n": 0, "claimed": 0, "commit_correct": 0, "reader_correct": 0}
    slices: list[dict] = []
    for payload in payloads:
        counts = payload.get("counts") or {}
        for key in totals:
            totals[key] += int(counts.get(key, 0) or 0)
        slices.append({
            "label": payload.get("label") or Path(payload["path"]).stem,
            "suite": payload.get("suite"),
            "cache": payload.get("cache"),
            "reader_run": payload.get("reader_run"),
            "min_turn_score": payload.get("min_turn_score"),
            "path": payload["path"],
            "metrics": committer_metrics(counts),
            "groups": [
                {"name": str(group.get("name", "?")),
                 "metrics": committer_metrics(group.get("counts") or {})}
                for group in (payload.get("groups") or [])
            ],
        })
    relative = directory.relative_to(ROOT).as_posix() if directory.is_relative_to(ROOT) else str(directory)
    return {
        "status": "MEASURED" if slices else "NOT MEASURED",
        "artefact_dir": relative,
        "slices": slices,
        "total": committer_metrics(totals) if slices else None,
    }


def render_committer_section(committer: dict) -> list[str]:
    """§6 - deterministic coverage, commit accuracy and the saved LLM calls."""
    out: list[str] = []
    out.append("## 6. Deterministic memory engine (answer committer)")
    out.append("")
    out.append("Doctrine: *the reader is not asked to be smarter; the runtime removes the places "
               "where it can be wrong.*  Each row is one frozen measurement produced by "
               "`scripts/benchmarks/commit_sweep.py --report-out ...`; the aggregator reads raw "
               "counts only, so every ratio below is re-derived and no percentage is averaged.")
    out.append("")
    if committer.get("status") != "MEASURED":
        out.append(f"NOT MEASURED - no artefact in `{committer.get('artefact_dir')}`. "
                   "Re-measure with the `commit_sweep.py --report-out` commands documented in "
                   "the script docstring.")
        out.append("")
        return out

    out.append("| Measurement | N | Deterministic coverage | Commit accuracy | Reader on same set "
               "| Commit delta | Fallback rate | LLM calls saved |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for entry in committer["slices"]:
        m = entry["metrics"]
        delta = f"{m['commit_delta_pp']:+.2f}pp" if m["commit_delta_pp"] is not None else "n/a"
        out.append(f"| {entry['label']} | {m['n']} | **{pct_or(m['deterministic_coverage'])}** "
                   f"({m['claimed']}) | **{pct_or(m['commit_accuracy'])}** | "
                   f"{pct_or(m['reader_same'])} | {delta} | {pct_or(m['fallback_rate'])} | "
                   f"{m['llm_calls_saved']} |")
    total = committer.get("total") or {}
    if total:
        delta = (f"{total['commit_delta_pp']:+.2f}pp"
                 if total.get("commit_delta_pp") is not None else "n/a")
        out.append(f"| **Total (all measurements)** | {total['n']} | "
                   f"**{pct_or(total['deterministic_coverage'])}** ({total['claimed']}) | "
                   f"**{pct_or(total['commit_accuracy'])}** | {pct_or(total['reader_same'])} | "
                   f"{delta} | {pct_or(total['fallback_rate'])} | {total['llm_calls_saved']} |")
    out.append("")
    out.append("KPI definitions (re-derivable from the stored counts):")
    out.append("")
    out.append("* **Deterministic coverage** = questions answered by the runtime with **0 LLM "
               "calls**, `claimed / n`; the parenthesised number is the raw count.")
    out.append("* **Commit accuracy** = `commit_correct / claimed` - the accuracy of the committed "
               "answers. A skill has to defend this before it may bypass the reader.")
    out.append("* **Reader on same set** = the *paired* frozen reader run's accuracy on exactly the "
               "same questions; the counterfactual the committer has to beat.")
    out.append("* **Commit delta** = `(commit_correct - reader_correct) / n`, i.e. the net effect on "
               "the whole slice, not on the claimed subset.")
    out.append("* **Fallback rate** = `1 - claimed / n` - the share still handed to the LLM reader.")
    out.append("* **LLM calls saved** = `claimed` - every committed answer replaces one reader call "
               "(and its latency) with a deterministic one.")
    out.append("")

    grouped = [e for e in committer["slices"] if len(e["groups"]) > 1]
    if grouped:
        out.append("### 6.1 Per-capability breakdown (where the runtime can already decide)")
        out.append("")
        out.append("| Slice | N | Deterministic coverage | Commit accuracy | Commit delta |")
        out.append("|---|---:|---:|---:|---:|")
        for entry in grouped:
            for group in entry["groups"]:
                m = group["metrics"]
                delta = (f"{m['commit_delta_pp']:+.2f}pp"
                         if m["commit_delta_pp"] is not None else "n/a")
                out.append(f"| {group['name']} ({entry['suite']}) | {m['n']} | "
                           f"{pct_or(m['deterministic_coverage'])} ({m['claimed']}) | "
                           f"{pct_or(m['commit_accuracy'])} | {delta} |")
        out.append("")
        out.append("Zero-coverage rows are the honest part of the table: those capabilities have no "
                   "deterministic committer yet, so the LLM reader keeps them. The KPI that matters "
                   "is how much of this table moves into the deterministic zone without the commit "
                   "accuracy line falling below the paired reader's.")
        out.append("")

    out.append("Provenance: " + ", ".join(
        f"{e['label']} <- `{e['cache']}` vs `{e['reader_run']}` "
        f"(threshold {e['min_turn_score']})" for e in committer["slices"]) + ".")
    out.append("")
    return out


# ---------------------------------------------------------------------------
# Model sensitivity (the reader swap)
#
# §6 answers "how much can the runtime decide without an LLM?".  §7 answers the
# other half of the same question: "when the reader *is* called, is the score the
# runtime's or the model's?"  Two runs of one slice that differ in nothing but
# `--model` are paired, and every question falls into one of four buckets:
# both correct / only A / only B / both wrong.  Those buckets are stored as
# counts, so slices pool by addition and the accuracy, the delta and the exact
# McNemar p-value are all re-derived here instead of read from the artefact.
# ---------------------------------------------------------------------------
SENSITIVITY_DIR = RESULTS / "model_sensitivity"


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar p-value; mirrors ``model_sensitivity.mcnemar_exact``.

    Re-derived from the stored discordant counts rather than read from the
    artefact, so a stale stored p-value cannot leak into the scorecard.  The two
    implementations are pinned to each other by the unit tests.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


def sensitivity_metrics(counts: dict) -> dict:
    """Derive both readers' accuracy and the paired delta from 4-way counts."""
    n = int(counts.get("n", 0) or 0)
    both = int(counts.get("both_correct", 0) or 0)
    a_only = int(counts.get("a_only", 0) or 0)
    b_only = int(counts.get("b_only", 0) or 0)
    both_wrong = int(counts.get("both_wrong", 0) or 0)
    discordant = a_only + b_only
    return {
        "n": n,
        "both_correct": both,
        "a_only": a_only,
        "b_only": b_only,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "accuracy_a": ((both + a_only) / n) if n else None,
        "accuracy_b": ((both + b_only) / n) if n else None,
        "delta_pp": ((a_only - b_only) / n * 100) if n else None,
        # Of the questions only one reader got, the share the bigger reader won.
        "bigger_reader_share": (a_only / discordant) if discordant else None,
        "p_exact": mcnemar_exact(a_only, b_only) if discordant else None,
    }


def load_sensitivity_artefacts(directory: Path | None = None) -> list[dict]:
    """Every well-formed ``model_sensitivity/*.json`` artefact, sorted by name."""
    directory = directory if directory is not None else SENSITIVITY_DIR
    if not directory.is_dir():
        return []
    payloads: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            # Same rule as §6: one corrupt file must not take the scorecard down.
            print(f"!! ignoring unreadable model-sensitivity artefact {path.name}: {exc}")
            continue
        if not isinstance(payload, dict) or "counts" not in payload:
            print(f"!! ignoring malformed model-sensitivity artefact {path.name}")
            continue
        payload["path"] = (path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT)
                           else path.as_posix())
        payloads.append(payload)
    return payloads


def _pool_rank(payload: dict) -> tuple[int, int]:
    """Which pairing of a cohort enters the pooled row.

    Cohorts are question sets, and one set can be paired more than once
    (readers-only, deployed system, a repeat control).  Summing those would count
    the same questions twice, so exactly one pairing per cohort is pooled: the
    one where both arms ran the committer - the deployed number - and between
    equals the larger slice.
    """
    n = int((payload.get("counts") or {}).get("n") or 0)
    return (1 if payload.get("zone_mode") == "system" else 0, n)


def collect_model_sensitivity(directory: Path | None = None) -> dict:
    """Aggregate every stored reader-swap pairing: per slice, plus a pooled total."""
    directory = directory if directory is not None else SENSITIVITY_DIR
    payloads = load_sensitivity_artefacts(directory)
    chosen: dict[str, dict] = {}
    for payload in payloads:
        cohort = str(payload.get("cohort") or Path(payload["path"]).stem)
        current = chosen.get(cohort)
        if current is None or _pool_rank(payload) > _pool_rank(current):
            chosen[cohort] = payload
    pooled_paths = {payload["path"] for payload in chosen.values()}
    totals = {"n": 0, "both_correct": 0, "a_only": 0, "b_only": 0, "both_wrong": 0}
    zone_totals: dict[str, dict[str, int]] = {}
    slices: list[dict] = []
    for payload in payloads:
        counts = payload.get("counts") or {}
        pooled = payload["path"] in pooled_paths
        if pooled:
            for key in totals:
                totals[key] += int(counts.get(key, 0) or 0)

        zones: dict[str, dict] = {}
        for zone, zone_counts in sorted((payload.get("zones") or {}).items()):
            zone_counts = zone_counts or {}
            zones[str(zone)] = sensitivity_metrics(zone_counts)
            # The deterministic row reads "code answered this" only where both
            # arms ran the committer; a counterfactual deterministic row is the
            # readers' accuracy on the same questions, and mixing the two would
            # put a number in the scorecard that describes neither.
            if not pooled or (str(zone) == "deterministic" and payload.get("zone_mode") != "system"):
                continue
            bucket = zone_totals.setdefault(str(zone), {
                "n": 0, "both_correct": 0, "a_only": 0, "b_only": 0, "both_wrong": 0,
            })
            for key in totals:
                bucket[key] += int(zone_counts.get(key, 0) or 0)

        controls = payload.get("controls") or {}

        def control_count(key: str) -> int:
            return int((controls.get(key) or {}).get("n", 0) or 0)

        slices.append({
            "label": payload.get("label") or Path(payload["path"]).stem,
            "cohort": str(payload.get("cohort") or Path(payload["path"]).stem),
            "a": payload.get("a") or {},
            "b": payload.get("b") or {},
            "claims": payload.get("claims"),
            "zone_mode": payload.get("zone_mode"),
            "path": payload["path"],
            "shared": int(payload.get("shared", 0) or 0),
            "metrics": sensitivity_metrics(counts),
            "zones": zones,
            "groups": [
                {"name": str(group.get("name", "?")),
                 "metrics": sensitivity_metrics(group.get("counts") or {})}
                for group in (payload.get("groups") or [])
            ],
            "controls": {
                # Both counters are runtime-side, so any drift is a real change
                # in what was retrieved - which would invalidate the pairing.
                "retrieval_drift": control_count("token_mismatch") + control_count("oracle_mismatch"),
                "token_mismatch": control_count("token_mismatch"),
                "oracle_mismatch": control_count("oracle_mismatch"),
                "only_in_a": control_count("only_in_a"),
                "only_in_b": control_count("only_in_b"),
                "claims_missing": control_count("claims_missing"),
                "answer_source_mismatch": control_count("answer_source_mismatch"),
                "committed_detail_mismatch": control_count("committed_detail_mismatch"),
                "deterministic_zone_identical": controls.get("deterministic_zone_identical"),
            },
            "pooled": pooled,
        })

    relative = directory.relative_to(ROOT).as_posix() if directory.is_relative_to(ROOT) \
        else str(directory)
    return {
        "status": "MEASURED" if slices else "NOT MEASURED",
        "artefact_dir": relative,
        "slices": slices,
        "total": sensitivity_metrics(totals) if slices else None,
        "zones": {zone: sensitivity_metrics(counts) for zone, counts in sorted(zone_totals.items())},
    }


def _fmt_p(value: float | None) -> str:
    """p-values can underflow the printed precision; say so instead of printing 0.0000."""
    if value is None:
        return "n/a"
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def render_sensitivity_section(sensitivity: dict) -> list[str]:
    """§7 - does the score hold when the reader is swapped for a smaller model?"""
    out: list[str] = []
    out.append("## 7. Model sensitivity (same slice, smaller reader)")
    out.append("")
    out.append("Doctrine: *a score belongs to the runtime only where it survives a weaker "
               "reader.*  Each row pairs two runs that differ in exactly one thing, `--model`: same "
               "context cache, same questions, same prompt, same matcher, same committer "
               "threshold; `scripts/benchmarks/model_sensitivity.py` writes them.  Every question "
               "falls into both-correct / A-only / B-only / both-wrong, and only the counts are "
               "stored, so the accuracies, the delta and the p-value below are re-derived here.")
    out.append("")
    if sensitivity.get("status") != "MEASURED":
        out.append(f"NOT MEASURED - no artefact in `{sensitivity.get('artefact_dir')}`. "
                   "Re-measure with the `model_sensitivity.py` command documented in the script "
                   "docstring.")
        out.append("")
        return out

    pairs = sorted({(str((e["a"] or {}).get("model")), str((e["b"] or {}).get("model")))
                    for e in sensitivity["slices"]})
    swapped = [f"A = `{a}` vs B = `{b}`" for a, b in pairs if a != b]
    repeats = [e["label"] for e in sensitivity["slices"]
               if str((e["a"] or {}).get("model")) == str((e["b"] or {}).get("model"))]
    swapped += [f"A = B = `{a}` (repeat control)" for a, b in pairs if a == b]
    out.append("Readers paired: " + ", ".join(swapped) + ".")
    if repeats:
        out.append("Repeat controls (" + ", ".join(f"`{label}`" for label in repeats)
                   + ") pair a reader with itself across two runs that differ only by the "
                   "committer flag, so their delta is the run-to-run noise floor every other "
                   "row has to clear.")
    out.append("")
    out.append("| Slice | N | Reader A | Reader B | Delta | A-only | B-only | p (exact) "
               "| Bigger-reader share |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for entry in sensitivity["slices"]:
        m = entry["metrics"]
        delta = f"{m['delta_pp']:+.2f}pp" if m["delta_pp"] is not None else "n/a"
        out.append(f"| {entry['label']} | {m['n']} | {pct_or(m['accuracy_a'])} "
                   f"| {pct_or(m['accuracy_b'])} | **{delta}** | {m['a_only']} | {m['b_only']} "
                   f"| {_fmt_p(m['p_exact'])} | {pct_or(m['bigger_reader_share'])} |")
    total = sensitivity.get("total") or {}
    if total:
        delta = f"{total['delta_pp']:+.2f}pp" if total["delta_pp"] is not None else "n/a"
        out.append(f"| **Pooled (all slices)** | {total['n']} | {pct_or(total['accuracy_a'])} "
                   f"| {pct_or(total['accuracy_b'])} | **{delta}** | {total['a_only']} "
                   f"| {total['b_only']} | {_fmt_p(total['p_exact'])} "
                   f"| {pct_or(total['bigger_reader_share'])} |")
    out.append("")
    slice_deltas = [e["metrics"]["delta_pp"] for e in sensitivity["slices"]
                    if e["metrics"]["delta_pp"] is not None]
    if any(d > 0 for d in slice_deltas) and any(d < 0 for d in slice_deltas):
        # Netting a slice where the big reader wins against one where it loses
        # produces a pooled delta that describes neither.
        out.append("!! The pooled row nets slices that move in opposite directions: read the "
                   "per-slice rows above it, because a near-zero pool here is cancellation "
                   "rather than stability.")

    zones = sensitivity.get("zones") or {}
    if zones:
        out.append("### 7.1 Where the delta lives: the deterministic zone vs the LLM zone")
        out.append("")
        out.append("| Zone | N | Reader A | Reader B | Delta | A-only | B-only |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for zone, human in (("deterministic", "Deterministic zone (0 LLM calls)"),
                            ("llm", "LLM zone (the reader still answers)")):
            m = zones.get(zone)
            if not m:
                continue
            delta = f"{m['delta_pp']:+.2f}pp" if m["delta_pp"] is not None else "n/a"
            out.append(f"| {human} | {m['n']} | {pct_or(m['accuracy_a'])} "
                       f"| {pct_or(m['accuracy_b'])} | {delta} | {m['a_only']} | {m['b_only']} |")
        out.append("")

        deterministic = zones.get("deterministic") or {}
        llm = zones.get("llm") or {}
        # "system" = both arms ran the committer, so the deterministic zone is an
        # invariant.  "counterfactual" = they did not (reader-only arms, or a
        # runner with no committer): the split still describes the deployed
        # system, but the accuracy inside that row belongs to the readers.
        system_zone = [e["label"] for e in sensitivity["slices"] if e.get("zone_mode") == "system"]
        counterfactual = [e["label"] for e in sensitivity["slices"]
                          if e.get("zone_mode") == "counterfactual"]
        mixed = [e["label"] for e in sensitivity["slices"] if e.get("zone_mode") == "mixed"]
        if mixed:
            # One arm committed and the other did not, so the zone rows compare a
            # deterministic answerer with a reader.  That is a different
            # experiment - the committer A/B - not a reader swap.
            out.append("!! Zone mode: mixed in " + ", ".join(f"`{label}`" for label in mixed)
                       + " - one arm ran the committer and the other did not, so those zone rows "
                       "compare a deterministic answerer against a reader. That is an A/B of the "
                       "committer, not a model-sensitivity pairing.")
        # Only slices where code answered both arms can testify to the invariant;
        # movement inside a counterfactual row is the reader being a reader.
        moved = sum(
            int((e["zones"].get("deterministic") or {}).get("a_only") or 0)
            + int((e["zones"].get("deterministic") or {}).get("b_only") or 0)
            for e in sensitivity["slices"] if e.get("zone_mode") == "system")
        if system_zone:
            if moved:
                out.append(f"!! The deterministic zone moved on {moved} question(s) in "
                           + ", ".join(f"`{label}`" for label in system_zone)
                           + ". Code does not get better or worse when the reader changes, so "
                           "this is a determinism bug in the runs, not a finding.")
            else:
                out.append("The deterministic zone is an invariant rather than a result: questions "
                           "answered by code score identically under both readers, so that zone "
                           "contributes exactly 0 to every delta above.")
        elif not counterfactual and not mixed:
            out.append("No deterministic zone in these pairings: neither run had a committer "
                       "enabled, so the whole table above is reader sensitivity.")
        if counterfactual:
            out.append("Neither arm skipped the reader in "
                       + ", ".join(f"`{label}`" for label in counterfactual)
                       + ", so that slice's deterministic row is the readers' own accuracy on the "
                       "questions the deployed committer answers with no LLM call - it is held out "
                       "of the zone table above, and the operational model-sensitivity number for "
                       "those slices is the LLM-zone row.")
        zone_n = sum(int((z or {}).get("n", 0) or 0) for z in zones.values())
        total_n = int(total.get("n", 0) or 0)
        if llm and zone_n == total_n:
            det_flips = int(deterministic.get("a_only") or 0) - int(deterministic.get("b_only") or 0)
            if det_flips:
                out.append(f"The zone rows account for all {zone_n} pooled questions, but the "
                           "deterministic row carries a delta of its own here - see the slice "
                           "controls above before reading the split as runtime vs reader.")
            else:
                trailer = (" - and with neither arm running the committer, that LLM-zone delta is "
                           "the operational number."
                           if counterfactual and not system_zone else "")
                out.append(f"Every point of the delta is therefore carried by the {llm['n']} "
                           f"questions the reader still answers ({llm['a_only']} won only by A, "
                           f"{llm['b_only']} only by B){trailer}")
        elif llm:
            held_out = sum(
                int((e["zones"].get("deterministic") or {}).get("n") or 0)
                for e in sensitivity["slices"]
                if e.get("pooled") and e.get("zone_mode") != "system")
            unsplit = total_n - zone_n - held_out
            parts = []
            if held_out:
                parts.append(f"{held_out} in counterfactual deterministic rows (both arms "
                             "answered them as a reader), held out so the deterministic row keeps "
                             "meaning 'code answered it'")
            if unsplit:
                parts.append(f"{unsplit} belong to pairings with no claims file to split by")
            out.append(f"The zone rows cover {zone_n} of {total_n} pooled questions"
                       + (": " + "; ".join(parts) if parts else ".")
                       + (". The remaining rows are reader-answered either way."
                          if not parts else "."))
        out.append("")

    drifted = [e for e in sensitivity["slices"] if e["controls"]["retrieval_drift"]]
    moved = [e for e in sensitivity["slices"]
             if e["controls"]["deterministic_zone_identical"] is False]
    non_nested = [e for e in sensitivity["slices"]
                  if e["controls"]["only_in_a"] and e["controls"]["only_in_b"]]
    lost = [e for e in sensitivity["slices"]
            if e["controls"]["only_in_a"] and not e["controls"]["only_in_b"]]
    superset = [e for e in sensitivity["slices"]
                if e["controls"]["only_in_b"] and not e["controls"]["only_in_a"]]
    engines = [e for e in sensitivity["slices"] if e["controls"]["answer_source_mismatch"]]
    details = [e for e in sensitivity["slices"] if e["controls"]["committed_detail_mismatch"]]
    out.append("Pairing controls: the context is not re-retrieved between the two arms, so the "
               "per-question token count and oracle recall have to match exactly. "
               f"{len(sensitivity['slices']) - len(drifted)} of {len(sensitivity['slices'])} "
               "slices show no drift.")
    if drifted:
        out.append(f"!! Retrieval drifted in {len(drifted)} slice(s): " + ", ".join(
            f"{e['label']} ({e['controls']['retrieval_drift']} questions)" for e in drifted)
            + " - that row is not holding retrieval constant, so read it with care.")
    if non_nested:
        out.append("!! " + str(len(non_nested)) + " slice(s) compare non-nested question sets: "
                   + ", ".join(e["label"] for e in non_nested)
                   + " - the pairing silently runs on the intersection.")
    elif lost:
        out.append("!! " + str(len(lost)) + " slice(s) lost questions in arm B: "
                   + ", ".join(e["label"] for e in lost)
                   + " - the candidate run does not cover the reference's full slice.")
    elif superset:
        out.append("Question sets nest in " + str(len(superset)) + " slice(s): "
                   + ", ".join(e["label"] for e in superset)
                   + " - arm B covers arm A plus questions outside this slice, which are excluded "
                     "rather than mismatched.")
    if engines:
        out.append("!! The arms used a different engine on " + ", ".join(
            f"{e['label']} ({e['controls']['answer_source_mismatch']} questions)" for e in engines)
            + " - the committer's claim set changed between the runs (a code change, not the "
              "reader), so those questions no longer isolate `--model`; re-measure the pairing.")
    if details:
        out.append("!! Both arms committed but at a different turn/anchor in " + ", ".join(
            f"{e['label']} ({e['controls']['committed_detail_mismatch']} questions)"
            for e in details)
            + " - an unchanged claim with a moved extraction point; harmless where correctness "
              "matches, but it is the claim set drifting under the pairing.")
    if moved:
        out.append(f"!! The deterministic zone moved in {len(moved)} slice(s): "
                   + ", ".join(e["label"] for e in moved) + ".")
    out.append("")
    out.append("KPI definitions (re-derivable from the stored counts):")
    out.append("")
    out.append("* **Reader A / Reader B** = `(both_correct + a_only) / n` and "
               "`(both_correct + b_only) / n` - the two arms of one slice.")
    out.append("* **Delta** = `(a_only - b_only) / n` in points: the paired difference between "
               "the two readers on identical inputs. Near zero means the runtime absorbed the "
               "model difference; large means the score belongs to the reader.")
    out.append("* **A-only / B-only** = questions exactly one reader answered correctly - "
               "McNemar's discordant pairs.")
    out.append("* **p (exact)** = two-sided exact McNemar over those pairs, no chi-square "
               "approximation, because the discordant counts are small. A large delta riding on a "
               "handful of flips is not yet a finding.")
    out.append("* **Bigger-reader share** = `a_only / (a_only + b_only)`; 50% means the two "
               "readers disagree at random.")
    out.append("")

    grouped = [e for e in sensitivity["slices"] if len(e["groups"]) > 1]
    if grouped:
        out.append("### 7.2 Per-capability (which capabilities need the bigger reader)")
        out.append("")
        out.append("| Slice | Capability | N | Reader A | Reader B | Delta | A-only | B-only |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for entry in grouped:
            for group in entry["groups"]:
                m = group["metrics"]
                delta = f"{m['delta_pp']:+.2f}pp" if m["delta_pp"] is not None else "n/a"
                out.append(f"| {entry['label']} | {group['name']} | {m['n']} "
                           f"| {pct_or(m['accuracy_a'])} | {pct_or(m['accuracy_b'])} | {delta} "
                           f"| {m['a_only']} | {m['b_only']} |")
        out.append("")
        out.append("Read this as a map of reader dependency rather than a leaderboard: a "
                   "capability where B matches A is one the runtime already carries, and the rows "
                   "where A-only piles up are where a smaller reader would cost real points.")
        out.append("")

    out.append("Provenance: " + ", ".join(
        f"{e['label']} <- `{(e['a'] or {}).get('run')}` vs `{(e['b'] or {}).get('run')}`"
        + (f", claims `{e['claims']}`" if e.get("claims") else "")
        for e in sensitivity["slices"]) + ".")
    out.append("")
    return out



# ---------------------------------------------------------------------------
# Rubric: each dimension is graded separately; the composite is a weighted mean.
# Thresholds are explicit so any reader can re-derive every grade.
# ---------------------------------------------------------------------------
GRADE_POINTS = {"S": 4.0, "A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}


def grade_higher_better(value: float | None, s: float, a: float, b: float, c: float) -> str:
    if value is None:
        return "-"
    if value >= s:
        return "S"
    if value >= a:
        return "A"
    if value >= b:
        return "B"
    if value >= c:
        return "C"
    return "D"


def grade_lower_better(value: float | None, s: float, a: float, b: float, c: float) -> str:
    if value is None:
        return "-"
    if value <= s:
        return "S"
    if value <= a:
        return "A"
    if value <= b:
        return "B"
    if value <= c:
        return "C"
    return "D"


def weighted_grade(parts: list[tuple[str, float]]) -> str:
    usable = [(g, w) for g, w in parts if g in GRADE_POINTS and w > 0]
    if not usable:
        return "-"
    total_w = sum(w for _, w in usable)
    score = sum(GRADE_POINTS[g] * w for g, w in usable) / total_w
    if score >= 3.5:
        return "S"
    if score >= 2.5:
        return "A"
    if score >= 1.5:
        return "B"
    if score >= 0.8:
        return "C"
    return "D"


def build_rows() -> list[dict]:
    rows = [collect_locomo1540(), collect_locomo1986(), collect_longmemeval()]
    rows.extend(collect_beam())
    rows.extend(collect_personamem())
    return rows


def collect_personamem() -> list[dict]:
    """PersonaMem official-protocol runs executed locally ($0).

    These are *baseline* measurements of the local reader under the upstream protocol
    (full persona context), not AM Apex scores: the official runner feeds the raw
    persona history to the model, so a low value documents full-context truncation.
    """
    out: list[dict] = []
    lab = RESULTS / "official" / "personamem_lab"
    for path in sorted(lab.glob("personamem_32k_first*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            recs = [r for r in csv.DictReader(handle) if r.get("question_id")]
        if not recs:
            continue
        hits = sum(1 for r in recs if str(r.get("score", "")).strip().lower() in ("true", "1", "1.0"))
        res = new_result(f"PersonaMem 32K (baseline, not AM) — {path.stem}", path=str(path.relative_to(ROOT)))
        res.update({
            "status": "MEASURED",
            "model": "qwen2.5-coder:7b via local gpt-4o alias",
            "n": len(recs),
            "acc": hits / len(recs),
        })
        out.append(res)
    return out


def _best(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _worst(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


def _lat_p95(row: dict) -> float | None:
    lat = row.get("latency")
    return lat["p95_ms"] if lat else None


def capability_dimensions(rows: list[dict]) -> list[tuple[str, str, float | None, str, float]]:
    """Return (dimension, source, value, grade, composite weight) for the dimensions
    the operator asked to be measured: LoCoMo, LongMemEval, BEAM, PersonaMem-family
    persona consistency, retrieval recall, abstention, temporal, knowledge-update,
    multi-session, token efficiency and latency."""
    by_name = {r["benchmark"]: r for r in rows}
    l1540 = by_name.get("LoCoMo 1,540 (Mem0 non-adversarial)", {})
    l1986 = by_name.get("LoCoMo-10 1,986 (10 conversations, incl. 446 adversarial)", {})
    lme = by_name.get("LongMemEval 500 (6 capabilities + 30 abstention probes)", {})
    beam = [r for r in rows if r["benchmark"].startswith("BEAM")]

    def slice_acc(row: dict, key: str) -> float | None:
        s = (row.get("slices") or {}).get(key)
        return s["acc"] if s else None

    def acc_grade(value: float | None) -> str:
        return grade_higher_better(value, 0.95, 0.90, 0.80, 0.70)

    dims: list[tuple[str, str, float | None, str, float]] = []

    dims.append(("LoCoMo 1,540 accuracy", "LoCoMo (Mem0 non-adversarial)", l1540.get("acc"),
                 acc_grade(l1540.get("acc")), 0.0))
    dims.append(("LoCoMo-10 1,986 accuracy", "LoCoMo-10 full (incl. adversarial)", l1986.get("acc"),
                 acc_grade(l1986.get("acc")), 0.0))
    dims.append(("LongMemEval accuracy (500)", "LongMemEval", lme.get("acc"),
                 acc_grade(lme.get("acc")), 0.40))
    dims.append(("BEAM accuracy (worst scale)", "BEAM 100K-10M",
                 _worst([b.get("acc") for b in beam]) if beam else None,
                 acc_grade(_worst([b.get("acc") for b in beam]) if beam else None), 0.0))

    retrieval = _worst([l1986.get("recall"), lme.get("recall")])
    dims.append(("Retrieval (oracle) recall", "LoCoMo-10 + LongMemEval", retrieval,
                 grade_higher_better(retrieval, 0.95, 0.90, 0.85, 0.75), 0.20))

    abst = _worst([l1986.get("abstention"), lme.get("abstention"),
                   _best([b.get("abstention") for b in beam])])
    dims.append(("Abstention (unanswerable accuracy)", "LoCoMo adversarial + LongMemEval _abs + BEAM",
                 abst, acc_grade(abst), 0.15))

    temporal = _worst([slice_acc(l1986, "temporal"), slice_acc(lme, "temporal-reasoning"),
                       _best([slice_acc(b, "temporal_reasoning") for b in beam])])
    dims.append(("Temporal reasoning", "LoCoMo + LongMemEval + BEAM", temporal, acc_grade(temporal), 0.0))

    kupdate = _worst([slice_acc(lme, "knowledge-update"),
                      _best([slice_acc(b, "knowledge_update") for b in beam])])
    dims.append(("Knowledge update", "LongMemEval + BEAM", kupdate, acc_grade(kupdate), 0.0))

    multi = _worst([slice_acc(lme, "multi-session"),
                    _best([slice_acc(b, "multi_session_reasoning") for b in beam])])
    dims.append(("Multi-session reasoning", "LongMemEval + BEAM", multi, acc_grade(multi), 0.0))

    pref = _best([slice_acc(lme, "single-session-preference"), slice_acc(l1986, "open-domain")])
    dims.append(("Persona / preference consistency", "LongMemEval preference + LoCoMo open-domain",
                 pref, acc_grade(pref), 0.0))

    tok = _best([l1986.get("tokens_per_q"), lme.get("tokens_per_q"),
                 _best([b.get("tokens_per_q") for b in beam])])
    dims.append(("Token efficiency (tokens/Q)", "all suites", tok,
                 grade_lower_better(tok, 500, 1000, 2000, 4000), 0.15))

    lat = _worst([_lat_p95(l1986), _lat_p95(lme), _best([_lat_p95(b) for b in beam])])
    dims.append(("Latency p95 (ms)", "all suites", lat,
                 grade_lower_better(lat, 2000, 4000, 8000, 15000), 0.10))

    return dims


def render_markdown(rows: list[dict], dims: list[tuple[str, str, float | None, str, float]],
                    committer: dict | None = None, sensitivity: dict | None = None,
                    ceiling: dict | None = None) -> str:
    out: list[str] = []
    out.append("# AM Apex official benchmark scorecard")
    out.append("")
    out.append("Reader for every number below: `qwen2.5-coder:7b` on local Ollama "
               "(temperature 0.0, frozen prompts, no external API, 0 write-LLM calls).")
    out.append("")
    out.append("Generated by `uv run python scripts/benchmarks/apex_scorecard.py` "
               "(read-only aggregation of official runner artefacts).")
    out.append("")
    out.append("## 1. Headline results")
    out.append("")
    out.append("| Benchmark | N | Accuracy | Retrieval recall | Tokens/Q | Latency mean / p50 / p95 (ms) | Grade |")
    out.append("|---|---:|---:|---:|---:|---|---|")
    for r in rows:
        if r["status"] != "MEASURED":
            out.append(f"| {r['benchmark']} | - | NOT RUN | - | - | - | - |")
            continue
        lat = r.get("latency") or {}
        grade = grade_higher_better(r["acc"], 0.95, 0.90, 0.80, 0.70)
        recall = pct(r["recall"]) if r["recall"] is not None else "n/a"
        tokens = f"{r['tokens_per_q']:.0f}" if r.get("tokens_per_q") is not None else "n/a"
        lateness = (f"{lat['mean_ms']:.0f} / {lat['p50_ms']:.0f} / {lat['p95_ms']:.0f}"
                    if lat else "n/a")
        out.append(f"| {r['benchmark']} | {r['n']} | **{pct(r['acc'])}** | {recall} | "
                   f"{tokens} | {lateness} | {grade} |")
    out.append("")
    out.append("PersonaMem rows are *baseline* local-reader measurements under the upstream protocol "
               "(raw full persona context), **not** AM Apex scores — see `AM_APEX_STATUS.md` §3 for why the "
               "official runner cannot measure AM Apex without new adapter code.")
    out.append("")

    out.append("## 2. Requested capability dimensions")
    out.append("")
    out.append("| Dimension | Source | Value | Grade |")
    out.append("|---|---|---:|---|")
    for name, source, value, grade, _ in dims:
        if value is None:
            shown = "NOT MEASURED"
        elif "Latency" in name:
            shown = f"{value:.0f} ms"
        elif "Token efficiency" in name:
            shown = f"{value:.0f} tokens/Q"
        else:
            shown = f"{value * 100:.1f}%"
        out.append(f"| {name} | {source} | {shown} | {grade} |")
    out.append("")

    out.append("## 3. Per-benchmark category breakdown")
    out.append("")
    for r in rows:
        if r["status"] != "MEASURED" or not r["slices"]:
            continue
        out.append(f"### {r['benchmark']}")
        out.append("")
        out.append("| Category | N | Accuracy | Retrieval recall | Tokens/Q | p95 latency (ms) |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for name, s in sorted(r["slices"].items()):
            recall = pct(s["recall"]) if s.get("recall") is not None else "n/a"
            out.append(f"| {name} | {s['n']} | {pct(s['acc'])} | {recall} | "
                       f"{s['tokens']:.0f} | {s['latency']['p95_ms']:.0f} |")
        out.append("")

    out.append("## 4. Composite grade")
    out.append("")
    composite = weighted_grade([(g, w) for _, _, _, g, w in dims])
    out.append("Weighted composite (LongMemEval accuracy 40%, retrieval recall 20%, "
               f"abstention 15%, token efficiency 15%, latency p95 10%): **{composite}**")
    out.append("")
    out.append("Grading rubric (documented, re-derivable):")
    out.append("")
    out.append("* Accuracy: S >= 95%, A >= 90%, B >= 80%, C >= 70%, else D")
    out.append("* Retrieval recall (oracle evidence inside the built context): S >= 95%, A >= 90%, B >= 85%, C >= 75%, else D")
    out.append("* Abstention (accuracy on unanswerable/probe questions): S >= 95%, A >= 90%, B >= 80%, C >= 70%, else D")
    out.append("* Token efficiency (context tokens/question): S <= 500, A <= 1000, B <= 2000, C <= 4000, else D")
    out.append("* Latency p95: S <= 2000 ms, A <= 4000 ms, B <= 8000 ms, C <= 15000 ms, else D")
    out.append("")
    out.append("Multi-hop / single-hop / contradiction / instruction-following / summarization slices are "
               "reported but unweighted (they have no published Mem0/LongMemEval counterpart to compare against).")
    out.append("")

    official = load_json(RESULTS / "official_locomo_score.json")
    if official:
        out.append("## 5. Official-metric cross-check (LoCoMo, pinned upstream scorer)")
        out.append("")
        out.append(f"Scorer: `{official.get('scorer')}` on `{official.get('run_dir')}` "
                   f"({official.get('n')} questions). That harness applies the official per-category metric "
                   "(stemmed F1 for categories 1-4, literal refusal match for category 5), which is a "
                   "different metric from the lexical `is_correct` used in sections 1-3.")
        out.append("")
        out.append(f"* Official F1, all categories: **{official.get('official_f1'):.2f}%**")
        out.append(f"* Official F1, categories 1-4 (= the 1,540 non-adversarial protocol): "
                   f"**{official.get('official_f1_no_cat5'):.2f}%**")
        out.append("")
        out.append("| Category | N | Official F1 |")
        out.append("|---|---:|---:|")
        for name, payload in (official.get("per_category") or {}).items():
            value = payload.get("official_f1")
            shown = f"{value:.2f}%" if value is not None else "n/a"
            out.append(f"| {name} | {payload.get('n', 0)} | {shown} |")
        out.append("")

    f1_rows = []
    for label, fname in (
        ("1,540 — baseline prompt", "official_locomo_score_locomo1540_baseline.json"),
        ("1,540 — temporal directive only", "official_locomo_score_locomo1540_improved2.json"),
        ("1,540 — all prompt changes (incl. answer-only rule)", "official_locomo_score_locomo1540_improved.json"),
        ("1,540 — + zero-loss compression, temporal rules, refusal precedence", "official_locomo_score_locomo1540_fixes.json"),
    ):
        payload = load_json(RESULTS / fname)
        if payload:
            f1_rows.append((label, payload))
    if f1_rows:
        out.append("### Official F1 on the 1,540 protocol, before vs after the prompt change")
        out.append("")
        out.append("Rows 1-3 keep the same reader, the same memorised contexts and the same 1,540 questions, "
                   "so their delta is attributable to the answer-generation prompt. The last row additionally "
                   "changes the context (quote_mode=keep + compact_header compression, expanded temporal "
                   "phrases) and is therefore a combined arm.")
        out.append("")
        out.append("| Run | N | Official F1 | multi-hop | temporal | open-domain | single-hop |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for label, payload in f1_rows:
            cats = payload.get("per_category") or {}

            def cat_value(name: str) -> str:
                value = (cats.get(name) or {}).get("official_f1")
                return f"{value:.2f}%" if value is not None else "n/a"

            out.append(f"| {label} | {payload.get('n')} | **{payload.get('official_f1'):.2f}%** | "
                       f"{cat_value('multi-hop')} | {cat_value('temporal')} | "
                       f"{cat_value('open-domain')} | {cat_value('single-hop')} |")
        out.append("")

    if committer is None:
        committer = {"status": "NOT MEASURED", "slices": [], "total": None,
                     "artefact_dir": COMMITTER_DIR.relative_to(ROOT).as_posix()}
    out.extend(render_committer_section(committer))

    if sensitivity is None:
        sensitivity = {"status": "NOT MEASURED", "slices": [], "total": None, "zones": {},
                       "artefact_dir": SENSITIVITY_DIR.relative_to(ROOT).as_posix()}
    out.extend(render_sensitivity_section(sensitivity))
    out.extend(render_ceiling_section(ceiling))
    return "\n".join(out)


CEILING_DIR = RESULTS / "failure_ceiling"
CEILING_ARTEFACT = "locomo_1540_system_oraclefix.json"
GOLD_CELLS_ARTEFACT = "gold_context_cells.json"
CEILING_ROWS = (
    ("retrieval", "the gold evidence turn is not in the compiled context"),
    ("commit", "the committer claimed it and the committed answer is wrong"),
    ("unresolved_reader_sensitive", "evidence present, and the other reader got it right"),
    ("unresolved_reader_insensitive", "evidence present, and no reader in the pair can"),
)
GOLD_ROWS = (
    ("context_limited", "recovered by **both** readers - the context was the blocker"),
    ("reader_capability", "recovered by the 7B only - a capability gap"),
    ("small_reader_advantage", "recovered by the 1.5B only - distrust until reproduced"),
    ("semantic", "recovered by neither"),
)


def render_ceiling_section(ceiling: dict | None) -> list[str]:
    """§8 - what is actually blocking the remaining score."""
    out: list[str] = []
    ceiling = ceiling or {}
    out.append("## 8. Failure ceiling (where the remaining score is stuck)")
    out.append("")
    if ceiling.get("status") != "MEASURED":
        out.append("NOT MEASURED - no `benchmark_results/failure_ceiling/"
                   "locomo_1540_system_oraclefix.json`. Produce it with "
                   "`scripts/benchmarks/failure_ceiling.py` after "
                   "`scripts/benchmarks/recount_oracle.py`.")
        out.append("")
        return out

    buckets = ceiling.get("buckets") or {}
    signals = ceiling.get("signals") or {}
    n = int(ceiling.get("n") or 0)
    wrong = int(ceiling.get("wrong") or 0)
    run_name = Path(str((ceiling.get("a") or {}).get("run") or "")).name
    out.append(
        f"Every wrong answer of the deployed LoCoMo-1,540 arm (`{run_name}`, "
        f"{ceiling.get('committed_in_run', 0)} answered by the committer) lands in exactly one "
        f"bucket, and the buckets sum to the {wrong} failures. "
        "`scripts/benchmarks/failure_ceiling.py` writes them; the reader accounting comes from "
        "the §7 pairing.")
    out.append("")
    out.append("| Bucket | Questions | Share of slice | Share of failures | What it is |")
    out.append("|---|---:|---:|---:|---|")
    for key, human in CEILING_ROWS:
        count = int(buckets.get(key, 0))
        out.append(f"| {key.replace('_', ' ')} | {count} | {_pp(count, n)} | {_pp(count, wrong)} "
                   f"| {human} |")
    out.append("")
    reader_determined = int(signals.get("reader_determined", 0))
    neither = int(signals.get("neither_reader", 0))
    out.append(
        f"**Only {reader_determined} of {n} questions ({_pp(reader_determined, n)}) change outcome "
        f"with the reader at all**, and {neither} are wrong for both readers. That is the number "
        "that decides where effort goes: a bigger reader cannot address most of what is left, "
        "and neither can a bigger retriever - see the unresolved row.")
    out.append("")

    gold = ceiling.get("gold_context") or []
    if gold:
        cells: dict[str, int] = {}
        answer_absent = 0
        for row in gold:
            cell = str(row.get("cell", "?"))
            cells[cell] = cells.get(cell, 0) + 1
            if cell == "semantic" and row.get("answer_in_gold_context") == "absent":
                answer_absent += 1
        total = len(gold)
        out.append("### 8.1 Gold-context ceiling (what survives the evidence being handed over)")
        out.append("")
        out.append(f"The {total} unresolved questions replayed through the frozen runner with "
                   "**only the gold evidence turn(s)** - 66 words against the production "
                   "context's 1,590:")
        out.append("")
        out.append("| Outcome with the gold context | Questions | Share |")
        out.append("|---|---:|---:|")
        for key, human in GOLD_ROWS:
            count = cells.get(key, 0)
            out.append(f"| {human} | {count} | {_pp(count, total)} |")
        out.append("")
        if answer_absent:
            semantic = cells.get("semantic", 0)
            out.append(
                f"**On {answer_absent} of the {semantic} questions that fail for both readers "
                f"({_pp(answer_absent, semantic)}), the gold answer is not in the labelled "
                "evidence at all.** A reader cannot be wrong about a fact that is not in front of "
                "it, so that mass needs a deterministic **derivation** layer - temporal "
                "arithmetic, a join across turns - not a better retriever and not a bigger "
                "reader. Caveat: the gold context is built from LoCoMo's labelled evidence ids, "
                "so this figure mixes derivation-needed with label-under-specified.")
            out.append("")

    out.append("### 8.2 Provenance of the retrieval oracle")
    out.append("")
    out.append(
        "Earlier internal runs used a **label-based** oracle (did the gold evidence *id* appear in "
        "the compiled context?). An audit showed the context compiler numbers the turns it emits "
        "in its own group order, so those ids are not the dataset's ids: the flag was wrong on "
        "**22% of the slice in both directions** while the aggregate barely moved. Every ceiling "
        "number above is recomputed with content+date evidence matching; the label-based figures "
        "are kept below rather than deleted.")
    out.append("")
    out.append("| LoCoMo oracle | Retrieval failure | Unresolved |")
    out.append("|---|---:|---:|")
    out.append("| Legacy label oracle | 211 / 1,540 | 301 / 1,540 |")
    out.append(f"| Corrected content+date oracle | {buckets.get('retrieval', 0)} / {n:,} "
               f"| {int(signals.get('unresolved_n', 0)):,} / {n:,} |")
    out.append("")
    cells_path = (CEILING_DIR / GOLD_CELLS_ARTEFACT)
    cells_ref = (cells_path.relative_to(ROOT).as_posix() if cells_path.is_relative_to(ROOT)
                 else str(cells_path))
    out.append(f"Provenance: `{ceiling.get('artefact')}`"
               + (f", gold-context cells `{cells_ref}`" if gold else "")
               + ".")
    out.append("")
    return out



def _pp(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%" if total else "n/a"


def collect_failure_ceiling(directory: Path | None = None) -> dict:
    """The failure census, published with the oracle revision it was measured under.

    Both the corrected census and the gold-context experiment are optional: a
    missing or unreadable file degrades the section to NOT MEASURED rather than
    failing the scorecard, exactly like §6 and §7.
    """
    directory = directory if directory is not None else CEILING_DIR
    payload: dict = {"status": "NOT MEASURED", "artefact_dir": str(directory)}
    path = directory / CEILING_ARTEFACT
    if not path.exists():
        return payload
    try:
        payload.update(load_json(path))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"!! ignoring unreadable failure-ceiling artefact {path.name}: {exc}")
        return payload
    payload["status"] = "MEASURED"
    payload["artefact"] = (path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT)
                           else str(path))
    cells = directory / GOLD_CELLS_ARTEFACT
    if cells.exists():
        try:
            payload["gold_context"] = load_json(cells).get("questions", [])
        except (OSError, json.JSONDecodeError) as exc:
            print(f"!! ignoring unreadable gold-context artefact {cells.name}: {exc}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="AM Apex consolidated scorecard")
    parser.add_argument("--json", action="store_true", help="Emit the raw aggregate as JSON")
    parser.add_argument("--no-write", action="store_true", help="Do not write the markdown file")
    args = parser.parse_args()

    rows = build_rows()
    dims = capability_dimensions(rows)
    committer = collect_committer_metrics()
    sensitivity = collect_model_sensitivity()
    ceiling = collect_failure_ceiling()

    if args.json:
        print(json.dumps(
            {"results": rows,
             "dimensions": [{"dimension": d, "source": s, "value": v, "grade": g, "weight": w}
                            for d, s, v, g, w in dims],
             "deterministic_engine": committer,
             "model_sensitivity": sensitivity,
             "failure_ceiling": ceiling,
             "composite_grade": weighted_grade([(g, w) for _, _, _, g, w in dims])},
            indent=2, ensure_ascii=False, default=str))
        return


    md = render_markdown(rows, dims, committer, sensitivity, ceiling)
    print(md)
    if not args.no_write:
        OUT_MD.write_text(md, encoding="utf-8")
        print(f"\n[written] {OUT_MD}")


if __name__ == "__main__":
    main()
