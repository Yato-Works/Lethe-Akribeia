#!/usr/bin/env python3
"""Offline "would the runtime beat the reader?" check for a deterministic committer.

The question every AM-side skill has to answer is not "does it look right?" but
"on the questions it claims, is it more often right than the frozen reader, and
is the difference worth the code?".  This tool answers that **without calling an
LLM at all**: it replays a frozen context cache through
:mod:`artificial_memory.skills.answer_committer`, scores the committed answers
with the *official* per-category matchers, and pairs the result against a reader
run's own per-question correctness for exactly the questions the committer
claimed.

    claimed        how many questions the committer refused to abstain on
    commit_acc     its accuracy on those questions
    reader_same    the reader's accuracy on the *same* questions
    net_pp         commit_acc - reader_same, over the whole category

A positive ``net_pp`` is the only licence to ship a skill; a negative one means
the LLM stays in charge.  ``--sweep`` prints the whole threshold curve so the
committer's default is a measurement rather than a guess.

``--report-out`` turns the same run into a durable artefact under
``benchmark_results/committer_metrics/`` so the consolidated scorecard
(``apex_scorecard.py`` §6) can report the deterministic-engine KPIs - coverage,
commit accuracy, reader-on-the-same-set, commit delta, fallback rate and saved
LLM calls - from raw counts instead of from prose.  Raw counts (``n``,
``claimed``, ``commit_correct``, ``reader_correct``) are what is stored, so the
scorecard can add slices up without averaging percentages.

``--dump-committed`` writes the per-question attribution instead of just the
totals: one entry per question saying whether the committer answered it and
whether that answer was right.  ``model_sensitivity.py`` joins it onto two
reader runs, because a question answered by code scores the same under every
reader and therefore has to be held constant when the reader is swapped - the
delta can only come from the questions left over.

      --cache benchmark_results/locomo_context_cache_rules.jsonl \
      --llm-run temporal321_rules --sweep
  uv run python scripts/benchmarks/commit_sweep.py --category 2 \
      --cache benchmark_results/locomo_context_cache_rules.jsonl \
      --llm-run temporal321_rules --report-out \
      benchmark_results/committer_metrics/locomo_cat2_temporal.json
  uv run python scripts/benchmarks/commit_sweep.py --suite lme --qtype all \
      --cache benchmark_results/lme_context_cache.jsonl \
      --llm-run grand_longmemeval_report_coder7b_apex_ctx8192 --report-out \
      benchmark_results/committer_metrics/lme_all_types.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import (  # noqa: E402
    LoCoMoAdapter,
)
from artificial_memory.skills.answer_committer import (  # noqa: E402
    CommittedAnswer,
    extract_certificate_answer,
    extract_temporal_answer,
)

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def commit_for(category: int, question: str, context: str) -> CommittedAnswer:
    """Return a CommittedAnswer for a category, or a documented abstention.

    Only the temporal committer exists so far; the other categories deliberately
    return ``used=False`` so this tool reports "no committer" instead of silently
    scoring nothing.  Adding a skill means adding one branch here.
    """
    if category == 2:
        return extract_temporal_answer(question, context)
    if category == "temporal-reasoning":
        # LongMemEval contexts open with a [Temporal Calculation/Ordering]
        # certificate the runtime already computed, so transcribe that first and
        # only fall back to the date-span extractor when it is absent.
        committed = extract_certificate_answer(question, context)
        if committed.used:
            return committed
        return extract_temporal_answer(question, context)
    return CommittedAnswer(used=False, detail=f"no committer for category {category}")


def is_correct(category: int, expected: str, actual: str) -> bool:
    """Official per-category matchers, so numbers stay comparable across arms."""
    exp = expected.lower().strip()
    act = actual.lower().strip()
    if category == 2:
        return LoCoMoAdapter._temporal_answer_matches(exp, act)
    if category == 3:
        return LoCoMoAdapter._open_domain_answer_matches(exp, act)
    if category == 4:
        return LoCoMoAdapter._single_hop_answer_matches(exp, act)
    return exp in act or act in exp


def total_counts(stats: dict[str, dict[str, int]]) -> dict[str, int]:
    """Sum raw counts across slices - ratios are never averaged, they are re-derived."""
    total = {"n": 0, "claimed": 0, "commit_correct": 0, "reader_correct": 0}
    for counts in stats.values():
        for key in total:
            total[key] += counts.get(key, 0)
    return total


def default_label(args) -> str:
    if args.suite == "lme":
        scope = "all types" if args.qtype == "all" else args.qtype
        return f"LongMemEval - {scope}"
    name = CATEGORY_NAMES.get(args.category, "unknown")
    return f"LoCoMo 1,540 - category {args.category} ({name})"


def reader_artefact(args) -> str:
    """Repo-relative path of the reader run the committer is paired against."""
    if args.suite == "lme":
        return f"benchmark_results/longmemeval/{args.llm_run}.json"
    return f"benchmark_results/locomo1540/{args.llm_run}.json"


def write_report(args, stats: dict[str, dict[str, int]]) -> Path:
    """Persist one measurement as a scorecard-readable artefact.

    Only raw counts are stored (``n`` / ``claimed`` / ``commit_correct`` /
    ``reader_correct``); the scorecard derives coverage, commit accuracy, the
    paired reader accuracy, the commit delta and the fallback rate from them, so
    two artefacts can be added up without re-deriving anything by hand.
    """
    import artificial_memory.skills.answer_committer as committer

    payload = {
        "label": args.label or default_label(args),
        "suite": args.suite,
        "cache": args.cache,
        "reader_run": reader_artefact(args),
        "min_turn_score": committer.MIN_TURN_SCORE,
        "counts": total_counts(stats),
        "groups": [{"name": name, "counts": counts} for name, counts in sorted(stats.items())],
    }
    path = REPO / args.report_out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def dump_claims(args, records: list[dict]) -> Path:
    """Persist per-question attribution so a reader swap can be decomposed.

    Stored next to the count artefacts rather than inside them: §6 of the
    scorecard needs counts and nothing else, while this file is the join key for
    the model-sensitivity analysis (which questions are answered by code).
    """
    import artificial_memory.skills.answer_committer as committer

    payload = {
        "label": args.label or default_label(args),
        "suite": args.suite,
        "cache": args.cache,
        "reader_run": reader_artefact(args),
        "min_turn_score": committer.MIN_TURN_SCORE,
        "counts": {
            "n": len(records),
            "claimed": sum(1 for r in records if r["claimed"]),
            "commit_correct": sum(1 for r in records if r["commit_correct"]),
        },
        "records": records,
    }
    path = REPO / args.dump_committed
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_locomo(args) -> tuple[list[dict], dict[str, str], dict[str, bool]]:
    """Rows / ground truth / reader correctness for a LoCoMo context cache."""
    rows: list[dict] = []
    with open(REPO / args.cache, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("category") == args.category:
                rows.append(row)
    gt_map = load_ground_truth()
    reader_ok: dict[str, bool] = {}
    llm_path = REPO / "benchmark_results" / "locomo1540" / f"{args.llm_run}.json"
    if llm_path.exists():
        with open(llm_path, encoding="utf-8") as fh:
            for r in json.load(fh)["results"]:
                reader_ok[r["question_id"]] = bool(r["is_correct"])
    else:
        print(f"!! reader artefact {llm_path.name} missing; reader_same treated as 0")
    return rows, gt_map, reader_ok


def load_lme(args) -> tuple[list[dict], dict[str, str], dict[str, bool]]:
    """Rows / ground truth / reader correctness for the LongMemEval cache.

    The LME cache carries its own ``gt`` and ``qtype`` columns and its context uses
    the ``[answer_xxx on 2023/05/30 (Tue) 17:27] user: ...`` dialect, which
    ``answer_committer.parse_turns`` understands.  The reader baseline is the
    grand report under ``benchmark_results/longmemeval/``.

    ``--qtype all`` keeps every question type and lets each row's own ``qtype``
    drive both the committer shape and the official matcher, which is what the
    coverage report wants: it answers "which types can the runtime decide at all?".
    """
    from artificial_memory.research.benchmarks.external.longmemeval_adapter import (
        LongMemEvalAdapter,
    )

    rows: list[dict] = []
    with open(REPO / args.cache, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if args.qtype == "all" or row.get("qtype") == args.qtype:
                rows.append(row)
    gt_map = {row["qid"]: str(row.get("gt", "")) for row in rows}

    reader_ok: dict[str, bool] = {}
    report = REPO / "benchmark_results" / "longmemeval" / f"{args.llm_run}.json"
    if report.exists():
        with open(report, encoding="utf-8") as fh:
            for r in json.load(fh)["results"]:
                reader_ok[r["question_id"]] = bool(r["is_correct"])
    else:
        print(f"!! reader report {report.name} missing; reader_same treated as 0")
    # Expose the official matcher so scoring is identical to the published number.
    args._lme_scorer = LongMemEvalAdapter.score_answer
    return rows, gt_map, reader_ok


def load_ground_truth() -> dict[str, str]:
    """Ground truth by ``{sample_id}-qa-{index}``, joined onto the cache by qid.

    The context caches deliberately store no answers (they are retrieval
    artefacts), so the scorer side has to join the dataset - same source
    ``scripts/run_locomo_1540.py`` uses, so both agree on what the truth is.
    """
    gt: dict[str, str] = {}
    with open(REPO / "datasets/external/locomo10.json", encoding="utf-8") as fh:
        for conv in json.load(fh):
            sid = conv.get("sample_id", "")
            for i, qa in enumerate(conv.get("qa", [])):
                gt[f"{sid}-qa-{i:03d}"] = str(qa.get("answer", ""))
    return gt


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline committer-vs-reader comparison")
    ap.add_argument("--cache", default="benchmark_results/locomo_context_cache.jsonl")
    ap.add_argument("--category", type=int, default=2)
    ap.add_argument("--llm-run", default="locomo_instruct_full",
                    help="reader artefact used as the paired baseline")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the committer threshold and report the net effect")
    ap.add_argument("--show-misses", type=int, default=0)
    ap.add_argument("--show-committed", type=int, default=0)
    ap.add_argument("--suite", choices=("locomo", "lme"), default="locomo")
    ap.add_argument("--qtype", default="temporal-reasoning",
                    help="LongMemEval question type, or 'all' (--suite lme)")
    ap.add_argument("--report-out", default=None, metavar="PATH",
                    help="write the raw counts as a scorecard artefact "
                         "(e.g. benchmark_results/committer_metrics/locomo_cat2_temporal.json)")
    ap.add_argument("--label", default=None,
                    help="human label for the report artefact (default: derived from the slice)")
    ap.add_argument("--dump-committed", default=None, metavar="PATH",
                    help="write per-question attribution (claimed / commit_correct) for "
                         "the model-sensitivity analysis "
                         "(e.g. benchmark_results/committer_metrics/lme_all_types_claims.json)")
    args = ap.parse_args()

    if args.suite == "lme" and args.cache.startswith("benchmark_results/locomo"):
        args.cache = "benchmark_results/lme_context_cache.jsonl"

    if args.suite == "lme":
        rows, gt_map, reader_ok = load_lme(args)
        label = args.qtype
    else:
        rows, gt_map, reader_ok = load_locomo(args)
        label = f"{args.category} ({CATEGORY_NAMES.get(args.category)})"
    if not rows:
        print(f"!! no rows for suite={args.suite} in {args.cache}")
        return 2

    def context_of(row: dict) -> str:
        return str(row.get("context") or row.get("context_text") or "")

    def scored(row: dict, answer: str) -> bool:
        if args.suite == "lme":
            # Per-row type: with --qtype all every row must use its own matcher.
            return bool(args._lme_scorer(
                question_type=str(row.get("qtype", args.qtype)),
                gt=gt_map.get(row["qid"], ""),
                predicted_answer=answer,
                is_abstention_gt=bool(row.get("is_abstention_gt")),
                pcc_is_abstention=False,
            ))
        return is_correct(args.category, gt_map.get(row["qid"], ""), answer)

    def shape_of(row: dict):
        """Which committer answers this row."""
        if args.suite == "lme":
            return str(row.get("qtype", args.qtype))
        return args.category

    def group_of(row: dict) -> str:
        """Slice label used for the per-group breakdown."""
        if args.suite == "lme":
            return str(row.get("qtype", args.qtype))
        return f"{args.category} ({CATEGORY_NAMES.get(args.category, 'unknown')})"

    def evaluate(records: list[dict] | None = None) -> dict[str, dict[str, int]]:
        """Per-group raw counts.  Every KPI downstream is derived from these.

        Passing ``records`` also appends one entry per question saying who
        answered it (the deterministic committer or the reader fallback) and
        whether that answer was right.  ``--dump-committed`` writes them out so a
        reader-vs-reader comparison can subtract the deterministic zone: those
        questions are answered by code, not by a model, so any accuracy a reader
        swap moves must come from the questions left over.
        """
        stats: dict[str, dict[str, int]] = {}
        for row in rows:
            bucket = stats.setdefault(
                group_of(row),
                {"n": 0, "claimed": 0, "commit_correct": 0, "reader_correct": 0},
            )
            bucket["n"] += 1
            out = commit_for(shape_of(row), str(row["question"]), context_of(row))
            if records is not None:
                records.append({
                    "question_id": row["qid"],
                    "group": group_of(row),
                    "claimed": bool(out.used),
                    "commit_correct": bool(out.used) and scored(row, out.answer),
                    "reader_correct": bool(reader_ok.get(row["qid"], False)),
                    # What the *cache* thought the retrieval contained.  The reader
                    # runs record their own oracle recall, so comparing the two
                    # measures how faithful this offline replay is to the runs it
                    # is joined onto - the cache is what the committer saw, and the
                    # reader is what the prompt actually held.
                    "oracle_recall": row.get("oracle_recall"),
                })
            if not out.used:
                continue
            bucket["claimed"] += 1
            if scored(row, out.answer):
                bucket["commit_correct"] += 1
            if reader_ok.get(row["qid"], False):
                bucket["reader_correct"] += 1
        return stats

    if args.sweep:
        import artificial_memory.skills.answer_committer as committer

        original = committer.MIN_TURN_SCORE
        print(f"{'thresh':>7}{'claimed':>9}{'commit_acc':>12}{'reader_same':>13}{'net_pp':>9}")
        for threshold in (0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            committer.MIN_TURN_SCORE = threshold
            totals = total_counts(evaluate())
            claimed, correct = totals["claimed"], totals["commit_correct"]
            if not claimed:
                print(f"{threshold:>7.2f}{0:>9}{'-':>12}{'-':>13}{'-':>9}")
                continue
            net = (correct - totals["reader_correct"]) / len(rows) * 100
            print(f"{threshold:>7.2f}{claimed:>9}{correct / claimed:>11.1%}"
                  f"{totals['reader_correct'] / claimed:>12.1%}{net:>+9.2f}")
        committer.MIN_TURN_SCORE = original
        if args.report_out or args.dump_committed:
            print("!! --report-out/--dump-committed ignored with --sweep "
                  "(the threshold is mutated during the sweep)")
        return 0

    claim_records: list[dict] | None = [] if args.dump_committed else None
    stats = evaluate(claim_records)
    totals = total_counts(stats)
    claimed, correct, reader_correct = (totals["claimed"], totals["commit_correct"],
                                        totals["reader_correct"])
    print(f"cache={args.cache}")
    print(f"suite={args.suite}  slice={label}  n={len(rows)}")
    if not claimed:
        print(f"  committer abstained on all {len(rows)} questions - nothing to compare")
    else:
        print(f"  claimed            : {claimed:>5}  "
              f"({100 * claimed / len(rows):5.1f}% of the slice)")
        print(f"  committer accuracy : {correct:>5}  ({100 * correct / claimed:5.1f}% of claimed)")
        print(f"  reader on same set : {reader_correct:>5}  "
              f"({100 * reader_correct / claimed:5.1f}%)")
        print(f"  => net on slice    : {(correct - reader_correct) / len(rows) * 100:+.2f}pp")
    if len(stats) > 1:
        print("  by slice:")
        for name, counts in sorted(stats.items()):
            share = counts["claimed"] / counts["n"] * 100 if counts["n"] else 0.0
            acc = counts["commit_correct"] / counts["claimed"] if counts["claimed"] else 0.0
            print(f"    {name:<28} n={counts['n']:>4}  claimed={counts['claimed']:>4} "
                  f"({share:5.1f}%)  commit_acc={acc:5.1%}")

    if args.report_out:
        path = write_report(args, stats)
        print(f"  report             : {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")

    if args.dump_committed:
        path = dump_claims(args, claim_records or [])
        print(f"  claims             : {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")

    shown_ok = shown_bad = 0
    for row in rows:
        if shown_ok >= args.show_committed and shown_bad >= args.show_misses:
            break
        out = commit_for(shape_of(row), str(row["question"]), context_of(row))
        if not out.used:
            continue
        ok = scored(row, out.answer)
        if ok and shown_ok < args.show_committed:
            shown_ok += 1
            print(f"\n  [ok] {row['qid']} Q={str(row['question'])[:70]}")
            print(f"       GT={gt_map.get(row['qid'], '')!r} ANS={out.answer!r} ({out.detail})")
        elif not ok and shown_bad < args.show_misses:
            shown_bad += 1
            print(f"\n  [!!] {row['qid']} Q={str(row['question'])[:70]}")
            print(f"       GT={gt_map.get(row['qid'], '')!r} ANS={out.answer!r} ({out.detail})")
            print(f"       other candidates: {out.candidates[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
