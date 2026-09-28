#!/usr/bin/env python3
"""Where the remaining errors actually live: a failure-ceiling census.

Step 2 measured the reader's *sensitivity*: swapping ``qwen2.5:7b-instruct`` for
``qwen2.5:1.5b`` moved the LoCoMo 1,540 suite by 0.13pp, so the reader's size is
not what caps the score.  That result is only useful if the next question is
answered the same way - mechanically, from stored evidence, rather than by
argument:

    a score of 64.7% limits the runtime only if the missing 35.3% is semantic
    reasoning.  If it is retrieval, context construction or a lost commit, then
    none of it is waiting on a better reader.

So this tool classifies **every wrong answer of a frozen run** into buckets that
are decidable without a new experiment, and refuses to guess on the ones that
are not:

  1. ``retrieval``  - the answer session is not in the compiled context
                      (``oracle_recall`` false).  No reader, however large, can
                      answer from evidence that is not there.
  2. ``commit``     - the deterministic committer claimed the question and its
                      answer was wrong.  A runtime defect, not a reader defect,
                      and the cheapest kind to fix because the claim set is
                      already in code.
  3. ``unresolved`` - the evidence is present, the reader answered, the answer is
                      wrong.  That one bucket is *three* hypotheses at once -
                      context too noisy to read (B), reasoning failure (C), or
                      genuinely semantic (E) - and **no artefact separates
                      them**: "the answer is in the context and the reader missed
                      it" is exactly what all three look like from here.

``unresolved`` is therefore reported as unresolved, never allocated, with the
signals that *do* discriminate attached instead:

  * reader-sensitive  - the second reader (``--run-b``) got it right, so the
                        question is decided by the reader and is not a ceiling
                        for the runtime.
  * reader-insensitive- no reader in the pair can do it: a ceiling for any
                        reader, i.e. runtime, context shape or ground truth.
  * gt_verbatim       - the ground-truth string survives verbatim in the compiled
                        context (``--cache``), separating "the fact is there"
                        from "the fact is there in another wording".

The ceilings are re-derived from the counts, which is the point: how much a
*bigger* reader could ever recover, how much a *smaller* one already recovers,
how much is retrieval, and how much is still unresolved and needs one controlled
experiment (re-read the same questions with the gold evidence only - the only
thing that separates B from C).

Questions the committer already owns are counted separately and are **not**
headroom: no reader call is spent on them.

Usage:
  uv run python scripts/benchmarks/failure_ceiling.py \\
      --run-a benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json \\
      --run-b benchmark_results/locomo1540/temporal321_rules_commit_15b.json \\
      --cache benchmark_results/locomo_context_cache_rules.jsonl \\
      --label "LoCoMo 1,540 - deployed system (7B)" \\
      --out benchmark_results/failure_ceiling/locomo_1540_system.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
BUCKETS = ("retrieval", "commit", "unresolved_reader_sensitive", "unresolved_reader_insensitive")


def _norm(text: str) -> str:
    """Ground truths and context lines are compared case/punctuation-insensitively."""
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _is_committed(source: str) -> bool:
    """``answer_source`` values are ``reader`` or ``committed: <detail>``."""
    return bool(source) and source != "reader"


def load_run(path: str) -> dict:
    """One reader run keyed by question id, with the fields the census needs."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: dict[str, dict] = {}
    for r in data.get("results", []):
        qid = str(r.get("question_id") or r.get("id") or "")
        if not qid:
            continue
        category = r.get("category")
        rows[qid] = {
            "correct": bool(r.get("is_correct")),
            "oracle": r.get("oracle_recall") if isinstance(r.get("oracle_recall"), bool) else None,
            "committed": _is_committed(str(r.get("answer_source") or "")),
            "category": int(category) if isinstance(category, int) else None,
            "group": (str(r.get("question_type")) if r.get("question_type")
                      else CATEGORY_NAMES.get(category, "unknown")),
            "ground_truth": str(r.get("ground_truth") or ""),
        }
    return {"path": str(path),
            "model": str(data.get("model") or data.get("reader_model") or Path(path).stem),
            "rows": rows}


def load_cache(path: str) -> dict[str, dict]:
    """Compiled-context cache: per qid the context the reader was actually shown."""
    cache: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row.get("qid") or row.get("question_id") or "")
            if qid:
                cache[qid] = row
    return cache


def _tally() -> dict:
    return {name: 0 for name in BUCKETS}



def classify(a: dict, b: dict | None = None, cache: dict[str, dict] | None = None,
             claims: dict[str, bool] | None = None) -> dict:
    """Bucket every wrong answer of run ``a``.  Counts only.

    ``b`` is a second reader run of the *same* slice: it is what turns
    "unresolved" into reader-sensitive vs reader-insensitive.  ``cache`` supplies
    the compiled context for the ground-truth-verbatim signal, and ``claims`` (a
    ``commit_sweep.py --dump-committed`` attribution) can stand in for a runner
    that does not record ``answer_source`` in band.
    """
    shared = sorted(set(a["rows"]) & set(b["rows"])) if b else sorted(a["rows"])
    buckets = _tally()
    per_group: dict[str, dict] = {}
    correct_n = 0
    committed_total = 0
    oracle_known = 0
    gt_verbatim = 0
    # Reader accounting.  The two questions that matter are "the other reader got
    # it right" (reader choice is the limit here) and "neither did" (no reader in
    # the pair can).  `reference_only` is answered correctly by the reference and
    # wrongly by the other, so it is *not* headroom for a bigger reader - it is
    # headroom for a smaller one, and is counted separately.
    other_reader_only = 0
    neither_reader = 0
    reference_only = 0
    retrieval_other_right = 0
    both_right = 0
    unresolved_n = 0
    commit_without_evidence = 0

    for qid in shared:
        row = a["rows"][qid]
        group = per_group.setdefault(row["group"], _tally())
        if row["oracle"] is not None:
            oracle_known += 1
        committed = row["committed"] or bool((claims or {}).get(qid))
        if committed:
            committed_total += 1
        other_right = bool(b and b["rows"][qid]["correct"])
        if row["correct"]:
            correct_n += 1
            if b is not None and not other_right:
                reference_only += 1
            else:
                both_right += 1
            continue
        if other_right:
            other_reader_only += 1
        else:
            neither_reader += 1

        # Order matters, and it is not cosmetic.  A committed question is an
        # explicit claim that code can answer it with no reader at all; if that
        # claim is wrong the runtime is at fault even when the evidence was never
        # in the context, because a committer that claims a question it cannot
        # see is over-claiming.  So `commit` is tested first, and the subset that
        # committed without evidence is reported on its own.
        if committed:
            bucket = "commit"
            if row["oracle"] is False:
                commit_without_evidence += 1
        elif row["oracle"] is False:
            bucket = "retrieval"
        else:
            # Evidence present, the reader answered, the answer is wrong: three
            # hypotheses (context / reasoning / semantic) with one signature.
            bucket = ("unresolved_reader_sensitive" if other_right
                      else "unresolved_reader_insensitive")
            unresolved_n += 1

        buckets[bucket] += 1
        group[bucket] += 1
        if bucket == "retrieval" and other_right:
            # The context lacks the answer session yet the other reader answered
            # correctly: either the oracle flag is a false negative, or that reader
            # answered from parametric knowledge.  Either way "retrieval" is not
            # the whole story for this question.
            retrieval_other_right += 1
        if bucket != "retrieval" and cache is not None:
            cached = cache.get(qid)
            truth = _norm(row["ground_truth"])
            if cached and truth and truth in _norm(cached.get("context", "")):
                gt_verbatim += 1

    n = len(shared)
    return {
        "n": n,
        "correct": correct_n,
        "wrong": n - correct_n,
        "committed_in_run": committed_total,
        "oracle_measured": oracle_known,
        "buckets": buckets,
        "per_group": [{"name": name, "buckets": per_group[name]} for name in sorted(per_group)],
        "signals": {
            # Of the reference reader's failures: the other reader solves these.
            "other_reader_only": other_reader_only,
            # Of the reference reader's failures: neither reader solves these.
            "neither_reader": neither_reader,
            # Reference right, other wrong - the part of the reader-determined mass
            # a *smaller* reader would give away.
            "reference_only": reference_only,
            # Every question whose outcome depends on which reader is used.
            "reader_determined": other_reader_only + reference_only,
            "both_right": both_right,
            "retrieval_but_other_right": retrieval_other_right,
            "commit_without_evidence": commit_without_evidence,
            "gt_verbatim_in_context": gt_verbatim,
            "unresolved_n": unresolved_n,
        },
    }


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "n/a"


def render_report(label: str, a: dict, b: dict | None, analysis: dict) -> str:
    n = analysis["n"]
    buckets = analysis["buckets"]
    sig = analysis["signals"]
    lines = [
        f"### Failure ceiling - {label}",
        f"A = `{a['model']}`" + (f", B = `{b['model']}`" if b else ", no second reader run"),
        f"{n} shared questions, {analysis['correct']} correct ({_pct(analysis['correct'], n)}), "
        f"{analysis['wrong']} wrong ({_pct(analysis['wrong'], n)}); "
        f"{analysis['committed_in_run']} were answered by the deterministic committer",
        "",
        "| Bucket | Questions | Share of slice | What it means |",
        "|---|---:|---:|---|",
        f"| Retrieval | {buckets['retrieval']} | {_pct(buckets['retrieval'], n)} "
        "| the answer session is not in the compiled context |",
        f"| Commit | {buckets['commit']} | {_pct(buckets['commit'], n)} "
        "| the committer claimed it and the committed answer is wrong |",
        f"| Unresolved, reader-sensitive | {buckets['unresolved_reader_sensitive']} "
        f"| {_pct(buckets['unresolved_reader_sensitive'], n)} "
        "| evidence present, and the other reader got it right |",
        f"| Unresolved, reader-insensitive | {buckets['unresolved_reader_insensitive']} "
        f"| {_pct(buckets['unresolved_reader_insensitive'], n)} "
        "| evidence present, and no reader in the pair can do it |",
        "",
        "Unresolved is not a verdict: context noise (B), reasoning failure (C) and",
        "genuinely semantic questions (E) are indistinguishable from stored evidence."
        + (f"  {sig['gt_verbatim_in_context']} of the {sig['unresolved_n']} have the ground-truth"
           " string verbatim in the compiled context." if b else ""),
        "",
    ]
    if b is not None:
        lines += [
            "Reader accounting over the whole slice - the 2x2 that bounds any reader work:",
            "",
            "| | Other reader right | Other reader wrong |",
            "|---|---:|---:|",
            f"| **Reference reader right** | {sig['both_right']} | {sig['reference_only']} |",
            f"| **Reference reader wrong** | {sig['other_reader_only']} | {sig['neither_reader']} |",
            "",
            f"Only **{sig['reader_determined']} of {n} questions ({sig['reader_determined'] / n * 100:.1f}pp)**"
            f" change outcome with the reader at all; the {sig['neither_reader']} the reference reader"
            " gets wrong are wrong for the other reader too.",
            "",
        ]
    lines += [
        "| Ceiling | Questions | pp of slice | Reachable by |",
        "|---|---:|---:|---|",
        f"| Retrieval | {buckets['retrieval']} | {buckets['retrieval'] / n * 100:.1f} "
        "| retriever / context construction |",
        f"| Commit | {buckets['commit']} | {buckets['commit'] / n * 100:.1f} "
        "| the runtime, and the cheapest bucket to fix |",
        f"| Unresolved, reader-sensitive | {buckets['unresolved_reader_sensitive']} "
        f"| {buckets['unresolved_reader_sensitive'] / n * 100:.1f} "
        "| decided by the reader |",
        f"| Unresolved, reader-insensitive | {buckets['unresolved_reader_insensitive']} "
        f"| {buckets['unresolved_reader_insensitive'] / n * 100:.1f} "
        "| no reader in the pair can do it |",
        f"| Still unresolved | {sig['unresolved_n']} | {sig['unresolved_n'] / n * 100:.1f} "
        "| one gold-evidence re-read splits B from C |",
    ]
    if sig["commit_without_evidence"]:
        lines.append("")
        lines.append(f"!! {sig['commit_without_evidence']} committed questions were answered "
                     "wrong *and* had no answer session in the context: the committer over-claims, "
                     "which is a threshold defect rather than a retrieval one.")
    if sig["retrieval_but_other_right"]:
        lines.append("")
        lines.append(f"!! {sig['retrieval_but_other_right']} questions are filed as retrieval "
                     "failures yet the other reader answered them correctly, so the oracle flag "
                     "is a false negative (or that reader used parametric knowledge) on them.")
    if analysis["per_group"]:
        lines += [
            "",
            "| Group | Wrong | Retrieval | Commit | Unresolved (sensitive) | Unresolved (blind) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for group in analysis["per_group"]:
            g = group["buckets"]
            lines.append(
                f"| {group['name']} | {sum(g.values())} | {g['retrieval']} | {g['commit']} "
                f"| {g['unresolved_reader_sensitive']} | {g['unresolved_reader_insensitive']} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify every wrong answer of a frozen run")
    ap.add_argument("--run-a", required=True, help="reference run artefact (the deployed reader)")
    ap.add_argument("--run-b", default=None,
                    help="second reader run of the same slice, for reader-dependence")
    ap.add_argument("--cache", default=None, help="compiled-context cache jsonl")
    ap.add_argument("--claims", default=None,
                    help="per-question attribution from `commit_sweep.py --dump-committed`")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None, metavar="PATH", help="write counts as a JSON artefact")
    args = ap.parse_args()

    a = load_run(str(REPO / args.run_a))
    if not a["rows"]:
        print(f"!! run artefact carries no results: {args.run_a}")
        return 2
    b = load_run(str(REPO / args.run_b)) if args.run_b else None
    if b is not None and not b["rows"]:
        print(f"!! second run artefact carries no results: {args.run_b}")
        return 2
    cache = load_cache(str(REPO / args.cache)) if args.cache else None
    claims: dict[str, bool] | None = None
    if args.claims:
        data = json.loads((REPO / args.claims).read_text(encoding="utf-8"))
        claims = {str(r["question_id"]): bool(r["claimed"]) for r in data.get("records", [])}

    analysis = classify(a, b, cache, claims)
    print(render_report(args.label or Path(args.run_a).stem, a, b, analysis))

    if args.out:
        payload = {
            "label": args.label or Path(args.run_a).stem,
            "a": {"run": args.run_a, "model": a["model"]},
            "b": {"run": args.run_b, "model": b["model"]} if b else None,
            "cache": args.cache,
            "claims": args.claims,
            **analysis,
        }
        path = REPO / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved failure-ceiling artefact to {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
