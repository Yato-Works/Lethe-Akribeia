#!/usr/bin/env python3
"""Does Lethe hold its score when the reader is swapped for a smaller model?

Step 1 measured how much of a benchmark the deterministic engine can answer with
**0 LLM calls**.  That left the other half of the question open: on LongMemEval
the covered 11.8% is a much smaller number than the 88.2% that still runs through
the reader, so how much of Lethe's accuracy is the *runtime* and how much is
simply the reader model's ability?

This tool answers it by pairing two runs of the **same slice** that differ in
exactly one field - the reader model:

  * same context cache (same retrieved evidence, same token counts),
  * same questions in the same order,
  * same prompt template, same matcher, same committer threshold,
  * different ``--model``.

Each question lands in exactly one of four buckets, and the buckets are the
finding:

    both_correct    the runtime carried the question under either reader
    a_only          the question needed the bigger reader
    b_only          the small reader did better (noise, or a genuinely easier one)
    both_wrong      neither reader could do it; the runtime has to improve here

``a_only`` against ``b_only`` is the argument in one line: a large ``a_only``
means Lethe is riding on the reader, and near-equal counts mean the runtime
already absorbs most of the model difference.  They are also precisely McNemar's
discordant pairs, so the tool reports the exact two-sided binomial p-value
instead of a vibe - exact, because a 500-question slice produces small counts
where the chi-square approximation lies.

Three controls guard the "only the model changed" claim, because a pairing is
worthless if retrieval drifted underneath it:

  * per-question ``tokens`` / ``tokens_used`` must be equal in both arms;
  * per-question ``oracle_recall`` must be equal in both arms;
  * when ``--claims`` supplies an offline replay, the cache's own ``oracle_recall``
    has to agree with the runs': where it does not, the committer judged a context
    the reader never saw - a caveat on the zone split, never on the readers.

Both are computed by the runtime from the retrieved context and never by the
reader, so a difference there is a real difference in what was retrieved.  Any
mismatch is counted, printed with a ``!!`` marker and stored in the artefact
rather than averaged away.

With ``--claims`` (written by ``commit_sweep.py --dump-committed``) the table
splits into a **deterministic zone** (questions the committer answers in code)
and the **LLM zone** (everything else).  The deterministic zone is an invariant:
the same questions must score identically under both readers, so a mismatch
there is a determinism bug, not a result.  The LLM zone is where the delta
actually lives, and it is the number that decides the next move - widen the
committer, or accept that the reader is the bottleneck.

Usage:
  uv run python scripts/benchmarks/model_sensitivity.py \
      --a benchmark_results/locomo1540/temporal321_rules_commit.json \
      --b benchmark_results/locomo1540/temporal321_rules_commit_15b.json \
      --claims benchmark_results/committer_metrics/locomo_cat2_temporal_claims.json \
      --label "LoCoMo 1,540 - category 2 (temporal) - deployed system" \
      --out benchmark_results/model_sensitivity/locomo_cat2_temporal_system.json

Two pairings of the *same* question set share a cohort (default: the --claims file
stem, override with --cohort) so the scorecard's pooled row counts them once, and
--restrict-both-reader pairs two runs of the same reader to measure the
run-to-run noise floor instead of a model delta.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def group_name(row: dict) -> str:
    """Slice label of a result row: LongMemEval type, or LoCoMo category."""
    if row.get("question_type"):
        return str(row["question_type"])
    category = row.get("category")
    if category is None:
        return "unknown"
    return f"{category} ({CATEGORY_NAMES.get(category, 'unknown')})"


def load_run(path: str) -> dict:
    """One reader run keyed by question id, plus the metadata needed to name it."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: dict[str, dict] = {}
    for r in data.get("results", []):
        rows[str(r["question_id"])] = {
            "correct": bool(r.get("is_correct")),
            "tokens": int(r.get("tokens") or r.get("tokens_used") or 0),
            "oracle": bool(r.get("oracle_recall")),
            "prediction": str(r.get("prediction") or r.get("predicted_answer") or ""),
            "group": group_name(r),
            # Who answered.  The LoCoMo runner records this in-band; the
            # LongMemEval runner has no committer at all, so there the key is
            # empty and the arm is reader-only by construction.
            "source": str(r.get("answer_source") or ""),
        }
    model = data.get("model") or data.get("reader_model") or Path(path).stem
    return {"path": str(path), "model": str(model), "rows": rows}


def load_claims(path: str) -> tuple[dict[str, bool], dict[str, bool]]:
    """Two maps keyed by question id: answered by the committer, and the cache's oracle.

    The second map measures how faithful the offline replay is: the claims file is
    generated from a *cache* of retrieved context, while the runs it gets joined
    onto retrieved their own.  Where the two disagree, the committer judged a
    context the reader never saw.  Questions whose cache row carries no oracle
    flag are simply left out.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("records", [])
    claimed = {str(r["question_id"]): bool(r["claimed"]) for r in records}
    oracle = {str(r["question_id"]): bool(r["oracle_recall"])
              for r in records if isinstance(r.get("oracle_recall"), bool)}
    return claimed, oracle


def tally(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    """The 4-way contingency table as *counts*; every ratio is derived, never stored."""
    counts = {"n": len(pairs), "both_correct": 0, "a_only": 0, "b_only": 0, "both_wrong": 0}
    for a, b in pairs:
        if a and b:
            counts["both_correct"] += 1
        elif a:
            counts["a_only"] += 1
        elif b:
            counts["b_only"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def accuracy_a(counts: dict[str, int]) -> float:
    """Accuracy of arm A, derived from counts so slices add without averaging."""
    return (counts["both_correct"] + counts["a_only"]) / counts["n"] if counts["n"] else 0.0


def accuracy_b(counts: dict[str, int]) -> float:
    """Accuracy of arm B, derived the same way (the `b_only` bucket swaps in)."""
    return (counts["both_correct"] + counts["b_only"]) / counts["n"] if counts["n"] else 0.0


def delta_pp(counts: dict[str, int]) -> float:
    """Accuracy(A) - accuracy(B) in percentage points, straight from the counts."""
    if not counts["n"]:
        return 0.0
    return (counts["a_only"] - counts["b_only"]) / counts["n"] * 100


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs.

    Under the null (the two readers are equivalent on this slice) every
    discordant question is a fair coin, so the observed split is judged against
    ``Binomial(a_only + b_only, 0.5)``.  Exact rather than chi-square: with a few
    dozen discordant questions the approximation is the difference between a
    claim and a guess.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _sample(ids: list[str], limit: int = 12) -> dict:
    """Count plus a bounded sample, so one drifted question cannot bloat a report."""
    return {"n": len(ids), "sample": sorted(ids)[:limit]}


def _is_committed(source: str) -> bool:
    """``answer_source`` values are ``reader`` or ``committed: <detail>``."""
    return bool(source) and source != "reader"


def _resolve_claims(a: dict, b: dict, shared: list[str],
                    claims: dict[str, bool] | None) -> tuple[dict[str, bool] | None, str | None]:
    """Decide which questions the deterministic engine answered, and from where.

    An explicit ``--claims`` file wins.  Otherwise a run that recorded
    ``answer_source`` per question (the LoCoMo runner does) describes itself, and
    that is the stronger evidence - it is what the deployed path actually did,
    not what a separate offline replay says it would have done.
    """
    if claims is not None:
        return claims, "file"
    # Prefer whichever arm actually ran the committer: an arm that never skipped
    # the reader has nothing to say about who answered.
    for role, run in (("a", a), ("b", b)):
        if any(_is_committed(run["rows"][q]["source"]) for q in shared):
            return ({q: _is_committed(run["rows"][q]["source"]) for q in shared},
                    f"run:answer_source ({role})")
    return None, None


def pair_runs(a: dict, b: dict, claims: dict[str, bool] | None,
              claims_oracle: dict[str, bool] | None = None,
              restrict_both_reader: bool = False) -> dict:
    """Join two runs question by question.  Counts only - every ratio is derived.

    The result is exactly what the scorecard re-derives its percentages from, so
    the printed table and the stored artefact can never disagree.
    """
    ids_a, ids_b = set(a["rows"]), set(b["rows"])
    shared = sorted(ids_a & ids_b)
    if restrict_both_reader:
        # Repeat control: keep only questions where *neither* arm let the
        # committer skip the reader, so what is left is pure reader behaviour -
        # the same model on the same input twice, or two models once each.
        shared = [q for q in shared
                  if not _is_committed(a["rows"][q]["source"])
                  and not _is_committed(b["rows"][q]["source"])]
    pairs = [(a["rows"][q]["correct"], b["rows"][q]["correct"]) for q in shared]

    claims, claims_source = _resolve_claims(a, b, shared, claims)

    # The committer never calls the model, so where both arms ran it they must
    # claim exactly the same questions.  An arm that never committed (a
    # reader-only run, or one with the committer switched off) is a *different
    # pipeline*, not a disagreement, so it cannot be compared this way.
    committed_a = any(_is_committed(a["rows"][q]["source"]) for q in shared)
    committed_b = any(_is_committed(b["rows"][q]["source"]) for q in shared)
    source_compared = committed_a and committed_b
    # Two levels, because they mean two different things: a question one arm
    # committed and the other did not is a changed *decision*; a question both
    # committed with a different turn/anchor is a changed *extraction point* on
    # an unchanged decision.
    engine_mismatch = ([q for q in shared
                        if _is_committed(a["rows"][q]["source"])
                        != _is_committed(b["rows"][q]["source"])]
                       if source_compared else [])
    detail_mismatch = ([q for q in shared
                        if _is_committed(a["rows"][q]["source"])
                        and _is_committed(b["rows"][q]["source"])
                        and a["rows"][q]["source"] != b["rows"][q]["source"]]
                       if source_compared else [])

    zones: dict[str, dict[str, int]] = {}
    claims_missing: list[str] = []
    zone_identical: bool | None = None
    zone_mode: str | None = None
    zones_committed_in_run = 0
    if claims is not None:
        claims_missing = [q for q in shared if q not in claims]
        # A question the claims file does not know is treated as reader-answered,
        # which is the honest default: only the committer may skip the reader.
        for zone, wanted in (("deterministic", True), ("llm", False)):
            zones[zone] = tally([
                (a["rows"][q]["correct"], b["rows"][q]["correct"])
                for q in shared
                if claims.get(q, False) is wanted
            ])
        committed_in_run = sum(
            1 for q in shared
            if claims.get(q, False) and (_is_committed(a["rows"][q]["source"])
                                        or _is_committed(b["rows"][q]["source"]))
        )
        # A reader-only arm (or a runner that has no committer at all, like
        # LongMemEval's) makes the zone counterfactual: the split is still the
        # deployed system's, but the accuracy inside it is the readers' own, not
        # the committer's - so that row is not the operational number.  One arm of
        # each kind is neither: it is an A/B of the committer, not of the reader.
        if committed_a and committed_b:
            zone_mode = "system"
        elif committed_a or committed_b:
            zone_mode = "mixed"
        else:
            zone_mode = "counterfactual"
        # Invariant, not a finding: code does not get better or worse when the
        # reader changes.  It only *is* an invariant when both arms actually ran
        # the committer - in the other two modes the reader answered those
        # questions, so movement there is reader behaviour, not a bug.
        if zone_mode == "system":
            zone_identical = (zones["deterministic"]["a_only"] == 0
                              and zones["deterministic"]["b_only"] == 0)
        zones_committed_in_run = committed_in_run

    by_group: dict[str, list[tuple[bool, bool]]] = {}
    for q in shared:
        row_a = a["rows"][q]
        by_group.setdefault(row_a["group"], []).append((row_a["correct"], b["rows"][q]["correct"]))

    counts = tally(pairs)
    return {
        "shared": len(shared),
        "claims_source": claims_source,
        "zone_mode": zone_mode,
        "zones_committed_in_run": zones_committed_in_run,
        "counts": counts,
        "zones": {name: z for name, z in sorted(zones.items())},
        "groups": [{"name": name, "counts": tally(p)} for name, p in sorted(by_group.items())],
        "controls": {
            "only_in_a": _sample(sorted(ids_a - ids_b)),
            "only_in_b": _sample(sorted(ids_b - ids_a)),
            "token_mismatch": _sample([
                q for q in shared if a["rows"][q]["tokens"] != b["rows"][q]["tokens"]
            ]),
            "oracle_mismatch": _sample([
                q for q in shared if a["rows"][q]["oracle"] != b["rows"][q]["oracle"]
            ]),
            "claims_missing": _sample(claims_missing),
            # The offline claims cache vs what the runs actually retrieved: either
            # arm disagreeing means the replay judged a context a reader did not see.
            "claims_oracle_mismatch": _sample([] if not claims_oracle else [
                q for q in sorted(claims_oracle)
                if (q in a["rows"] and a["rows"][q]["oracle"] != claims_oracle[q])
                or (q in b["rows"] and b["rows"][q]["oracle"] != claims_oracle[q])
            ]),
            "claims_oracle_compared": len(claims_oracle or {}),
            "answer_source_mismatch": _sample(engine_mismatch),
            "committed_detail_mismatch": _sample(detail_mismatch),
            "answer_source_compared": source_compared,
            "deterministic_zone_identical": zone_identical,
        },
        "mcnemar": {
            "discordant_a_only": counts["a_only"],
            "discordant_b_only": counts["b_only"],
            "p_exact_two_sided": mcnemar_exact(counts["a_only"], counts["b_only"]),
        },
    }


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_report(label: str, a: dict, b: dict, analysis: dict) -> str:
    """Markdown tables, ready to paste into the CHANGELOG or the scorecard."""
    counts = analysis["counts"]
    rows_a, rows_b = len(a.get("rows") or {}), len(b.get("rows") or {})
    if rows_a and rows_b and not (rows_a == rows_b == analysis["shared"]):
        overlap = f"paired on {analysis['shared']} of {rows_a} (A) / {rows_b} (B) rows"
    else:
        overlap = f"paired on {analysis['shared']} shared questions"
    lines = [
        f"### Model sensitivity - {label}",
        f"A = `{a['model']}`, B = `{b['model']}`, {overlap}",
        "",
        "| Zone | N | A | B | Delta | A-only | B-only | p (exact McNemar) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def row(name: str, c: dict[str, int]) -> str:
        p = mcnemar_exact(c["a_only"], c["b_only"])
        return (f"| {name} | {c['n']} | {_pct(accuracy_a(c))} | {_pct(accuracy_b(c))} "
                f"| {delta_pp(c):+.2f}pp | {c['a_only']} | {c['b_only']} | {p:.4f} |")

    lines.append(row("**All questions**", counts))
    for zone, human in (("deterministic", "deterministic zone (0 LLM calls)"),
                        ("llm", "LLM zone (reader answers)")):
        if zone in analysis["zones"]:
            lines.append(row(human, analysis["zones"][zone]))

    lines += ["", "| Group | N | A | B | Delta | A-only | B-only |", "|---|---:|---:|---:|---:|---:|---:|"]
    for g in analysis["groups"]:
        c = g["counts"]
        lines.append(f"| {g['name']} | {c['n']} | {_pct(accuracy_a(c))} | {_pct(accuracy_b(c))} "
                     f"| {delta_pp(c):+.2f}pp | {c['a_only']} | {c['b_only']} |")

    ctl = analysis["controls"]
    lines += [
        "",
        f"Controls: tokens mismatched {ctl['token_mismatch']['n']}, "
        f"oracle mismatched {ctl['oracle_mismatch']['n']}, "
        f"only in A {ctl['only_in_a']['n']}, only in B {ctl['only_in_b']['n']}, "
        f"claims missing {ctl['claims_missing']['n']}",
    ]
    zone_mode = analysis.get("zone_mode")
    if zone_mode == "mixed":
        lines.append("!! Zone mode: mixed - one arm ran the committer and the other did not, so the "
                     "deterministic-zone row compares a deterministic answerer against a reader. "
                     "That is an A/B of the committer, not a model-sensitivity pairing: pair two "
                     "committing arms or two reader arms.")
    elif ctl["deterministic_zone_identical"] is True:
        lines.append("Deterministic zone identical under both readers: yes (the committer is model-independent).")
    elif ctl["deterministic_zone_identical"] is False:
        lines.append("!! Deterministic zone MOVED between readers - that is a determinism bug, not a result.")
    if zone_mode == "counterfactual":
        # The claims describe the deployed system, but neither arm skipped the
        # reader here, so the row is the readers' accuracy on those questions.
        lines.append("Zone mode: counterfactual - the "
                     f"{(analysis.get('zones', {}).get('deterministic') or {}).get('n', 0)} "
                     "deterministic-zone questions were answered by the reader in both arms, so "
                     "that row is the readers' own accuracy on questions the deployed committer "
                     "answers in code. The operational delta is the LLM-zone row.")
    if ctl.get("claims_oracle_compared"):
        compared = ctl["claims_oracle_compared"]
        if ctl["claims_oracle_mismatch"]["n"]:
            lines.append(f"!! the offline claims cache disagrees with what the runs retrieved on "
                         f"{ctl['claims_oracle_mismatch']['n']} of {compared} questions (the "
                         "committer judged a context the reader never saw); the zone split is a "
                         "caveat there, the reader comparison is not")
        else:
            lines.append("The offline claims cache agrees with the runs' own oracle recall on "
                         f"every one of the {compared} questions it covers.")
    for key, human in (("token_mismatch", "retrieval token counts differ"),
                       ("oracle_mismatch", "oracle recall differs"),
                       ("claims_missing", "questions absent from the claims file")):
        if ctl[key]["n"]:
            lines.append(f"!! {ctl[key]['n']} {human}: {', '.join(ctl[key]['sample'])}")
    engine_controls = (
        ("answer_source_mismatch", "questions answered by a different engine in the two arms"),
        ("committed_detail_mismatch", "questions committed by both arms at a different anchor"),
    )
    for key, human in engine_controls:
        if ctl[key]["n"]:
            lines.append(f"!! {ctl[key]['n']} {human}: {', '.join(ctl[key]['sample'])}")
    # Question-set overlap: one arm covering more than the other is a *different*
    # slice having been run alongside this one (the other LoCoMo categories, say),
    # which costs nothing - the pairing runs on the overlap.  Only a question B
    # lost, or a set that is not nested at all, threatens the comparison.
    only_a, only_b = ctl["only_in_a"]["n"], ctl["only_in_b"]["n"]
    if only_a and only_b:
        lines.append(f"!! {only_a} questions only A covers and {only_b} only B covers: the arms "
                     "are not nested, so the pairing silently runs on the intersection")
    elif only_a:
        lines.append(f"!! {only_a} questions missing from B: {', '.join(ctl['only_in_a']['sample'])}")
    elif only_b:
        lines.append(f"B is a superset of A (+{only_b} questions outside this slice); the pairing "
                     "runs on A's questions only")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pair two reader runs of the same slice to measure model sensitivity",
    )
    ap.add_argument("--a", required=True, help="reference run artefact (the bigger reader)")
    ap.add_argument("--b", required=True, help="candidate run artefact (the smaller reader)")
    ap.add_argument("--claims", default=None,
                    help="per-question attribution from `commit_sweep.py --dump-committed`; "
                         "splits the table into the deterministic zone and the LLM zone")
    ap.add_argument("--label", default=None, help="slice label (default: the --a file stem)")
    ap.add_argument("--cohort", default=None,
                    help="question set this pairing belongs to (default: the --claims file stem, "
                         "else the --a file stem); the scorecard pools one pairing per cohort, "
                         "so two pairings of the same questions share a cohort")
    ap.add_argument("--restrict-both-reader", action="store_true",
                    help="count only questions neither arm let the committer answer: a repeat "
                         "control, i.e. what the reader does with identical inputs twice")
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="write the raw counts as a scorecard artefact "
                         "(e.g. benchmark_results/model_sensitivity/locomo_cat2_temporal.json)")
    args = ap.parse_args()

    for path in (args.a, args.b):
        if not (REPO / path).exists():
            print(f"!! missing run artefact {path}")
            return 2

    a = load_run(str(REPO / args.a))
    b = load_run(str(REPO / args.b))
    if not a["rows"] or not b["rows"]:
        print(f"!! a run artefact carries no results: {args.a if not a['rows'] else args.b}")
        return 2

    claims: dict[str, bool] | None = None
    claims_oracle: dict[str, bool] = {}
    if args.claims:
        if (REPO / args.claims).exists():
            claims, claims_oracle = load_claims(str(REPO / args.claims))
        else:
            # Degrade to "not measured" rather than to a wrong zone split.
            print(f"!! missing claims file {args.claims}; reporting without the zone split")

    label = args.label or Path(args.a).stem
    cohort = args.cohort or (Path(args.claims).stem if claims is not None
                             else Path(args.a).stem)
    analysis = pair_runs(a, b, claims, claims_oracle,
                         restrict_both_reader=args.restrict_both_reader)
    print(render_report(label, a, b, analysis))

    if args.out:
        payload = {
            "label": label,
            "cohort": cohort,
            "a": {"run": args.a, "model": a["model"]},
            "b": {"run": args.b, "model": b["model"]},
            "claims": args.claims if claims is not None else None,
            **analysis,
        }
        path = REPO / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        shown = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        print(f"\nSaved model-sensitivity artefact to {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


