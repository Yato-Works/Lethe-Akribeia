#!/usr/bin/env python3
"""Build a gold-evidence context cache for the questions nothing explained yet.

Step 3a/b left 378 of the 1,540 questions in one bucket: the gold evidence turn
**is** in the context, a reader was asked, and the answer was wrong.  That bucket
is three hypotheses wearing one coat - context too noisy to read, reasoning
failure, genuinely semantic - and no stored artefact can tell them apart.

One controlled experiment can: hand the reader the *gold evidence and nothing
else* and see what it does.  Same frozen prompt, same matcher, same verifier, same
committer switch; the only difference is that the context now contains the
answer's own turns and no distractors.  Whatever it recovers is a ceiling for the
context pipeline rather than a score the system has earned.

The output is a context cache in the runner's own format, so
``run_locomo_1540.py --cache-file <this>`` replays it through the frozen path
with no code change and no special-casing.  The gold context renders each
evidence turn exactly as the compiler does - ``[D1:3 on <session date>] Speaker:
text`` - so the prompt looks the same and the corrected oracle recognises it.

Usage:
  uv run python scripts/benchmarks/gold_context_cache.py \\
      --run benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.oraclefix.json \\
      --out benchmark_results/locomo_gold_context_cache.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_audit = _load("_oracle_fn_audit_shared", REPO / "scripts" / "benchmarks" / "oracle_fn_audit.py")


def unresolved_qids(run_path: str | Path, dataset: dict, contexts: dict) -> list[str]:
    """Wrong for the reference reader, not committed, and the evidence was there.

    Deliberately the same three conditions the failure ceiling uses, spelled out
    here so the experiment's input cannot drift from the census that produced it.
    """
    path = Path(run_path)
    if not path.is_absolute():
        path = REPO / path
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in data.get("results", []):
        qid = str(row.get("question_id") or "")
        if row.get("is_correct") or str(row.get("answer_source") or "reader") != "reader":
            continue
        item = dataset.get(qid)
        if item is None or not item.get("evidence_ids"):
            continue
        here, _ = _audit.evidence_present(contexts.get(qid, ""), item.get("turns") or [],
                                          item["evidence_ids"])
        if here:
            out.append(qid)
    return sorted(out)


def gold_context(item: dict) -> str:
    """The evidence turns, in conversation order, rendered like a compiled context."""
    wanted = {str(e) for e in item.get("evidence_ids") or []}
    lines = [f"[{t['id']} on {t['date']}] {t['text']}" for t in item.get("turns") or []
             if t["id"] in wanted]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="reference run (use the oracle-corrected copy)")
    ap.add_argument("--out", required=True, help="gold context cache (jsonl)")
    ap.add_argument("--qids-out", default=None, help="also write the question ids (one per line)")
    ap.add_argument("--dataset", default="datasets/external/locomo10.json")
    ap.add_argument("--cache", default="benchmark_results/locomo_context_cache_rules.jsonl",
                    help="production context cache, for the size comparison")
    args = ap.parse_args()

    dataset = _audit.load_dataset(str(REPO / args.dataset))
    contexts = _audit.load_contexts(str(REPO / args.cache))
    qids = unresolved_qids(args.run, dataset, contexts)
    if not qids:
        print("!! no unresolved questions found - is the run the oracle-corrected copy?")
        return 2

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gold_words = prod_words = 0
    by_category: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as handle:
        for qid in qids:
            item = dataset[qid]
            context = gold_context(item)
            handle.write(json.dumps({
                "qid": qid, "question": item["question"], "category": item["category"],
                "context": context, "token_cost": len(context.split()),
                "is_abstention": False,
            }, ensure_ascii=False) + "\n")
            gold_words += len(context.split())
            prod_words += len(contexts.get(qid, "").split())
            key = str(item["category"])
            by_category[key] = by_category.get(key, 0) + 1

    if args.qids_out:
        (REPO / args.qids_out).write_text("\n".join(qids) + "\n", encoding="utf-8")

    print(f"wrote {len(qids)} gold-evidence contexts to {out_path.relative_to(REPO)}")
    print(f"  by category: {dict(sorted(by_category.items()))}")
    print(f"  gold context {gold_words / len(qids):.0f} words/question vs the production "
          f"context {prod_words / len(qids):.0f} - the distractor budget is "
          f"{prod_words / max(gold_words, 1):.1f}x the evidence")
    print("  every one of these questions had the evidence present and was still answered wrong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
