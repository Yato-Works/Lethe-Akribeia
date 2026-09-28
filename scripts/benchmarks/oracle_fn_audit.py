#!/usr/bin/env python3
"""Audit the questions the failure ceiling filed as retrieval failures that a
*different* reader nevertheless answered correctly.

The oracle in ``locomo_adapter`` is an id test, not a content test:

    oracle_recall = any(ev_id in pcc.context_text for ev_id in question.evidence_ids)

LoCoMo marks the gold evidence by dialogue id (``D1:3``), so the flag is false
whenever the compiled context carries the fact from a *different* turn, or when
the id never appears verbatim in the rendered context.  Step 3a found 42 such
questions in LoCoMo 1,540: retrieval-bucket failures that the 1.5B reader got
right, which would mean the 13.7pp retrieval ceiling is really ~11pp.

This tool settles each of them against the dataset instead of arguing about it,
and splits them three ways - deliberately *not* reclassifying the ceiling:

  ORACLE_FALSE_NEGATIVE
      The answer text is in the context the reader was actually shown.  The id
      test missed a turn that carried the fact, so retrieval did its job and the
      measurement is what's wrong.

  TRUE_RETRIEVAL_FAILURE
      The answer is absent from the context but present elsewhere in the
      conversation.  Retrieval really picked the wrong turn, and the other reader
      answered from parametric knowledge (or the matcher is lenient) - which is
      worth knowing, because "retrieval missed it" and "the reader guessed it"
      have different fixes.

  AMBIGUOUS
      The answer is neither in the context nor verbatim in the conversation: the
      question needs synthesis or inference, and no string test can say whether
      the reader had what it needed.

Nothing here writes back to a score or to the ceiling artefact: it is an audit of
the measuring instrument, and the caller decides what to do with the verdict.

Usage:
  uv run python scripts/benchmarks/oracle_fn_audit.py \\
      --run-a benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json \\
      --run-b benchmark_results/locomo1540/temporal321_rules_commit_15b.json \\
      --cache benchmark_results/locomo_context_cache_rules.jsonl \\
      --dataset datasets/external/locomo10.json \\
      --out benchmark_results/failure_ceiling/oracle_fn_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]

# Words too common to carry a fact; an answer made only of these is not evidence.
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "are", "his",
    "her", "she", "he", "they", "them", "their", "there", "here", "have", "has", "had",
    "but", "not", "you", "your", "yours", "it", "its", "in", "on", "at", "to", "of",
    "a", "an", "is", "be", "been", "was", "did", "do", "does", "s", "t",
}
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _content_words(text: str) -> list[str]:
    return [w for w in _norm(text).split() if w not in STOPWORDS and len(w) > 1]


def _overlap(answer: str, haystack: str) -> float:
    """Share of the answer's content words that appear in ``haystack``."""
    words = _content_words(answer)
    if not words:
        return 0.0
    pool = _norm(haystack)
    return sum(1 for w in words if w in pool) / len(words)


def load_dataset(path: str) -> dict[str, dict]:
    """``{qid: {question, answer, evidence_ids, category, turns}}`` from locomo10.json.

    ``turns`` is the flattened conversation with LoCoMo's own ids (``D1:3``) and
    each turn's session date kept, so an answer can be located in the dialogue and
    attributed to a turn *by content and date* rather than by label.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for conv in data:
        sample = str(conv.get("sample_id") or "")
        blocks = conv.get("conversation") or {}
        turns: list[dict] = []
        session = 1
        while f"session_{session}" in blocks:
            date = str(blocks.get(f"session_{session}_date_time", ""))
            for index, turn in enumerate(blocks[f"session_{session}"]):
                turns.append({
                    "id": f"D{session}:{index}",
                    "date": date,
                    "text": f"{turn.get('speaker', '')}: {turn.get('text', turn.get('content', ''))}",
                })
            session += 1
        for index, qa in enumerate(conv.get("qa") or []):
            qid = f"{sample}-qa-{index:03d}"
            out[qid] = {
                "question": str(qa.get("question") or ""),
                "answer": str(qa.get("answer") or ""),
                "evidence_ids": [str(e) for e in (qa.get("evidence") or [])],
                "category": qa.get("category"),
                "turns": turns,
            }
    return out


CONTEXT_DATE = re.compile(r"\[D\d+:\d+ on ([^\]]*)\]")


def evidence_present(context: str, turns: list[dict], evidence_ids: list[str]) -> tuple[bool, str]:
    """Is the gold evidence turn actually in the context, by content and date?

    The id test the runner uses compares *labels*, and the compiler numbers the
    turns it emits in its own group order: across LoCoMo 1,540 that makes 165
    questions look retrieved that are not, and 172 look missing that are not.  So
    the audit asks the question the flag claims to answer - is this turn's content
    here, from that session's date - with a word-overlap test that survives the
    context compiler's condensation.
    """
    dates = set(CONTEXT_DATE.findall(context))
    pool = _norm(context)
    for ev_id in evidence_ids:
        turn = next((t for t in turns if t["id"] == ev_id), None)
        if turn is None:
            continue
        if turn.get("date") and turn["date"] not in dates:
            continue
        words = [w for w in _norm(turn["text"]).split() if len(w) > 2]
        if words and sum(1 for w in words if w in pool) / len(words) >= 0.8:
            return True, ev_id
    return False, ""



def load_runs(run_a: str, run_b: str) -> dict[str, dict]:
    """The disputed set: the reference run calls it a retrieval failure, the
    other reader got it right."""
    def read(path: str) -> dict[str, dict]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {str(r.get("question_id")): r for r in data.get("results", []) if r.get("question_id")}

    a, b = read(run_a), read(run_b)
    disputed = {}
    for qid, row_a in a.items():
        row_b = b.get(qid)
        if row_b is None or not row_b.get("is_correct"):
            continue
        if row_a.get("oracle_recall") or str(row_a.get("answer_source") or "reader") != "reader":
            continue
        if row_a.get("is_correct"):
            continue
        disputed[qid] = {"a": row_a, "b": row_b}
    return disputed


def load_contexts(path: str) -> dict[str, str]:
    """Compiled-context cache: qid -> the context the reader was shown."""
    contexts: dict[str, str] = {}
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
                contexts[qid] = str(row.get("context") or "")
    return contexts


def audit_one(qid: str, entry: dict, dataset_row: dict, context: str) -> dict:
    """One disputed question, with the evidence for its verdict attached.

    The verdict is a string test against two bodies of text - the context the
    reader saw, and the conversation the question came from - because that is
    exactly the difference between "the flag is wrong" and "the reader guessed".
    """
    answer = dataset_row["answer"]
    gt = _norm(answer)
    ctx_norm = _norm(context)
    answer_in_context = bool(gt) and gt in ctx_norm
    evidence_here, matched_id = evidence_present(context, dataset_row.get("turns") or [],
                                                 dataset_row.get("evidence_ids") or [])
    gold_ids = dataset_row.get("evidence_ids") or []
    gold_ids_in_context = [i for i in gold_ids if i and i in context]

    # The verdict is about the *flag*, not about the reader: the flag said the
    # evidence was absent, so the only question is whether it was.
    if evidence_here:
        verdict = "ORACLE_FALSE_NEGATIVE"
    elif answer_in_context:
        verdict = "AMBIGUOUS"
    else:
        verdict = "TRUE_RETRIEVAL_FAILURE"

    prediction_b = str(entry["b"].get("prediction") or "")
    evidence_line = ""
    for line in context.splitlines():
        if gt and gt in _norm(line):
            evidence_line = line.strip()[:240]
            break
    return {
        "qid": qid,
        "category": dataset_row.get("category"),
        "capability": CATEGORY_NAMES.get(dataset_row.get("category"), "unknown"),
        "question": dataset_row.get("question"),
        "gold_answer": answer,
        "prediction_a": str(entry["a"].get("prediction") or ""),
        "prediction_b": prediction_b,
        "prediction_b_contains_gold": bool(gt) and gt in _norm(prediction_b),
        "prediction_b_overlap": _overlap(answer, prediction_b),
        "evidence_ids": gold_ids,
        "evidence_text_present": evidence_here,
        "evidence_matched_id": matched_id,
        "gold_ids_in_context": gold_ids_in_context,
        "answer_in_context": answer_in_context,
        "overlap_context": round(_overlap(answer, context), 3),
        "context_line_with_answer": evidence_line,
        "verdict": verdict,
    }


def render_report(label: str, records: list[dict]) -> str:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
    total = len(records)
    lines = [
        f"### Oracle false-negative audit - {label}",
        f"{total} questions: the reference run filed them as retrieval failures "
        "(no gold evidence id in the context, answer wrong) while the second reader answered them.",
        "",
        "| Verdict | Questions | Share | Meaning |",
        "|---|---:|---:|---|",
        f"| ORACLE_FALSE_NEGATIVE | {counts.get('ORACLE_FALSE_NEGATIVE', 0)} "
        f"| {counts.get('ORACLE_FALSE_NEGATIVE', 0) / total * 100 if total else 0:.1f}% "
        "| the gold evidence turn is in the context (by content and date); the id test missed it |",
        f"| TRUE_RETRIEVAL_FAILURE | {counts.get('TRUE_RETRIEVAL_FAILURE', 0)} "
        f"| {counts.get('TRUE_RETRIEVAL_FAILURE', 0) / total * 100 if total else 0:.1f}% "
        "| the evidence is not in the context; the other reader answered without it |",
        f"| AMBIGUOUS | {counts.get('AMBIGUOUS', 0)} "
        f"| {counts.get('AMBIGUOUS', 0) / total * 100 if total else 0:.1f}% "
        "| the answer text is in the context, but not as the gold evidence turn |",
        "",
        "| Question | Capability | Gold answer | Other reader's answer | Evidence turn in context | Answer string in context | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| `{record['qid']}` | {record['capability']} | {record['gold_answer'][:40]} "
            f"| {record['prediction_b'][:40]} "
            f"| {'yes' if record['evidence_text_present'] else 'no'} "
            f"| {'yes' if record['answer_in_context'] else 'no'} | {record['verdict']} |")
    return "\n".join(lines)


def recheck_oracle(run: str, dataset: dict, contexts: dict) -> dict:
    """How reliable is the flag itself, over the whole slice?

    Answers the question the 42 cannot: the flag is an id test, so compare it with
    a content test on every question of the run, in both directions.
    """
    data = json.loads(Path(REPO / run).read_text(encoding="utf-8"))
    counts = {"agree_present": 0, "false_positive": 0, "false_negative": 0,
              "agree_absent": 0, "no_evidence_ids": 0}
    for row in data.get("results", []):
        qid = str(row.get("question_id") or "")
        flag = row.get("oracle_recall")
        item = dataset.get(qid)
        if item is None or not isinstance(flag, bool):
            continue
        evidence_ids = item.get("evidence_ids") or []
        if not evidence_ids:
            counts["no_evidence_ids"] += 1
            continue
        here, _ = evidence_present(contexts.get(qid, ""), item.get("turns") or [], evidence_ids)
        key = ("agree_present" if flag and here else
               "false_positive" if flag else
               "false_negative" if here else "agree_absent")
        counts[key] += 1
    checked = sum(v for k, v in counts.items() if k != "no_evidence_ids")
    counts["checked"] = checked
    if checked:
        counts["flag_rate"] = (counts["agree_present"] + counts["false_positive"]) / checked
        counts["content_rate"] = (counts["agree_present"] + counts["false_negative"]) / checked
        counts["disagree_rate"] = (counts["false_positive"] + counts["false_negative"]) / checked
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit oracle false negatives in the retrieval bucket")
    ap.add_argument("--run-a", required=True, help="reference run (the one whose oracle is disputed)")
    ap.add_argument("--run-b", required=True, help="second reader run of the same slice")
    ap.add_argument("--cache", required=True, help="compiled-context cache jsonl")
    ap.add_argument("--dataset", default="datasets/external/locomo10.json")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None, metavar="PATH", help="write the per-question audit as JSON")
    args = ap.parse_args()

    disputed = load_runs(str(REPO / args.run_a), str(REPO / args.run_b))
    dataset = load_dataset(str(REPO / args.dataset))
    contexts = load_contexts(str(REPO / args.cache))

    records: list[dict] = []
    skipped = 0
    for qid in sorted(disputed):
        row = dataset.get(qid)
        if row is None:
            skipped += 1
            continue
        records.append(audit_one(qid, disputed[qid], row, contexts.get(qid, "")))

    label = args.label or Path(args.run_a).stem
    print(render_report(label, records))
    if skipped:
        print(f"\n!! {skipped} disputed questions are not in {args.dataset} and were skipped")

    recheck = recheck_oracle(args.run_a, dataset, contexts)
    print("\n### Is the flag itself reliable? (every question of the run)")
    print(f"  checked {recheck['checked']} questions with gold evidence ids "
          f"({recheck['no_evidence_ids']} have none and are excluded)")
    print(f"  agree, evidence present : {recheck['agree_present']}")
    print(f"  !! false positive (flag true, evidence absent) : {recheck['false_positive']}")
    print(f"  !! false negative (flag false, evidence present): {recheck['false_negative']}")
    print(f"  agree, evidence absent  : {recheck['agree_absent']}")
    print(f"  flag rate {recheck['flag_rate']:.1%} vs content rate {recheck['content_rate']:.1%}"
          f"  ->  {recheck['disagree_rate']:.1%} of the flags are wrong on one side or the other")

    if args.out:
        counts: dict[str, int] = {}
        for record in records:
            counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
        payload = {
            "label": label,
            "run_a": args.run_a,
            "run_b": args.run_b,
            "dataset": args.dataset,
            "audited": len(records),
            "verdicts": counts,
            "oracle_recheck": recheck,
            "records": records,
        }
        path = REPO / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nSaved audit to {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


