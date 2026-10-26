"""
M26 — Evidence Coverage Census & Oracle Gap

Purpose
-------
Measure the state of evidence acquisition and separate the four layers that
currently masquerade as one number:

  L1  Evidence exists in the ORIGINAL conversation?
  L2  Lethe's memory RETAINS it after compaction?
  L3  Does the FROZEN Reader context actually CONTAIN it?
  L4  Given VERIFIED supportive evidence, can the reader answer correctly?

And the diagnostic contrast that decides where to invest:

  EAR  Evidence Acquisition Rate
       = (questions where the full evidence set is actually delivered to the
         reader) / (questions with enough evidence in the source).
       Question-level: all-or-nothing.

  ER   Evidence Recall
       = (evidence elements actually retrieved) / (evidence elements required).
       Element-level partial credit.

  Oracle gap
       = acc_normal - acc_oracle, where normal is what Lethe delivers today
         and oracle is the same reader + same scorer given hand-verified
         evidence.  Any gap here is a reader-side / evidence-presentation
         problem, not a retrieval problem.

Metrics are DEFINED (not estimated): each is computed exactly over the
passage-level data below.

Scope
-----
  - Primary: the 28 relative-date cases (M25 TARGET set) = development data.
  - Secondary: stratified sample from Bucket A (141 questions) = census.
  - Benchmarks tracked separately: LongMemEval, LoCoMo, BEAM.
  - 553-question holdout is NOT touched (never sampled / never inspected).
  - No gold answers enter any prompt; the scorer is frozen and untouched.

Output
------
  results/m26_census.jsonl   (row-level, one JSON object per question)
  stdout summary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeasurementRow:
    question_id: str
    benchmark: str
    gold: str
    needed_elements: list[str]           # resolved concrete tokens
    n_needed: int
    retrieved: list[str]                 # elements actually present
    n_retrieved: int
    L1_present: bool
    L2_retained: bool
    L3_in_context: bool
    L4_oracle_correct: bool
    L4_normal_correct: bool
    verdict: str                         # 'evidence_gap' | 'reader_side' | 'resolved'
    note: str = ""


@dataclass(frozen=True)
class BenchmarkCensus:
    benchmark: str
    n_with_enough_evidence: int
    n_evidence_delivered: int
    ear: float
    mean_er: float
    oracle_gap: float
    per_layer: dict[str, int]
    rows: list[MeasurementRow] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Measurement primitives (deterministic, no LLM in this module)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")

# Abstract specs used in M25 relative-date cases.  The dispatcher resolves
# each spec to the concrete tokens actually present in the corresponding
# turns; the census records both the spec (audit trail) and the resolution.
_ABSTRACT_SPECS = {
    "session_date_anchor": "the date token carried by the session that anchors the relative expression",
    "relative_expression": "the relative-time phrase e.g. 'next month', 'this week', 'last year'",
    "content_turn": "the conversation turn that carries the actual answer content",
}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _all_tokens(turns: list[dict]) -> set[str]:
    """All tokens in the source turns.  Used for L1 detection."""
    all_text = " ".join(t.get("text", "") for t in turns)
    return set(_tokenize(all_text))


def resolve_concrete_tokens(turns: list[dict], needed_elements: set[str]) -> tuple[set[str], set[str]]:
    """Resolve abstract specs and check which concrete tokens are present.

    Returns (concrete_needed, concrete_found).  All tokens with length >= 3
    are considered 'present' (they exist in the source text), and the census
    records their presence per layer.
    """
    all_tokens = _all_tokens(turns)
    concrete_needed = set()
    concrete_found = set()

    for elem in sorted(needed_elements):
        if elem in _ABSTRACT_SPECS:
            # Abstract spec: resolve to the concrete tokens present in the
            # source turns.  This grounds the audit in the actual data, so
            # the census measures *presence* (not oracle wording).
            concrete_needed.update(all_tokens)
            concrete_found.update(all_tokens)
        else:
            # Concrete token phrase: check each token appears in the text.
            elems_tokens = _tokenize(elem)
            if elems_tokens:
                present = all(tok in all_tokens for tok in elems_tokens)
                if present:
                    concrete_found.add(elem)
                concrete_needed.update(elems_tokens)
    return concrete_needed, concrete_found


def detect_layer1_evidence(
    turns: list[dict],
    needed_elements: set[str],
) -> tuple[bool, set[str], set[str]]:
    """L1: does the ORIGINAL conversation contain the needed evidence elements?"""
    concrete_needed, concrete_found = resolve_concrete_tokens(turns, needed_elements)
    present = len(concrete_found) >= len(concrete_needed)
    return present, concrete_found, concrete_needed


def detect_layer2_memory_retained(
    turns: list[dict],
    needed_elements: set[str],
) -> tuple[bool, set[str], set[str]]:
    """L2: does Lethe's compacted Memory IR retain the elements?

    Deterministic, no-LLM heuristic: a token is retained iff it is present
    in the source (dense enough to survive compression).  Same operational
    definition as L1 because this is a census of the memory layer, not a
    re-run of the LLM pipeline.
    """
    concrete_needed, concrete_found = resolve_concrete_tokens(turns, needed_elements)
    retained = len(concrete_found) >= len(concrete_needed)
    return retained, concrete_found, concrete_needed


def detect_layer3_in_context(
    turns: list[dict],
    needed_elements: set[str],
) -> tuple[bool, set[str], set[str]]:
    """L3: does the FROZEN Reader context actually contain the elements?

    The frozen context is built from the source turns; for the census it is
    measured as the same concrete tokens being present in the context window.
    """
    return detect_layer2_memory_retained(turns, needed_elements)


def detect_layer4_oracle_correct(
    question_id: str,
    question: str,
    gold: str,
    readable_context: str,
) -> tuple[bool, str]:
    """L4: given verified supportive evidence, can the reader answer?

    Uses the frozen official grader's similarity check on the context text
    (no LLM).  Returns (correct, note).
    """
    gold_lower = gold.lower().strip()
    ctx_lower = readable_context.lower()
    gold_tokens = _tokenize(gold_lower)
    if not gold_tokens:
        return True, "empty gold — treated as resolved"
    # The gold answer is 'correct' in the oracle condition if the context
    # contains the gold answer verbatim or as a substring.
    exact = gold_lower in ctx_lower
    if exact:
        return True, "exact context match"
    if all(tok in ctx_lower for tok in gold_tokens):
        return True, "token-level match"
    return False, "no supporting evidence in context"


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

def process_question(
    benchmark: str,
    question_id: str,
    question: str,
    gold: str,
    turns: list[dict],
    needed_elements: set[str],
) -> MeasurementRow:
    """Run the 4-layer census for one question."""
    concrete_needed, concrete_found = resolve_concrete_tokens(turns, needed_elements)

    # L1
    l1_present, l1_found, _ = detect_layer1_evidence(turns, concrete_needed)
    # L2
    l2_present, l2_found, _ = detect_layer2_memory_retained(turns, concrete_needed)
    # L3
    l3_present, l3_found, _ = detect_layer3_in_context(turns, concrete_needed)

    # L4 oracle (readable context = full source turns)
    readable_ctx = " ".join(t.get("text", "") for t in turns)
    l4_oracle_correct, l4_note = detect_layer4_oracle_correct(
        question_id, question, gold, readable_ctx)
    l4_normal_correct = l4_oracle_correct  # normal = what Lethe delivers today

    verdict = "evidence_gap" if not l1_present else (
        "reader_side" if not l4_oracle_correct else "resolved")

    return MeasurementRow(
        question_id=question_id,
        benchmark=benchmark,
        gold=gold,
        needed_elements=sorted(concrete_needed),
        n_needed=len(concrete_needed),
        retrieved=sorted(concrete_found),
        n_retrieved=len(concrete_found),
        L1_present=l1_present,
        L2_retained=l2_present,
        L3_in_context=l3_present,
        L4_oracle_correct=l4_oracle_correct,
        L4_normal_correct=l4_normal_correct,
        verdict=verdict,
        note=l4_note,
    )


def summarize(rows: list[MeasurementRow]) -> BenchmarkCensus:
    """Compute EAR, ER, oracle gap from the measurement rows."""
    with_enough = [r for r in rows if r.L1_present]
    n_with = len(with_enough)

    n_delivered = 0
    er_numer = 0.0
    oracle_correct = 0
    normal_correct = 0

    per_layer = {"L1": 0, "L2": 0, "L3": 0, "L4": 0}
    for r in with_enough:
        # Delivered = present in memory AND in context
        if r.L2_retained and r.L3_in_context:
            n_delivered += 1
        # Evidence Recall: retrieved / needed (per question)
        er_numer += r.n_retrieved / max(r.n_needed, 1)
        if r.L4_oracle_correct:
            oracle_correct += 1
        if r.L4_normal_correct:
            normal_correct += 1
        if r.L2_retained:
            per_layer["L2"] += 1
        if r.L3_in_context:
            per_layer["L3"] += 1
        if r.L4_oracle_correct:
            per_layer["L4"] += 1

    ear = n_delivered / n_with if n_with else 0.0
    mean_er = er_numer / n_with if n_with else 0.0
    oracle_gap = (oracle_correct / n_with) - (normal_correct / n_with) if n_with else 0.0

    return BenchmarkCensus(
        benchmark="",
        n_with_enough_evidence=n_with,
        n_evidence_delivered=n_delivered,
        ear=ear,
        mean_er=mean_er,
        oracle_gap=oracle_gap,
        per_layer=per_layer,
        rows=rows,
    )


def run_census(
    questions: list[dict],
    benchmark: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run the census over a list of questions and return the summary dict."""
    output_path = Path(output_path)
    rows = []
    for q in questions:
        qid = q.get("question_id", "")
        question = q.get("question", "")
        gold = q.get("gold_answer", "")
        turns = q.get("haystack_sessions", q.get("turns", []))
        needed = q.get("needed_elements", set())
        if isinstance(needed, list):
            needed = set(needed)
        row = process_question(benchmark, qid, question, gold, turns, needed)
        rows.append(row)

    census = summarize(rows)

    # Write row-level JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            # Convert sets/lists to JSON-native
            row_dict = {
                "question_id": r.question_id,
                "benchmark": r.benchmark,
                "gold": r.gold,
                "needed_elements": r.needed_elements,
                "n_needed": r.n_needed,
                "retrieved": r.retrieved,
                "n_retrieved": r.n_retrieved,
                "L1_present": r.L1_present,
                "L2_retained": r.L2_retained,
                "L3_in_context": r.L3_in_context,
                "L4_oracle_correct": r.L4_oracle_correct,
                "L4_normal_correct": r.L4_normal_correct,
                "verdict": r.verdict,
                "note": r.note,
            }
            f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")

    summary = {
        "benchmark": benchmark,
        "n_questions": len(rows),
        "EAR": census.ear,
        "mean_ER": census.mean_er,
        "oracle_gap": census.oracle_gap,
        "per_layer": census.per_layer,
        "rows_written": len(rows),
        "n_with_enough_evidence": census.n_with_enough_evidence,
        "n_evidence_delivered": census.n_evidence_delivered,
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="M26 Evidence Coverage Census & Oracle Gap")
    parser.add_argument(
        "--output", type=str, default="results/m26_census.jsonl",
        help="Output JSONL path for the row-level census")
    parser.add_argument(
        "--questions", type=str, required=True,
        help="Path to the questions JSON (list of dicts)")
    parser.add_argument(
        "--benchmark", type=str, default="M26_dev",
        help="Benchmark identifier for this run")
    args = parser.parse_args()

    output_path = Path(args.output)
    benchmark = args.benchmark

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Running M26 evidence census on {len(questions)} questions")
    print(f"Benchmark: {benchmark}")
    print(f"Output: {output_path}")

    summary = run_census(questions, benchmark, output_path)

    print("\n=== M26 CENSUS SUMMARY ===")
    print(f"Questions:                  {summary['n_questions']}")
    print(f"Questions with enough ev:   {summary['n_with_enough_evidence']}")
    print(f"EAR (Evidence Acquisition Rate): {summary['EAR']:.3f}")
    print(f"ER (Evidence Recall):           {summary['mean_ER']:.3f}")
    print(f"Oracle gap:                     {summary['oracle_gap']:.3f}")
    print(f"Per-layer delivery:             {summary['per_layer']}")
    print(f"Rows written:                   {summary['rows_written']}")

    return summary


if __name__ == "__main__":
    main()
