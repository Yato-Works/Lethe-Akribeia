"""Measure a refusal gate on frozen contexts, paired against a stored run.

Why this exists
    The LoCoMo adversarial category (5) is correct ONLY when the system declines
    in an accepted form.  After the protocol fix (commit fc22eea) the system
    already refuses 248/446; the remaining 198 are answered and score 0.  A gate
    that refuses them is worth up to +9.97pt of official F1 -- but every false
    refusal on categories 1-4 costs a correct answer.

    This script prices both sides on the SAME frozen contexts, using the pinned
    official F1, and reports the delta per scope.  It is a measurement, not a
    promotion: nothing here changes the runtime.

What "before" means
    The stored run's predictions, with refusals normalised to the accepted form
    (the protocol fix).  Category 5 questions that were answered score 0.

What "after" means
    The same predictions, except that a question the gate refuses is replaced by
    the canonical refusal.  So the delta is exactly
        (#category-5 newly refused) - (official F1 lost on categories 1-4).

Usage
    python scripts/benchmarks/measure_refusal_gate.py \
        --contexts benchmark_results/frozen_ctx/locomo_r7_20261024/contexts.jsonl \
        --run benchmark_results/locomo_full_20261023_010055.json \
        --scope dev

Sweeps the gate's overlap threshold and prints every variant, because a single
threshold chosen after seeing the number would be fitting.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

OFFICIAL_DIR = ROOT / "third_party" / "benchmarks" / "locomo"
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
#: Conversations reserved for validation; never used to select a threshold.
HOLDOUT = {"conv-42", "conv-48"}

CANONICAL_REFUSAL = "not mentioned"
ABSTENTIONS = {"i don't know.", "", "not mentioned", "none"}

#: Question words and light verbs that carry no topical content.
STOP = frozenset("""
what when where who whom which why how does did do was were is are have has had
would could should will can may might must
many much long often
the and but for with from about that this these those there here
she he they them her his their its him
caroline melanie
""".split())


def load_official():
    import types

    if importlib.util.find_spec("bert_score") is None:
        stub = types.ModuleType("bert_score")
        stub.score = lambda *a, **k: (None, None, [0.0])
        sys.modules["bert_score"] = stub
    spec = importlib.util.spec_from_file_location(
        "official_locomo_evaluation", OFFICIAL_DIR / "task_eval" / "evaluation.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(OFFICIAL_DIR))
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ text utils
_LINE_RE = re.compile(r"^\[[^\]]+\]\s*([A-Z][A-Za-z]+):\s*(.*)$")
NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")


def stem(word: str) -> str:
    """Crude suffix stripping, enough to make 'realizing' meet 'realize'."""
    for suffix in ("ing", "ed", "es", "s", "e"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def dialogue_lines(context: str) -> list[tuple[str, str]]:
    """(speaker, text) per dialogue line; certificate blocks are skipped."""
    out = []
    for raw in context.splitlines():
        line = raw.strip()
        if not line or line.startswith("[SPEAKER") or line.startswith("[/"):
            continue
        m = _LINE_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def speakers_in(lines: list[tuple[str, str]]) -> list[str]:
    seen: list[str] = []
    for speaker, _ in lines:
        if speaker not in seen:
            seen.append(speaker)
    return seen


def asked_subject(question: str, speakers: list[str]) -> str | None:
    """The single speaker the question is about, or None when ambiguous."""
    low = question.lower()
    hits = {s: len(re.findall(rf"\b{re.escape(s.lower())}\b", low)) for s in speakers}
    hits = {s: n for s, n in hits.items() if n}
    if len(hits) != 1:
        return None
    return next(iter(hits))


def content_stems(question: str, speakers: list[str]) -> set[str]:
    words = re.findall(r"[a-z]{4,}", question.lower())
    drop = {s.lower() for s in speakers}
    return {stem(w) for w in words if w not in STOP and w not in drop}


# ----------------------------------------------------------------------- gates
def gate_subject_absent(question: str, context: str, prediction: str,
                        threshold: float) -> bool:
    """Refuse when the best-matching evidence line never mentions the subject.

    Rationale: for an attribution-swap question ("What did X do after Y's Z?")
    the line that carries Y and Z is predicated of someone else and does not
    mention X at all.  A line that DOES mention X in any role is treated as
    ambiguous and never refused here.
    """
    lines = dialogue_lines(context)
    if not lines:
        return False
    speakers = speakers_in(lines)
    subject = asked_subject(question, speakers)
    if subject is None:
        return False
    terms = content_stems(question, speakers)
    if not terms:
        return False
    best = (0.0, "")
    for _speaker, text in lines:
        low = text.lower()
        overlap = sum(1 for t in terms if t in low) / len(terms)
        if overlap > best[0]:
            best = (overlap, text)
    overlap, text = best
    if overlap < threshold:
        return False
    return subject.lower() not in text.lower()


def gate_predicate_absent(question: str, context: str, prediction: str,
                          threshold: float) -> bool:
    """Stricter: the subject is named in the line, but not as the actor of it.

    Approximated on the name-normalised text (first person is rewritten to the
    speaker): the subject must appear in the line's opening clause, where its
    actor is stated.
    """
    lines = dialogue_lines(context)
    if not lines:
        return False
    speakers = speakers_in(lines)
    subject = asked_subject(question, speakers)
    if subject is None:
        return False
    terms = content_stems(question, speakers)
    if not terms:
        return False
    best = (0.0, "")
    for _speaker, text in lines:
        low = text.lower()
        overlap = sum(1 for t in terms if t in low) / len(terms)
        if overlap > best[0]:
            best = (overlap, text)
    overlap, text = best
    if overlap < threshold:
        return False
    head = text.lower()[: max(40, len(text) // 4)]
    return subject.lower() not in head


AUX = ("did", "does", "do", "has", "have", "had")


def predicate_stem(question: str, speakers: list[str]) -> str:
    """The verb the question asks about, from an auxiliary construction.

    "What did Caroline realize after ...?" -> "realiz".  Returns "" when the
    question has no auxiliary verb, because noun-predicate questions ("What are
    Melanie's pets' names?") would otherwise be read as a verb and refused.
    """
    words = re.findall(r"[a-z']+", question.lower())
    names = {s.lower() for s in speakers} | {s.lower() + "'s" for s in speakers}
    for i, word in enumerate(words):
        if word in AUX and i + 1 < len(words):
            j = i + 1
            while j < len(words) and (words[j] in STOP or words[j] in names):
                j += 1
            if j < len(words) and len(words[j]) >= 4:
                return stem(words[j])
    return ""


def gate_predicate_object(question: str, context: str, prediction: str,
                          threshold: float) -> bool:
    """Refuse when no line of the SUBJECT binds the asked predicate to its object.

    This is the attribution-swap detector.  For "What did Caroline realize after
    her charity race?", Caroline's turns mention the charity race and mention a
    realisation, but never in the same turn: the binding is Melanie's.  A single
    turn of the subject carrying predicate + objects means the evidence exists,
    and the gate stays out of the way.
    """
    lines = dialogue_lines(context)
    if not lines:
        return False
    speakers = speakers_in(lines)
    subject = asked_subject(question, speakers)
    if subject is None:
        return False
    pred = predicate_stem(question, speakers)
    if not pred:
        return False
    objects = {t for t in content_stems(question, speakers) if t != pred}
    if not objects:
        return False

    subject_lines = [text for spk, text in lines if spk.lower() == subject.lower()]
    if not subject_lines:
        subject_lines = [text for _spk, text in lines
                         if subject.lower() in text.lower()]
    for text in subject_lines:
        low = text.lower()
        if pred not in low:
            continue
        hit = sum(1 for obj in objects if obj in low) / len(objects)
        if hit >= threshold:
            return False
    return True


_MONTHS = frozenset("""
january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec
""".split())
_WEEKDAYS = frozenset("""
monday tuesday wednesday thursday friday saturday sunday
""".split())


def looks_temporal(text: str) -> bool:
    """A date-shaped answer.  Temporal questions are scored as dates, and a date
    legitimately appears in turns of BOTH speakers, so provenance attribution is
    the wrong test for them (measured: it cost 7.8pt on the temporal category)."""
    low = text.lower()
    if re.search(r"\b(19|20)\d{2}\b", low):
        return True
    if any(m in low for m in _MONTHS):
        return True
    if any(d in low for d in _WEEKDAYS):
        return True
    if re.fullmatch(r"[\d\s,./-]+", low.strip()):
        return True
    return False


def subject_is_actor(question: str, subject: str) -> bool:
    """True when the question's grammatical actor is `subject`.

    "What did Caroline realize ...?" -> yes.
    "What was grandma's gift to Caroline?" -> no: the actor is grandma, and the
    answer legitimately comes from grandma's turn.  Attributing the ANSWER to the
    named person is only a valid test when that person is the actor
    (measured: this precondition removes most of the single-hop false positives).
    """
    pat = rf"\b(?:did|does|do|has|have|had)\s+{re.escape(subject)}\b"
    return re.search(pat, question, re.I) is not None


def gate_answer_attribution(question: str, context: str, prediction: str,
                            threshold: float) -> bool:
    """Refuse when the ANSWER's content comes from someone else's turn.

    This is the attribution-swap signature, but anchored on the answer instead of
    on the question.  For "What did Caroline realize after her charity race?" the
    reader answers "self-care is important" -- a string that exists verbatim in
    MELANIE's turn.  Wrong-person provenance is direct evidence that the answer
    does not transfer to the asked subject.

    Questions that name no single speaker, answers whose content is not found in
    the context (paraphrase), and best-matches below the threshold are all left
    alone: the gate only fires when it can point at another person's line.
    """
    lines = dialogue_lines(context)
    if not lines or not prediction:
        return False
    if looks_temporal(prediction):
        return False
    speakers = speakers_in(lines)
    subject = asked_subject(question, speakers)
    if subject is None:
        return False
    if not subject_is_actor(question, subject):
        return False
    terms = {stem(w) for w in re.findall(r"[a-z]{4,}", prediction.lower())
             if w not in STOP and w not in {s.lower() for s in speakers}}
    if not terms:
        return False
    best = (0.0, "")
    for speaker, text in lines:
        low = text.lower()
        overlap = sum(1 for t in terms if t in low) / len(terms)
        if overlap > best[0]:
            best = (overlap, speaker)
    overlap, speaker = best
    if overlap < threshold:
        return False
    return speaker.lower() != subject.lower()


def gate_answer_attributed_dominant(question: str, context: str, prediction: str,
                                    threshold: float) -> bool:
    """As above, but also requires that NO subject turn carries the answer terms.

    Stricter: refuses only when the answer's wording exists in other people's
    turns and never in the subject's own.
    """
    lines = dialogue_lines(context)
    if not lines or not prediction:
        return False
    if looks_temporal(prediction):
        return False
    speakers = speakers_in(lines)
    subject = asked_subject(question, speakers)
    if subject is None:
        return False
    terms = {stem(w) for w in re.findall(r"[a-z]{4,}", prediction.lower())
             if w not in STOP and w not in {s.lower() for s in speakers}}
    if len(terms) < 2:
        return False
    other_best = 0.0
    own_best = 0.0
    for speaker, text in lines:
        low = text.lower()
        overlap = sum(1 for t in terms if t in low) / len(terms)
        if speaker.lower() == subject.lower():
            own_best = max(own_best, overlap)
        else:
            other_best = max(other_best, overlap)
    return other_best >= threshold and own_best < threshold


GATES = {
    "subject_absent": gate_subject_absent,
    "predicate_absent": gate_predicate_absent,
    "predicate_object": gate_predicate_object,
    "answer_attribution": gate_answer_attribution,
    "answer_attribution_dominant": gate_answer_attributed_dominant,
}


# ------------------------------------------------------------------- scoring
def load_contexts(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row.get("question_id") or row.get("qid")
        out[qid] = row
    return out


def normalise(pred: str) -> str:
    low = str(pred).strip().lower()
    if low in ABSTENTIONS or "no information available" in low:
        return CANONICAL_REFUSAL
    return str(pred).strip()


def load_predictions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", payload)
    return {r["question_id"]: normalise(r.get("predicted_answer", "")) for r in rows}


def _cat(qa: dict) -> int:
    return int(qa.get("category", 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="datasets/external/locomo10.json")
    ap.add_argument("--scope", choices=["dev", "full"], default="dev")
    ap.add_argument("--gate", default="subject_absent", choices=sorted(GATES))
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--out", default="")
    ap.add_argument("--runtime-gate", action="store_true",
                    help="use the shipped AnswerProvenanceGate instead of a local gate")
    args = ap.parse_args()

    official = load_official()
    contexts = load_contexts(ROOT / args.contexts)
    preds = load_predictions(ROOT / args.run)
    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    gate = GATES[args.gate]

    print(f"gate={args.gate} scope={args.scope} contexts={len(contexts)} "
          f"predictions={len(preds)}")

    results = []
    for threshold in [float(t) for t in args.thresholds.split(",")]:
        before = defaultdict(list)
        after = defaultdict(list)
        fired = defaultdict(int)
        for sample in dataset:
            sid = sample["sample_id"]
            # scope=dev keeps the eight development conversations; scope=full
            # keeps everything.  The earlier form of this test was inverted and a
            # "--scope full" run silently measured ONLY the holdout (499 rows).
            # That leak is recorded in V4_PROGRAM.md 4.9; do not tune on it.
            if args.scope == "dev" and sid in HOLDOUT:
                continue
            qas_b, qas_a = [], []
            for i, qa in enumerate(sample["qa"]):
                qid = f"lc_{sid}_{i}"
                pred = preds.get(qid)
                if pred is None:
                    continue
                item = dict(qa)
                if "answer" not in item:
                    item["answer"] = item.get("adversarial_answer", "")
                item["am_prediction"] = pred
                qas_b.append(item)

                ctx = contexts.get(qid, {}).get("context", "")
                if args.runtime_gate:
                    # Exercise the SHIPPED module, not a copy of its logic: the
                    # numbers must be reproducible from the runtime code.
                    from artificial_memory.skills.refusal_gate import (
                        AnswerProvenanceGate,
                    )
                    refuse = (bool(ctx) and AnswerProvenanceGate()
                              .decide(qa["question"], ctx, pred).refuse)
                else:
                    refuse = gate(qa["question"], ctx, pred, threshold) if ctx else False
                if refuse:
                    fired[_cat(qa)] += 1
                    item_after = dict(item)
                    item_after["am_prediction"] = CANONICAL_REFUSAL
                    qas_a.append(item_after)
                else:
                    qas_a.append(item)

            ems_b, _, _ = official.eval_question_answering(qas_b, "am_prediction", metric="f1")
            ems_a, _, _ = official.eval_question_answering(qas_a, "am_prediction", metric="f1")
            for qa_b, b, a in zip(qas_b, ems_b, ems_a):
                before[_cat(qa_b)].append(float(b))
                after[_cat(qa_b)].append(float(a))

        flat_b = [v for vals in before.values() for v in vals]
        flat_a = [v for vals in after.values() for v in vals]
        row = {
            "threshold": threshold,
            "n": len(flat_b),
            "f1_before": round(sum(flat_b) / len(flat_b) * 100, 3),
            "f1_after": round(sum(flat_a) / len(flat_a) * 100, 3),
            "delta": round((sum(flat_a) - sum(flat_b)) / len(flat_b) * 100, 3),
            "fired": {CAT.get(k, str(k)): v for k, v in sorted(fired.items())},
            "cat2_before": round(sum(before[2]) / max(1, len(before[2])) * 100, 2),
            "cat2_after": round(sum(after[2]) / max(1, len(after[2])) * 100, 2),
            "cat4_before": round(sum(before[4]) / max(1, len(before[4])) * 100, 2),
            "cat4_after": round(sum(after[4]) / max(1, len(after[4])) * 100, 2),
            "cat5_before": round(sum(before[5]) / max(1, len(before[5])) * 100, 2),
            "cat5_after": round(sum(after[5]) / max(1, len(after[5])) * 100, 2),
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

    if args.out:
        (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
