"""Deterministic answer-provenance refusal gate.

The problem it solves
    In an attributed-memory corpus a question can ask about one person while the
    evidence that matches it belongs to another.  A reader that is not tracking
    attribution returns the other person's fact, fluently and confidently.  The
    adversarial category of LoCoMo is built exactly this way.

Why provenance instead of the question
    Three gates keyed on the QUESTION failed (V4_PROGRAM.md 4.5): the evidence for
    one fact is often spread across several turns, so "does the subject have a
    turn containing the question's predicate" refuses correct answers.  The
    ANSWER is a much sharper probe: if the answer's own wording only ever appears
    in another speaker's turn, it did not transfer to the asked subject.

What it does NOT do
    It never rewrites an answer.  It only answers the yes/no question "should this
    be replaced by a refusal?", and the caller decides what a refusal is.  It
    leaves alone: questions that name no single speaker, date-shaped answers (a
    date legitimately appears in both speakers' turns), and answers whose wording
    is not found in the context at all (paraphrase, usually a reader inference).

Measured on LoCoMo dev (1,487 questions, pinned official F1): +4.33pt, of which
+25.5pt on the adversarial category and -2.6pt on single-hop.  The full set moves
+2.68pt.  See design_docs/v4_world_class/V4_PROGRAM.md 4.9, including the
holdout-leak note for this gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: A date can legitimately appear in both speakers' turns, so provenance is the
#: wrong test for a date-shaped answer (measured: -7.8pt on temporal without this).
_MONTHS = frozenset("""
january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec
""".split())
_WEEKDAYS = frozenset("""
monday tuesday wednesday thursday friday saturday sunday
""".split())

#: Function words plus the verbs that carry no topic.  Deliberately generic: no
#: corpus, entity, or benchmark vocabulary appears here.
_STOP = frozenset("""
a an the and or but if then than that this these those there here
i me my we our you your he him his she her it its they them their
is are was were be been being am do does did doing done
have has had having will would can could should may might must
not no yes none all any some more most other another such
what when where who whom which why how while
about above after again against all also because before below between both
during each few from further into just more once only out over own same
so than too under until up very with within without
about across around back down even ever every first last less lot many much
new next now off on onto put really right said say says see seen still take
taken tell than thing things think thought time times two use used using want
was way well went were what when where which while who why will with
""".split())

#: "I don't know." and friends are already refusals; the gate has nothing to add.
_REFUSALS = frozenset({"i don't know.", "not mentioned", "none", ""})


@dataclass(frozen=True)
class RefusalDecision:
    """The gate's verdict, with the provenance that produced it."""

    refuse: bool
    reason: str
    subject: str = ""
    source_speaker: str = ""
    overlap: float = 0.0


def _stem(word: str) -> str:
    """Crude suffix stripping, enough for 'realizing' to meet 'realize'."""
    for suffix in ("ing", "ed", "es", "s", "e"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _dialogue_lines(context: str) -> list[tuple[str, str]]:
    """(speaker, text) for every dialogue line; certificate blocks are skipped."""
    out: list[tuple[str, str]] = []
    for raw in context.splitlines():
        line = raw.strip()
        if not line or line.startswith("[SPEAKER") or line.startswith("[/"):
            continue
        # Speaker token may be a NAME (Caroline) or a ROLE (user / assistant).
        # LongMemEval renders roles; requiring an initial capital made the gate
        # structurally unable to read 128/128 of its contexts.
        m = re.match(r"^\[[^\]]+\]\s*([A-Za-z][A-Za-z0-9_ -]{0,24}?):\s*(.*)$", line)
        if m:
            out.append((m.group(1).strip(), m.group(2)))
    return out


#: Role labels used by agent-memory corpora instead of names.
_ROLES = frozenset({"user", "assistant", "system", "human", "ai", "agent", "bot"})
_FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself)\b", re.I)


def _speakers(lines: list[tuple[str, str]]) -> list[str]:
    seen: list[str] = []
    for speaker, _ in lines:
        if speaker not in seen:
            seen.append(speaker)
    return seen


def is_temporal_answer(text: str) -> bool:
    low = text.lower()
    if re.search(r"\b(19|20)\d{2}\b", low):
        return True
    if any(m in low for m in _MONTHS):
        return True
    if any(d in low for d in _WEEKDAYS):
        return True
    return re.fullmatch(r"[\d\s,./-]+", low.strip()) is not None


class AnswerProvenanceGate:
    """Refuse an answer whose wording only ever comes from another speaker."""

    #: Share of the answer's content stems that a line must contain.
    THRESHOLD = 0.7

    def __init__(self, allow_roles: bool = False, min_terms: int = 1,
                 threshold: float | None = None) -> None:
        #: When True, a first-person question in a role-labelled corpus
        #: (user/assistant) is attributed to the user.  OFF by default because it
        #: was measured as harmful: on LongMemEval it refuses 2 correct answers and
        #: costs -1.56pt (82.03% -> 80.47%), since the assistant legitimately
        #: restates the user's facts.  The provenance rule is valid where distinct
        #: persons exist; agent memory does not give it that separation.
        self.allow_roles = allow_roles
        #: How many content stems the answer must carry before the gate may fire.
        #: Measured on the merged full run: 1 -> 165 fires, 2 -> 102, 3 -> 46.
        self.min_terms = max(1, int(min_terms))
        #: Share of the answer's content stems the best-matching line must carry.
        #: Exposed so a threshold sweep actually sweeps it (it previously read the
        #: class constant, so every sweep row was identical).
        self.threshold = self.THRESHOLD if threshold is None else float(threshold)

    def decide(self, question: str, context: str, answer: str) -> RefusalDecision:
        answer = (answer or "").strip()
        if not answer or answer.lower() in _REFUSALS:
            return RefusalDecision(False, "already a refusal")
        if is_temporal_answer(answer):
            return RefusalDecision(False, "date-shaped answer: provenance not applicable")

        lines = _dialogue_lines(context)
        if not lines:
            return RefusalDecision(False, "no dialogue lines parsed")
        speakers = _speakers(lines)

        low_q = question.lower()
        named = [s for s in speakers if re.search(rf"\b{re.escape(s.lower())}\b", low_q)]
        if len(named) == 1:
            subject = named[0]
        elif (self.allow_roles and not named and _FIRST_PERSON.search(question)):
            # Agent-memory corpora ask in the first person ("Where did I redeem
            # the coupon?"), so "I" denotes the user.  Only accepted when the
            # context actually has a single user-like speaker.
            users = [s for s in speakers if s.lower() in {"user", "human"}]
            if len(users) != 1:
                return RefusalDecision(
                    False, "first-person question without a single user speaker")
            subject = users[0]
        else:
            return RefusalDecision(False, "question names no single speaker")

        names = {s.lower() for s in speakers}
        terms = {_stem(w) for w in re.findall(r"[a-z]{4,}", answer.lower())
                 if w not in _STOP and w not in names}
        if len(terms) < self.min_terms:
            return RefusalDecision(
                False, f"answer carries {len(terms)} content terms "
                       f"(< {self.min_terms})", subject)

        best_overlap = 0.0
        best_speaker = ""
        for speaker, text in lines:
            low = text.lower()
            overlap = sum(1 for t in terms if t in low) / len(terms)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, speaker

        if best_overlap < self.threshold:
            return RefusalDecision(
                False, "answer wording not traceable in the context", subject,
                best_speaker, best_overlap)
        if best_speaker.lower() == subject.lower():
            return RefusalDecision(
                False, "answer wording comes from the asked person", subject,
                best_speaker, best_overlap)
        return RefusalDecision(
            True,
            f"answer wording comes from {best_speaker}, not {subject}",
            subject, best_speaker, best_overlap,
        )


__all__ = ["AnswerProvenanceGate", "RefusalDecision", "is_temporal_answer"]
