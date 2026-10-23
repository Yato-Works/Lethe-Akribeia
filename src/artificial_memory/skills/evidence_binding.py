"""Evidence Binding -- does a candidate value belong to the thing that was asked?

WHY THIS EXISTS.  The DURATION path answers with the right TYPE of value but sometimes
the wrong SUBJECT:

    Q  : 'How long has John been surfing?'          gold 'five years'
    OK : 'I started surfing five years ago (2018)'
    BUG: 'We were teammates for four years in high school'  -> answered '4 years'

Both sentences are about John, so a speaker/entity check passes.  The difference is the
EVENT: the asked event is 'surfing', and 'four years' attaches to 'teammates'.  Nothing
in the existing path compares the event, so the first duration in the highest-scoring
turn wins.

THE CONTRACT (phase 1 of the mission).  Deliberately small, and it reuses the data
already in circulation rather than inventing a parallel model:

    BindingRequest   what was asked  : the entity, the event/relation, the value type
    BindingCandidate a piece of evidence: its text, its turn id, and where in that text
                   the value sits
    BindingVerdict   MATCH / NO_MATCH / AMBIGUOUS, plus the supporting overlap and the
                   reason -- so the answer keeps provenance

It returns a verdict, never an answer: deciding WHICH value to commit stays with the
caller, which keeps this a shared check rather than a new engine.

WHAT IT DOES NOT DO, on purpose.  It does not resolve paraphrase, and it does not carry
a word-by-word exception table.  When the event cannot be tied to the value it says
AMBIGUOUS rather than guessing, because the mission's rule is that an unsupported
commit is worse than an abstention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

#: Words that carry no event content, so they cannot support a binding either way.
#: Kept small and closed: this list decides when evidence is INSUFFICIENT, so every
#: word here is a word that must never be able to create a match.
_NON_EVENT_WORDS = frozenset({
    "the", "and", "for", "was", "were", "with", "that", "this", "from", "have",
    "has", "had", "been", "being", "are", "will", "would", "could", "should",
    "did", "does", "about", "into", "over", "than", "then", "when", "what",
    "who", "why", "how", "which", "there", "here", "his", "her", "its", "their",
    "our", "your", "him", "she", "they", "them", "not", "but", "all", "any",
    "out", "many", "much", "long", "ago", "now", "just", "very", "really",
    "also", "still", "some", "such", "only", "even", "back", "get", "got",
    "make", "made", "take", "took", "give", "gave", "went", "come", "came",
})

#: Inflection shapes for a crude but predictable lemma, used only to compare two words.
_SUFFIXES = ("ings", "ing", "edly", "ed", "es", "s", "ly", "er", "est")


class BindingVerdicts(str, Enum):
    """Whether the value belongs to what was asked."""
    MATCH = "MATCH"          # the event terms of the ask appear on the value
    NO_MATCH = "NO_MATCH"    # the ask's event terms are absent from the value's clause
    AMBIGUOUS = "AMBIGUOUS"  # not enough to decide -- the caller should abstain


def event_terms(question: str) -> frozenset[str]:
    """The content words that name WHAT the question is about.

    The entity is deliberately NOT included: the caller supplies it separately, because
    'the same person' is exactly the condition that must not be sufficient.
    """
    words = re.findall(r"[a-z]{3,}", question.lower())
    stems = set()
    for w in words:
        if w in _NON_EVENT_WORDS:
            continue
        stems.add(w)
        stems.add(_stem(w))
    return frozenset(stems)


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def clause_of(text: str, start: int, end: int) -> str:
    """The clause around a value span -- the unit an event actually attaches to.

    A value's subject lives in its own clause, not in the whole sentence: in
    'We were teammates for four years in high school, so we've played together for
    quite some time', 'four years' attaches to 'teammates' in the first clause.  Taking
    the whole sentence would let the second clause's 'played' create a false match.
    """
    left = text.rfind(".", 0, start)
    left = max(left, text.rfind("!", 0, start), text.rfind("?", 0, start))
    left = max(left, text.rfind(";", 0, start), text.rfind(",", 0, start))
    right_candidates = [text.find(c, end) for c in ".;!?,"
                        if text.find(c, end) != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1: right].strip()


@dataclass
class BindingRequest:
    """What the question asked for."""
    question: str
    entity: str | None = None          # 'john' -- necessary but NOT sufficient
    value_type: str = "DURATION"       # DURATION | CARDINALITY | DATE | ATTRIBUTE
    event: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def for_question(cls, question: str, entity: str | None = None,
                     value_type: str = "DURATION") -> BindingRequest:
        return cls(question=question, entity=(entity or "").lower() or None,
                   value_type=value_type, event=event_terms(question))

    @property
    def has_event(self) -> bool:
        return bool(self.event)


@dataclass
class BindingCandidate:
    """One piece of evidence, and where the value sits inside it."""
    text: str
    turn_id: str = ""
    value: str = ""
    value_start: int = -1
    value_end: int = -1

    @property
    def clause(self) -> str:
        """The clause the value actually belongs to."""
        if self.value_start < 0 or self.value_end <= self.value_start:
            return self.text
        return clause_of(self.text, self.value_start, self.value_end)


@dataclass
class BindingVerdict:
    verdict: BindingVerdicts
    overlap: frozenset[str]
    reason: str
    clause: str = ""
    #: True when the entity agrees, recorded separately because the mission requires
    #: that entity agreement alone must never be treated as a match.
    entity_agrees: bool = False


def bind(request: BindingRequest, candidate: BindingCandidate) -> BindingVerdict:
    """Decide whether the candidate's value belongs to the asked event.

    The rule, stated so it can be argued with:
      MATCH       -- at least one of the ask's EVENT terms appears in the value's own
                     clause.  This is what 'I started surfing five years ago' has and
                     'We were teammates for four years' does not.
      NO_MATCH    -- the clause has content words, but none of them are the ask's event.
      AMBIGUOUS   -- the clause has no content words at all, so there is nothing to
                     compare.  Not a match, and not a refusal either.
    """
    clause = candidate.clause
    clause_words = {w for w in re.findall(r"[a-z]{3,}", clause.lower())
                    if w not in _NON_EVENT_WORDS}
    clause_stems = {_stem(w) for w in clause_words} | clause_words

    entity_agrees = True
    if request.entity:
        low = candidate.text.lower()
        entity_agrees = request.entity in low

    # The VALUE's own words are not evidence.  'It's been about four months.' has no
    # content words once 'four' and 'months' -- the value itself -- are removed, and
    # that is precisely the AMBIGUOUS case: the clause states a span and names nothing
    # it belongs to.  Counting the value's own tokens as content made this look like a
    # NO_MATCH, which would have been an over-confident refusal.
    value_tokens = {_stem(w) for w in re.findall(r"[a-z]{3,}", candidate.value.lower())}
    value_tokens |= {w for w in re.findall(r"[a-z]{3,}", candidate.value.lower())}
    evidence_stems = clause_stems - value_tokens

    overlap = frozenset(request.event & evidence_stems)

    if overlap:
        return BindingVerdict(
            BindingVerdicts.MATCH, overlap,
            f"event term(s) {sorted(overlap)} appear in the value's own clause",
            clause=clause, entity_agrees=entity_agrees)
    if evidence_stems:
        return BindingVerdict(
            BindingVerdicts.NO_MATCH, frozenset(),
            f"the clause '{clause[:60]}' names {sorted(evidence_stems)[:6]} but none "
            f"is the asked event {sorted(request.event)[:6]}",
            clause=clause, entity_agrees=entity_agrees)
    return BindingVerdict(
        BindingVerdicts.AMBIGUOUS, frozenset(),
        "the value's clause carries no content words besides the value itself, so the "
        "event cannot be tied to it",
        clause=clause, entity_agrees=entity_agrees)


def best_bound(request: BindingRequest,
               candidates: list[BindingCandidate]) -> BindingVerdict | None:
    """The first MATCH among candidates, in the caller's order.

    Returns None when nothing matches, which the caller turns into an abstention --
    deliberately, so that 'right type, unproven subject' never becomes a commit.
    """
    for c in candidates:
        v = bind(request, c)
        if v.verdict is BindingVerdicts.MATCH:
            return v
    return None
