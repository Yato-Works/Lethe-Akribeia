"""Evidence-Commit Gate -- refuse to commit an answer the evidence does not support.

Motivation, measured.  Six when_date answers were committed whose gold is absent from
the frozen context (trace_overcommit.py).  Their chosen turns show why a naive gate
fails:

  lc_conv-47_59  gold 'October 30-31, 2022'  gave '27 April, 2022'
                 chosen turn is about strategy BOARD GAMES
  lc_conv-50_17  gold 'Friday before 21 June, 2023'  gave 'week before 16 May 2023'
                 gold's date IS in the context, but the chosen turn is about a
                 FLOODED APARTMENT, not a car incident

So neither check works alone:
  * a date-PRESENCE check passes lc_conv-50_17 and commits a wrong event,
  * a PERSON check passes 5 of 6 (the person is usually in the chosen turn),
  * an EVENT-RELEVANCE check catches both, because it asks whether the chosen turn
    actually describes the asked event rather than merely mentioning the subject.

The gate therefore requires, before a date answer is committed:
  1  an evidence turn was actually cited          (no unattributed commits)
  2  the asked person appears in that turn        (subject continuity)
  3  the turn carries a date expression at all    (it is a date-bearing turn)
  4  EVENT OVERLAP: the question's content terms appear in the turn, so the turn is
     about the asked event and not merely adjacent to it
  5  no stronger competing date turn exists whose event overlap is higher

Check 4 is the one that matters.  It is deliberately a CONTENT-TERM overlap with a
threshold, not a loosened keyword match: the gate only ever makes an existing answer
SAFER by refusing to commit, so a false positive costs an abstention (safe) and never
invents an answer.  That asymmetry is why this gate can be applied broadly without the
blast radius that loosening the upstream event gate would carry.

Derived answers stay auditable: the gate records which checks passed, so a committed
answer can be traced to its evidence rather than merely asserted.

MEASURED LIMIT, stated up front rather than discovered later.  Calibration
(scripts/benchmarks/calibrate_commit_gate.py) against the 6 measured over-commits and
the 20 date answers the engine gets right today shows this gate rejects 3 of the 6
while blocking 4 of the 20.  It is NOT a clean filter -- no single event-overlap
threshold separates the two sets, because lc_conv-50_17 (wrong: 'flooded apartment'
answering a car-incident question) and lc_conv-30_12 (correct) both score 0.33.

So the gate is shipped as a PARTIAL safety net with its numbers published, not as a
solved problem.  What it reliably catches is the structural failure -- no evidence
cited, or the asked person absent from both speaker and text -- which is 2 of the 6
and, more importantly, is a check whose false-positive cost is bounded to an
abstention.  Anything stronger requires turn selection to improve upstream, which is
the 83%-semantic barrier Phase 1 of the previous mission measured.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Words that carry no event content.  Same spirit as the committer's stopword set.
_STOP = {
    "the", "a", "an", "and", "or", "but", "i", "you", "he", "she", "it", "we",
    "they", "my", "your", "his", "her", "its", "our", "their", "me", "him",
    "them", "is", "are", "was", "were", "be", "been", "being", "am", "do",
    "does", "did", "have", "has", "had", "to", "of", "in", "on", "at", "for",
    "with", "that", "this", "these", "those", "so", "if", "then", "than", "as",
    "by", "from", "about", "just", "really", "very", "too", "also", "not", "no",
    "yes", "what", "when", "where", "who", "why", "how", "which", "there",
    "here", "go", "went", "get", "got", "take", "took", "did",
}

_DATE_IN_TURN = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+20\d\d\b"
    r"|\b(?:19|20)\d\d\b"
    r"|\b(?:yesterday|today|tomorrow|last\s+night|last\s+week|last\s+"
    r"(?:mon|tues?|wed|wednes|thur?s?|fri|satur|sun)[a-z]*|the\s+other\s+day)\b"
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|"
    r"weekend|month|year)s?\s+(?:before|after|of)\s+\d", re.I)


def content_terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", str(text).lower())
            if w not in _STOP}


@dataclass(frozen=True)
class GateDecision:
    """Whether a commit may proceed, and why."""

    allowed: bool
    checks: dict[str, bool]
    reason: str
    #: Populated when the gate overrides an answer to abstention, for audit.
    evidence_turn_id: str = ""
    event_overlap: float = 0.0
    competing_turns: tuple[str, ...] = field(default_factory=tuple)


#: Minimum fraction of the question's content terms that must appear in the chosen
#: turn for it to count as describing the asked event.
#:
#: MEASURED, not chosen.  calibrate_commit_gate.py sweeps this on real frozen data
#: against 6 measured over-commits and 20 answers the engine gets right today:
#:
#:   threshold   over-commits allowed   correct blocked
#:        0.00                     5                  2
#:        0.30                     4                  2
#:        0.34                     3                  4
#:        0.60                     2                 10
#:
#: NO threshold separates the two sets.  At 0.34 the gate still admits 3 of the 6
#: over-commits while blocking 4 correct answers, and two cases sit a hair apart --
#: lc_conv-50_17 (an over-commit) scores 0.33 while lc_conv-30_12 (correct) also
#: scores 0.33.  A single overlap cut cannot tell them apart.
#:
#: 0.34 is therefore kept as a WEAK signal only, contributing to the decision and
#: never sufficient on its own.  The gate is honest about this: it rejects what its
#: checks can justify and reports each case's reason, rather than claiming a
#: separation the measurement disproves.  An earlier version of this docstring
#: asserted 0.34 rejected all six with zero collateral damage; that claim was
#: fabricated before the sweep existed and was removed.
EVENT_OVERLAP_MIN = 0.34


class EvidenceCommitGate:
    """Refuse unsupported date commits.  Only ever makes answers safer."""

    def __init__(self, event_overlap_min: float = EVENT_OVERLAP_MIN) -> None:
        self.event_overlap_min = event_overlap_min

    def check(self, question: str, answer: str, evidence_turn: str,
              evidence_turn_id: str = "", person: str | None = None,
              speaker: str = "",
              all_turns: list | None = None) -> GateDecision:
        checks: dict[str, bool] = {}

        # 1. An evidence turn must be cited at all.
        checks["has_evidence"] = bool(str(evidence_turn).strip())
        if not checks["has_evidence"]:
            return GateDecision(False, checks, "no evidence turn cited")

        # 2. The asked person must be the SUBJECT of that turn.
        #    The person may appear as the turn's SPEAKER or inside its text.  LoCoMo
        #    turn text does not repeat the speaker's own name, so checking text alone
        #    wrongly rejects first-person turns -- calibration blocked 19 of 20 known
        #    correct answers that way.  Both are valid attributions.
        if person:
            p = person.lower()
            in_text = p in str(evidence_turn).lower()
            in_speaker = p in str(speaker).lower()
            checks["person_in_evidence"] = in_text or in_speaker
            if not checks["person_in_evidence"]:
                return GateDecision(
                    False, checks,
                    f"asked person {person!r} is neither the speaker nor named in "
                    f"the evidence turn",
                    evidence_turn_id=evidence_turn_id)

        # 3. The turn must actually carry a date expression.
        checks["turn_has_date"] = bool(_DATE_IN_TURN.search(str(evidence_turn)))
        if not checks["turn_has_date"]:
            return GateDecision(False, checks,
                                "evidence turn carries no date expression",
                                evidence_turn_id=evidence_turn_id)

        # 4. Event relevance: the turn must describe the asked event.
        q_terms = content_terms(question)
        t_terms = content_terms(evidence_turn)
        overlap = (len(q_terms & t_terms) / len(q_terms)) if q_terms else 0.0
        checks["event_overlap"] = overlap >= self.event_overlap_min
        if not checks["event_overlap"]:
            return GateDecision(
                False, checks,
                f"event overlap {overlap:.2f} < {self.event_overlap_min} -- the "
                f"evidence turn is not about the asked event",
                evidence_turn_id=evidence_turn_id, event_overlap=overlap)

        # 5. No competing turn describes the event better while carrying a date.
        competing: list[str] = []
        if all_turns:
            best_other = 0.0
            for t in all_turns:
                text = getattr(t, "text", t)
                tid = getattr(t, "dia_id", "")
                if tid == evidence_turn_id:
                    continue
                if not _DATE_IN_TURN.search(str(text)):
                    continue
                o = len(q_terms & content_terms(text)) / max(1, len(q_terms))
                if o > overlap:
                    best_other = max(best_other, o)
                    competing.append(str(tid))
            if best_other > overlap + 0.15:
                return GateDecision(
                    False, checks,
                    f"a date-bearing turn describes the event better "
                    f"({best_other:.2f} > {overlap:.2f}) -- turn selection is unsafe",
                    evidence_turn_id=evidence_turn_id, event_overlap=overlap,
                    competing_turns=tuple(competing[:5]))

        return GateDecision(
            True, checks, "evidence supports the commit",
            evidence_turn_id=evidence_turn_id, event_overlap=overlap)
