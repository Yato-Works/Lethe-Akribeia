"""LoCoMo full-corpus benchmark adapter for Lethe-Akribeia.

LongMemEval only exercises 250 questions across 6 types.  LoCoMo is a much
larger and differently-shaped corpus (1,986 questions over 10 long multi-session
conversations) and — critically — includes two categories LongMemEval does not
have at all:

    cat4 open-domain  (841 q, 42.3%)  free-form answers
    cat5 adversarial  (446 q, 22.5%)  questions whose answer is NOT in the
                                      conversation (unanswerable / misleading)

Running the full corpus is how we find weaknesses that a 250-question smoke
suite structurally cannot reveal.

The adapter converts LoCoMo's conversation format into the (sid, date, speaker,
content) turn tuples the V4 pipeline already consumes, so the SAME frozen
pipeline runs unchanged.

Usage:
    python scripts/benchmarks/eval_locomo_full.py --cond R7 --limit 0
    python scripts/benchmarks/eval_locomo_full.py --cond R7 --cats 4,5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

# The strict rubric is the single source of truth for "is this gold an
# enumeration answer".  Sharing it keeps the production scorer and the audit in
# agreement instead of letting them drift apart.
from artificial_memory.compiler.eval.audit_strict_grade import is_list_gold  # noqa: E402

#: LoCoMo's own category numbering.  This mapping was previously rotated by one
#: position (1=single-hop, 2=multi-hop, 3=temporal, 4=open-domain), which
#: mislabelled every printed report: the 282-question category is multi-hop, not
#: single-hop.  Verified against the pinned harness, which splits sub-answers for
#: category 1 only (evaluation.py:212-214) and treats 2/3/4 uniformly, and against
#: the dataset counts (282/321/96/841/446).
CAT_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


# --------------------------------------------------------------------------- IO
def load_locomo(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def conversation_turns(conv: dict) -> tuple[list[tuple[str, str, str, str]], dict[str, str]]:
    """Flatten a LoCoMo conversation into pipeline turn tuples.

    Returns (turns, dia_id_to_text) where turns are
    (session_id, session_date, speaker, content) and dia_id_to_text maps each
    LoCoMo dialogue id ('D1:3') to its text so evidence can be resolved.
    """
    turns: list[tuple[str, str, str, str]] = []
    dia_map: dict[str, str] = {}
    speaker_a = conv.get("speaker_a", "User")
    speaker_b = conv.get("speaker_b", "Assistant")

    for key in conv.keys():
        m = re.match(r"session_(\d+)$", key)
        if not m:
            continue
        session_idx = int(m.group(1))
        sid = f"{key}"
        date_key = f"session_{session_idx}_date_time"
        date = str(conv.get(date_key, ""))
        for turn in conv[key]:
            spk = turn.get("speaker", speaker_a)
            text = turn.get("text", "")
            did = turn.get("dia_id", "")
            if did:
                dia_map[did] = text
            # Preserve actual speaker name so SpeakerAttributionNormalizer
            # and deterministic attribution engines can resolve agent/user ownership.
            turns.append((sid, date, spk, text))
    return turns, dia_map


def load_questions(conv: dict) -> list[dict]:
    out = []
    sample_id = conv.get("sample_id", "x")
    for i, qa in enumerate(conv.get("qa", [])):
        # cat5 (adversarial) has NO 'answer'.  It stores only
        # 'adversarial_answer', which is the DISTRACTOR the pinned official
        # harness scores 0 for emitting (evaluation.py:217-221).  Using it as
        # gold inverted the category: 126 questions were credited for producing
        # the wrong option while the correct behaviour (refusing) scored 0.
        # See design_docs/v4_world_class/V4_PROGRAM.md section 3.
        gold = qa.get("answer")
        out.append({
            "question_id": f"lc_{sample_id}_{i}",
            "sample_id": sample_id,
            "question": qa["question"],
            "answer": "" if gold is None else str(gold),
            "adversarial_answer": str(qa.get("adversarial_answer", "")),
            "category": int(qa.get("category", 0)),
            "evidence": qa.get("evidence", []),
        })
    return out


# ------------------------------------------------------------------ scoring
_YESNO = {"yes", "no"}

#: Month names, used only to recognise a day-of-month inside a date string.
_MONTH_WORDS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}

#: Day-of-month patterns: '10 May' / 'May 10', with an optional ordinal suffix.
_DATE_DAY_BEFORE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)")
_DATE_DAY_AFTER = re.compile(r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\b")


def _date_days(text: str) -> set[str]:
    """Day-of-month numbers this string commits to, as normalised strings.

    Only numbers adjacent to a MONTH NAME count, so a stray number in prose
    ('she has 3 cats') is never mistaken for a day.
    """
    days: set[str] = set()
    for m in _DATE_DAY_BEFORE.finditer(text.lower()):
        num, word = m.group(1), m.group(2)
        if word in _MONTH_WORDS:
            days.add(str(int(num)))
    for m in _DATE_DAY_AFTER.finditer(text.lower()):
        word, num = m.group(1), m.group(2)
        if word in _MONTH_WORDS:
            days.add(str(int(num)))
    return days


def _date_contradiction(p: str, g: str) -> bool:
    """True when both sides name a day-of-month and they disagree.

    A disagreement on the day is a factual contradiction: '10 May, 2023' is not
    '27 May, 2023' however much month and year they share.  Without this guard the
    60%-token rule silently drops 1-2 digit days and credits the wrong date.
    """
    pd, gd = _date_days(p), _date_days(g)
    if not pd or not gd:
        return False
    return not (pd & gd)


#: Cardinal words the corpus uses in counting answers.
_NUMERAL_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "twice": 2,
}

#: Counted nouns.  A numeral only counts as a QUANTITY when it quantifies
#: something; a lone year or a house number is not a count of the asked thing.
_COUNT_NOUNS = (
    "times?|dogs?|cats?|pets?|kids?|children|childs?|weeks?|months?|years?|"
    "days?|hours?|minutes?|birds?|fish|rabbits?|hamsters?|turtles?|pupp(?:y|ies)|"
    "books?|cars?|trips?|roadtrips?|tournaments?|marathons?|hikes?|sessions?"
)

#: A quantity = a cardinal (word or digits) followed by an optional counted noun.
#: The noun is optional so '2' vs '3' is caught too, but the ANCHOR requirement
#: below is what keeps this from firing on every digit in a sentence.
_QUANTITY_RE = re.compile(
    rf"\b(?:{ '|'.join(_NUMERAL_WORDS) }|\d+)\b(?:\s+(?:{_COUNT_NOUNS})\b)?",
    re.I,
)

#: A bare 4-digit year is not a counted quantity, so exclude it from the
#: quantity scan.  '22 August, 2023' must still be caught by the YEAR rule below,
#: which compares years explicitly.
_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")


def _quantity_values(text: str) -> set[int]:
    """Cardinal quantities this string commits to.

    Numerals adjacent to a counted noun are included; a bare 4-digit year is
    excluded here (handled by _year_values) because 'in 2023' is a date, not a
    count.  A bare numeral with no noun is kept so '2' vs '3' is still caught.
    """
    out: set[int] = set()
    lowered = text.lower()
    for m in _QUANTITY_RE.finditer(lowered):
        token = m.group(0).split()[0]
        if token.isdigit():
            value = int(token)
            if len(token) == 4 and 1000 <= value <= 2999:
                continue        # a year, not a count
        else:
            value = _NUMERAL_WORDS[token]
        out.add(value)
    return out


def _year_values(text: str) -> set[int]:
    """Four-digit years this string names."""
    return {int(m.group(1)) for m in _YEAR_RE.finditer(text)}


def _numeral_contradiction(p: str, g: str) -> bool:
    """True when both sides state a quantity and the quantities disagree.

    'two dogs' is not 'three dogs', and '2 weeks' is not 'three weeks', however
    much the rest of the sentence agrees.  The 50%-either-direction token-overlap
    rule below credits those as correct, because 'dogs' or 'weeks' carries the
    overlap and the differing numeral is never compared.  Measured on 46 stored
    runs / 17,604 pred-gold pairs: 382 pairs state conflicting numerals and 32
    of them (8.4%) were credited correct.

    Narrow on purpose, so it cannot eat legitimate credits:
      * only fires when BOTH sides name a quantity, and
      * only when the quantity sets are DISJOINT (a prediction naming several
        quantities, one of which matches, is not a contradiction).
    """
    pq, gq = _quantity_values(p), _quantity_values(g)
    if pq and gq and not (pq & gq):
        return True
    py, gy = _year_values(p), _year_values(g)
    if py and gy and not (py & gy):
        return True
    return False


#: Month name -> number.  Local to the guards so they do not depend on the
#: definition order of the relative-date resolver's own map further down.
_MONTH_NUM = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def _month_values(text: str) -> set[int]:
    """Month numbers this string names, 1-12.

    Uses a local map rather than _MONTHS, which is defined further down the file for
    the relative-date resolver; depending on definition order across a 1,000-line
    scorer is how a guard ends up silently inert.
    """
    return {_MONTH_NUM[w] for w in re.findall(r"[a-z]+", str(text).lower())
            if w in _MONTH_NUM}


def _month_contradiction(p: str, g: str) -> bool:
    """True when both sides name a month and the months disagree.

    THIRD defect of the same family, and the largest of the three.  The date guard
    (c449aa0) fixed day-of-month; the numeral guard (e6a305f) fixed counts.  Months
    were still unguarded, and the 50%-token-overlap rule credited them because its
    `len(w) > 2` filter drops the 1-2 digit DAY, leaving month+year to carry the
    score while the differing month is invisible:

        pred 'March 5 2023'      gold '5 May 2023'        -> was scored CORRECT
        pred '15 December 2023'  gold 'August 15, 2023'   -> was scored CORRECT
        pred '26 August 2023'    gold 'November 2023'     -> was scored CORRECT

    Measured across all 46 stored runs / 2,083 month-naming pairs: 318 pairs state
    conflicting months and 109 of them (34.3%) were credited correct.  That is 3.4x
    the numeral defect and 4x the day defect in absolute count.

    Narrow by construction, same as its siblings: fires only when BOTH sides name a
    month AND the month sets are disjoint.  A prediction naming several months where
    one matches gold is not a contradiction.
    """
    pm, gm = _month_values(p), _month_values(g)
    if pm and gm and not (pm & gm):
        return True
    return False


#: Tokens that carry no content.  Used to stop an article from standing in for a
#: real answer.
_NON_CONTENT = {
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "his",
    "her", "its", "their", "our", "your", "him", "she", "they", "them", "not",
    "but", "all", "any", "out", "about", "into", "over", "than", "then", "when",
    "what", "who", "why", "how", "which", "there", "here", "have", "has", "had",
    "been", "being", "are", "will", "would", "could", "should", "did", "does",
}


def _has_content_overlap(p: str, g: str) -> bool:
    """True when pred and gold share at least one CONTENT word.

    FOURTH defect of the same family, and the broadest of them.  The day guard
    (c449aa0), numeral guard (e6a305f) and month guard (75b1f2f) each required a
    disagreeing VALUE before refusing credit.  This one needs no disagreement at
    all -- only a shared article:

        pred 'the'                    gold 'The Alchemist'  -> was scored CORRECT
        pred 'The Name of the Wind'   gold 'The Alchemist'  -> was scored CORRECT

    Mechanism: both token-overlap rules keep tokens with len(w) > 2, which ADMITS
    'the' (3 chars).  When every gold content word is unmatched, 'the' alone carries
    100% of the surviving gold tokens and the rule credits.  LoCoMo golds are full of
    titles, artworks and band names of exactly this shape.

    Measured across all 46 stored runs: 1,050 pairs have a 'The X' gold and 704 are
    credited; of those, 3 are attributable specifically to this defect (no
    content-word overlap at all).  The other 701 reach credit through other rules,
    so this guard fixes 3 and is deliberately narrow rather than reaching further --
    the honest figure is 3, not 704, and the two are not the same claim.

    Deliberately NOT applied to the containment or prefix rules, which compare whole
    strings and are not vulnerable to this.
    """
    pc = {w for w in re.findall(r"[a-z]+", str(p).lower())
          if len(w) > 2 and w not in _NON_CONTENT}
    gc = {w for w in re.findall(r"[a-z]+", str(g).lower())
          if len(w) > 2 and w not in _NON_CONTENT}
    if not gc:
        # Gold has no content words at all; nothing to verify against, so do not
        # let this guard veto a credit the other rules legitimately earned.
        return True
    return bool(pc & gc)


def score(pred: str, gold: str) -> bool:
    """Deterministic containment scorer.

    LoCoMo answers are free-form text (93% of them), so exact match is far too
    strict.  We accept when the prediction contains the gold answer (or the gold
    contains the prediction) after light normalisation, mirroring the tolerant
    LongMemEval matcher.
    """
    def norm(s: str) -> str:
        s = str(s).lower().strip()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    p, g = norm(pred), norm(gold)
    if not g:
        return False

    # DATE CONTRADICTION GUARD -- placed FIRST, before every credit rule.
    #
    # Two rules below both credit a wrong DATE when only the day differs:
    #   pred '10 May, 2023'  gold '27 May, 2023'   -> was scored CORRECT
    #   pred '25 June, 2023' gold '24 June, 2023'  -> was scored CORRECT
    #   pred 'The Tuesday before 20 July 2023'
    #        gold '7 May 2023'                     -> was scored CORRECT
    # The 60%-token rule drops 1-2 digit day numbers via its len(w) > 2 filter,
    # so 'may' and '2023' carry the whole score; and the either-direction
    # containment rule credits a lone shared year.  Measured on 188 committer
    # claims: 8 of 159 credits were spurious this way (5.0%).
    #
    # When BOTH sides name a day-of-month and they disagree, that is a factual
    # contradiction, and no amount of month/year agreement can rescue it.  It must
    # therefore run BEFORE the credit rules, not after -- an earlier placement
    # after the containment checks let 'The week before 13 October 2023' credit
    # against 'On the night of October 30, 2023' through the shared month.
    if _date_contradiction(p, g):
        return False

    # NUMERAL CONTRADICTION GUARD -- also placed FIRST, before every credit rule.
    #
    # Same disease as the date guard, different unit.  The either-direction
    # 50%-overlap rule below shares one content word ('dogs', 'weeks') and
    # credits the answer while the DISAGREEING NUMERAL is never compared:
    #   pred 'two dogs'   gold 'three dogs'  -> was scored CORRECT
    #   pred '5 dogs'     gold '2 dogs'      -> was scored CORRECT
    #   pred 'ten times'  gold 'two times'   -> was scored CORRECT
    #   pred '22 August, 2023'  gold 'August 2022'  -> was scored CORRECT
    # Measured on 46 stored runs / 17,604 pairs: 382 conflicting-numeral pairs,
    # 32 credited correct (8.4%).  A count is a factual claim; a wrong count is
    # wrong however many other words it shares with gold.
    #
    # Deliberately narrow: both sides must state a quantity, and the quantity
    # sets must be disjoint.  A prediction naming several quantities, one of
    # which matches, is not a contradiction and still gets its credit.
    if _numeral_contradiction(p, g):
        return False

    # MONTH CONTRADICTION GUARD -- third of the family, placed with its siblings.
    #
    # The date guard fixed day-of-month and the numeral guard fixed counts.  Months
    # were still unguarded, and this is the LARGEST of the three: measured over all
    # 46 stored runs / 2,083 month-naming pairs, 318 state conflicting months and
    # 109 of them (34.3%) were credited correct -- 3.4x the numeral defect.
    #
    #   pred 'March 5 2023'      gold '5 May 2023'       -> was scored CORRECT
    #   pred '26 August 2023'    gold 'November 2023'    -> was scored CORRECT
    #   pred '15 December 2023'  gold 'August 15, 2023'  -> was scored CORRECT
    #
    # Mechanism: the 50%-token-overlap rule's `len(w) > 2` filter drops the 1-2 digit
    # DAY, so month+year carry the score and the differing month is never compared.
    # Narrow like its siblings: both sides must name a month, and the month sets must
    # be disjoint.
    if _month_contradiction(p, g):
        return False

    # Yes/No polar evaluation: when pred is "yes" or "no"
    if p in _YESNO:
        g_words = set(re.findall(r"\b[a-z]+\b", g))
        if p == "no":
            if "yes" in g_words and not any(w in g_words for w in ("no", "not", "never", "unlikely")):
                return False
            if any(w in g_words for w in ("no", "not", "unlikely")) or g.startswith("likely no") or g.startswith("probably no"):
                return True
        elif p == "yes":
            if any(w in g_words for w in ("no", "not", "unlikely", "never")):
                return False
            if "yes" in g_words:
                return True

    # A gold is a LIST when it enumerates >=2 items.
    #
    # A comma alone is NOT sufficient.  Ordinary prose contains commas
    # ("Likely no, though she likes reading") and would be misread as an
    # enumeration, which routes it into the list branch below and blocks the
    # containment credit it legitimately earns.  Measured: 365 golds in the
    # LoCoMo corpus are misclassified by the comma-only rule.
    #
    # Reuse the strict rubric's definition so the production scorer and the
    # audit can never disagree about what counts as an enumeration answer.
    _is_list = is_list_gold(gold) and p not in _YESNO

    if _is_list:
        # Permutation match: every gold item must appear in the prediction.
        # Split gold items on commas AND on 'and', mirroring _items() in the
        # strict rubric.  With a comma-only split, an 'and'-enumerated gold
        # ('Dramas and emotionally-driven films') stays a single chunk, never
        # reaches the >=2 check, and a correct comma-rephrased answer
        # ('dramas, emotionally-driven films') is scored WRONG.
        # Drops leading/trailing articles/pronouns from each gold item.
        _stop_articles = {"her", "his", "my", "their", "our", "the", "a", "an"}
        raw_gold_items = [x.strip() for x in re.split(r",|\band\b", str(gold)) if x.strip()]
        gold_item_core = []
        for it in raw_gold_items:
            words = [w for w in norm(it).split() if w not in _stop_articles]
            if words:
                gold_item_core.append(words)

        if len(gold_item_core) >= 2:
            p_words = set(p.split())
            all_hit = True
            for core in gold_item_core:
                if not any(w in p_words for w in core):
                    all_hit = False
                    break
            if all_hit:
                return True

    # containment. The reverse direction (prediction inside gold) is the
    # subset credit, so it is disabled for list golds.
    if g in p:
        return True
    # The reverse direction requires a CONTENT overlap, not a substring hit.
    #
    # A bare substring test lets a 3-character string inside a longer gold credit:
    # 'the' is a substring of 'the alchemist', so pred 'the' against gold 'The
    # Alchemist' scored CORRECT.  Same family as the day/numeral/month guards -- a
    # degenerate token carrying the whole verdict.  Measured across all 46 stored
    # runs: 16 pairs were credited with no content-word overlap.
    if p in g and not _is_list and _has_content_overlap(p, g):
        return True
    # yes/no exact
    if g in _YESNO:
        return p in _YESNO and p == g

    # numeric equality.  GATED on _is_list: gold 'The week before 2 May, 2022'
    # and pred '2 May, 2022' share the first number, but the prediction dropped
    # the offset, so it is WRONG.  Only the containment rules below (which
    # compare full strings) may settle those.
    pn = re.findall(r"\d+", p)
    gn = re.findall(r"\d+", g)
    if gn and pn and gn[0] == pn[0] and not _is_list:
        return True
    # token overlap >= 60% of gold tokens (never for list golds)
    gt = [w for w in g.split() if len(w) > 2]
    if gt and not _is_list:
        pt = set(p.split())
        hit = sum(1 for w in gt if w in pt)
        if hit / len(gt) >= 0.6 and _has_content_overlap(p, g):
            return True
    # TOKEN OVERLAP (either direction).  LoCoMo golds are often short labels
    # ('lactose intolerance', 'a handwritten letter', 'Roasted Chicken') while a
    # correct answer restates them ('She is lactose intolerant', 'a letter',
    # 'Chicken dish with garlic, lemon, and herbs').  Requiring 60% of the GOLD
    # tokens to appear is far too strict here: measured 85 of 886 failures
    # (9.5%) were exactly this shape and are plainly correct to a human.
    # Accept when at least half of the SHORTER side's tokens are shared.
    if gt and not _is_list:
        pt_all = [w for w in p.split() if len(w) > 2]
        if pt_all:
            shorter = gt if len(gt) <= len(pt_all) else pt_all
            longer_set = set(pt_all) if len(gt) <= len(pt_all) else set(gt)
            hit = sum(1 for w in shorter if w in longer_set)
            if hit / len(shorter) >= 0.5 and _has_content_overlap(p, g):
                return True
    # FUZZY TOKEN MATCH: tolerate the model's spelling of a proper noun
    # ('Xeonoblade' vs 'Xenoblade').  Also excluded for list golds for the same
    # reason as above.
    if gt and not _is_list:
        import difflib
        pt = set(w for w in p.split() if len(w) > 3)
        for gw in gt:
            if len(gw) < 4:
                continue
            for pw in pt:
                if difflib.SequenceMatcher(None, gw, pw).ratio() >= 0.85:
                    return True
    # PREFIX MATCH: the prediction is the opening of the gold answer.
    # LoCoMo gold answers are often verbose hedged sentences ('Likely no;
    # though she likes reading...') while a correct short answer is 'No' or
    # 'national park'.  GATED on _is_list: for a list gold, a one-item prefix
    # ('Summer Sounds' for 'Summer Sounds, Matt Patterson') is incomplete
    # enumeration, which must not be credited.
    if len(p) >= 3 and g.startswith(p) and not _is_list \
            and _has_content_overlap(p, g):
        return True
    # leading-token match for hedged golds: gold opens with a hedge word
    # ('likely', 'probably', 'yes', 'no', 'somewhat') — if the prediction
    # carries the same stance word, accept.  GATED on _is_list: a hedge-word
    # rule must not become a back door for crediting one item of a list.
    _hedge = ("likely", "probably", "yes", "no", "somewhat", "perhaps")
    gp = g.split()[0] if g.split() else ""
    pp = p.split()[0] if p.split() else ""
    if gp in _hedge and pp == gp and not _is_list:
        return True
    # hedge-word appears anywhere in gold and pred carries the same stance
    if gp in _hedge and any(h == pp for h in _hedge if h in g) and not _is_list:
        return True
    return False


# ------------------------------------------------ relative-date resolution
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def _parse_session_date(header: str):
    """Extract a (year, month, day) from a context line header.

    LoCoMo headers look like '[session_1 on 1:56 pm on 8 May, 2023]'; the
    calendar date is the part after the final 'on'.  LongMemEval headers look
    like '[sid on 2023/05/22]'.
    """
    from datetime import date
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", header)
    if m and m.group(2).lower() in _MONTHS:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
        except ValueError:
            return None
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", header)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


_REL_PATTERNS = [
    (re.compile(r"\byesterday\b", re.I), -1, "day"),
    (re.compile(r"\btoday\b", re.I), 0, "day"),
    (re.compile(r"\btomorrow\b", re.I), 1, "day"),
    (re.compile(r"\blast week\b", re.I), -7, "day"),
    (re.compile(r"\bnext week\b", re.I), 7, "day"),
    (re.compile(r"\blast month\b", re.I), -30, "day"),
    (re.compile(r"\bnext month\b", re.I), 30, "day"),
    (re.compile(r"\blast year\b", re.I), -365, "day"),
]


def resolve_relative_dates(answer: str, context: str, question: str = "") -> str:
    """Deterministically convert a relative-time answer to an absolute date.

    Weakness found on the LoCoMo full corpus: ~10% of questions ask 'when' and
    the conversation states a relative time while the gold answer is absolute.
    The 7B reader kept echoing 'yesterday' / 'last week' even when told to
    resolve it (11 of 93 failures on a 150-q sample), because calendar
    arithmetic is unreliable at that size.  This post-processor does the
    arithmetic deterministically using the session date in the evidence header.

    Only rewrites when the answer is (almost entirely) a relative phrase, so a
    correct substantive answer is never touched.
    """
    if not answer:
        return answer
    a = answer.strip().lower().rstrip(".")
    # only act on short, purely-relative answers
    if len(a.split()) > 4:
        return answer
    # find which relative pattern the answer is
    chosen = None
    for pat, delta, unit in _REL_PATTERNS:
        if pat.search(a):
            chosen = (pat, delta, unit)
            break
    if not chosen:
        return answer
    pat, delta, _unit = chosen
    # Pick the RIGHT evidence line, not merely the first dated one.  Using the
    # first dated line was measurably harmful: 71 of 167 when-failures on the
    # full corpus were dates this resolver produced from the wrong line
    # (e.g. gold '7 May 2023' answered '26 June 2023').
    # Require the candidate line to share a content word with the QUESTION, so
    # we resolve against the line the question is actually about.
    q_words = {w for w in re.findall(r"[a-z]{4,}", question.lower())
               if w not in {"when", "what", "where", "which", "does", "did",
                            "have", "has", "was", "were", "the", "and"}}
    fallback = None
    for line in context.split("\n"):
        if not line.strip().startswith("["):
            continue
        d = _parse_session_date(line)
        if d is None:
            continue
        if fallback is None:
            fallback = d
        low = line.lower()
        if any(w in low for w in q_words):
            from datetime import timedelta
            resolved = d + timedelta(days=delta)
            return f"{resolved.day} {resolved.strftime('%B')} {resolved.year}"
    # No question-matching dated line: refuse to guess rather than emit a
    # wrong absolute date.  A wrong date scores the same as a relative leak
    # but is harder to spot, so abstain and keep the original answer.
    if fallback is None:
        return answer
    return answer


# ------------------------------------------------------------------ pipeline
def _second_pass_turns(question: str, all_turns, already_retrieved,
                       retriever, max_extra: int = 20) -> list:
    """Re-query the turns the first pass DID NOT retrieve, keyed on subject.

    The first pass cuts at base_top_k=45, which is already ~99% full of
    lexically-matching turns.  Evidence that shares no vocabulary with the
    question is therefore never a candidate -- a quota over the same 45 slots
    cannot reach it (measured: pool 44.7 -> 44.8).  This widens the cut.

    Selection is by SPEAKER, not by name-in-text: when Melanie lists her own
    hobbies she says "I went camping", never "Melanie went camping".  The corpus
    populates the speaker field with the real name, so that is the only usable
    signal.  Returns at most max_extra turns, newest-session last.
    """
    subjects = {
        m.group(1).lower()
        for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", question)
    }
    if not subjects:
        return []

    retrieved = {(t[0], t[3]) for t in already_retrieved}
    unseen = [t for t in all_turns if (t[0], t[3]) not in retrieved]
    if not unseen:
        return []

    # Rank the UNSEEN turns with a subject-shaped query.  The probe must be
    # CAPITALISED: rank_turns derives proper_nouns from the query with
    # [A-Z][a-z...], so a lowercase probe yields an empty set and the
    # subject_fallback path never fires (measured: 0 turns added).
    probe = " ".join(sorted(s.capitalize() for s in subjects))
    try:
        extra, _ = retriever.rank_turns(probe, unseen, top_k=max_extra * 4,
                                        subject_fallback=True)
    except TypeError:
        extra = [t for t in unseen if t[2].lower() in subjects][:max_extra]

    out = []
    for t in extra:
        if t[2].lower() not in subjects:
            continue
        key = (t[0], t[3])
        if key in retrieved:
            continue
        out.append(t)
        retrieved.add(key)
        if len(out) >= max_extra:
            break
    return out


def _scaled_budget(base_budget: int, pool_tokens: int,
                   ratio: float = 0.12, cap: int = 2400, floor: int = 1400) -> int:
    """Scale a per-mode token budget to the candidate pool actually retrieved.

    The per-mode budgets are calibrated for SHORT conversations.  On LoCoMo a
    typical candidate pool is ~290 turns / ~9,000 tokens, so the fixed 550-token
    budget admits only ~6% of it and structurally excludes the rest -- measured
    as the single largest cause of the answer turn never reaching the reader.
    """
    scaled = int(pool_tokens * ratio)
    return int(max(base_budget, min(cap, max(floor, scaled))))


def _question_subject(question: str, turns) -> str | None:
    """The single speaker the question is about, or None when ambiguous."""
    speakers: list[str] = []
    for t in turns:
        if len(t) > 2 and t[2] and t[2] not in speakers:
            speakers.append(t[2])
    low = question.lower()
    hits = [s for s in speakers if re.search(rf"\b{re.escape(s.lower())}\b", low)]
    return hits[0] if len(hits) == 1 else None


def _enumeration_probe_turns(question: str, all_turns, retriever,
                             already: set, max_extra: int = 6) -> list:
    """Extra candidate turns for ENUMERATION questions.

    A list-gold question ('what activities does Melanie do?') has its answer
    items spread across many turns, and those turns are frequently lexically
    unrelated to the question -- measured: the missing item's turn sits OUTSIDE
    top-45 retrieval and carries anchor score 0.0, so neither a larger
    retrieval window (base_top_k 45 -> 150 changed nothing) nor a bigger token
    budget (550 -> 2400 still admitted only 3 of 4 items) can surface it.

    Only 38% of list-gold questions have ALL their items in the reader context,
    so the model returns a SUBSET -- and it cannot do otherwise, because the
    remaining items were never shown to it.

    Strategy: run a second retrieval pass using the question's PLURAL HEAD as
    the query ('activities', 'events', 'books').  Each item-bearing turn usually
    mentions that head noun, which the original question-only query does not
    weight heavily.
    """
    import re as _re
    m = _re.search(r"\b(activities|events|things|kinds|types|places|books|songs|movies|foods|dishes|hobbies|sports|games|names|cities|countries|states)\b",
                   question.lower())
    if not m:
        return []
    head = m.group(1)
    extra, _scores = retriever.rank_turns(head, all_turns, top_k=max_extra * 3)
    out = []
    for t in extra:
        key = (t[0], t[3])
        if key in already:
            continue
        out.append(t)
        if len(out) >= max_extra:
            break
    return out


def build_reader_prompt(question: str, context: str, enum_contract: str = "off",
                        no_abstain: bool = False) -> str:
    """Evidence-first reader prompt.

    NO BENCHMARK ANSWERS ARE EMBEDDED HERE.

    An earlier version of this prompt hardcoded dataset gold answers and
    per-question disambiguation branches keyed on question wording.  That is
    the same class of contamination that invalidated the R8/R8.1 LongMemEval
    results, and it was quietly inflating this score.  A corpus audit found
    20 gold answers written into the prompt, worth up to 2.7pt.  All of it is
    removed.  See git log for commit fa00a9c for the exact list.

    What remains is GENERIC reading guidance only: how to resolve relative
    dates from the session header, how to separate two speakers, how to
    enumerate exhaustively.  No question-, topic- or answer-specific content.
    """
    base_prompt = (
        f"[TARGET GOAL]\n"
        f"You are answering this exact question: \"{question}\"\n\n"
        f"RULES:\n"
        f"1. SPEAKER ATTRIBUTION:\n"
        f"   - The dialogue has two speakers. Only attribute events, dates and traits to the EXACT person asked about. Never merge one speaker's facts with the other's.\n"
        f"2. MULTI-HOP EVIDENCE & UNNAMED REFERENCES:\n"
        f"   - Connect pronouns and demonstratives ('that book', 'that trip') to the specific dates and details stated elsewhere in the evidence.\n"
        f"   - When asked where someone moved FROM, output the specific place name that was stated, not a generic phrase like 'their home country'.\n"
        f"3. EXHAUSTIVE ENUMERATION:\n"
        f"   - If the question asks about a person's activities, events, hobbies, coping methods, preferences, or multiple items, list EVERY distinct item found across all evidence lines, comma-separated. Do not stop at one.\n"
        f"4. SPECIFICITY:\n"
        f"   - When asked what kind of art/music/writing someone makes, give the specific style if the evidence states it.\n"
        f"5. DATE RESOLUTION:\n"
        f"   - The bracket header timestamp is the CONVERSATION date, not the event date. Resolve relative times ('yesterday', 'last Friday', 'two weekends ago') against that header date and give the ABSOLUTE date.\n"
        f"   - Preserve the stated weekday ('last Sunday' -> 'The Sunday before <header date>').\n"
        f"   - If the question names a month, only select events in that month.\n"
        f"   - For elapsed-time questions ('how long ago'), locate the stated interval in the evidence and report it.\n"
        f"6. Reply with the concise final answer only (a few words or a comma-separated list). Never show reasoning.\n"
        f"7. INFERENCE QUESTIONS:\n"
        f"   - When asked whether someone 'would likely' do or have something, answer from what the evidence actually establishes about them. Use 'Yes'/'No'/'Likely no' with a short reason.\n"
        f"   - For counterfactual questions, answer only if the evidence explicitly supports it; otherwise give the most supported answer.\n\n"
        f"=== VERIFIED EVIDENCE ===\n"
        f"{context}\n\n"
        f"=== FINAL ANSWER ===\n"
    )
    # Generic output-format guidance only -- keyed on question TYPE, never on
    # question content, and never naming an answer.
    hints = []
    ql = question.lower()
    if ql.startswith("where"):
        hints.append("- Output ONLY the place name(s) (comma-separated if multiple), never a date or a time.")
    if ql.startswith("how many") or ql.startswith("how much"):
        hints.append("- Output ONLY the number (and its unit if the question asks for one).")
    if ql.startswith("when"):
        hints.append("- Output ONLY the date or time expression.")

    hint_text = ("FORMAT RULES:\n" + "\n".join(hints) + "\n") if hints else ""

    # ENUMERATION OUTPUT CONTRACT (opt-in, one clause at a time).
    #
    # Rule 3 already tells the model to enumerate exhaustively, but it gives the
    # model nothing to CHECK against, and 63% of enumeration questions come back
    # with a subset.  Each option below is a single additional clause, selectable
    # so they can be measured one at a time rather than shipped as a bundle.
    #
    # 'count'     -- re-read the evidence and state how many distinct items exist
    #                before listing them.
    # 'evidence'  -- list only items the evidence actually states, and never
    #                invent an item to reach a rounder count.
    # 'attribute' -- keep the person/date/object each item belongs to attached.
    #
    # Off by default: the standing result is that "tell the model to reason"
    # made things WORSE once already, so prompt elaboration is not assumed to help.
    contract = ""
    if enum_contract == "count":
        contract = (
            "ENUMERATION CHECK:\n"
            "Before answering, count how many DISTINCT items the evidence states for this "
            "question. Then list exactly those items, comma-separated. If you list fewer "
            "than you counted, re-read the evidence and add the missing items.\n\n"
        )
    elif enum_contract == "evidence":
        contract = (
            "ENUMERATION CHECK:\n"
            "List only items the evidence explicitly states. Never add an item you did not "
            "read, and never drop an item because the list already looks long enough.\n\n"
        )
    elif enum_contract == "attribute":
        contract = (
            "ENUMERATION CHECK:\n"
            "When listing items, keep attached which person each item belongs to and when it "
            "happened, so items from different people are never merged.\n\n"
        )

    # ABSTENTION GUIDANCE (opt-in, off by default).
    #
    # Measured: 468 of 1,986 predictions (23.6%) are abstentions, and 310 of
    # those had the gold evidence IN the compiled context (gold-word coverage
    # >= 0.5).  The prompt neither permits nor forbids abstaining, so the model
    # decides on its own and gets it wrong two times in three.
    #
    # This clause asks for the best supported answer from the evidence shown,
    # without inventing anything.  It does NOT tell the model to guess beyond the
    # evidence, and it does not forbid an honest "not stated" -- the point is to
    # stop it abstaining when the answer is sitting in the context.
    abstain_clause = ""
    if no_abstain:
        abstain_clause = (
            "ANSWER FROM THE EVIDENCE:\n"
            "Answer using what the evidence above states. If it states the "
            "answer, give it -- do not decline because you are unsure of the "
            "wording. Say the information is not stated only when the evidence "
            "genuinely does not cover it. Never invent a detail you did not "
            "read.\n\n"
        )

    return (
        base_prompt
        + abstain_clause
        + contract
        + hint_text
        + "Answer with ONLY the concise value (place, date, number, name, or comma-separated list). Do not output conversational filler or full sentences.\n"
        + f"Question: {question}\n"
        + "Answer:"
    )


ABSTENTIONS = {"i don't know.", "", "not mentioned", "none"}

#: The pinned official harness accepts a refusal only in these forms for the
#: adversarial category (third_party/.../evaluation.py:217-221).  This is a FORMAT
#: contract, not a capability claim: the decision to refuse is made elsewhere and
#: is measured separately.  Aligning the lexical form is worth +12.50pt of
#: official F1 on the frozen 2026-10-23 run (V4_PROGRAM.md section 3.2).
REFUSAL_FORMS = ("no information available", "not mentioned")


def is_refusal(text: str) -> bool:
    """True when the output declines to answer (any accepted or legacy form).

    Used to decide what must be NORMALISED.  It is deliberately wider than the
    official contract so that a refusal phrased in the model's own words is
    still put into the accepted form before scoring.
    """
    low = str(text).strip().lower()
    return low in ABSTENTIONS or _has_official_refusal(low)


def _has_official_refusal(text: str) -> bool:
    """Strictly the forms the pinned harness recognises (evaluation.py:217-221)."""
    low = str(text).strip().lower()
    return any(form in low for form in REFUSAL_FORMS)


def canonical_refusal() -> str:
    """The refusal string the official protocol recognises."""
    return "not mentioned"


def score_question(pred: str, question: dict) -> bool:
    """Score one prediction under the official LoCoMo protocol.

    Category 5 (adversarial) is correct iff the output refuses in a form the
    upstream harness accepts; its ``adversarial_answer`` is a distractor and is
    never compared as gold.  Categories 1-4 keep the in-house matcher unchanged.

    The category-5 check is STRICTER than :func:`is_refusal` on purpose: it mirrors
    upstream exactly, so this scorer cannot be more lenient than the official one.
    The credit for a legacy-worded refusal comes from normalising it in the
    harness, not from a forgiving matcher.
    """
    if int(question.get("category", 0)) == 5:
        return _has_official_refusal(pred)
    return score(pred, question.get("answer", ""))


def reorder_context_by_speaker(context: str, question: str, conv: dict) -> str:
    """Group evidence lines by speaker, putting the question's target speaker first.

    Only changes PRESENTATION order of already-retrieved evidence.  Does not
    add, remove, or reorder evidence selection in any way.
    """
    spk_a = conv.get("speaker_a", "")
    spk_b = conv.get("speaker_b", "")
    # Conversation may be nested under 'conversation' key
    if not spk_a and not spk_b:
        inner = conv.get("conversation", {})
        if isinstance(inner, dict):
            spk_a = inner.get("speaker_a", "")
            spk_b = inner.get("speaker_b", "")
    q_names = {m.group(1).lower() for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", question)}
    target = None
    for spk in (spk_a, spk_b):
        if spk and spk.lower() in q_names:
            target = spk
            break
    if not target:
        return context

    parts = context.split("\n\n")
    other_parts = []
    evidence_block = None
    for p in parts:
        if p.startswith("[VERIFIED CONVERSATION EVIDENCE]"):
            evidence_block = p
        elif p:
            other_parts.append(p)

    if evidence_block is None:
        return context

    header = "[VERIFIED CONVERSATION EVIDENCE]"
    body_lines = evidence_block[len(header):].strip("\n").split("\n")

    target_lines: list[str] = []
    other_lines: list[str] = []
    for line in body_lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"\[.*?\]\s+(\w+):\s+.*", line)
        if m:
            if m.group(1).lower() == target.lower():
                target_lines.append(line)
            else:
                other_lines.append(line)
        else:
            target_lines.append(line)

    if not other_lines:
        return context

    reordered_body = (
        f"[VERIFIED CONVERSATION EVIDENCE -- {target} (target speaker)]\n"
        + "\n".join(target_lines)
        + "\n\n"
        + "[VERIFIED CONVERSATION EVIDENCE -- other speaker]\n"
        + "\n".join(other_lines)
    )

    result_parts: list[str] = []
    for p in other_parts:
        result_parts.append(p)
    result_parts.append(reordered_body)
    return "\n\n".join(result_parts)


def _write_frozen_contexts(out_dir: Path, rows: list[dict], args) -> None:
    """Write the compiled context of every question, with no reader call.

    This is the shared bench for engine work: retrieval + compilation are
    deterministic, so freezing them once gives every subsequent engine change the
    SAME context to answer from.  A paired before/after on this file isolates the
    engine, which is what `integration_gate.py` requires.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "contexts.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = Counter((r["category"], bool(r["committer_claimed"])) for r in rows)
    manifest = {
        "kind": "frozen_contexts",
        "n": len(rows),
        "config": {
            "cond": args.cond, "cats": args.cats, "second_pass": getattr(args, "second_pass", False),
            "data": args.data, "llm_calls": 0,
        },
        "committer_claimed": sum(1 for r in rows if r["committer_claimed"]),
        "per_category": {str(k): sum(v for (c, _), v in stats.items() if c == k)
                         for k in sorted({r["category"] for r in rows})},
        "context_tokens_mean": round(
            sum(r["context_tokens"] for r in rows) / max(1, len(rows)), 1),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nfrozen contexts -> {path}  ({len(rows)} questions, 0 LLM calls)")
    print(f"manifest        -> {out_dir / 'manifest.json'}")


def run(args) -> None:
    args.use_committer = not getattr(args, "no_committer", False)
    freeze_dir = Path(args.freeze_contexts) if getattr(args, "freeze_contexts", "") else None
    freeze_rows: list[dict] = []
    manifest = []
    if getattr(args, "manifest", ""):
        with open(args.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
    args.manifest = manifest
    data = load_locomo(Path(args.data))
    wanted = set(int(x) for x in args.cats.split(",")) if args.cats else None

    # Collect all (conversation, question) pairs
    jobs: list[tuple[dict, dict]] = []
    for conv in data:
        for q in load_questions(conv):
            if wanted and q["category"] not in wanted:
                continue
            jobs.append((conv, q))
    if args.limit:
        jobs = jobs[: args.limit]
    if args.manifest:
        # Select by the MANIFEST's (conv_idx, q_idx) pair, not by position.
        # `--limit N` is a conversation-CONTIGUOUS PREFIX: with conv-26 first
        # (199 qa), `--limit 100` selects 100 questions from ONE conversation,
        # which is not a stratified sample and cannot generalise.  A manifest
        # pins the exact cases so a dev set is actually the dev set.
        sel: list[tuple[dict, dict]] = []
        by_idx = {}
        for ci, conv in enumerate(data):
            for qi, q in enumerate(load_questions(conv)):
                by_idx[(ci, qi)] = (conv, q)
        for e in args.manifest:
            key = (int(e["conv_idx"]), int(e["q_idx"]))
            if key not in by_idx:
                raise SystemExit(f"manifest entry not in dataset: {key}")
            sel.append(by_idx[key])
        jobs = sel

    print(f"LoCoMo full benchmark: {len(jobs)} questions "
          f"(from {len(data)} conversations)")
    if wanted:
        print(f"  filtered to categories: {sorted(wanted)}")

    from eval_v4_pipeline_dev import V4CandidateRetriever  # reuse frozen retriever
    from artificial_memory_gen.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
    from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder
    from artificial_memory_gen.compiler.budgeted_compiler import BudgetedCognitiveCompiler
    from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
    from artificial_memory_gen.recall.fact_index import FactIndex
    from artificial_memory.research.benchmarks.llm import OllamaAnswerer

    retriever = V4CandidateRetriever()
    coverage = MultiSessionCoverageEngine()
    graph_builder = GraphBuilder()
    compiler = BudgetedCognitiveCompiler()
    planner = CognitiveExecutionPlanner()
    answerer = OllamaAnswerer(num_ctx=args.num_ctx)

    results: list[dict] = []
    by_cat: dict[int, list[bool]] = defaultdict(list)
    t_start = time.perf_counter()

    # cache the flattened turns per sample_id (conversations are large)
    turn_cache: dict[str, list] = {}
    fact_cache: dict[str, FactIndex] = {}

    for idx, (conv, q) in enumerate(jobs):
        t_q = time.perf_counter()
        sid = conv.get("sample_id", "x")
        if sid not in turn_cache:
            turn_cache[sid] = conversation_turns(conv["conversation"])
        all_turns, _ = turn_cache[sid]

        try:
            plan = planner.plan(q["question"], [], reference_date=None)
            token_budget = plan.recommended_budget

            top_turns, turn_scores, guaranteed = coverage.retrieve_convergent_evidence(
                query=q["question"], all_turns=all_turns, retriever=retriever,
                reference_date=None, base_top_k=45)

            # SECOND-PASS RECALL (opt-in).  base_top_k=45 is already ~99% full
            # of lexically-matching turns, so evidence that shares no
            # vocabulary with the question never becomes a candidate at all --
            # measured: candidate pool 44.7 -> 44.8 turns when a subject quota
            # was added, i.e. the quota only reshuffled existing candidates.
            # A quota cannot fix a cut; widening the cut can.  This re-queries
            # ONLY the turns the first pass did NOT retrieve, keyed on the
            # queried subject, and merges the survivors into the candidate set.
            if getattr(args, "second_pass", False):
                _sp = _second_pass_turns(q["question"], all_turns, top_turns,
                                         retriever, max_extra=60)
                if _sp:
                    top_turns = top_turns + _sp

            base_limit = 32 if plan.dominant_mode == "aggregation" else 28
            top_sids = {t[0] for t in top_turns[:base_limit]} | guaranteed
            cand = [t for t in all_turns if t[0] in top_sids]
            if not cand:
                cand = list(top_turns[:25])

            # FACT INDEX RECALL: Generic Cross-Session Fact, Action, and Coreference Recall
            if sid not in fact_cache:
                _fidx = FactIndex()
                _fidx.build(all_turns)
                fact_cache[sid] = _fidx
            fact_idx = fact_cache[sid]

            fact_turns = fact_idx.query(q["question"], all_turns, max_turns=16)
            already_cand = {(t[0], t[3]) for t in cand}
            for ft in fact_turns:
                key = (ft[0], ft[3])
                if key not in already_cand:
                    cand.append(ft)
                    already_cand.add(key)
                turn_scores[key] = max(turn_scores.get(key, 0.0), 25.0)
                if ft not in top_turns:
                    top_turns.append(ft)

            # ENUMERATION RECALL.  For list-gold questions the answer items
            # live in turns that rank outside top-45 and score 0.0, so they are
            # invisible to both the retriever and the knapsack.  Add a second
            # pass keyed on the question's plural head so the missing items can
            # compete at all.
            _probe = _enumeration_probe_turns(
                q["question"], all_turns, retriever,
                already={(t[0], t[3]) for t in cand})
            if _probe:
                cand = cand + _probe
                # The probe turns were pooled but never became anchors, so the
                # compiler could only reach them through bundle adjacency.  This
                # flag promotes them to anchors so the selection stage can
                # actually choose them (measurement first; see B0 in
                # design_docs/v4_world_class/V4_PROGRAM.md).
                if getattr(args, "probe_anchors", False):
                    for t in _probe:
                        turn_scores[(t[0], t[3])] = max(
                            turn_scores.get((t[0], t[3]), 0.0), 20.0)
                        if t not in top_turns:
                            top_turns.append(t)

            # SUBJECT-SCOPED ENUMERATION (measurement).  A list answer is the
            # union of what the asked person said about the category, and those
            # turns are pooled but rank too low to be selected.  Promoting the
            # subject's own turns to anchors is the direct test of that claim.
            if getattr(args, "subject_anchors", False):
                try:
                    from artificial_memory.skills.autonomous_engines.counting_engine import (
                        CountingEngine,
                    )
                    _is_list_q = CountingEngine.is_aggregation_question(q["question"])
                except Exception:
                    _is_list_q = False
                if _is_list_q:
                    _person = _question_subject(q["question"], all_turns)
                    if _person:
                        _own = [t for t in cand if t[2].lower() == _person.lower()]
                        _quota = int(getattr(args, "subject_anchor_quota", 12))
                        for t in _own[:_quota]:
                            turn_scores[(t[0], t[3])] = max(
                                turn_scores.get((t[0], t[3]), 0.0), 30.0)
                            if t not in top_turns:
                                top_turns.append(t)

            # BUDGET CALIBRATION.  The per-mode budget is calibrated for SHORT
            # conversations; LoCoMo candidate pools run ~290 turns / ~9,000
            # tokens, so a 550-token budget admits only ~6% and structurally
            # excludes the rest.  Scale the budget to the pool we actually
            # retrieved (this was the single largest cause of evidence loss).
            _pool_tokens = sum(len(t[3].split()) * 4 // 3 for t in cand)
            if _pool_tokens:
                token_budget = _scaled_budget(token_budget, _pool_tokens)
            # MEASUREMENT KNOB (default 1.0 = unchanged).  Used to test whether
            # the budget or the candidate pool binds enumeration recall.
            _bscale = float(getattr(args, "budget_scale", 1.0) or 1.0)
            if _bscale != 1.0:
                token_budget = int(token_budget * _bscale)

            t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(cand)}
            anchor_ids, anchor_scores = [], {}
            for t in top_turns:
                nid = t_map.get((t[0], t[3]))
                if nid and nid not in anchor_ids:
                    anchor_ids.append(nid)
                    anchor_scores[nid] = turn_scores.get((t[0], t[3]), 1.0)

            edg = graph_builder.build(cand, [])
            bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
            # MEASUREMENT: admit the queried person's turns even when the
            # question's form is not enumeration-shaped (B0 probe).
            _admit_names = None
            if getattr(args, "subject_admit", False):
                _admit_names = {
                    m.group(1) for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", q["question"])
                } or None
            compiled = compiler.compile(
                question=q["question"], bundles=bundles,
                token_budget=token_budget, anchor_scores=anchor_scores,
                is_multi_session=len(top_sids) > 1, adaptive_escalation=True,
                admit_subject_names=_admit_names)

            # DETERMINISTIC COMMITTER (fallback, not override).
            #
            # Eli's architecture: 8 deterministic engines already exist in
            # skills/autonomous_engines/ (temporal algebra, temporal anchor,
            # arithmetic difference, boolean verifier, counting, entity
            # attribute, event action, relational traverser) with 82 tests --
            # and none was wired into this harness.  Every number reported this
            # session came from 7B free generation alone.
            #
            # commit_answer claims a question only when an engine can answer it
            # deterministically, and carries turn-level provenance the reader
            # cannot produce.  Measured offline on cat2/temporal (n=321): claims
            # 118 at 69.5% accuracy, agreeing with the reader on 75, uniquely
            # solving 7, losing 7 -- so OVERRIDING the reader would net +0.
            # Hence FALLBACK: committer answers what it claims, reader answers
            # the rest, which preserves the reader's 7 unique wins.
            committed = None
            if getattr(args, "use_committer", True):
                try:
                    from artificial_memory.skills.answer_committer import (
                        commit_answer, post_process_answer,
                    )
                    _c = commit_answer(q["question"], compiled.context_text,
                                       q["category"])
                    if _c.used and str(_c.answer).strip():
                        committed = post_process_answer(
                            q["question"], str(_c.answer).strip(), q["category"])
                except Exception:
                    committed = None

            # FREEZE MODE: record the exact context the reader would have seen,
            # then move on without a reader call.  Used to build the shared bench
            # for engine work (0 LLM calls, deterministic).
            if freeze_dir is not None:
                freeze_rows.append({
                    "question_id": q["question_id"],
                    "category": q["category"],
                    "question": q["question"],
                    "gold": q["answer"],
                    "adversarial_answer": q.get("adversarial_answer", ""),
                    "context": compiled.context_text,
                    "context_tokens": getattr(compiled, "total_tokens", 0),
                    "committer_claimed": committed is not None,
                    "committer_answer": committed or "",
                    "plan_mode": getattr(plan, "dominant_mode", ""),
                    "token_budget": token_budget,
                    "anchor_ids": anchor_ids[:8],
                    # Optional: the candidate pool, so an offline analysis can tell
                    # a turn that never entered the pool (retrieval loss) from one
                    # that was pooled but not compiled (selection loss).
                    "pool": ([{"s": str(t[0]), "k": t[2], "t": str(t[3])[:400]}
                              for t in cand]
                             if getattr(args, "freeze_pool", False) else None),
                })
                continue

            if committed:
                pred = committed
            else:
                ctx_text = compiled.context_text
                if getattr(args, "speaker_first", False):
                    ctx_text = reorder_context_by_speaker(
                        ctx_text, q["question"], conv)
                pred = answerer.answer(q["question"],
                                       build_reader_prompt(q["question"], ctx_text,
                                                           enum_contract=getattr(args, "enum_contract", "off"),
                                                           no_abstain=getattr(args, "no_abstain", False))).text.strip()
                if pred.endswith(".") and len(pred.split()) <= 2:
                    pred = pred[:-1].strip()

                # Deterministic relative-date resolution (LoCoMo weakness).  Only
                # for 'when' questions, so a substantive answer is never rewritten.
                if re.search(r"\bwhen\b", q["question"], re.I):
                    pred = resolve_relative_dates(pred, ctx_text, q["question"])

            # PROTOCOL: emit a refusal in the form the official harness accepts.
            # The raw output is retained so the lexical change stays auditable
            # and is never confused with a capability gain (V4_PROGRAM.md 3.2).
            raw_pred = pred
            if is_refusal(pred):
                pred = canonical_refusal()
            ok = score_question(pred, q)
            ctx_tokens = getattr(compiled, "total_tokens", 0)
        except Exception as e:  # a single failure must not kill a 1986-q run
            pred = f"<ERROR: {type(e).__name__}: {e}>"
            ok = False
            ctx_tokens = 0
            committed = None
            raw_pred = None

        results.append({
            "question_id": q["question_id"], "category": q["category"],
            "question": q["question"], "ground_truth": q["answer"],
            # cat5 carries the distractor here for audit; it is never gold.
            "adversarial_answer": q.get("adversarial_answer", ""),
            "predicted_answer": pred, "is_correct": ok,
            # The model's own output before the refusal was put into the form the
            # official protocol accepts.  Kept so a format fix can never be
            # reported as a capability gain.
            "predicted_answer_raw": raw_pred,
            "refusal_normalized": raw_pred is not None and pred != raw_pred,
            # Recorded so evidence retention and cost can be measured from the
            # artifact alone.  Without these, every retention or latency question
            # needed a separate no-LLM script re-running the whole pipeline.
            "context_tokens": ctx_tokens,
            "latency_seconds": round(time.perf_counter() - t_q, 3),
            "gold_is_list": is_list_gold(str(q["answer"])),
            "abstained": bool(raw_pred is not None and is_refusal(raw_pred)),
            # True when the deterministic committer answered instead of the 7B
            # reader.  Recorded because the date-boost lesson is that "wired in"
            # is not evidence -- only a non-zero count plus a changed prediction
            # count proves the path executes.
            "committer_claimed": bool(committed),
        })
        by_cat[q["category"]].append(ok)

        if (idx + 1) % 50 == 0 or idx + 1 == len(jobs):
            elapsed = time.perf_counter() - t_start
            rate = elapsed / (idx + 1)
            eta = rate * (len(jobs) - idx - 1)
            print(f"[{idx+1}/{len(jobs)}] {elapsed:.0f}s elapsed, "
                  f"ETA {eta/60:.1f}min")

    if freeze_dir is not None:
        _write_frozen_contexts(freeze_dir, freeze_rows, args)
        return

    elapsed = time.perf_counter() - t_start
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])

    print()
    print("=" * 72)
    print(f"  LoCoMo FULL: {correct}/{total} ({correct/total*100:.2f}%)")
    print(f"  elapsed {elapsed/60:.1f}min")
    print("=" * 72)
    print(f"{'category':22s} {'acc':>8s} {'correct/total':>16s}")
    print("-" * 72)
    for cat in sorted(by_cat):
        v = by_cat[cat]
        print(f"cat{cat} {CAT_NAMES.get(cat,'?'):18s} "
              f"{sum(v)/len(v)*100:7.1f}% {sum(v):8d}/{len(v):<6d}")

    out = REPO / "benchmark_results" / "locomo_full.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write to a UNIQUE path per run.  Every run previously wrote to the same
    # locomo_full.json, so a stale background run finishing late would silently
    # clobber a newer, more-correct result (this actually happened: an old
    # 32.98% run overwrote the 47.13% artifact).  A timestamped file makes each
    # run's output independently verifiable and undeletable by a late arrival.
    import datetime as _dt
    _stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # RUN MANIFEST.  Gaming-catalog surface #12 (Audit Partition
    # Fingerprinting): a hold-out derived only from public identity is fully
    # guessable, so anyone iterating on --conv-idx/--offset tunes against a
    # known partition.  Stamp every artifact with the exact configuration and
    # the code revision that produced it, so a number can always be traced to
    # what made it -- and a tuned-on partition is visible as such.
    import hashlib
    import subprocess
    try:
        _rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5
                              ).stdout.strip() or "unknown"
    except Exception:
        _rev = "unknown"

    _config = {
        "cond": args.cond, "cats": args.cats, "limit": args.limit,
        "num_ctx": args.num_ctx, "second_pass": getattr(args, "second_pass", False),
        "enum_contract": getattr(args, "enum_contract", "off"),
        "speaker_first": getattr(args, "speaker_first", False),
        "manifest": args.manifest or "",
        "n_questions": len(jobs),
        "data": args.data,
    }
    _salt = hashlib.sha256(
        (_stamp + _rev + repr(sorted(_config.items()))).encode()
    ).hexdigest()[:16]

    _manifest = {
        "run_stamp": _stamp,
        "code_revision": _rev,
        "config": _config,
        "run_salt": _salt,
        # Which scoring contract produced `is_correct`.  v1 (legacy) compared
        # cat5 against `adversarial_answer`; v1-official uses the refusal contract
        # of the pinned harness.  A number is unreadable without this label.
        "scorer_protocol": "locomo_official_v1",
        "accuracy": correct / total,
        "per_category": {str(k): {"correct": sum(v), "total": len(v)}
                         for k, v in by_cat.items()},
        "scorer": "eval_locomo_full.score_question + is_list_gold (audit_strict_grade)",
    }

    out = out.with_name(f"locomo_full_{_stamp}.json")
    # Also keep the stable 'latest' pointer for convenience.
    latest = out.parent / "locomo_full.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({**_manifest, "results": results},
                  f, ensure_ascii=False, indent=2)
    with open(latest, "w", encoding="utf-8") as f:
        json.dump({**_manifest, "source": out.name},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")
    print(f"Latest -> {latest}")
    print(f"Run manifest: revision={_rev} salt={_salt} config={_config}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/external/locomo10.json")
    ap.add_argument("--cond", default="R7")
    ap.add_argument("--cats", default="", help="comma-separated categories")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--second-pass", action="store_true",
                    help="re-query turns the first pass did not retrieve, keyed "
                         "on the queried subject (off by default)")
    ap.add_argument("--enum-contract", default="off",
                    choices=["off", "count", "evidence", "attribute"],
                    help="one extra enumeration clause in the reader prompt; "
                         "each is measured separately (off by default)")
    ap.add_argument("--no-abstain", action="store_true",
                    help="add the answer-from-the-evidence clause (off by default)")
    ap.add_argument("--no-committer", action="store_true",
                    help="disable the deterministic committer fallback")
    ap.add_argument("--manifest", default="",
                    help="JSON file listing {conv_idx,q_idx} pairs to evaluate "
                         "(a real sample); overrides --limit, which is only a "
                         "conversation-contiguous prefix")
    ap.add_argument("--speaker-first", action="store_true",
                    help="reorder reader context so the question's target speaker's "
                         "evidence appears first (off by default)")
    ap.add_argument("--freeze-contexts", default="",
                    help="write the compiled context of every question to DIR and "
                         "make no reader call (the shared bench for engine work)")
    ap.add_argument("--budget-scale", type=float, default=1.0,
                    help="measurement knob: multiply the compiled token budget "
                         "(1.0 = unchanged; used to test whether the budget binds)")
    ap.add_argument("--freeze-pool", action="store_true",
                    help="with --freeze-contexts: also record the candidate pool, "
                         "to separate retrieval loss from selection loss")
    ap.add_argument("--probe-anchors", action="store_true",
                    help="promote enumeration-probe turns to anchors so the "
                         "compiler can select them (off by default)")
    ap.add_argument("--subject-anchors", action="store_true",
                    help="promote the asked person's own turns to anchors for "
                         "list-shaped questions (off by default)")
    ap.add_argument("--subject-anchor-quota", type=int, default=12,
                    help="how many of the subject's turns to promote (default 12)")
    ap.add_argument("--subject-admit", action="store_true",
                    help="admit the queried person's turns even without lexical "
                         "overlap (off by default; measured for B0)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
