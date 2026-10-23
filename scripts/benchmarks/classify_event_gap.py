"""Phase 1 -- classify the vocabulary gap, and check the BINDING, not just presence.

size_event_gate_gap.py established that _date_event_matches rejects every turn for 27
of the 33 actionable abstentions, and that in all 27 a turn carries an explicit full
date.  What it did NOT establish is the thing that decides whether a fix is safe:

    does the date actually BIND to the asked event?

A date sitting in a turn that happens to share a few words with the question is not
evidence of a binding.  If the carrying turn does not describe the asked event, then
widening the gate would commit a date from an unrelated event -- which is exactly the
failure the strict gate exists to prevent, and exactly what the rejected branch did
33 times.

So each missing keyword is classified by WHY it is missing:

    INFLECTION   a stem-variant is in the turn  (buy/bought, visit/visited)
    ALIAS        an existing LEMMA_MAP/_EVENT_TERM_ALIASES entry covers it
    PHRASAL      a closed idiomatic pair (return home / got back, go to / visit)
    SEMANTIC     no rule-based relation findable -- needs judgement
    ABSENT       the concept genuinely is not in the carrying turn

and each case records a BINDING verdict:

    CLEAN    person present AND >=60% of event terms present in the carrying turn
    PARTIAL  person present but below that
    UNBOUND  person absent from the carrying turn

Reported per pattern with counts, plus the concrete (question term, turn evidence)
pairs for the SEMANTIC bucket so the model dependency can be judged rather than
assumed.

Nothing here adds a rule.  This measures whether a small set of GENERALISABLE rules
could close the gap, and whether closing it would bind correctly.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory.skills.answer_committer import parse_turns  # noqa: E402
from artificial_memory.skills.autonomous_engines.temporal_algebra_engine import (  # noqa: E402
    LEMMA_MAP,
    TemporalAlgebraEngine,
    _EVENT_TERM_ALIASES,
    _NON_EVENT_DATE_TERMS,
    _date_event_matches,
)

MAP = ROOT / "benchmark_results/when_date_map/when_date_failure_map.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/when_date_map/event_gap_classified.json"

EXPLICIT_DATE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+20\d\d\b")


def _stem(word: str) -> str:
    """Crude stem.  Handles the inflections that actually occur in this corpus.

    Written carefully because an UNDER-stemming classifier inflates the SEMANTIC
    bucket: 'hike' vs 'hiking' is morphology, not semantics, and mislabelling it
    overstates the model dependency.

    The trick that avoids a hand-listed exception set: compare on the E-LESS form.
    A silent 'e' is exactly the thing that is unpredictable ('hike' but 'dance',
    'stop' but 'keep'), so instead of deciding where to re-attach it, the stem is
    defined as the form with a trailing silent 'e' already removed.  Then

        hike   -> hik        dance  -> danc       stop -> stop
        hiking -> hik        dancing-> danc       stopped -> stop

    all collapse without any per-word list, so this generalises and cannot overfit
    this corpus.  Geminate undoubling ('running' -> 'run') is handled first.
    """
    w = word
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        base = w[:-2]
    elif w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    elif w.endswith("es") and len(w) > 4:
        base = w[:-2]
    elif w.endswith("s") and len(w) > 3:
        base = w[:-1]
    else:
        base = w
    if len(base) >= 2 and base[-1] == base[-2]:      # running -> run
        base = base[:-1]
    if len(base) >= 2 and base.endswith("e"):        # compare e-less
        base = base[:-1]
    return base


#: Closed idiomatic pairs.  DELIBERATELY NOT a synonym table -- these are
#: grammatical alternations (phrasal verb vs simple verb), each of which
#: generalises across many verbs, not a list of words that happen to co-occur.
#: If the SEMANTIC bucket turns out large, this file does not grow.
_PHRASAL = {
    "return": {"back", "got", "get", "came", "come", "arrive", "arrived"},
    "go": {"went", "visit", "visited", "attend", "attended", "head"},
    "buy": {"got", "get", "purchase", "purchased", "pick", "picked"},
    "visit": {"go", "went", "see", "saw", "stop"},
    "sell": {"sold", "gave", "give", "let"},
    "meet": {"met", "saw", "see", "ran", "run"},
    "receive": {"got", "get", "was", "given"},
    "finish": {"completed", "done", "wrapped"},
    "start": {"began", "begin", "opened", "kicked"},
    "lose": {"lost", "let"},
}


def carrying_turns(gold: str, context: str, turns) -> list:
    """Turns that state the gold date, as (index, text)."""
    g = str(gold).strip().strip('"').strip("'")
    hits = []
    for i, t in enumerate(turns):
        if g and g.lower() in t.text.lower():
            hits.append((i, t))
    if hits:
        return hits
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)", g)
    if m:
        pat = re.compile(
            rf"\b{re.escape(m.group(1))}\s+{re.escape(m.group(2))}\b"
            rf"|\b{re.escape(m.group(2))}\s+{re.escape(m.group(1))}(?:st|nd|rd|th)?\b",
            re.I)
        for i, t in enumerate(turns):
            if pat.search(t.text):
                hits.append((i, t))
    return hits


def classify_gap(term: str, turn_words: set[str]) -> str:
    if term in turn_words:
        return "PRESENT"
    if _stem(term) in {_stem(w) for w in turn_words}:
        return "INFLECTION"
    if (LEMMA_MAP.get(term, {term}) | _EVENT_TERM_ALIASES.get(term, set())) & turn_words:
        return "ALIAS"
    if _PHRASAL.get(term, set()) & turn_words:
        return "PHRASAL"
    return "SEMANTIC"


def main() -> int:
    data = json.loads(MAP.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row

    actionable = [r for r in data["records"]
                  if r["engine_class"] == "ENGINE_ABSTAIN"
                  and r["gold_in_context"]]
    print(f"actionable abstentions : {len(actionable)}")
    print()

    pattern_counts = Counter()
    binding_counts = Counter()
    per_case: list[dict] = []
    semantic_pairs: list[tuple] = []

    for r in actionable:
        context = (cache.get(r["cache_qid"]) or {}).get("context") or ""
        turns = parse_turns(context)
        if not turns:
            binding_counts["NO_CONTEXT"] += 1
            continue
        person = TemporalAlgebraEngine._extract_person(r["question"], turns)
        q_kws = TemporalAlgebraEngine._extract_keywords(r["question"])
        event_terms = q_kws - _NON_EVENT_DATE_TERMS - (
            {person.lower()} if person else set())

        carriers = carrying_turns(r["gold"], context, turns)
        if not carriers:
            binding_counts["NO_CARRYING_TURN"] += 1
            continue
        # Prefer a carrier that mentions the person; that is the candidate binding.
        chosen = None
        for i, t in carriers:
            if person and person.lower() in (t.speaker.lower() + " " + t.text.lower()):
                chosen = (i, t)
                break
        if chosen is None:
            chosen = carriers[0]
        idx, turn = chosen

        turn_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", turn.text.lower()))
        speaker_blob = (turn.speaker + " " + turn.text).lower()
        person_in = bool(person) and person.lower() in speaker_blob

        missing = [t_ for t_ in event_terms if t_ not in turn_words]
        buckets = Counter(classify_gap(t_, turn_words) for t_ in missing)
        present = sum(1 for t_ in event_terms if t_ in turn_words)
        frac = present / len(event_terms) if event_terms else 1.0

        for t_ in missing:
            if classify_gap(t_, turn_words) == "SEMANTIC":
                if len(semantic_pairs) < 25:
                    ctx_snip = re.sub(r"\s+", " ", turn.text)[:110]
                    semantic_pairs.append((r["question_id"], t_, ctx_snip))

        # Which single rule class, if applied, would let this turn through?
        rule_needed = None
        for cls in ("INFLECTION", "PHRASAL", "ALIAS", "SEMANTIC"):
            if buckets.get(cls):
                rule_needed = cls
                break

        pattern_counts[rule_needed or "NONE_MISSING"] += 1
        if not person_in:
            binding_counts["UNBOUND_person_absent"] += 1
        elif frac >= 0.6:
            binding_counts["CLEAN"] += 1
        else:
            binding_counts["PARTIAL"] += 1

        per_case.append({
            "question_id": r["question_id"],
            "question": r["question"],
            "gold": r["gold"],
            "person": person,
            "carrying_turn_id": turn.dia_id,
            "carrying_turn": re.sub(r"\s+", " ", turn.text)[:200],
            "event_terms": sorted(event_terms),
            "missing": sorted(missing),
            "missing_by_rule": dict(buckets),
            "rule_that_would_pass": rule_needed,
            "event_term_fraction_in_turn": round(frac, 2),
            "person_in_carrying_turn": person_in,
            "binding": ("UNBOUND" if not person_in
                        else "CLEAN" if frac >= 0.6 else "PARTIAL"),
        })

    print("=" * 74)
    print("MISSING-KEYWORD PATTERNS (which rule class would let the turn through)")
    print("=" * 74)
    for k, n in pattern_counts.most_common():
        print(f"  {k:<16} {n:>3}  ({n / max(1, len(per_case)) * 100:.0f}%)")
    print()

    print("=" * 74)
    print("BINDING VERDICT -- does the date actually attach to the asked event?")
    print("=" * 74)
    for k, n in binding_counts.most_common():
        print(f"  {k:<24} {n:>3}")
    print()
    clean = binding_counts.get("CLEAN", 0)
    partial = binding_counts.get("PARTIAL", 0)
    unbound = binding_counts.get("UNBOUND_person_absent", 0)
    print(f"  CLEAN   {clean:>3}  <- widening the gate would bind correctly here")
    print(f"  PARTIAL {partial:>3}  <- person right, event only partly described")
    print(f"  UNBOUND {unbound:>3}  <- person NOT in the date-bearing turn; widening")
    print(f"                   here would bind an unrelated event")
    print()

    # Aggregate the missing-keyword classes across all cases.
    agg = Counter()
    for c in per_case:
        for cls, n in c["missing_by_rule"].items():
            agg[cls] += n
    print("=" * 74)
    print("ALL MISSING KEYWORDS, BY RULE CLASS (aggregate)")
    print("=" * 74)
    for k, n in agg.most_common():
        print(f"  {k:<16} {n:>3}")
    total_missing = sum(agg.values())
    coverable = agg.get("INFLECTION", 0) + agg.get("PHRASAL", 0) + agg.get("ALIAS", 0)
    print()
    print(f"  total missing keywords        : {total_missing}")
    print(f"  coverable by general rules    : {coverable} "
          f"({coverable / max(1, total_missing) * 100:.0f}%)")
    print(f"  INFLECTION only               : {agg.get('INFLECTION', 0)} "
          f"<- pure morphology, fully generalisable")
    print(f"  PHRASAL (closed idioms)       : {agg.get('PHRASAL', 0)}")
    print(f"  SEMANTIC (needs judgement)    : {agg.get('SEMANTIC', 0)} "
          f"<- whack-a-mole risk lives here")
    print()

    print("=" * 74)
    print("SEMANTIC BUCKET -- the concrete pairs, so the dependency is judged")
    print("=" * 74)
    for qid, term, snip in semantic_pairs:
        print(f"  {qid:<17} missing={term!r}")
        print(f"      turn: {snip}")
    print()

    print("HOW TO READ THIS")
    print("  INFLECTION and PHRASAL are the tractable part: they are regular")
    print("  alternations, so a handful of rules covers many words.  If most missing")
    print("  keywords land there AND the binding is CLEAN, a small general change is")
    print("  justified and Phase 2 has a real target.")
    print()
    print("  A large SEMANTIC bucket is the model dependency again.  Adding a synonym")
    print("  per word is whack-a-mole and is explicitly out of scope.")
    print()
    print("  UNBOUND cases are the safety floor: no amount of gate widening should")
    print("  commit a date whose turn does not even name the person.")

    OUT.write_text(json.dumps({
        "label": "when_date event-gap classification, phase 1",
        "n": len(per_case),
        "patterns": dict(pattern_counts),
        "binding": dict(binding_counts),
        "missing_by_rule_aggregate": dict(agg),
        "coverable_by_general_rules": coverable,
        "total_missing": total_missing,
        "cases": per_case,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
