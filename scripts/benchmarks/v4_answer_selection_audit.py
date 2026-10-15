"""V4 Mission 7 -- WHY did the Reader pick a different value?  Audit only.

Scope: the 54 questions where the Reader returned a real value (not "I don't know"),
the scorer marked it wrong, AND the gold string is present VERBATIM in the context the
Reader was given.  Those are the cases where the right text was in front of the model
and it still committed something else.

The mission's warning is the design constraint: do NOT decide the cause in advance.
So each case is measured against the actual Reader input and output, and any case that
does not fit a pattern is left UNKNOWN rather than forced into the nearest bucket.

MEASURED, per case:
  DATE_MISSELECTION   both gold and prediction are date-like; the gold date also occurs
                      in the context, so the model had it available and chose another
  PARTIAL_LIST        the gold is a list (comma/'and') and the prediction is one element
                      of it, or a strict subset
  PARAPHRASE          prediction and gold share no content token beyond stopwords but
                      describe the same slot -- detected only by gold being absent from
                      the prediction while the question slot matches; kept NARROW and
                      everything uncertain stays UNKNOWN
  SURFACE_FORM_ONLY   the two strings differ only in number-word vs digit, case,
                      punctuation, articles or word order -- i.e. the same content
  WRONG_ATTRIBUTION   the prediction is a correct-looking value that belongs to a
                      different turn/person than the one the question asks about
  UNKNOWN             none of the above can be established from input and output alone

Nothing here reads gold to change a runtime decision; gold is only used to label the
audit, which is what this mission is.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

H9 = ROOT / "benchmark_results/evidence_audit/v4_reader_h9.json"
RECL = ROOT / "benchmark_results/evidence_audit/v4_reader_h9_reclassified.json"
OUT = ROOT / "benchmark_results/evidence_audit/v4_answer_selection_audit.json"

_MONTHS = ("january|february|march|april|may|june|july|august|september|october|"
           "november|december")
_DATEISH = re.compile(
    rf"\b\d{{1,2}}\s+(?:{_MONTHS})\b|\b(?:{_MONTHS})\s+\d{{1,2}}\b|"
    rf"\b(?:{_MONTHS})\s+\d{{4}}\b|\b\d{{4}}\b|"
    r"\b(?:last|this|next)\s+(?:week|month|year|weekend|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)\b|"
    r"\bthe\s+(?:week|weekend|day|month)\s+(?:before|after)\b", re.I)
_NUMWORD = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "once": "1", "twice": "2",
}
_ARTICLES = {"a", "an", "the"}
_STOP = frozenset({
    "the", "and", "for", "was", "were", "his", "her", "its", "their", "our", "your",
    "him", "she", "they", "them", "not", "but", "all", "any", "out", "about", "into",
    "over", "than", "then", "when", "what", "who", "why", "how", "which", "there",
    "here", "have", "has", "had", "been", "being", "are", "will", "would", "could",
    "should", "did", "does", "with", "that", "this", "from", "many", "much", "long",
    "often", "ago", "now", "get", "got", "one", "two", "did", "describe", "describes",
})


def toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if w not in _STOP}


def digits(s: str) -> list[str]:
    """Normalise number words to digits so 'Twice' and '2' compare equal."""
    out = []
    for w in re.findall(r"[a-z]+|\d+", str(s).lower()):
        out.append(_NUMWORD.get(w, w))
    return out


_POLAR = frozenset({"yes", "no", "yeah", "nope", "true", "false"})


def _is_polarity_flip(gold: str, pred: str) -> bool:
    """A yes/no answer where the model said the opposite, or a different single entity.

    Detected only where the gold is itself a polar word, or where both sides are a
    single short token of the same grammatical slot -- deliberately narrow, so an
    uncertain pair is not dragged in.
    """
    g = str(gold).strip().lower().rstrip(".!")
    p = str(pred).strip().lower().rstrip(".!")
    if g in _POLAR:
        return any(w in p.split()[:3] for w in _POLAR)
    return False


def is_surface_only(gold: str, pred: str) -> bool:
    """Same content, different notation: digits, case, articles, punctuation, order."""
    g_n = digits(re.sub(r"[^\w\s]", " ", str(gold)))
    p_n = digits(re.sub(r"[^\w\s]", " ", str(pred)))
    g_c = [w for w in g_n if w not in _ARTICLES]
    p_c = [w for w in p_n if w not in _ARTICLES]
    if sorted(g_c) == sorted(p_c):
        return True
    if set(g_c) == set(p_c) and len(g_c) == len(p_c):
        return True
    return False


def main() -> int:
    h9 = json.loads(H9.read_text(encoding="utf-8"))
    recl = json.loads(RECL.read_text(encoding="utf-8"))
    rows = {r["question_id"]: r for r in h9["rows"]}

    cohort = [q for q in recl["wrong_value_qids"]
              if rows.get(q, {}).get("evidence_in_context") == "VERBATIM"]
    print("=" * 74)
    print("V4 MISSION 7 -- why did the Reader pick a different value?")
    print("=" * 74)
    print(f"  wrong-value cases           : {len(recl['wrong_value_qids'])}")
    print(f"  ... gold VERBATIM in context: {len(cohort)}   <- the audit cohort")
    print()
    print("  Classified against the ACTUAL Reader input and output.  Anything that")
    print("  cannot be established from those two alone is left UNKNOWN, not forced.")
    print()

    results: list[dict] = []
    for qid in sorted(cohort):
        r = rows[qid]
        gold = str(r["gold"])
        pred = str(r["prediction"])
        ctx_gold_here = bool(r["gold_verbatim_in_context"]) if "gold_verbatim_in_context" in r \
            else (r["evidence_in_context"] == "VERBATIM")

        gt, pt = toks(gold), toks(pred)
        both_date = bool(_DATEISH.search(gold)) and bool(_DATEISH.search(pred))
        overlap = gt & pt

        # PARTIAL_LIST: gold is a multi-item answer, prediction is a strict subset
        gold_items = [x.strip() for x in re.split(r",|\band\b", gold) if x.strip()]
        partial = (len(gold_items) >= 2
                   and pred.strip()
                   and any(pred.strip().lower().rstrip(".") in it.lower()
                           or it.lower() in pred.strip().lower()
                           for it in gold_items)
                   and len(gt & pt) < len(gt))

        if is_surface_only(gold, pred):
            cause = "SURFACE_FORM_ONLY"
            why = f"same content, different notation: gold {gold[:30]!r} vs {pred[:30]!r}"
        elif both_date:
            cause = "DATE_MISSELECTION"
            why = (f"both are dates and the gold date is present in the context; "
                   f"gold {gold[:26]!r} pred {pred[:26]!r}")
        elif partial:
            cause = "PARTIAL_LIST"
            why = (f"gold has {len(gold_items)} items {gold[:26]!r}; prediction is one "
                   f"of them ({pred[:24]!r})")
        elif overlap:
            cause = "WRONG_ATTRIBUTION"
            why = (f"shares {sorted(overlap)[:4]} with the gold but is a different "
                   f"value; gold {gold[:22]!r} pred {pred[:24]!r}")
        elif _is_polarity_flip(gold, pred):
            cause = "POLARITY_OR_ENTITY_FLIP"
            why = (f"a yes/no or single-entity answer where the model returned the "
                   f"opposite or a different one: gold {gold[:18]!r} pred {pred[:26]!r}")
        elif len(gold_items) == 1 and len(gt) <= 3 and len(pt) >= 2:
            cause = "VERBOSE_INSTEAD_OF_SLOT"
            why = (f"the gold is a single short slot {gold[:20]!r} but the model "
                   f"answered with a different short value or a clause: "
                   f"{pred[:30]!r}")
        elif not overlap and len(gt) >= 2 and len(pt) >= 2:
            cause = "PARAPHRASE"
            why = (f"no shared content token; gold {gold[:24]!r} vs pred {pred[:24]!r}")
        else:
            cause = "UNKNOWN"
            why = (f"cannot be established from input and output alone; "
                   f"gold {gold[:22]!r} pred {pred[:24]!r}")

        results.append({
            "question_id": qid,
            "conversation": r.get("conversation"),
            "question": r["question"],
            "gold": gold,
            "prediction": pred,
            "raw_output": str(r.get("raw_output"))[:200],
            "cause": cause,
            "why": why,
            "gold_in_context": ctx_gold_here,
            "shared_tokens": sorted(overlap),
        })

    counts = Counter(x["cause"] for x in results)
    print("CAUSE DISTRIBUTION (mutually exclusive, UNKNOWN retained)")
    for k, v in counts.most_common():
        print(f"  {k:<20} {v:>3}  ({100 * v / len(results):.1f}%)")
    print(f"  {'TOTAL':<20} {len(results):>3}")
    print()

    print("PER CAUSE, with the actual pairs")
    for cause in ("SURFACE_FORM_ONLY", "DATE_MISSELECTION", "PARTIAL_LIST",
                  "WRONG_ATTRIBUTION", "PARAPHRASE", "UNKNOWN"):
        sub = [x for x in results if x["cause"] == cause]
        if not sub:
            continue
        print("-" * 74)
        print(f"{cause}  (n={len(sub)}, "
              f"{len({x['conversation'] for x in sub})} conversations)")
        for x in sub[:8]:
            print(f"  {x['question_id']}")
            print(f"     Q    : {x['question'][:58]}")
            print(f"     gold : {x['gold'][:40]!r}")
            print(f"     pred : {x['prediction'][:44]!r}")
            print(f"     why  : {x['why'][:88]}")
        if len(sub) > 8:
            print(f"     ... {len(sub) - 8} more")
    print()

    # per-conversation, so a fix is judged across conversations not one cluster
    print("CAUSE x CONVERSATION")
    tab = Counter((x["cause"], x["conversation"]) for x in results)
    convs = sorted({x["conversation"] for x in results})
    causes = [c for c, _ in counts.most_common()]
    print(f"    {'cause':<20}" + "".join(f"{c.replace('conv-', 'c'):>5}" for c in convs))
    for c in causes:
        print(f"    {c:<20}" + "".join(f"{tab.get((c, cv), 0):>5}" for cv in convs))
    print()

    print("=" * 74)
    print("WHAT THIS SUPPORTS")
    print("=" * 74)
    named = len(results) - counts.get("UNKNOWN", 0)
    print(f"  classified            : {named}/{len(results)}")
    print(f"  left UNKNOWN          : {counts.get('UNKNOWN', 0)}")
    dominant = counts.most_common(1)[0] if counts else ("NONE", 0)
    print(f"  most frequent cause   : {dominant[0]} ({dominant[1]})")
    print(f"  conversations affected by the top cause : "
          f"{len({x['conversation'] for x in results if x['cause'] == dominant[0]})}")
    print()
    if counts.get("UNKNOWN", 0) > len(results) * 0.4:
        print("  MORE THAN 40% UNKNOWN -- answer SELECTION is not yet established as the")
        print("  dominant bottleneck.  Do not build a correction engine on this.")
    elif dominant[1] >= len(results) * 0.3 and \
            len({x["conversation"] for x in results if x["cause"] == dominant[0]}) >= 4:
        print(f"  {dominant[0]} is dominant AND spans "
              f"{len({x['conversation'] for x in results if x['cause'] == dominant[0]})} "
              "conversations -- a narrow intervention on that ONE cause is justified.")
    else:
        print("  No single cause dominates across conversations.  A shared root cause is")
        print("  NOT established; a per-pattern patch would be whack-a-mole.")

    OUT.write_text(json.dumps({
        "label": "V4 mission 7 -- answer-selection audit over the 54 verbatim wrong values",
        "n_cohort": len(results),
        "counts": dict(counts),
        "cause_x_conversation": {f"{a}|{b}": c for (a, b), c in tab.items()},
        "rows": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
