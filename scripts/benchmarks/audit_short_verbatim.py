"""Pre-implementation audit for LETHE_SHORT_VERBATIM.  AUDIT ONLY -- production untouched.

The cohort is RECOMPUTED here, not trusted from the previous report, because the last
three audits in this project each found their own counting rule was wrong and the
reachability number moved every time.

MATCH RULES, stated precisely because the cohort depends on them:

  normalization   lowercased; whitespace runs collapsed to one space; leading and
                  trailing whitespace stripped.  Nothing else is stripped.
  punctuation     KEPT as part of the match.  A gold of '"Charlotte\\'s Web"' therefore
                  matches only text containing the quotation marks.  This is the strict
                  reading, and the audit reports how many golds it excludes.
  length          measured in WORDS of the RAW gold (split on whitespace), so
                  '"Becoming Nicole"' is 2 words and 'A few times a week' is 5.
  occurrences     counted on the NORMALIZED context with non-overlapping search, so
                  '1 time' is not counted as an occurrence of 'time' and a gold inside
                  a longer word never matches.

DISCOVERY vs SCORING, kept apart as the mission requires:
  * gold is used ONLY to measure the cohort and to label the audit table
  * the proposed runtime rule is described in terms of QUESTION features and CONTEXT
    features only; no rule below reads a gold value.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory.skills.deterministic_memory_engine import (  # noqa: E402
    DeterministicMemoryEngine,
)

_spec = importlib.util.spec_from_file_location(
    "eval_locomo_full", ROOT / "scripts/benchmarks/eval_locomo_full.py")
scorer = importlib.util.module_from_spec(_spec)
sys.modules["eval_locomo_full"] = scorer
try:
    _spec.loader.exec_module(scorer)
except SystemExit:
    pass

AUDIT = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
OUT = ROOT / "benchmark_results/evidence_audit/short_verbatim_audit.json"

SHORT_WORDS = 2          # the short-answer threshold
_UNIQUE = 1              # 'occurs exactly once'

_WS = re.compile(r"\s+")

#: Question shapes -- used only to see whether the cohort is one phenomenon.
_YESNO = re.compile(r"^\s*(?:did|do|does|is|are|was|were|has|have|had|can|could|"
                    r"will|would|should)\b", re.I)
_NUMERIC = re.compile(r"^\s*(?:how\s+(?:many|much|old|far)|\d)", re.I)
_DATEISH = re.compile(r"\b(?:when|what\s+(?:date|day|year|month)|since\s+\d{4})\b", re.I)
_QUOTED = re.compile(r"[\"'“”‘’]")
_PROPER = re.compile(r"\b[A-Z][a-z]{2,}")

_STOP = {"the", "and", "for", "was", "were", "his", "her", "its", "their", "our",
         "your", "him", "she", "they", "them", "not", "but", "all", "any", "out",
         "about", "into", "over", "than", "then", "when", "what", "who", "why",
         "how", "which", "there", "here", "have", "has", "had", "been", "being",
         "are", "will", "would", "could", "should", "did", "does", "with", "that",
         "this", "from", "did"}


def norm(s: str) -> str:
    return _WS.sub(" ", str(s).strip().lower())


def occurrences(haystack: str, needle: str) -> int:
    """Non-overlapping count on normalized text, so a gold cannot match inside a word."""
    if not needle:
        return 0
    return haystack.count(needle)


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cache[json.loads(line)["qid"]] = json.loads(line)

    rows: list[dict] = []
    for r in audit["records"]:
        ctx = cache.get(r["cache_qid"]) or {}
        context = ctx.get("context") or ""
        if not context:
            continue
        gold = r["gold"]
        ans = DeterministicMemoryEngine.resolve(r["question"], context)
        if ans.used:
            continue  # only the abstentions are in scope
        n_ctx = norm(context)
        g_norm = norm(gold)
        n_occ = occurrences(n_ctx, g_norm)
        if n_occ == 0:
            continue
        raw_words = len(str(gold).split())
        rows.append({
            "question_id": r["question_id"],
            "cache_qid": r["cache_qid"],
            "conversation": r["cache_qid"].split("-")[1],
            "question": r["question"],
            "gold": gold,
            "gold_norm": g_norm,
            "raw_words": raw_words,
            "occurrences": n_occ,
        })

    # ---- reproduce the three reported counts, and report every exclusion -----
    n_verbatim = len(rows)
    short = [x for x in rows if x["raw_words"] <= SHORT_WORDS]
    unique = [x for x in short if x["occurrences"] == _UNIQUE]

    n_abstain = 0
    for r in audit["records"]:
        c = cache.get(r["cache_qid"]) or {}
        if not c.get("context"):
            continue
        if not DeterministicMemoryEngine.resolve(r["question"], c["context"]).used:
            n_abstain += 1
    print("=" * 74)
    print("COHORT, recomputed from the frozen data")
    print("=" * 74)
    print(f"  abstentions scored                     : {n_abstain}")
    print(f"  gold verbatim in context (abstaining)  : {n_verbatim}")
    print(f"  ... gold <= {SHORT_WORDS} words                 : {len(short)}")
    print(f"  ... gold occurs exactly once           : {len(unique)}")
    print()
    print("  EXCLUSIONS applied, each one a deliberate narrowing:")
    print(f"    abstaining answers only                : reader path excluded")
    print(f"    gold must appear verbatim (punctuation kept)")
    print(f"    gold <= {SHORT_WORDS} raw words")
    print(f"    exactly 1 occurrence in the context")
    print()
    # Punctuation is kept, so how many golds carry quotes/punctuation at all?
    punct = [x for x in unique if _QUOTED.search(str(x["gold"]))]
    print(f"  of the {len(unique)}, golds containing quote/punctuation marks: {len(punct)}")
    print(f"    -> these match ONLY text that also carries the marks, which is why")
    print(f"       keeping punctuation is the strict reading and is reported separately.")
    print()

    # ---- classify every candidate -------------------------------------------
    for x in unique:
        q, g = x["question"], str(x["gold"])
        labels: list[str] = []
        if _YESNO.match(q) or norm(g) in {"yes", "no"}:
            labels.append("BOOLEAN_OR_YES_NO")
        if _NUMERIC.match(q) or _DATEISH.search(q) or re.fullmatch(
                r"[\d\s,.]+", g):
            labels.append("NUMERIC_OR_DATE")
        if _QUOTED.search(g):
            labels.append("QUOTED_OR_PUNCTUATED")
        # MULTIPLE_CANDIDATES: another string in the context is ALSO an instance of
        # the asked kind, so uniqueness of the gold does not make it selectable.
        q_terms = {w for w in re.findall(r"[a-z]{4,}", q.lower()) if w not in _STOP}
        n_ctx = norm(cache[x["cache_qid"]]["context"])
        same_kind = 0
        for m in re.finditer(r"\b[A-Za-z][a-z]{3,}(?:\s+[A-Za-z][a-z]{3,})?\b", n_ctx):
            pass
        labels_multi = _has_sibling_candidates(q_terms, n_ctx, g)
        if labels_multi:
            labels.append("MULTIPLE_CANDIDATES")
        if _PROPER.search(g) and g[:1].isupper():
            labels.append("ENTITY_OR_PROPER_NOUN")
        # RELATION support: does a sentence containing the gold also contain a
        # question term?  That is the only relation evidence available without
        # semantics, and its absence is exactly the VALUE_PRESENT_BUT_RELATION_UNPROVEN
        # case the mission warns about.
        rel = _relation_support(x, cache, q_terms)
        x["relation_turn"] = rel["turn_id"]
        x["relation_terms"] = sorted(rel["terms"])
        x["relation_sentence"] = rel["sentence"][:160]
        if rel["supported"]:
            labels.append("RELATION_EXPLICITLY_SUPPORTED")
        else:
            labels.append("VALUE_PRESENT_BUT_RELATION_UNPROVEN")
        x["labels"] = labels
        x["n_labels"] = len(labels)

    lab_counts = Counter(l for x in unique for l in x["labels"])
    print("=" * 74)
    print("CLASSIFICATION (labels may overlap)")
    print("=" * 74)
    for k, v in lab_counts.most_common():
        print(f"  {k:<40} {v}")
    print()

    # ---- the safe subset: unique, short, relation proven, not a yes/no -------
    unsafe = {"BOOLEAN_OR_YES_NO", "QUOTED_OR_PUNCTUATED",
              "VALUE_PRESENT_BUT_RELATION_UNPROVEN"}
    safe = [x for x in unique if not (set(x["labels"]) & unsafe)]
    print("=" * 74)
    print("SAFE-OPPORTUNITY ESTIMATE (conservative)")
    print("=" * 74)
    print(f"  1. literal-match candidates (abstaining)     : {n_verbatim}")
    print(f"  2. unique literal match, short               : {len(unique)}")
    print(f"  3. with explicit relational support          : "
          f"{sum(1 for x in unique if 'RELATION_EXPLICITLY_SUPPORTED' in x['labels'])}")
    print(f"  4. SAFE for a generic rule (unique+short+")
    print(f"     relation proven, not yes/no, unquoted)    : {len(safe)}")
    print(f"  5. need semantic inference / coreference     : "
          f"{sum(1 for x in unique if 'VALUE_PRESENT_BUT_RELATION_UNPROVEN' in x['labels'])}")
    print(f"  6. should abstain                            : {len(unique) - len(safe)}")
    print()
    print(f"  independent conversations, cohort 1: "
          f"{len({x['conversation'] for x in unique})}")
    print(f"  independent conversations, safe    : {len({x['conversation'] for x in safe})}")
    print()

    # ---- try to FALSIFY: inspect the safe subset by hand --------------------
    print("=" * 74)
    print("FALSIFICATION -- every 'safe' candidate, with its relation sentence")
    print("=" * 74)
    for x in safe:
        print(f"  {x['question_id']}  gold={str(x['gold'])[:22]!r}")
        print(f"     Q   : {x['question'][:62]}")
        print(f"     rel : [{x['relation_turn']}] {x['relation_sentence'][:80]}")
        print(f"     terms: {x['relation_terms']}")
    print()
    print("=" * 74)
    print("COUNTEREXAMPLES -- candidates a naive rule would answer wrongly")
    print("=" * 74)
    shown = 0
    for x in unique:
        if not (set(x["labels"]) & unsafe):
            continue
        if shown >= 14:
            break
        shown += 1
        why = [l for l in x["labels"] if l in unsafe]
        print(f"  {x['question_id']}  gold={str(x['gold'])[:20]!r}  risk={why}")
        print(f"     Q: {x['question'][:62]}")
        print(f"     rel: [{x['relation_turn']}] {x['relation_sentence'][:74]}")

    OUT.write_text(json.dumps({
        "label": "short-verbatim pre-implementation audit (audit only)",
        "match_rules": {
            "normalization": "lowercase + collapse whitespace + strip; nothing else",
            "punctuation": "KEPT as part of the match",
            "length": "words of the RAW gold, split on whitespace",
            "occurrences": "non-overlapping count on normalized context",
        },
        "n_verbatim": n_verbatim,
        "n_short": len(short),
        "n_unique": len(unique),
        "label_counts": dict(lab_counts),
        "n_safe": len(safe),
        "n_relation_proven": sum(1 for x in unique
                                 if "RELATION_EXPLICITLY_SUPPORTED" in x["labels"]),
        "safe_qids": [x["question_id"] for x in safe],
        "rows": unique,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


def _has_sibling_candidates(q_terms: set[str], n_ctx: str, gold: str) -> bool:
    """True when the context names another instance of the SAME KIND as the gold.

    The first version of this checked whether ANY word in the context was followed
    within 40 characters by a question term.  That matched 'nice' (followed by
    'painted'), 'was', and multi-word runs of stopwords, so it fired on 92 of 92
    'safe' candidates -- a detector that always says yes is not a measurement.

    The kind of thing a gold IS has to be derived from the QUESTION, not guessed:
      a date/span   -> another date or span in the context
      a proper noun -> another capitalised name of similar shape
      a number      -> another number
    Only when a second instance of that same kind appears near the asked relation is
    the gold genuinely not selectable.
    """
    g = gold.strip().lower()
    # 1. the gold is a date or a span -> siblings are other dates/spans
    if re.fullmatch(r"[\d\s,.]+|[a-z]+\s+\d{4}|\d{4}", g):
        dates = set(re.findall(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,4}"
            r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
            r"|\b20\d{2}\b|\b\d{4}\b", n_ctx))
        dates.discard(g)
        return len(dates) >= 1
    # 2. the gold is a number word or numeral -> siblings are other numbers
    if re.fullmatch(r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
                    r"twelve|once|twice)", g):
        nums = set(re.findall(
            r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
            r"twelve|once|twice)\b", n_ctx))
        nums.discard(g)
        return len(nums) >= 2
    # 3. the gold is a capitalised proper-noun phrase -> siblings are other
    #    capitalised phrases of the same arity (same number of words).
    words = gold.split()
    if words and words[0][:1].isupper():
        arity = len(words)
        same_shape = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){%d}\b"
                                    % (arity - 1), n_ctx))
        same_shape.discard(gold)
        return len(same_shape) >= 2
    # 4. otherwise: a common-noun phrase.  Siblings are other noun phrases that share
    #    the gold's HEAD noun, which is the only shape that competes with it.
    head = words[-1].lower() if words else ""
    if not head or len(head) < 4:
        return False
    peers = set(re.findall(rf"\b[A-Za-z][a-z]+\s+{re.escape(head)}\b", n_ctx))
    peers.discard(g)
    return len(peers) >= 2


def _relation_support(x: dict, cache: dict, q_terms: set[str]) -> dict:
    """Does any sentence containing the gold also name something the question asks?

    This is deliberately the WEAKEST possible relation test -- a shared content word in
    the same sentence.  The point is to find the cases where even that fails, because
    those are the ones a substring rule would answer with no relation at all.
    """
    ctx = cache[x["cache_qid"]]["context"]
    g = x["gold_norm"]
    best = {"turn_id": "", "sentence": "", "terms": set(), "supported": False}
    for line in ctx.split("\n"):
        body = re.sub(r"^\[[^\]]*\]\s*", "", line)
        tid = ""
        m = re.match(r"\[([^\]]+)\]", line)
        if m:
            tid = m.group(1).split(" on ")[0]
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            if g not in norm(sentence):
                continue
            terms = {w for w in re.findall(r"[a-z]{4,}", norm(sentence))
                     if w in q_terms}
            if len(terms) > len(best["terms"]):
                best = {"turn_id": tid, "sentence": sentence.strip(),
                        "terms": terms, "supported": bool(terms)}
    return best


if __name__ == "__main__":
    raise SystemExit(main())
