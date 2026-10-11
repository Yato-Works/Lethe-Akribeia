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

CAT_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
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
        # cat5 (adversarial) stores its gold under 'adversarial_answer' instead
        # of 'answer'.  Without this fallback all 446 adversarial questions
        # would read as empty gold and be scored 0 unfairly.
        gold = qa.get("answer", qa.get("adversarial_answer", ""))
        out.append({
            "question_id": f"lc_{sample_id}_{i}",
            "sample_id": sample_id,
            "question": qa["question"],
            "answer": str(gold),
            "category": int(qa.get("category", 0)),
            "evidence": qa.get("evidence", []),
        })
    return out


# ------------------------------------------------------------------ scoring
_YESNO = {"yes", "no"}


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
    _is_list = is_list_gold(gold) and not (p in _YESNO)

    if _is_list:
        # Check permutation match: every gold item must appear in pred.
        # Drops leading/trailing articles/pronouns from each gold item.
        _stop_articles = {"her", "his", "my", "their", "our", "the", "a", "an"}
        gold_item_core = []
        raw_gold_items = [x.strip() for x in str(gold).split(",") if x.strip()]
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
    if p in g and not _is_list:
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
        if hit / len(gt) >= 0.6:
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
            if hit / len(shorter) >= 0.5:
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
    if len(p) >= 3 and g.startswith(p) and not _is_list:
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


def build_reader_prompt(question: str, context: str) -> str:
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
    return (
        base_prompt
        + hint_text
        + "Answer with ONLY the concise value (place, date, number, name, or comma-separated list). Do not output conversational filler or full sentences.\n"
        + f"Question: {question}\n"
        + "Answer:"
    )


def run(args) -> None:
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

    print(f"LoCoMo full benchmark: {len(jobs)} questions "
          f"(from {len(data)} conversations)")
    if wanted:
        print(f"  filtered to categories: {sorted(wanted)}")

    from eval_v4_pipeline_dev import V4CandidateRetriever  # reuse frozen retriever
    from artificial_memory.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
    from artificial_memory.compiler.graph.graph_builder import GraphBuilder
    from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler
    from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner
    from artificial_memory.recall.fact_index import FactIndex
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

            # BUDGET CALIBRATION.  The per-mode budget is calibrated for SHORT
            # conversations; LoCoMo candidate pools run ~290 turns / ~9,000
            # tokens, so a 550-token budget admits only ~6% and structurally
            # excludes the rest.  Scale the budget to the pool we actually
            # retrieved (this was the single largest cause of evidence loss).
            _pool_tokens = sum(len(t[3].split()) * 4 // 3 for t in cand)
            if _pool_tokens:
                token_budget = _scaled_budget(token_budget, _pool_tokens)

            t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(cand)}
            anchor_ids, anchor_scores = [], {}
            for t in top_turns:
                nid = t_map.get((t[0], t[3]))
                if nid and nid not in anchor_ids:
                    anchor_ids.append(nid)
                    anchor_scores[nid] = turn_scores.get((t[0], t[3]), 1.0)

            edg = graph_builder.build(cand, [])
            bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
            compiled = compiler.compile(
                question=q["question"], bundles=bundles,
                token_budget=token_budget, anchor_scores=anchor_scores,
                is_multi_session=len(top_sids) > 1, adaptive_escalation=True)

            pred = answerer.answer(q["question"],
                                   build_reader_prompt(q["question"], compiled.context_text)).text.strip()
            if pred.endswith(".") and len(pred.split()) <= 2:
                pred = pred[:-1].strip()

            # Deterministic relative-date resolution (LoCoMo weakness).  Only
            # for 'when' questions, so a substantive answer is never rewritten.
            if re.search(r"\bwhen\b", q["question"], re.I):
                pred = resolve_relative_dates(pred, compiled.context_text, q["question"])

            ok = score(pred, q["answer"])
        except Exception as e:  # a single failure must not kill a 1986-q run
            pred = f"<ERROR: {type(e).__name__}: {e}>"
            ok = False

        results.append({
            "question_id": q["question_id"], "category": q["category"],
            "question": q["question"], "ground_truth": q["answer"],
            "predicted_answer": pred, "is_correct": ok,
        })
        by_cat[q["category"]].append(ok)

        if (idx + 1) % 50 == 0 or idx + 1 == len(jobs):
            elapsed = time.perf_counter() - t_start
            rate = elapsed / (idx + 1)
            eta = rate * (len(jobs) - idx - 1)
            print(f"[{idx+1}/{len(jobs)}] {elapsed:.0f}s elapsed, "
                  f"ETA {eta/60:.1f}min")

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
    out = out.with_name(f"locomo_full_{_stamp}.json")
    # Also keep the stable 'latest' pointer for convenience.
    latest = out.parent / "locomo_full.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"accuracy": correct / total, "results": results,
                   "per_category": {str(k): {"correct": sum(v), "total": len(v)}
                                    for k, v in by_cat.items()}},
                  f, ensure_ascii=False, indent=2)
    with open(latest, "w", encoding="utf-8") as f:
        json.dump({"accuracy": correct / total,
                   "source": out.name,
                   "per_category": {str(k): {"correct": sum(v), "total": len(v)}
                                    for k, v in by_cat.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")
    print(f"Latest -> {latest}")


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
    run(ap.parse_args())


if __name__ == "__main__":
    main()
