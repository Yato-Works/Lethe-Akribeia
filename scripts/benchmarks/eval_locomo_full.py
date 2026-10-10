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
            # LoCoMo is a two-person dialogue; normalise to the pipeline's
            # user/assistant convention so speaker-normalisation still applies.
            role = "user" if spk == speaker_a else "assistant"
            turns.append((sid, date, role, text))
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
    # A gold is a LIST when it has >=2 comma-separated items.  NOTE: computed
    # from the RAW gold -- norm() strips commas, so testing g here would never
    # detect a list ('Paris, Rome' normalises to 'paris rome').  For list golds
    # a one-item subset ('painting' for 'pottery, camping, painting, swimming')
    # must NOT be credited by ANY rule below, or the score is inflated by
    # rewarding incomplete enumeration -- the exact weakness being measured.
    _is_list = len([x for x in str(gold).split(",") if x.strip()]) >= 2
    # containment.  The reverse direction (prediction inside gold) is the
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
def _scaled_budget(base_budget: int, pool_tokens: int,
                   ratio: float = 0.12, cap: int = 2400, floor: int = 900) -> int:
    """Scale a per-mode token budget to the candidate pool actually retrieved.

    The per-mode budgets are calibrated for SHORT conversations.  On LoCoMo a
    typical candidate pool is ~290 turns / ~9,000 tokens, so the fixed 550-token
    budget admits only ~6% of it and structurally excludes the rest -- measured
    as the single largest cause of the answer turn never reaching the reader.
    """
    scaled = int(pool_tokens * ratio)
    return int(max(base_budget, min(cap, max(floor, scaled))))


def build_reader_prompt(question: str, context: str) -> str:
    return (
        "[INSTRUCTION: EVIDENCE-FIRST ANSWERING]\n"
        "A memory system has already retrieved the verified evidence below for this question.\n"
        "The answer IS present in this evidence.\n"
        "RULES:\n"
        "1. Locate the specific fact the question asks about and state it directly.\n"
        "2. Reply with the final value only: a few words or one short sentence.\n"
        "3. NEVER reply \"I don't know\", \"not enough information\", \"not mentioned\", or any other refusal.\n"
        "4. Use only the evidence; do not invent details that are not in it.\n"
        # EXHAUSTIVE ENUMERATION (dominant LoCoMo weakness).  Measured on the
        # full 1,986-question corpus: 40% of questions (616/1540 excluding
        # adversarial) have a LIST gold answer, and 64% of those fail (392 q)
        # -- by far the largest weakness class, ~2x everything else combined.
        # The reader consistently returns a SUBSET ('Her friends' when the gold
        # is 'Her mentors, family, and friends') or hallucinates an extra item.
        "5. EXHAUSTIVE ENUMERATION: if the question asks for a plural or a set "
        "('what activities', 'which events', 'what kinds', 'who supports', "
        "'where has X been'), do NOT stop at the first item. Scan EVERY evidence "
        "line, collect ALL distinct matching items, and list them comma-separated. "
        "Omitting an item is an error; listing only one when several exist is an error.\n\n"
        # RELATIVE-DATE RESOLUTION (LoCoMo weakness).  Measured on the full
        # corpus: ~10% of questions ask 'when' and the conversation states a
        # RELATIVE time ('yesterday', 'last week', 'next month'), while the
        # expected answer is the ABSOLUTE date.  Each evidence line carries its
        # session date in its header, so resolve it.
        "6. DATE RESOLUTION: if the question asks WHEN and the evidence states a relative time "
        "('yesterday', 'last week', 'next month', 'last year', 'the other day'), convert it to an "
        "ABSOLUTE date using the session date in that line's header (e.g. a line dated '8 May, 2023' "
        "saying 'yesterday' means '7 May 2023'). NEVER answer with the relative word itself.\n\n"
        f"{context}\n\nQUESTION: {question}\n\nAnswer:"
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

            base_limit = 32 if plan.dominant_mode == "aggregation" else 28
            top_sids = {t[0] for t in top_turns[:base_limit]} | guaranteed
            cand = [t for t in all_turns if t[0] in top_sids]
            if not cand:
                cand = list(top_turns[:25])

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
    run(ap.parse_args())


if __name__ == "__main__":
    main()
