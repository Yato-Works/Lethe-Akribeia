"""AM current: the R7 research pipeline behind an arena player interface.

V4_PROGRAM.md 4.17 named the next measurement: "R7 retrieval + compile behind
the arena player".  The P2 arena compares what ACTUALLY SHIPS (``AM v0.2.0``,
the runtime facade), which the lexical channel lifted from 11.5% to 53.0% --
but the R7 pipeline reached 54.94% official F1 on LoCoMo and was never priced
in the same arena.  This player closes that gap honestly:

- ingest: the shared history is kept as raw pipeline turns (0 write-LLM
  calls -- the R7 chain is deterministic up to the reader);
- answer: the R7 chain (execution planner -> convergent coverage -> session
  candidate pool -> fact-index recall -> graph bundles -> budgeted compiler)
  compiles the context, and the FROZEN shared answerer -- the exact LLM every
  other player uses -- answers from it.

Budget discipline: the compiler's own token unit is ``word_tokens_floored``
(words x 4/3), the arena contract is the shared estimator ``len(text) // 4``;
measured 18.3s probe: a 1400-unit budget compiles to ~2367 arena tokens.
``compile_r7_context`` therefore re-compiles with a measured correction until
the compiled context honours the arena's 2,000-token contract (recorded in
metadata; never silently violated).

Measurement machinery only: this module lives in scripts/, never in the
shipped package, and the arena runner schema is unchanged.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = REPO / "scripts" / "benchmarks"
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from artificial_memory.recall.conflict_surface import (  # noqa: E402
    detect_conflicts as _canonical_detect,
    note_line as _note_line,
    query_gated_keys as _query_gated_keys,
)
from artificial_memory.research.benchmarks.player import (  # noqa: E402
    Answer,
    ControlledPlayer,
)
from artificial_memory.research.benchmarks.players import (  # noqa: E402
    AMv020Player,
    _SharedAnswerGeneration,
)

ARENA_TOKEN_ESTIMATE = 2000  # the shared Controlled-Arena context contract


def _load_v4_retriever():
    """The frozen V4 candidate retriever, loaded once by file path."""
    cached = getattr(_load_v4_retriever, "_retriever", None)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        "eval_v4_pipeline_dev", str(_SCRIPTS / "eval_v4_pipeline_dev.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    retriever = mod.V4CandidateRetriever()
    _load_v4_retriever._retriever = retriever
    return retriever


def _scaled_budget(base_budget: int, pool_tokens: int,
                   ratio: float = 0.12, cap: int = 2400, floor: int = 1400) -> int:
    """Copy of eval_locomo_full._scaled_budget (kept byte-identical)."""
    scaled = int(pool_tokens * ratio)
    return int(max(base_budget, min(cap, max(floor, scaled))))


def est_tokens(text: str) -> int:
    """The arena's shared estimator (chars/4)."""
    return len(text) // 4

# The runtime's lexical ranker, applied to raw pipeline turns (V4_PROGRAM.md 4.13).

def _word_stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "e"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _term_set(text: str) -> set[str]:
    return {_word_stem(w) for w in re.findall(r"[a-z0-9]{3,}", text.lower())}


def lexical_top_k(query: str, turns: list, top_k: int) -> list:
    """IDF-weighted overlap: a rare shared term counts, 'the user' does not."""
    docs = [_term_set(f"{t[2]}: {t[3]}") for t in turns]
    df = Counter()
    for doc in docs:
        df.update(doc)
    n = max(1, len(docs))
    query_terms = _term_set(query)
    max_idf = sum(math.log((n + 1) / (df[t] + 1)) for t in query_terms) or 1.0
    scored = []
    for turn, doc in zip(turns, docs):
        raw = sum(math.log((n + 1) / (df[t] + 1)) for t in query_terms & doc)
        scored.append((raw, turn))
    scored.sort(key=lambda p: (-p[0], p[1][0], p[1][3]))
    return [t for raw, t in scored[:top_k] if raw > 0]


def hybrid_context(r7_ctx: str, query: str, all_turns: list,
                   arena_budget: int, top_k: int) -> tuple[str, dict]:
    """R7 compiled context first, lexical top-k turns fill the remainder.

    The two channels are measured complementary; the union keeps every
    unique within-budget line of both, deduped by turn text.
    """
    used = est_tokens(r7_ctx)
    parts = [r7_ctx] if r7_ctx else []
    lex_lines: list[str] = []
    dropped = 0
    seen = set()
    for turn in lexical_top_k(query, all_turns, top_k):
        line = f"[{turn[0]}] {turn[2]}: {turn[3]}"
        key = turn[3].strip().lower()
        if key in seen or key in r7_ctx.lower():
            continue
        seen.add(key)
        cost = est_tokens(line) + 1
        if used + cost > arena_budget:
            dropped += 1
            continue
        lex_lines.append(line)
        used += cost
    ctx = "\n".join(parts + lex_lines)
    return ctx, {"lexical_lines": len(lex_lines), "lexical_dropped": dropped,
                 "hybrid_tokens": est_tokens(ctx)}


def fill_context_from_lines(r7_ctx: str, fill_lines: list,
                            arena_budget: int) -> tuple[str, dict]:
    """Append candidate lines after the compiled context, honouring the budget.

    Generic: the fill source is chosen by the caller (raw top-k, or the
    lexical runtime's own recall output as semantic-content lines).
    """
    used = est_tokens(r7_ctx)
    parts = [r7_ctx] if r7_ctx else []
    added = dropped = 0
    seen: set[str] = set()
    r7_low = r7_ctx.lower()
    for line in fill_lines:
        text = str(line).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen or key in r7_low:
            continue
        seen.add(key)
        cost = est_tokens(text) + 1
        if used + cost > arena_budget:
            dropped += 1
            continue
        parts.append(text)
        used += cost
        added += 1
    ctx = "\n".join(parts)
    return ctx, {"lexical_lines": added, "lexical_dropped": dropped,
                 "hybrid_tokens": est_tokens(ctx)}


def build_pipeline():
    """One instance of every R7 stage, constructed once per process."""
    from artificial_memory_gen.compiler.budgeted_compiler import (
        BudgetedCognitiveCompiler,
    )
    from artificial_memory_gen.compiler.coverage.coverage_engine import (
        MultiSessionCoverageEngine,
    )
    from artificial_memory_gen.compiler.execution_planner import (
        CognitiveExecutionPlanner,
    )
    from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder
    from artificial_memory_gen.recall.fact_index import FactIndex

    return (
        _load_v4_retriever(),
        MultiSessionCoverageEngine(),
        GraphBuilder(),
        BudgetedCognitiveCompiler(),
        CognitiveExecutionPlanner(),
        FactIndex(),
    )


def compile_r7_context(pipeline, all_turns: list, q: str,
                       arena_budget: int = ARENA_TOKEN_ESTIMATE,
                       fact_index=None) -> tuple[str, dict]:
    """Compile the R7 context for one question, honouring the arena budget.

    The compiler's unit (word_tokens_floored) is ~1.7x smaller than the
    arena's estimator (chars/4), measured on this dataset.  A first compile,
    then up to two corrected re-compiles, drive the arena-visible tokens
    under the shared contract.  Deterministic: the schedule depends only on
    the measured ratios of the compilations themselves.
    """
    retriever, coverage, graph_builder, compiler, planner = pipeline[:5]

    def _one(budget: int):
        plan = planner.plan(q, [], reference_date=None)
        token_budget = plan.recommended_budget
        top_turns, turn_scores, guaranteed = coverage.retrieve_convergent_evidence(
            query=q, all_turns=all_turns, retriever=retriever,
            reference_date=None, base_top_k=45)
        base_limit = 32 if plan.dominant_mode == "aggregation" else 28
        top_sids = {t[0] for t in top_turns[:base_limit]} | guaranteed
        cand = [t for t in all_turns if t[0] in top_sids]
        if not cand:
            cand = list(top_turns[:25])
        # Fact-index recall, exactly as the R7 harness does it.
        fact_turns = fact_index.query(q, all_turns, max_turns=16)
        already_cand = {(t[0], t[3]) for t in cand}
        for ft in fact_turns:
            key = (ft[0], ft[3])
            if key not in already_cand:
                cand.append(ft)
                already_cand.add(key)
            turn_scores[key] = max(turn_scores.get(key, 0.0), 25.0)
            if ft not in top_turns:
                top_turns.append(ft)
        # Budget calibration, exactly as the R7 harness does it.
        _pool_tokens = sum(len(t[3].split()) * 4 // 3 for t in cand)
        if _pool_tokens:
            token_budget = _scaled_budget(token_budget, _pool_tokens)
        token_budget = min(token_budget, budget)
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
            question=q, bundles=bundles,
            token_budget=token_budget, anchor_scores=anchor_scores,
            is_multi_session=len(top_sids) > 1, adaptive_escalation=True)
        meta = {
            "plan_mode": getattr(plan, "dominant_mode", ""),
            "token_budget": token_budget,
            "pool_turns": len(cand),
            "compiled_tokens": getattr(compiled, "total_tokens", 0),
            "web_ctx_tokens": len(compiled.context_text) // 4,
        }
        return compiled.context_text, meta

    ctx, meta = _one(arena_budget)
    meta["recompiles"] = 0
    # Measured-correction loop: scale the compiler budget by the observed
    # ratio between arena tokens and the compiler budget until it fits.
    for _ in range(2):
        if meta["web_ctx_tokens"] <= arena_budget:
            break
        b_prev = max(1, meta["token_budget"])
        meta2_budget = max(300, int(b_prev * (arena_budget - 25) / max(1, meta["web_ctx_tokens"])))
        ctx, meta2 = _one(meta2_budget)
        meta2["recompiles"] = meta["recompiles"] + 1
        meta = meta2
    # Last resort: the contract is absolute -- drop whole lines from the tail.
    if meta["web_ctx_tokens"] > arena_budget:
        lines: list[str] = []
        used = 0
        for line in ctx.splitlines():
            cost = len(line) // 4 + 1
            if used + cost > arena_budget:
                break
            lines.append(line)
            used += cost
        ctx = "\n".join(lines)
        meta["truncated"] = True
        meta["web_ctx_tokens"] = used
    return ctx, meta


# --- conflict surfacing (V4 4.24, M14 runtime port) --------------------------
# Thin adapters over the canonical runtime module
# (src/artificial_memory/recall/conflict_surface.py).  Single source of
# truth: the player and the shipped runtime class share one implementation,
# so benchmark and product stay in parity by construction.
#
# Boundary: the detected annotation uses ONLY the conversation turns; the
# scorer's ground_truth and CONFLICT_MARKERS vocabularies are used NOWHERE
# here (enforced by test_conflict_player_module_never_imports_scorer_vocabulary).

def detect_conflicts(all_turns) -> dict:
    """{(entity, property): [(value, day), ...]} for duplicated assertions."""
    documents = [(turn[3], str(turn[1])) for turn in all_turns]
    return _canonical_detect(documents)


def conflict_annotations(question_text: str, conflicts) -> list[str]:
    """Lines surfacing query-relevant conflicts, built only from the turns."""
    lines = []
    for proj, prop in _query_gated_keys(question_text, conflicts):
        # the player's provenance unit is the arena day index: pre-format it
        # so the note keeps the exact M13-measured surface ("day 45").
        entries = [(v, f"day {d}") for v, d in conflicts[(proj, prop)]]
        lines.append(_note_line(f"project {proj}", prop, entries))
    return lines


class AMCurrentPlayer(_SharedAnswerGeneration, ControlledPlayer):
    """S6: the R7 retrieval + compile chain with the frozen shared reader."""

    PLAYER_NAME = "AM current"

    def __init__(self, config: dict[str, Any] | None = None, answerer=None):
        super().__init__(self.PLAYER_NAME, config)
        self._answerer = answerer
        self._pipeline = None
        self._all_turns: list = []
        self._fact_index = None
        # Hybrid mode (probe_hybrid_ctx): R7 compile at a reduced budget,
        # then lexical top-k turns fill the remainder to the shared contract.
        self.hybrid = bool(self.config.get("hybrid", False))
        self.hybrid_budget_a = int(self.config.get("hybrid_budget_a", 1200))
        self.hybrid_top_k = int(self.config.get("hybrid_top_k", 30))

    def ingest(self, scenarios) -> None:
        start = time.perf_counter()
        turns: list = []
        for scenario in scenarios:
            date = str(scenario.day)
            for turn in scenario.turns:
                turns.append((scenario.scenario_id, date, turn.speaker, turn.content))
        self._all_turns = turns
        self._pipeline = build_pipeline()
        self._fact_index = self._pipeline[5]
        self._fact_index.build(turns)
        # Write cost is honest: the whole chain is deterministic, zero LLM.
        self._update_costs(self.costs.add_write_cost(
            calls=0, latency_ms=(time.perf_counter() - start) * 1000))

    def _answer_with_budget(self, question, budget: int) -> Answer:
        q = question.question
        t_retr = time.perf_counter()
        compile_budget = self.hybrid_budget_a if self.hybrid else self.context_budget
        ctx, meta = compile_r7_context(
            self._pipeline, self._all_turns, q, arena_budget=compile_budget,
            fact_index=self._fact_index)
        if getattr(self, "conflict_surface", False):
            if getattr(self, "_conflicts", None) is None:
                self._conflicts = detect_conflicts(self._all_turns)
            notes = conflict_annotations(q, self._conflicts)
            if notes:
                ctx, cmeta = fill_context_from_lines(
                    ctx, notes, arena_budget=self.context_budget)
                meta["conflict_lines"] = cmeta["lexical_lines"]
        if self.hybrid:
            if (self.config.get("hybrid_fill", "raw") == "runtime"
                    and getattr(self, "_lex_runtime", None) is not None):
                import asyncio
                remaining = self.context_budget - est_tokens(ctx)
                result = asyncio.run(self._lex_runtime.recall(
                    q, topic="Arena", level=2, max_tokens=remaining))
                fill = [m.semantic_content.content for m in result.memories]
                ctx, hmeta = fill_context_from_lines(
                    ctx, fill, arena_budget=self.context_budget)
                hmeta["fill_source"] = "runtime"
                hmeta["fill_candidates"] = len(fill)
            else:
                ctx, hmeta = hybrid_context(
                    ctx, q, self._all_turns,
                    arena_budget=self.context_budget, top_k=self.hybrid_top_k)
                hmeta["fill_source"] = "raw"
            meta.update(hmeta)
        t_ans = time.perf_counter()
        ctx_tokens = len(ctx) // 4
        retrieval_ms = (t_ans - t_retr) * 1000 - 0.0
        llm = self._generate_answer(q, ctx)
        answer_ms = (time.perf_counter() - t_ans) * 1000
        self._update_costs(self.costs.add_read_cost(
            retrieval_calls=1,
            retrieval_latency_ms=retrieval_ms,
            context_calls=1,
            context_latency_ms=0.0,
            answer_calls=1,
            answer_prompt_tokens=llm.prompt_tokens,
            answer_completion_tokens=llm.completion_tokens,
            answer_latency_ms=answer_ms,
            context_tokens_used=ctx_tokens,
        ))
        return Answer(
            text=llm.text,
            question_id=question.question_id,
            player_name=self.name,
            run_index=0,
            context_tokens_used=ctx_tokens,
            metadata=meta,
        )


class AMHybridPlayer(AMCurrentPlayer):
    """S7: R7 compile (budget A) + lexical fill, within one 2000-token context.

    The union of the two measured-complementary channels
    (V4_PROGRAM.md 4.13 lexical vs the AM current probe).

    ``hybrid_fill`` controls the fill source:
      - "raw": a re-implemented IDF top-k over raw turns (measured first;
        it does NOT reproduce the lexical runtime's retrieval wins);
      - "runtime": the lexical runtime's OWN recall stack -- the facade the
        AM v0.2.0 player answers from (4.13), driven synchronously; its
        semantic-content lines fill the remainder.  Measured diagnosis (4.21):
        the real lexical wins on indirect/distractor come from this stack.
    """

    PLAYER_NAME = "AM hybrid"

    def __init__(self, config: dict[str, Any] | None = None, answerer=None):
        config = dict(config or {})
        config.setdefault("hybrid", True)
        config.setdefault("hybrid_budget_a", 1200)
        config.setdefault("hybrid_top_k", 30)
        config.setdefault("hybrid_fill", "raw")
        super().__init__(config, answerer)
        self._lex_runtime = None

    def ingest(self, scenarios) -> None:
        super().ingest(scenarios)
        if self.config.get("hybrid_fill", "raw") != "runtime":
            return
        import asyncio
        from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

        runtime = ArtificialMemoryRuntime(RuntimeConfig(
            database_path=":memory:",
            memory_files_path=None,
            vector_index_path=None,
            retrieval_strategy="classic",
            lexical_recall=True,
        ))
        for scenario in scenarios:
            for turn in scenario.turns:
                if turn.speaker == "user":
                    asyncio.run(runtime.remember(turn.content, topic="Arena"))
        self._lex_runtime = runtime


class AMRuntimeConflictPlayer(AMv020Player):
    """S9 (M14): the PRODUCT read path with conflict surfacing priced E2E.

    Same facade ingest as AM v0.2.0 (user-turn ``remember``, lexical
    channel from the shared config knob), but the answer context is
    assembled by the runtime's own ``EnhancedContextBuilder`` -- the exact
    path ``LLMManager.chat`` takes in the product -- instead of the
    research ``recall(level=2)`` blob.  ``conflict_surfacing`` passes
    straight through to ``RuntimeConfig``.

    The paired comparable is the identical path with the flag off
    ("AM runtime"), so this measures the FLAG on the product path, never
    the R7 compile the benchmark player runs.
    """

    PLAYER_NAME = "AM runtime conflict"

    def __init__(self, config: dict[str, Any] | None = None, answerer=None):
        config = dict(config or {})
        config.setdefault("lexical_recall", True)
        config.setdefault("conflict_surfacing", True)
        super().__init__(config, answerer)
        # AMv020Player hard-codes its own name; the arena needs the flag
        # arms distinguishable, and ControlledPlayer takes the name
        # through the constructor string.  Name follows the flag: the
        # flag-off arm is the paired baseline ("AM runtime"), the
        # flag-on arm carries the surfacing name.
        if self.config.get("conflict_surfacing", False):
            self.name = self.PLAYER_NAME
        else:
            self.name = "AM runtime"
        # M15.2: the provenance-distinguishability arm gets its own name
        # so runs remain citable per identity.
        if self.config.get("provenance_days", False) \
                and self.config.get("conflict_surfacing", False):
            self.name = "AM runtime conflict dt"
        elif self.config.get("conflict_fill_ablation", False) \
                and self.config.get("conflict_surfacing", False):
            self.name = "AM runtime conflict bare"

    def _ingest_scenario(self, scenario) -> None:
        """Remember BOTH speakers for the conflict player.

        The AM v0.2.0 ingest protocol stores user turns only; that froze
        the product store behind a one-sided assertion corpus.  The first
        product-path 0-LLM pricing run measured the consequence: the
        second report of every planted conflict is an *assistant* turn
        ("a teammate reported..."), so no contradictory key ever existed
        in the store and the annotation fired 0/320 contexts even flag-on.
        The product itself stores both sides of a conversation (see
        CompilerPipelineWrapper.compile_conversation), so this player
        ingests the same way the product does.

        M15.2 ``provenance_days``: with the knob on, each turn's real
        scenario timestamp (the dataset's own day index, ``turn.timestamp``)
        is passed as ``occurred_at``, so ``valid_from`` -- and therefore
        every provenance label the builder prints -- reflects distinct,
        data-backed time points instead of one ingest-date collapse.
        Nothing else changes: same turns, same builder, same budget.
        """
        if not self._runtime:
            return
        for turn in scenario.turns:
            stamp = turn.timestamp if self.config.get(
                "provenance_days", False) else None
            stamp = turn.timestamp if self.config.get(
                "provenance_days", False) else None
            asyncio.run(self._runtime.remember(
                turn.content, topic="Arena", occurred_at=stamp))

    def _all_active_current(self, topic_id: int) -> list:
        """Page through the store: the full active current-memory corpus.

        The product chat path fetches ``get_memories(limit=100)`` -- the
        most-recently-updated 100 only.  The first product-path pricing
        runs measured what that does to conflict detection: with ~880
        stored memories the planted pairs sit outside the 100-window and
        the annotation fired 0/320 contexts.  Detection needs the whole
        active corpus, so this player pages it in.  The behavior change
        is scoped to this flag player; the paired baseline ("AM runtime")
        pages identically, so the delta isolates the flag.
        """
        assert self._runtime is not None
        page = 500
        out = []
        offset = 0
        while True:
            batch = self._runtime.store.get_memories(
                topic_id=topic_id, is_current=True, status="active",
                limit=page, offset=offset)
            if not batch:
                break
            out.extend(batch)
            if len(batch) < page:
                break
            offset += page
        return out

    def _answer_with_budget(self, question, budget: int) -> Answer:
        if not self._runtime:
            return self._empty_answer(question)
        runtime = self._runtime
        t_retr = time.perf_counter()
        topic = runtime.conversation_manager.get_or_create_topic("Arena")
        current = self._all_active_current(topic.id)
        ctx = runtime.context_builder.build_context(
            question.question, topic.id, max_tokens=budget,
            current_memories=current)
        meta = {
            "path": "product_builder",
            "conflict_surfacing": runtime.config.conflict_surfacing,
            "ctx_chars": len(ctx),
            "conflict_notes": ctx.count("NOTE: conflicting reports"),
        }
        t_ans = time.perf_counter()
        retrieval_ms = (t_ans - t_retr) * 1000
        ctx_tokens = len(ctx) // 4
        llm = self._generate_answer(question.question, ctx)
        answer_ms = (time.perf_counter() - t_ans) * 1000
        self._update_costs(self.costs.add_read_cost(
            retrieval_calls=1,
            retrieval_latency_ms=retrieval_ms,
            context_calls=1,
            context_latency_ms=0.0,
            answer_calls=1,
            answer_prompt_tokens=llm.prompt_tokens,
            answer_completion_tokens=llm.completion_tokens,
            answer_latency_ms=answer_ms,
            context_tokens_used=ctx_tokens,
        ))
        return Answer(
            text=llm.text,
            question_id=question.question_id,
            player_name=self.name,
            run_index=0,
            context_tokens_used=ctx_tokens,
            metadata=meta,
        )


class AMConflictPlayer(AMHybridPlayer):
    """S8: hybrid raw fill + deterministic conflict surfacing (V4 4.24).

    The contradiction category reaches its planted values in every compiled
    context yet is failed 0/20 by every player, because no context ever
    surfaces that the two reports conflict.  This player adds, ONLY when a
    query-relevant (entity, property) key was asserted with two distinct
    values, a single annotation line built from the turns:

        NOTE: conflicting reports for project P (prop): V1 (day d) vs V2
        (day d); which is correct is unverified.

    Priced at 0 LLM before being built: the detector captures the planted
    pairs 20/20 and the annotation fires on 0/180 non-contradiction
    questions, so the added text cannot ride into unrelated answers.
    """

    PLAYER_NAME = "AM conflict"

    def __init__(self, config: dict[str, Any] | None = None, answerer=None):
        config = dict(config or {})
        config.setdefault("conflict_surface", True)
        super().__init__(config, answerer)
        self.conflict_surface = bool(self.config.get("conflict_surface", False))
        self._conflicts: dict | None = None
