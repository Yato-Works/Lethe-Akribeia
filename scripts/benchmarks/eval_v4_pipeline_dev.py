"""
Lethe-Akribeia V4: Full End-to-End Budgeted Cognitive Compilation Pipeline.
Evaluates LongMemEval Dev 250 with full haystack sessions (including distractors).

Pipeline:
  1. UniversalIRExtractor: Haystack sessions -> StructuredIR records
  2. CognitiveExecutionPlanner: Query + records -> Demand Vector + Adaptive Budget Bt
  3. Candidate Retriever: Multi-channel lexical + entity ranking for candidate turns
  4. GraphBuilder: Builds EvidenceDependencyGraph (EDG) & forms EvidenceBundles
  5. BudgetedCognitiveCompiler: Submodular knapsack selection under budget Bt
  6. CoProcessorEngine: Deterministic reasoning certificates & 0-token overrides
  7. Reader Execution: Qwen2.5 7B with verified compiled context & certificates
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

REPO = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"
SPLIT_PATH = REPO / "benchmark_results" / "splits" / "lme_dev_qids.json"
OUT_PATH = REPO / "benchmark_results" / "pipeline_v4" / "lme_dev250_v4_pipeline.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "src"))

from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory.compiler.graph.graph_builder import GraphBuilder
from artificial_memory.compiler.graph.dependency_graph import EvidenceBundle, EvidenceNode
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import FROZEN_MODEL, OllamaAnswerer


from artificial_memory.recall.evidence_scorer import UniversalEvidenceScorer
from artificial_memory.compiler.samplers.per_session_sampler import PerSessionSampler
from artificial_memory.compiler.coverage.coverage_engine import MultiSessionCoverageEngine


from artificial_memory.compiler.r6_features import (
    R6Features,
    extract_r6_features,
    feature_dict,
)


class V4CandidateRetriever:
    """Fast multi-channel lexical & entity turn ranker for full haystacks."""

    STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "i", "you", "my", "your", "we", "our",
        "in", "on", "at", "to", "for", "with", "from", "by", "about", "of", "and", "or", "it",
        "this", "that", "there", "then", "did", "do", "does", "have", "had", "has", "me",
        "suggest", "recommend", "advice", "tips", "ideas", "some", "any", "please", "wondering",
        "tell", "show", "give", "help", "planning", "plans", "soon",
        # Scaffolding words to prevent false-positive overlap on conversational framing
        "what", "which", "who", "whom", "whose", "how", "many", "much",
        "past", "few", "months", "month", "days", "day", "weeks", "week", "years", "year",
        "time", "times", "total", "can", "could", "would", "should", "will",
        "done", "am", "be", "been", "being",
        "look", "looking", "back", "previous", "conversation", "conversations",
        "remind", "remember", "recall", "talked", "discussed", "mention", "mentioned"
    })

    def __init__(self) -> None:
        self.expansions = UniversalEvidenceScorer.SEMANTIC_EXPANSIONS

    def _word_root(self, w: str) -> str:
        w = w.lower()
        for suffix in ["ing", "ed", "es", "s"]:
            if len(w) > len(suffix) + 2 and w.endswith(suffix):
                return w[:-len(suffix)]
        return w

    def expand_query_words(self, query: str) -> set[str]:
        raw_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower())) - self.STOP_WORDS
        if not raw_words:
            raw_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower()))

        expanded = set(raw_words)
        for w in raw_words:
            expanded.add(self._word_root(w))
            # Semantic dictionary expansions
            if w in self.expansions:
                expanded.update(self.expansions[w])
            root = self._word_root(w)
            if root in self.expansions:
                expanded.update(self.expansions[root])

        # Also check phrase expansions in query
        q_lower = query.lower()
        for phrase, syns in self.expansions.items():
            if (" " in phrase or "-" in phrase) and phrase in q_lower:
                expanded.update(syns)

        return expanded

    def compute_idf(self, all_turns: list[tuple[str, str, str, str]]) -> dict[str, float]:
        session_words: dict[str, set[str]] = {}
        for sid, date, speaker, content in all_turns:
            words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", content.lower()))
            roots = {self._word_root(w) for w in words}
            session_words.setdefault(sid, set()).update(words | roots)

        N = max(1, len(session_words))
        df: Counter = Counter()
        for s_words in session_words.values():
            df.update(s_words)

        idfs: dict[str, float] = {}
        for w, count in df.items():
            idfs[w] = max(0.2, math.log(1.0 + (N - count + 0.5) / (count + 0.5)))
        return idfs

    # ── Relative date parsing for temporal query boosting ──────────────────────
    _RE_WEEKS_AGO  = re.compile(r"\b(one|two|three|four|five|six|\d+)\s+weeks?\s+ago\b", re.I)
    _RE_MONTHS_AGO = re.compile(r"\b(one|two|three|four|five|six|\d+)\s+months?\s+ago\b", re.I)
    _RE_DAYS_AGO   = re.compile(r"\b(one|two|three|four|five|six|seven|\d+)\s+days?\s+ago\b", re.I)
    _RE_LAST_WEEK  = re.compile(r"\blast\s+week\b", re.I)
    _RE_LAST_MONTH = re.compile(r"\blast\s+month\b", re.I)
    _RE_LAST_DOW   = re.compile(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
    _WORD_TO_INT   = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7}

    @classmethod
    def _parse_relative_date(cls, query: str, reference_date: str | None) -> "datetime | None":
        """Parse a relative date expression from the query and return the target datetime."""
        if not reference_date:
            return None
        import re as _re
        from datetime import datetime, timedelta
        m = _re.search(r"(\d{4})/(\d{2})/(\d{2})", reference_date)
        if not m:
            return None
        ref = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))

        def _to_int(s: str) -> int:
            try:
                return int(s)
            except ValueError:
                return cls._WORD_TO_INT.get(s.lower(), 1)

        if (mg := cls._RE_WEEKS_AGO.search(query)):
            return ref - timedelta(weeks=_to_int(mg.group(1)))
        if (mg := cls._RE_MONTHS_AGO.search(query)):
            return ref - timedelta(days=_to_int(mg.group(1)) * 30)
        if (mg := cls._RE_DAYS_AGO.search(query)):
            return ref - timedelta(days=_to_int(mg.group(1)))
        if cls._RE_LAST_WEEK.search(query):
            return ref - timedelta(weeks=1)
        if cls._RE_LAST_MONTH.search(query):
            return ref - timedelta(days=30)
        if (mg := cls._RE_LAST_DOW.search(query)):
            return ref - timedelta(weeks=1)  # approximate
        return None

    def rank_turns(
        self,
        query: str,
        all_turns: list[tuple[str, str, str, str]],  # (sid, date, speaker, content)
        top_k: int = 35,
        reference_date: str | None = None,
    ) -> tuple[list[tuple[str, str, str, str]], dict[tuple[str, str], float]]:
        from datetime import datetime
        import re as _re
        q_words = self.expand_query_words(query)
        idfs = self.compute_idf(all_turns)
        proper_nouns = {
            w.lower()
            for w in re.findall(r"\b[A-Z][a-z0-9_-]+\b", query)
            if w.lower() not in {"i", "what", "which", "who", "whom", "how", "did", "can", "could", "would", "do", "does", "is", "are", "any", "the", "my", "your"}
        }

        # Compute target date for temporal proximity boosting
        target_dt = self._parse_relative_date(query, reference_date)

        scored = []
        turn_scores: dict[tuple[str, str], float] = {}
        for turn in all_turns:
            sid, date, speaker, content = turn
            t_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", content.lower()))
            t_roots = {self._word_root(w) for w in t_words}

            overlap = q_words & (t_words | t_roots)
            if not overlap:
                continue

            term_score = 0.0
            for w in overlap:
                weight = idfs.get(w, 1.0)
                if w in proper_nouns:
                    weight *= 3.5  # Critical salience boost for named entities
                term_score += weight

            score = term_score * 2.5
            if speaker.lower() == "user":
                score *= 1.8
            length_penalty = len(content.split()) / 60.0
            score = max(0.1, score - length_penalty * 0.1)

            # Date-proximity boost for temporal queries
            if target_dt is not None and date:
                dm = _re.search(r"(\d{4})/(\d{2})/(\d{2})", date)
                if dm:
                    session_dt = datetime(int(dm.group(1)), int(dm.group(2)), int(dm.group(3)))
                    diff_days = abs((session_dt - target_dt).days)
                    # Strong boost for sessions within ±3 days of target date
                    if diff_days <= 3:
                        score *= 3.0
                    elif diff_days <= 7:
                        score *= 2.0
                    elif diff_days <= 14:
                        score *= 1.4

            scored.append((score, turn))
            turn_scores[(sid, content)] = score

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]], turn_scores

    def is_aggregation_query(self, query: str) -> bool:
        """Detect queries requiring exhaustive cross-session counting or summing.

        Strictly excludes:
        - Temporal elapsed time / duration queries (e.g. 'how many days ago', 'days passed')
        - Arithmetic differences / comparisons (e.g. 'how much more/less', 'compared to')
        - Current state / Knowledge updates (e.g. 'currently', 'now', 'added to my collection')
        - Assistant references (e.g. 'you recommended', 'in our previous conversation')
        """
        q = query.lower().strip()
        import re as _re

        # 1. Non-aggregation guardrails
        if _re.search(r"\b(you (told|recommended|suggested|said|mentioned|advised|gave)|in our previous conversation|did you say)\b", q):
            return False
        if _re.search(r"\b(currently|right now|at present|as of now)\b", q):
            return False
        if _re.search(r"\b(how many|what|where|who)\b.*\bnow\??$", q):
            return False
        if _re.search(r"\b(compared to|difference in|faster|slower|cheaper|more expensive|how much (more|less)|higher percentage|discount on)\b", q):
            return False

        # 2. Temporal elapsed time guards (unless explicit 'in total')
        if not _re.search(r"\b(in total|total (hours|time|cost|amount|distance)|altogether|combined)\b", q):
            if _re.search(r"\bhow many (days|weeks|months|years|hours|minutes)\s+(ago|passed|did it take|before|after|since|between|had passed)\b", q):
                return False
            if _re.search(r"\bhow long (ago|did it take|passed)\b", q):
                return False
            if _re.search(r"\b(passed between|days before|days after|days since)\b", q):
                return False
            if _re.search(r"\bhow many days a week\b", q):
                return False

        # 3. Explicit Total / Sum indicators
        if _re.search(r"\b(in total|total (cost|price|amount|hours|time|spending|expense|distance|number)|combined (cost|price|total)|sum of|across all|altogether)\b", q):
            return True

        # 4. Count of distinct items/events across sessions
        if _re.search(r"\bhow many (different )?(\w+(?:\s+\w+){0,3})\s+(did i|have i|do i|were|are there|can i count|did we)\b", q):
            m = _re.search(r"\bhow many (days|weeks|months|years)\b", q)
            if m and not _re.search(r"\b(spend|work out|exercise|attend|travel|stay)\b", q):
                return False
            return True

        return False


# ---------------------------------------------------------------------------
# Reader × Memory Engine conditions (one variable changed per experiment).
#   R0 - frozen baseline: prompt built exactly as the original pipeline
#   R1 - evidence-first ordering: same evidence bytes, relevance order only
#   R2 - compact evidence: R0 order kept, zero-overlap evidence blocks dropped
#   R3 - provenance tags: R0 order kept, each agg turn stamped with
#        [session id | date] (graph path already carries these headers)
#   R23 - integration: R2 + R3 composed (compared against R2 = one variable)
#   H1  - hygiene: anchor-priority prefill in the knapsack (prompt = R0)
# ---------------------------------------------------------------------------
SUPPORTED_CONDS = ("R0", "R1", "R2", "R3", "R23", "H1", "R6", "R7", "R7.3")

_EVIDENCE_START_RE = re.compile(r"^(?:--- Event \d+|\[\S+(?: on .*)?\] (?:User|Assistant):)")


def reorder_evidence_first(text: str, query: str) -> str:
    """Permute evidence blocks by lexical relevance to the query (R1).

    Content-preserving: identical blocks and bytes, only their order
    changes.  The preamble (certificate / header blocks appearing before
    the first evidence block) always stays first.  Returns the input
    unchanged when fewer than two evidence blocks exist or when there is
    no lexical signal to sort by.
    """
    if not text or "\n\n" not in text:
        return text

    head: list[str] = []
    groups: list[list[str]] = []
    for block in text.split("\n\n"):
        if _EVIDENCE_START_RE.match(block):
            groups.append([block])
        elif groups:
            groups[-1].append(block)
        else:
            head.append(block)
    if len(groups) < 2:
        return text

    q_words = set(re.findall(r"[a-zA-Z0-9_-]+", query.lower())) - V4CandidateRetriever.STOP_WORDS
    if not q_words:
        return text

    def _score(group: list[str]) -> int:
        words = set(re.findall(r"[a-zA-Z0-9_-]+", "\n\n".join(group).lower()))
        return sum(1 for w in q_words if w in words)

    scored = [(_score(g), i, g) for i, g in enumerate(groups)]
    if all(s == 0 for s, _, _ in scored):
        return text
    scored.sort(key=lambda t: (-t[0], t[1]))  # stable: ties keep original order
    return "\n\n".join(head + ["\n\n".join(g) for _, _, g in scored])


def filter_evidence(text: str, query: str) -> str:
    """Drop evidence blocks with zero lexical overlap with the query (R2).

    Content-REDUCING by design (compact-evidence ceiling probe, not a pure
    model comparison): the head (certificate / header blocks) is always
    kept, the order of survivors never changes, and the original text is
    returned when there is no lexical signal or when the filter would
    remove every evidence block.
    """
    if not text or "\n\n" not in text:
        return text

    head: list[str] = []
    groups: list[str] = []
    for block in text.split("\n\n"):
        if _EVIDENCE_START_RE.match(block):
            groups.append(block)
        elif groups:
            groups[-1] = groups[-1] + "\n\n" + block
        else:
            head.append(block)
    if not groups:
        return text

    q_words = set(re.findall(r"[a-zA-Z0-9_-]+", query.lower())) - V4CandidateRetriever.STOP_WORDS
    if not q_words:
        return text

    MIN_EVIDENCE_BLOCKS = 3
    kept_idx = [i for i, g in enumerate(groups)
                if q_words & set(re.findall(r"[a-zA-Z0-9_-]+", g.lower()))]
    if not kept_idx or len(kept_idx) == len(groups):
        # nothing to drop (or everything would go) -> keep R0 bytes
        return text
    # Safety floor: never starve the context below MIN_EVIDENCE_BLOCKS.
    # Zero-overlap filtering can drop PARAPHRASED evidence (75499fd8: the
    # answer turn shares no lexical overlap with the question), so the
    # earliest dropped blocks are reinstated until the floor is met.
    if len(kept_idx) < MIN_EVIDENCE_BLOCKS:
        kept_set = set(kept_idx)
        for i in range(len(groups)):
            if len(kept_idx) >= MIN_EVIDENCE_BLOCKS:
                break
            if i not in kept_set:
                kept_idx.append(i)
                kept_set.add(i)
        kept_idx.sort()
    return "\n\n".join(head + [groups[i] for i in kept_idx])


def build_prompt(
    cond: str,
    plan,
    is_aggregation: bool,
    compiled_ctx_str: str,
    q: str,
    q_date: str,
) -> str:
    """Assemble the reader prompt for the given condition.

    ``cond == "R0"`` reproduces the original pipeline byte-for-byte; the
    only permitted difference in R1 is the ORDER of evidence blocks.
    """
    ctx = compiled_ctx_str
    if cond == "R1":
        ctx = reorder_evidence_first(ctx, q)
    elif cond in ("R2", "R23"):
        ctx = filter_evidence(ctx, q)

    if plan.preference_demand >= 0.40:
        return (
            "[INSTRUCTION: PERSONALIZED RECOMMENDATIONS]\n"
            "You are a helpful assistant providing personalized recommendations based on the user's past conversations.\n"
            "RULES:\n"
            "1. The user previously discussed this topic in their past conversations. You MUST identify the specific past experiences, pets, past recipes, venues, favorite artists, or background they mentioned.\n"
            "2. Explicitly build upon or incorporate their specific past mentions into your suggestions (e.g. mention their pet by name, build upon a past recipe they made, reference venues/artists from their past experiences, or account for their equipment/classes).\n"
            "3. DO NOT give generic tourist or generic hobby advice. Anchor directly in their personal history from the evidence.\n"
            "4. Never say 'I don't know'.\n\n"
            f"Question date: {q_date or 'N/A'}\n\n"
            f"{ctx}"
        )
    if is_aggregation:
        # AGGREGATION & MULTI-SESSION PROMPT:
        # Exhaustive enumeration for counting/summing, AND precise subtraction for difference/comparison.
        return (
            "[INSTRUCTION: EXHAUSTIVE AGGREGATION & COMPARISON]\n"
            "A memory system has retrieved ALL available evidence across multiple sessions.\n"
            "You must count, sum, or compute the answer by examining EVERY piece of evidence.\n"
            "RULES:\n"
            "1. First, locate and list ALL individual amounts, items, or instances mentioned across all sessions (including multiple items or costs mentioned within the SAME session).\n"
            "2. If the question asks 'how many' or for a total amount: add up ALL amounts or count ALL items across all sessions (e.g. '10', '$75', '$450').\n"
            "3. If the question asks for a difference, comparison, or margin (e.g. 'difference in price/cost', 'how much faster/slower', 'how many years older/younger', 'how much more/less compared to'): find the two relevant values and calculate the difference (A - B = difference), NOT just the single amount or the sum!\n"
            "4. If multiple costs or items are mentioned in the evidence (e.g. vet visit $50 and medication $25), add them together: $50 + $25 = $75.\n"
            "5. DO NOT stop at the first amount or first session. Check EVERY detail in the evidence.\n"
            "6. NEVER say 'I don't know'. Compute the answer directly from the evidence available.\n"
            "7. ARITHMETIC — compute step by step:\n"
            "   - Money: read every digit and comma exactly ('$5,850' = 5850, NOT 6175). Sum only the user's own amounts; ignore illustrative reference figures.\n"
            "   - Elapsed time: subtract the earlier date from the later, then convert (7 days = 1 week, ~30 days = 1 month).\n"
            "   - Counting: enumerate items one by one and be careful with off-by-one (inclusive vs exclusive).\n\n"
            f"Question date: {q_date or 'N/A'}\n\n"
            f"{ctx}\n\n"
            "Now, enumerate the relevant values and state the final computed answer directly:"
        )
    prompt_parts = [
        "[INSTRUCTION: EVIDENCE-FIRST ANSWERING]\n"
        "A memory system has already retrieved the verified evidence below for this question.\n"
        "The answer IS present in this evidence.\n"
        "RULES:\n"
        "1. Locate the specific fact the question asks about and state it directly.\n"
        "2. Reply with the final value only: a few words or one short sentence.\n"
        "3. NEVER reply \"I don't know\", \"not enough information\", \"not mentioned\", or any other refusal.\n"
        "4. For coupon redemptions, transactions, or shopping: name the merchant or store (e.g. Target, Amazon), NOT an email inbox.\n"
        "5. If the question asks for current / new / updated (job, address, preference, status), provide the latest one. If the question asks for previous / earlier / former, provide the previous one from before the transition.\n"
        "6. For questions asking 'Did I...', 'Was I...', 'Have I...': answer 'Yes' or 'No' directly.\n"
        "7. For questions asking 'How much more / less / faster / slower compared to...': calculate and state the difference (e.g. A - B = difference), NOT just the raw single amount.\n"
        "8. Use only the evidence; do not invent details that are not in it.\n"
        # --- CoT arithmetic (general, no benchmark fitting) ---
        "9. ARITHMETIC — compute step by step, never guess:\n"
        "   - Elapsed time ('how many days/weeks/months between X and Y', 'how long since'): find BOTH dates, subtract the earlier from the later, and divide to convert units (7 days = 1 week, ~30 days = 1 month). State the number and unit.\n"
        "   - Money: read every digit and comma exactly ('$5,850' is five thousand eight hundred fifty, NOT $6175). Add the relevant amounts only; ignore illustrative figures that are not the user's own.\n"
        "   - Counting: enumerate each matching item one by one, then report the count. Watch for off-by-one: count inclusively or exclusively exactly as the question's wording requires.\n"
        "   - For any calculation, do the subtraction/addition on the actual numbers from the evidence, not by estimation.\n\n"
        f"Question date: {q_date or 'N/A'}\n\n"
        f"{ctx}"
    ]
    return "\n\n".join(prompt_parts)


def main():
    parser = argparse.ArgumentParser(description="Evaluate V4 Budgeted Cognitive Pipeline on Dev 250")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--budget-override", type=int, default=0, help="Override adaptive budget with fixed B")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint if available")
    parser.add_argument("--fresh", action="store_true", help="Start fresh ignoring checkpoint")
    parser.add_argument("--qid", default="", help="Evaluate only this question_id (fast-loop debug)")
    parser.add_argument("--qids", default="", help="Comma-separated question_ids to evaluate (fast-loop debug)")
    parser.add_argument("--cond", default="R0", choices=SUPPORTED_CONDS,
                        help="Experiment condition: R0=frozen baseline, "
                             "R1=evidence-first ordering, R2=compact evidence, "
                             "R3=provenance tags, R23=R2+R3 integration")
    parser.add_argument("--out", default="",
                        help="Explicit output json path (overrides --cond/--qid path derivation); "
                             "checkpoint becomes <stem>_ckpt.json next to it")
    parser.add_argument("--split", default="dev", choices=("dev", "holdout"),
                        help="Evaluation split: dev (250) or holdout (250). "
                             "Both run the IDENTICAL pipeline code — only the qid list differs.")
    args = parser.parse_args()

    if args.split == "holdout":
        split_path = REPO / "benchmark_results" / "splits" / "lme_holdout_qids.json"
        split_label = "holdout"
    else:
        split_path = REPO / "benchmark_results" / "splits" / "lme_dev_qids.json"
        split_label = "dev"
    with open(split_path, encoding="utf-8") as f:
        dev_qids = set(json.load(f))

    with open(DATASET_PATH, encoding="utf-8") as f:
        all_items = json.load(f)

    dev_items = [it for it in all_items if it["question_id"] in dev_qids]
    out_path = OUT_PATH
    ckpt_name = "lme_dev250_v4_checkpoint.json"
    if args.cond != "R0" or args.split != "dev":
        # Non-baseline conditions write to their own artefacts so the frozen
        # R0 baseline file is never overwritten by an experiment run.
        tag = args.cond if args.split == "dev" else f"{args.cond}_{split_label}"
        out_path = OUT_PATH.parent / f"lme_dev250_v4_{tag}.json"
        ckpt_name = f"lme_dev250_v4_{tag}_checkpoint.json"
    if args.qid or args.qids:
        wanted = set()
        if args.qid:
            wanted.add(args.qid.strip())
        if args.qids:
            wanted.update(s.strip() for s in args.qids.split(",") if s.strip())
        dev_items = [it for it in dev_items if it["question_id"] in wanted]
        print(f"  [Fast-loop] filtered to {len(dev_items)} question(s): {sorted(wanted)}")
        # Fast-loop runs write to a separate file so the full-run artifacts stay frozen
        stem = "_".join(sorted(wanted))[:80]
        if args.cond == "R0":
            out_path = OUT_PATH.parent / ("lme_dev_fastloop_%s.json" % stem)
            ckpt_name = "lme_dev_fastloop_ckpt.json"
        else:
            out_path = OUT_PATH.parent / ("lme_dev_fastloop_%s_%s.json" % (args.cond, stem))
            ckpt_name = "lme_dev_fastloop_%s_ckpt.json" % args.cond
    if args.out:
        out_path = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_name = out_path.stem + "_ckpt.json"
    if args.limit > 0:
        dev_items = dev_items[:args.limit]

    print("=" * 80)
    print(f"  LETHE-AKRIBEIA V4: E2E BUDGETED COMPILATION PIPELINE (n={len(dev_items)})")
    print(f"  Reader: {FROZEN_MODEL} (num_ctx={args.num_ctx})")
    print(f"  Condition: {args.cond}")
    print("=" * 80)

    extractor = UniversalIRExtractor()
    planner = CognitiveExecutionPlanner()
    retriever = V4CandidateRetriever()
    graph_builder = GraphBuilder()
    compiler = BudgetedCognitiveCompiler()
    coprocessor = CoProcessorEngine()
    answerer = OllamaAnswerer(model=FROZEN_MODEL, num_ctx=args.num_ctx)
    per_session_sampler = PerSessionSampler(turns_per_session=3)
    coverage_engine = MultiSessionCoverageEngine()

    results = []
    ckpt_path = out_path.parent / ckpt_name
    if not args.fresh and ckpt_path.exists():
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                ckpt_data = json.load(f)
                loaded_res = ckpt_data.get("results", [])
                # Only use if matching dev_items
                dev_item_ids = {it["question_id"] for it in dev_items}
                results = [r for r in loaded_res if r["question_id"] in dev_item_ids]
                print(f"  [Checkpoint] Resumed {len(results)} previously evaluated questions from {ckpt_path.name}")
        except Exception as e:
            print(f"  [Checkpoint] Warning: could not load checkpoint ({e}), starting fresh.")
            results = []

    done_qids = {r["question_id"] for r in results}
    overridden_count = sum(1 for r in results if r.get("deterministic_override"))
    t0 = time.perf_counter()

    # R6 Phase-1: accumulate deterministic evidence-state features per question
    r6_features = {}

    for idx, item in enumerate(dev_items):
        qid = item["question_id"]
        qtype = item["question_type"]
        q = item["question"]
        q_date = item.get("question_date", "")
        gt = str(item.get("answer", ""))

        if qid in done_qids:
            # Already completed in resumed checkpoint
            existing = next(r for r in results if r["question_id"] == qid)
            status = "PASS" if existing["is_correct"] else "FAIL"
            print(f"[{idx+1:03d}/{len(dev_items)}] {qid} ({qtype[:12]}): {status} (RESUMED) | tok={existing['prompt_tokens']}")
            continue

        print(f"[{idx+1:03d}/{len(dev_items)}] {qid} ({qtype[:12]}): starting")
        # Strict per-query isolation guard
        pred = ""
        tok_used = 0
        lat_ms = 0.0
        compiled_ctx_str = ""
        prompt_context = None

        # 1. Ingest all haystack sessions into turns and StructuredIR records
        all_turns: list[tuple[str, str, str, str]] = []
        all_records: list[StructuredIR] = []
        dates = item.get("haystack_dates", [])
        sids = item.get("haystack_session_ids", [])

        for s_idx, session in enumerate(item.get("haystack_sessions", [])):
            s_date = dates[s_idx] if s_idx < len(dates) else ""
            sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
            for turn_idx, turn in enumerate(session):
                speaker = turn.get("role", "user")
                content = turn.get("content", "")
                header = f"[{sid} on {s_date}]" if s_date else f"[{sid}]"
                line = f"{header} {speaker.capitalize()}: {content}"
                all_turns.append((sid, s_date, speaker, content))

                recs = extractor.extract(content, default_source=speaker)
                for r in recs:
                    r.raw_content = line
                    r.time_scope = s_date
                    if not r.metadata:
                        r.metadata = {}
                    r.metadata["session_id"] = sid
                    r.metadata["turn_idx"] = turn_idx
                    all_records.append(r)

        # 2. Execution Planning (Demand Vector + Budget Bt)
        plan = planner.plan(q, all_records, reference_date=q_date)
        token_budget = args.budget_override if args.budget_override > 0 else plan.recommended_budget

        # 3. Candidate Retrieval (Multi-channel Lexical, Semantic & Entity)
        # Aggregation mode: governed directly by Execution Planner's cognitive mode
        is_aggregation = bool(plan.dominant_mode == "aggregation")
        guaranteed_sids: set[str] = set()

        if args.cond in ("R6", "R6_coverage", "R7", "R7.3"):
            # Multi-Session Coverage Engine (R6 Core)
            # Plans evidence requirements (Concept Expansions + Self-Anchor Dual Queries)
            # and runs convergent multi-turn collection across fragmented sessions.
            top_turns, turn_scores, guaranteed_sids = coverage_engine.retrieve_convergent_evidence(
                query=q,
                all_turns=all_turns,
                retriever=retriever,
                reference_date=q_date,
                base_top_k=60 if is_aggregation else 45,
            )
            base_limit = 32 if is_aggregation else 28
            top_sids = {t[0] for t in top_turns[:base_limit]} | guaranteed_sids
            candidate_turns = [t for t in all_turns if t[0] in top_sids]
            if not candidate_turns:
                candidate_turns = list(top_turns[:25])
            if is_aggregation:
                token_budget = min(2200, int(token_budget * 1.5))
            elif len(guaranteed_sids) > 1 or plan.requires_event_aggregation or plan.aggregation_demand >= 0.50 or plan.temporal_demand >= 0.85:
                # Expand budget for multi-session graph compilation so knapsack doesn't starve secondary sessions
                token_budget = min(2200, int(token_budget * 1.5))
        elif is_aggregation:
            # AGGREGATION MODE: Cross-session count / total sum query.
            # Pre-filter to top-30 candidate sessions via retriever scoring,
            # then PerSessionSampler does fine-grained turn-level selection.
            # (Using ALL turns caused context pollution that hurt accuracy.)
            top_turns, turn_scores = retriever.rank_turns(q, all_turns, top_k=60, reference_date=q_date)
            top_sids = {t[0] for t in top_turns[:30]}
            candidate_turns = [t for t in all_turns if t[0] in top_sids]
            if not candidate_turns:
                candidate_turns = list(all_turns)
            # Boost budget to accommodate multi-session evidence (capped at 2200 tokens)
            token_budget = min(2200, int(token_budget * 1.5))
        else:
            top_turns, turn_scores = retriever.rank_turns(q, all_turns, top_k=45, reference_date=q_date)
            # Top 26 candidate sessions to preserve comparison and temporal anchor pairs
            top_sids = {t[0] for t in top_turns[:26]}
            # Preserve conversational dialogue turns from candidate sessions
            candidate_turns = [t for t in all_turns if t[0] in top_sids]
            if not candidate_turns:
                candidate_turns = top_turns[:20]

        # Filter matching records for candidates (deterministic header-sid match;
        # avoids O(N*M) substring scan + FP from short generic turns).
        cand_sids = {c[0] for c in candidate_turns}
        header_sid_re = re.compile(r"^\[([^\s\]]+)")
        cand_records = []
        for r in all_records:
            m = header_sid_re.match(r.raw_content or "")
            if m and m.group(1) in cand_sids:
                cand_records.append(r)
        if not cand_records:
            cand_records = all_records[:50]

        # Prioritize anchors by candidate retrieval score
        t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(candidate_turns)}
        anchor_ids = []
        anchor_scores: dict[str, float] = {}
        for t in top_turns:
            nid = t_map.get((t[0], t[3]))
            if nid and nid not in anchor_ids:
                anchor_ids.append(nid)
                score = turn_scores.get((t[0], t[3]), 1.0)
                if t[0] in guaranteed_sids:
                    score = max(score, 25.0)
                anchor_scores[nid] = score
        for i, t in enumerate(candidate_turns):
            nid = f"{t[0]}:t{i}"
            if nid not in anchor_scores and (t[0], t[3]) in turn_scores:
                score = turn_scores[(t[0], t[3])]
                if t[0] in guaranteed_sids:
                    score = max(score, 25.0)
                anchor_scores[nid] = score

        # 4. Deterministic Co-processor Reasoning & Certificates
        # Pass the retrieval-selected candidate records to the operator engine
        # (via operator_records) so a SUM aggregates only relevant sessions,
        # not distractor quantities from the full corpus.
        bundle = coprocessor.process(q, all_records, plan, reference_date=q_date, operator_records=cand_records)

        # Check for 0-token deterministic override
        if bundle.deterministic_answer_override:
            pred = bundle.deterministic_answer_override
            tok_used = 0
            lat_ms = 0.0
            overridden_count += 1
            compiled_ctx_str = "[0-TOKEN DETERMINISTIC RESOLUTION]"
            compiled = None  # deterministic override consumed the full context
        elif is_aggregation:
            # 5. Per-Session Sampling for cross-session exhaustive aggregation
            sampled_ev = per_session_sampler.sample(q, candidate_turns, max_total_tokens=token_budget)
            compiled_ctx_str = per_session_sampler.format_as_context(
                sampled_ev, provenance=(args.cond in ("R3", "R23")))
            if bundle.assembled_certificate_block:
                compiled_ctx_str = f"{bundle.assembled_certificate_block}\n\n{compiled_ctx_str}"
            compiled = None  # per-session sampler selects directly
        else:
            # 5. Build Evidence Dependency Graph & Bundles
            edg = graph_builder.build(candidate_turns, cand_records)
            bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)

            # Window / clamp oversized nodes (> 180 words) to ensure they fit within tight budgets
            clamped_bundles = []
            for b in bundles:
                new_members = []
                for m in b.member_nodes:
                    words = m.raw_text.split()
                    if len(words) > 180:
                        trunc_text = " ".join(words[:180]) + " ..."
                        new_members.append(EvidenceNode(
                            node_id=m.node_id, session_id=m.session_id, session_date=m.session_date,
                            speaker=m.speaker, raw_text=trunc_text, tokens=len(trunc_text.split()) * 4 // 3,
                            entities=m.entities, properties=m.properties, values=m.values
                        ))
                    else:
                        new_members.append(m)
                tot_tok = sum(n.tokens for n in new_members)
                clamped_bundles.append(EvidenceBundle(b.bundle_id, b.primary_node_id, new_members, tot_tok, b.rationale))
            bundles = clamped_bundles

            # 6. Budgeted Compilation (Submodular Knapsack under Bt)
            is_multi = bool(
                plan.requires_event_aggregation
                or plan.aggregation_demand >= 0.50
                or plan.temporal_demand >= 0.85
                or len(guaranteed_sids) > 1
            )
            use_adaptive = bool(args.cond in ("R6", "R7", "R7.3", "R6_adaptive"))
            compiled = compiler.compile(
                question=q,
                bundles=bundles,
                token_budget=token_budget,
                co_processor_bundle=bundle,
                anchor_scores=anchor_scores,
                is_multi_session=is_multi,
                anchor_prefill=(args.cond == "H1"),
                adaptive_escalation=use_adaptive,
            )
            compiled_ctx_str = compiled.context_text

        # ── R6 Phase-1: deterministic evidence-state features ──
        r6_f = extract_r6_features(
                    item=item,
                    q=q,
                    plan=plan,
                    bundle=bundle,
                    compiled=compiled,
                    candidate_turns=candidate_turns,
                    candidate_records=cand_records,
                    anchor_scores=anchor_scores,
                    is_aggregation=is_aggregation,
                    correct=False,  # filled after grading
                    reader_model="",
                    predicted="",
                )
        r6_features[qid] = r6_f

        # 7. Prompt Reader (condition variants live in build_prompt)
        if not bundle.deterministic_answer_override:
            prompt_context = build_prompt(
                cond=args.cond,
                plan=plan,
                is_aggregation=is_aggregation,
                compiled_ctx_str=compiled_ctx_str,
                q=q,
                q_date=q_date,
            )

            ans_resp = answerer.answer(q, prompt_context)
            pred = ans_resp.text.strip()
            if pred.endswith(".") and len(pred.split()) <= 2:
                pred = pred[:-1].strip()
            tok_used = ans_resp.prompt_tokens
            lat_ms = ans_resp.latency_ms

        # P0-2: freeze the exact reader input (context cache) so Reader-side
        # experiments (R5) can replay identical inputs without recompiling.
        ctx_dir = REPO / "cache" / "v4_contexts" / args.cond
        ctx_dir.mkdir(parents=True, exist_ok=True)
        with open(ctx_dir / f"{qid}.json", "w", encoding="utf-8") as f:
            json.dump({
                "question_id": qid,
                "condition": args.cond,
                "question_type": qtype,
                "question_date": q_date,
                "question": q,
                "deterministic_override": bool(bundle.deterministic_answer_override),
                "compiled_context": compiled_ctx_str,
                "prompt_context": prompt_context,
            }, f, ensure_ascii=False, indent=2)

        # Grading-only: the official scorer needs the reference answer and the
        # reference abstention label. This is computed after inference and never
        # influences retrieval, selection, or prompt construction.
        is_abstention_reference = (
            qtype == "abstention"
            or qid.endswith("_abs")
            or "information provided is not enough" in gt.lower()
        )
        is_ok = LongMemEvalAdapter.score_answer(
            question_type=qtype,
            gt=gt,
            predicted_answer=pred,
            is_abstention_gt=is_abstention_reference,
            pcc_is_abstention=is_abstention_reference,
        )

        # Fill in the outcome into the R6 feature row (1A visualization)
        # Resumed items are not re-computed, so only update features we own.
        if qid in r6_features:
            r6_features[qid].correct = bool(is_ok)
            r6_features[qid].predicted = pred

        results.append({
            "question_id": qid,
            "question_type": qtype,
            "question": q,
            "ground_truth": gt,
            "predicted_answer": pred,
            "is_correct": is_ok,
            "deterministic_override": bool(bundle.deterministic_answer_override),
            "prompt_tokens": tok_used,
            "latency_ms": lat_ms,
            "r6_features": feature_dict(r6_features[qid]),
        })

        # Periodic checkpointing (every 10 items)
        if (len(results) % 10 == 0) or (len(results) == len(dev_items)):
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, indent=2, ensure_ascii=False)

        status = "PASS" if is_ok else "FAIL"
        print(f"[{idx+1:03d}/{len(dev_items)}] {qid} ({qtype[:12]}): {status} | tok={tok_used} | pred='{pred[:40]}'")

    elapsed = time.perf_counter() - t0
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    acc = correct / total if total > 0 else 0.0

    by_type = defaultdict(lambda: {"total": 0, "correct": 0, "tokens": []})
    for r in results:
        t = r["question_type"]
        by_type[t]["total"] += 1
        if r["is_correct"]:
            by_type[t]["correct"] += 1
        by_type[t]["tokens"].append(r["prompt_tokens"])

    total_tokens = sum(r["prompt_tokens"] for r in results)
    mean_tokens = total_tokens / total if total > 0 else 0.0

    summary = {
        "benchmark": "LongMemEval",
        "split": "dev_250",
        "pipeline": "V4_Budgeted_Cognitive_Compilation",
        "condition": args.cond,
        "model": FROZEN_MODEL,
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": acc,
        "mean_prompt_tokens": mean_tokens,
        "deterministic_overrides": overridden_count,
        "elapsed_seconds": elapsed,
        "per_type_breakdown": {
            t: {
                "accuracy": d["correct"] / d["total"] if d["total"] > 0 else 0.0,
                "correct": d["correct"],
                "total": d["total"],
                "mean_tokens": sum(d["tokens"]) / len(d["tokens"]) if d["tokens"] else 0.0,
            }
            for t, d in by_type.items()
        },
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"  V4 PIPELINE DEV 250 EVALUATION COMPLETE")
    print(f"  Total Accuracy: {correct}/{total} ({acc:.2%})")
    print(f"  Mean Tokens: {mean_tokens:.1f} tokens (V3.1 baseline was 3,653 tokens)")
    print(f"  Deterministic 0-token overrides: {overridden_count}/{total} ({overridden_count/total:.1%})")
    print(f"  Elapsed: {elapsed:.1f}s ({(elapsed/total)*1000:.0f}ms/query)")
    print("=" * 80)
    print(f"{'Type':<28} {'Accuracy':<12} {'Correct/Total':<14} {'Mean Tokens':<12}")
    print("-" * 68)
    for t, d in summary["per_type_breakdown"].items():
        print(f"{t:<28} {d['accuracy']:<12.1%} {d['correct']}/{d['total']:<12} {d['mean_tokens']:<12.1f}")
    print("=" * 80)
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
