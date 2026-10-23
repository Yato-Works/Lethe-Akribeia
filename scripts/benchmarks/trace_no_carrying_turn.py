"""Phase 1 -- trace NO_CARRYING_TURN cases to the raw corpus, per layer.

The atlas labeled 10 when_date cases NO_CARRYING_TURN, meaning the gold date string
was not found in the frozen context the engine saw.  That label is a fact about the
COMPILED context, not about the conversation.  The gold date may exist in the raw
conversation but be dropped by the retriever, the graph builder, or the compiler.

This traces each of the 10 through the real pipeline:
  1  does the raw conversation contain the gold date? (raw corpus vs compiled context)
  2  which raw turns carry it?
  3  does the retriever surface those turns at all? (top-k candidates)
  4  after compiler/graph, is the turn still present with date + speaker + turn_id?
  5  what actually reaches the temporal engine?

Cause classification (Eli's taxonomy):
  A  no evidence in the raw corpus at all
  B  retriever did not select the turn
  C  retriever selected it, but compiler/graph dropped it
  D  evidence present but event-date linkage not represented
  E  undetermined

No fix here -- this only locates where the pipeline loses the evidence.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory_gen.skills.answer_committer import parse_turns  # noqa: E402

ATLAS = ROOT / "benchmark_results/failure_atlas/failure_atlas.jsonl"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
CORPUS = ROOT / "datasets/external/locomo10.json"
OUT = ROOT / "benchmark_results/when_date_map/raw_trace_no_carrying_turn.json"

EXPLICIT_DATE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+20\d\d\b")


def load_raw_conversations() -> dict[str, dict]:
    """The raw LoCoMo corpus, keyed by sample_id ('conv-26').

    Keyed by sample_id rather than by list position or by a regex guess: the cache
    qid 'conv-26-qa-000' embeds the same identifier, so this is an exact join and
    cannot silently mis-align a conversation.
    """
    if not CORPUS.exists():
        return {}
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return {c.get("sample_id"): c for c in data if c.get("sample_id")}


def raw_turns_of(conv: dict) -> list[dict]:
    """Every speaker turn with its session date, from a raw conversation.

    The corpus stores turns under 'session_N' with the date under
    'session_N_date_time' -- not under a 'sessions' list.  Walking the real keys
    matters because a wrong shape would report zero raw hits and classify every
    case as 'A_no_evidence', which is a fabricated cause.
    """
    turns: list[dict] = []
    conv_body = conv.get("conversation") or {}
    if not isinstance(conv_body, dict):
        return turns
    for key in sorted(conv_body.keys(),
                      key=lambda k: (len(k), k)):
        m = re.fullmatch(r"session_(\d+)", key)
        if not m:
            continue
        si = int(m.group(1))
        sdate = conv_body.get(f"session_{si}_date_time") or ""
        for turn in conv_body.get(key) or []:
            if isinstance(turn, dict):
                turns.append({
                    "session": si,
                    "session_date": sdate,
                    "speaker": turn.get("speaker") or "",
                    "dia_id": turn.get("dia_id") or "",
                    "text": turn.get("text") or "",
                })
    return turns


def _date_patterns(gdate: str) -> list[re.Pattern]:
    """Exact day+month patterns.  NO loose month-only fallback.

    A month-only fallback was tried and produced a false positive: gold
    'February 24' matched a context line reading '25 February', which is a
    DIFFERENT date about a different event.  Counting that as 'evidence present'
    would have inverted the cause classification for that case.  Day+month must
    match exactly or the case is reported as not-carried.
    """
    d, mon = gdate.split()[0], re.escape(gdate.split()[1])
    return [
        re.compile(rf"\b{d}\s+{mon}\b", re.I),
        re.compile(rf"\b{mon}\s+{d}(?:st|nd|rd|th)?\b", re.I),
    ]


def gold_date_str(gold: str) -> str | None:
    """The literal day-month the gold expects, if it names one."""
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)", str(gold).strip())
    if m and m.group(2).lower() in (
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december"):
        return f"{int(m.group(1))} {m.group(2).lower()}"
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2})", str(gold).strip())
    if m and m.group(1).lower() in (
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december"):
        return f"{int(m.group(2))} {m.group(1).lower()}"
    return None


def main() -> int:
    atlas = {json.loads(l)["question_id"]: json.loads(l)
             for l in ATLAS.read_text(encoding="utf-8").splitlines() if l.strip()}
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["qid"]] = row
    raw_convs = load_raw_conversations()

    # Two evidence-absent populations, named explicitly.  The mission's
    # 'NO_CARRYING_TURN 10' came from a binding-count label in an earlier script;
    # the map itself records no such class.  The real evidence-absent sets are:
    #   (a) the engine answered and was WRONG with gold absent from context (6),
    #   (b) the engine ABSTAINED with gold absent from context (85).
    # (a) is the harmful one -- it committed a date it had no evidence for.  (b) is
    # safe abstention.  Both are traced, and (a) is prioritised because a wrong
    # answer is worse than an honest abstention.
    map_data = json.loads(
        (ROOT / "benchmark_results/when_date_map/when_date_failure_map.json")
        .read_text(encoding="utf-8"))
    records = map_data["records"]
    wrong_absent = [r for r in records
                    if r["engine_class"] == "ENGINE_WRONG_EVIDENCE_ABSENT"]
    abstain_absent = [r for r in records
                      if r["engine_class"] == "ENGINE_ABSTAIN"
                      and r["gold_in_context"] is False]
    print(f"engine WRONG with gold absent from context : {len(wrong_absent)}")
    print(f"engine ABSTAIN with gold absent           : {len(abstain_absent)}")
    print(f"(the mission's 'NO_CARRYING_TURN 10' is not a class the map records;")
    print(f" these two are the real evidence-absent sets)")
    print()

    if not raw_convs:
        print(f"\nRAW CORPUS NOT FOUND at {CORPUS}")
        print("  cannot separate A (no evidence) from B/C (evidence lost downstream)")
        print("  -> stopping rather than guessing a cause")
        return 1
    print(f"raw corpus conversations    : {len(raw_convs)}")
    print()

    all_results: list[dict] = []
    for target, tag in ((wrong_absent, "WRONG_ABSENT"),
                        (abstain_absent, "ABSTAIN_ABSENT")):
        print("=" * 74)
        print(f"TRACING {tag}  n={len(target)}")
        print("=" * 74)
        results: list[dict] = []
        for r in target:
            qid = r["question_id"]
            ctx_row = cache.get(r.get("cache_qid")) or {}
            context = ctx_row.get("context") or ""
            gold = r["gold"]
            gdate = gold_date_str(gold)

            # Which conversation does this belong to?  Join on sample_id, exact.
            conv_id = None
            m = re.search(r"(conv-\d+)-qa-", r.get("cache_qid") or "")
            if m:
                conv_id = m.group(1)
            raw = raw_convs.get(conv_id) if conv_id else None
            raw_turns = raw_turns_of(raw) if raw else []

            # 1. Does the raw conversation contain the gold date?  Exact day+month.
            raw_hits: list[dict] = []
            pats = _date_patterns(gdate) if gdate else []
            for t in raw_turns:
                if any(p.search(t["text"]) for p in pats):
                    raw_hits.append(t)

            # 2. Does the compiled context contain it?  Same strict match.
            ctx_hits = [line.strip()[:160] for line in context.splitlines()
                        if any(p.search(line) for p in pats)]

            # 3. Does the retriever surface it?  Run the production retriever.
            retriever_selected = None
            try:
                from artificial_memory_gen.compiler.closure.planner import (
                    DeterministicQueryPlanner,
                )
                from artificial_memory_gen.compiler.closure.retriever import (
                    MultiViewRetriever,
                )
                from artificial_memory_gen.skills.answer_committer import Turn  # noqa

                qplan = DeterministicQueryPlanner().plan(r["question"])
                turns = parse_turns(context)
                raw4 = [(_s(t.dia_id), t.header_date, t.speaker, t.text)
                        for t in turns]
                cands = MultiViewRetriever().retrieve(qplan, raw4, top_k=35)
                retriever_selected = [c.turn_id for c in cands]
            except Exception as exc:                       # noqa: BLE001
                retriever_selected = f"ERROR: {type(exc).__name__}: {exc}"

            # ---- Cause classification.
            if not gdate:
                cause = "E_undetermined"
                why = "gold names no day-month to search for"
            elif not raw_hits:
                cause = "A_no_evidence_in_raw"
                why = f"gold date {gdate!r} absent from the raw conversation"
            elif not ctx_hits:
                cause = ("B_retriever_did_not_select"
                         if isinstance(retriever_selected, list)
                         and not any(gdate.split()[1] in str(x).lower()
                                     for x in retriever_selected)
                         else "C_compiler_dropped")
                why = ("date present in raw conversation, absent from compiled context"
                       f" (raw hits={len(raw_hits)}, ctx hits=0)")
            else:
                cause = "D_event_date_linkage"
                why = "date present in compiled context but engine did not bind it"

            results.append({
                "question_id": qid,
                "question": r["question"],
                "gold": gold,
                "gold_date_searched": gdate,
                "conv_id": conv_id,
                "raw_turn_count": len(raw_turns),
                "raw_hit_count": len(raw_hits),
                "raw_hit_samples": [
                    {"session": h["session"], "session_date": h["session_date"],
                     "speaker": h["speaker"], "text": h["text"][:150]}
                    for h in raw_hits[:3]],
                "compiled_context_hit_count": len(ctx_hits),
                "compiled_context_hit_samples": ctx_hits[:2],
                "retriever_top_n": (len(retriever_selected)
                                    if isinstance(retriever_selected, list) else None),
                "retriever_turn_ids": (retriever_selected[:15]
                                       if isinstance(retriever_selected, list)
                                       else retriever_selected),
                "cause": cause,
                "why": why,
            })

        for x in results:
            x["group"] = tag
        all_results.extend(results)

        counts = Counter(x["cause"] for x in results)
        print(f"CAUSES for {tag}: {dict(counts)}")
        for x in results:
            print(f"  {x['question_id']:<17} {x['cause']}")
            print(f"    Q    : {x['question'][:60]}")
            print(f"    gold : {x['gold']!r}   searching for "
                  f"{x['gold_date_searched']!r}")
            print(f"    raw hits={x['raw_hit_count']}  "
                  f"compiled hits={x['compiled_context_hit_count']}  "
                  f"retriever top-n={x['retriever_top_n']}")
            print(f"    why  : {x['why']}")
            for h in x["raw_hit_samples"][:2]:
                print(f"      raw s{h['session']} [{h['session_date']}] "
                      f"{h['speaker']}: {h['text'][:110]}")
            print()

    total_counts = Counter(x["cause"] for x in all_results)
    print("=" * 74)
    print("CAUSE CLASSIFICATION ACROSS BOTH EVIDENCE-ABSENT GROUPS")
    print("=" * 74)
    for k, n in total_counts.most_common():
        print(f"  {k:<26} {n:>3}")
    print(f"  {'TOTAL':<26} {len(all_results):>3}")

    OUT.write_text(json.dumps({
        "label": "phase 1 -- evidence-absent when_date cases traced to the raw corpus",
        "corpus": str(CORPUS.relative_to(ROOT)) if CORPUS.exists() else "MISSING",
        "note": "the mission's 'NO_CARRYING_TURN 10' is not a class the map records; "
                "the real evidence-absent sets are ENGINE_WRONG_EVIDENCE_ABSENT and "
                "ENGINE_ABSTAIN-with-gold-absent",
        "n": len(all_results),
        "causes": dict(total_counts),
        "cases": all_results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


def _s(dia_id: str) -> str:
    return dia_id.split(":", 1)[0] if ":" in dia_id else dia_id


if __name__ == "__main__":
    raise SystemExit(main())
