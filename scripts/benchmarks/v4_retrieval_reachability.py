"""V4 Mission 5 -- frozen retrieval reachability audit.  MEASUREMENT ONLY.

Runs the REAL gold-free retrieval pipeline -- DeterministicQueryPlanner + MultiViewRetriever
(reciprocal rank fusion) -- over the frozen 90-question pilot, freezes the ranked
candidates, and asks a single question:

    "Within the turns the retriever actually surfaces, is sufficient supporting
     evidence present?"

It does NOT claim to measure whether evidence exists somewhere in the original
conversation.  A turn that the retriever never returns is OUT OF SCOPE here, and that
boundary is respected everywhere below.

Gold, prior predictions and cohort labels are never used to build, filter or rerank a
candidate list.  Question-term matching is NOT substituted for the retriever: the whole
point is to measure the pipeline that production would actually run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from artificial_memory_gen.compiler.closure.planner import (  # noqa: E402
    DeterministicQueryPlanner,
)
from artificial_memory_gen.compiler.closure.retriever import MultiViewRetriever  # noqa: E402
from artificial_memory.skills.answer_committer import parse_turns  # noqa: E402

PILOT = ROOT / "benchmark_results/evidence_audit/v4_pilot_selection.json"
AUDIT = ROOT / "benchmark_results/evidence_audit/evidence_audit.json"
CACHE = ROOT / "benchmark_results/locomo_context_cache_rules.jsonl"
H9 = ROOT / "benchmark_results/evidence_audit/v4_reader_h9.json"
OUT = ROOT / "benchmark_results/evidence_audit/v4_retrieval_reachability.json"
BLIND = ROOT / "benchmark_results/evidence_audit/v4_pilot_annotator_view.json"

TOP_K_1 = 20
TOP_K_2 = 50

_STOP = frozenset({
    "the", "and", "for", "was", "were", "his", "her", "its", "their", "our", "your",
    "him", "she", "they", "them", "not", "but", "all", "any", "out", "about", "into",
    "over", "than", "then", "when", "what", "who", "why", "how", "which", "there",
    "here", "have", "has", "had", "been", "being", "are", "will", "would", "could",
    "should", "did", "does", "with", "that", "this", "from", "many", "much", "long",
    "often", "ago", "now", "get", "got", "one", "two", "did",
})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:20] if path.exists() else "MISSING"


def main() -> int:
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    meta = {r["question_id"]: r for r in audit["records"]}
    h9 = json.loads(H9.read_text(encoding="utf-8"))
    arm = {r["question_id"]: r for r in h9["rows"]}

    ctx = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ctx[r["qid"]] = r["context"]

    planner = DeterministicQueryPlanner()
    retriever = MultiViewRetriever()

    print("=" * 74)
    print("V4 MISSION 5 -- frozen retrieval reachability audit")
    print("=" * 74)
    print(f"  pilot questions       : {len(pilot['selected'])}")
    print(f"  retriever             : MultiViewRetriever (RRF, rrf_k=60)")
    print(f"  planner               : DeterministicQueryPlanner")
    print(f"  top-K passes          : {TOP_K_1} then {TOP_K_2}")
    print(f"  pilot sha             : {sha(PILOT)}")
    print(f"  cache sha             : {sha(CACHE)}")
    print(f"  gold / prediction / cohort used to rank : NO")
    print()

    rows: list[dict] = []
    for s in pilot["selected"]:
        qid = s["question_id"]
        m = meta.get(qid) or {}
        cq = m.get("cache_qid")
        if not cq or cq not in ctx:
            rows.append({"question_id": qid, "error": "no frozen context join"})
            continue
        q = m["question"]
        turns = parse_turns(ctx[cq])
        # The retriever's 4-tuple shape: (session_id, session_date, speaker, content).
        # session_id is the dia_id prefix, which is how the retriever groups sessions.
        tup = [(t.dia_id.split(":")[0], t.header_date, t.speaker, t.text)
               for t in turns]

        plan = planner.plan(q)
        rec = {
            "question_id": qid,
            "cache_qid": cq,
            "conversation": s["conversation"],
            "category": s.get("category"),
            "cohort_eval_only": s["cohort"],   # never used for retrieval or ranking
            "question": q,
            "n_source_turns": len(turns),
            "A_verdict": (arm.get(qid) or {}).get("mutually_exclusive"),
        }
        for label, k in (("top20", TOP_K_1), ("top50", TOP_K_2)):
            cands = retriever.retrieve(plan, tup, top_k=k)
            rec[f"{label}_n"] = len(cands)
            rec[f"{label}_turns"] = [{
                "rank": i + 1,
                "turn_id": c.turn_id,
                "rrf": round(float(c.rrf_score), 6),
                "speaker": c.speaker,
                "session_date": c.session_date,
                "content": c.content,
            } for i, c in enumerate(cands)]
        rows.append(rec)

    ok = [r for r in rows if "error" not in r]
    assert ok, "no question joined to a frozen context -- the audit cannot run"
    print(f"  joined to context     : {len(ok)}/{len(pilot['selected'])}")
    print(f"  avg candidates top20  : "
          f"{sum(r['top20_n'] for r in ok) / len(ok):.1f}")
    print(f"  avg candidates top50  : "
          f"{sum(r['top50_n'] for r in ok) / len(ok):.1f}")
    print()

    # ---- gold-free EVIDENCE PRESENCE probe, for triage only -----------------
    # This is deliberately NOT the annotation.  It is a coarse, reproducible signal
    # used to route cases to the annotation surface, and it is labelled as such.
    for r in ok:
        gold = str((meta[r["question_id"]].get("gold") or "")).strip().lower()
        toks = [t for t in re.findall(r"[a-z0-9]{3,}", gold) if t not in _STOP]
        for label in ("top20", "top50"):
            hits = []
            for c in r[f"{label}_turns"]:
                body = c["content"].lower()
                if gold and gold in body:
                    hits.append(c["turn_id"])
                elif toks and sum(1 for t in toks if t in body) >= max(1, len(toks) // 2):
                    hits.append(c["turn_id"])
            r[f"{label}_goldstring_turns"] = hits
        r["reach20"] = bool(r["top20_goldstring_turns"])
        r["reach50"] = bool(r["top50_goldstring_turns"])
        r["in_20_not_50"] = False
        r["only_in_50"] = bool(r["reach50"] and not r["reach20"])

    print("=" * 74)
    print("TRIAGE SIGNAL -- does a gold-like string appear inside the candidates?")
    print("=" * 74)
    print("  This is a routing probe, NOT the evidence annotation.  It cannot judge")
    print("  whether the relation holds, so 'not found' here does NOT mean the")
    print("  evidence is absent -- only that this string test did not fire.")
    print()
    print(f"  gold-like string in top-20 : {sum(1 for r in ok if r['reach20'])}/{len(ok)}")
    print(f"  gold-like string in top-50 : {sum(1 for r in ok if r['reach50'])}/{len(ok)}")
    print(f"  newly surfaced by top-50   : {sum(1 for r in ok if r['only_in_50'])}")
    print()

    # ---- the annotation surface the annotator actually receives ------------
    # Question + the retrieved candidate turns ONLY.  No gold, no prediction, no cohort.
    surface = []
    for r in ok:
        surface.append({
            "question_id": r["question_id"],
            "question": r["question"],
            "conversation": r["conversation"],
            "candidates_top20": [
                {"rank": c["rank"], "turn_id": c["turn_id"],
                 "speaker": c["speaker"], "session_date": c["session_date"],
                 "content": c["content"]}
                for c in r["top20_turns"]
            ],
        })
    # Cases needing the top-50 pass, decided WITHOUT looking at gold: any case whose
    # top-20 annotation is still open.  The list is emitted so the annotator can be
    # given the deeper window for exactly those.
    OUT.write_text(json.dumps({
        "label": "V4 mission 5 -- frozen retrieval reachability audit (measurement only)",
        "baseline": {"head": "3b60ba0", "production_changed": False,
                     "evidence_binding": "unwired, preserved"},
        "retriever": {"class": "MultiViewRetriever", "rrf_k": 60,
                      "planner": "DeterministicQueryPlanner"},
        "top_k": [TOP_K_1, TOP_K_2],
        "hashes": {"pilot": sha(PILOT), "cache": sha(CACHE), "audit": sha(AUDIT),
                   "h9": sha(H9)},
        "gold_used_for_ranking": False,
        "n_joined": len(ok),
        "triage": {
            "goldstring_in_top20": sum(1 for r in ok if r["reach20"]),
            "goldstring_in_top50": sum(1 for r in ok if r["reach50"]),
            "newly_in_top50": sum(1 for r in ok if r["only_in_50"]),
        },
        "rows": ok,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    surf_out = OUT.with_name("v4_annotation_surface.json")
    surf_out.write_text(json.dumps({
        "label": "annotator surface: question + retrieved candidates ONLY. "
                 "No gold, no prior prediction, no cohort label.",
        "n": len(surface),
        "annotation_fields_required": [
            "sufficient_evidence_in_top20: yes|no|ambiguous|NOT_FOUND_IN_CANDIDATES",
            "supporting_turn_ids",
            "subject_entity_attribution",
            "supported_relation_or_predicate",
            "multi_turn_or_coreference_required: yes|no",
            "temporal_change_or_correction_or_contradiction: yes|no|unsure",
            "uncertainty_and_rationale",
        ],
        "cases": surface,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"written: {OUT}")
    print(f"written: {surf_out}   <- the annotator surface (blind)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
