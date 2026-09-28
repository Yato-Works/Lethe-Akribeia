#!/usr/bin/env python3
"""Anatomy of LoCoMo oracle-recall misses (Layer 1 failure isolation).

Failure-driven step 1-4: isolate the failing examples, reproduce them,
classify the failure, identify the dependency.

For every question where ``oracle_recall`` is False, this answers:

  A. ``source_missing``   the ground-truth evidence id does not resolve to any
                          turn in the dataset at all -> annotation / harness
                          artefact, NOT a Lethe retrieval failure.
  B. ``label_absent``     the source turn exists but its ``[Dn:m ...]`` label is
                          absent from the context Lethe built -> genuine
                          retrieval/selection failure (Lethe-attributable).
  C. ``content_absent``   the label is absent AND the turn's text does not
                          appear either -> the turn was never a candidate.
  D. ``measured_wrong``   the label IS in the context yet oracle said False ->
                          the oracle predicate itself is broken.

Usage:
  uv run python scripts/benchmarks/oracle_miss_anatomy.py \
      --results benchmark_results/locomo1540/locomo_final.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "datasets" / "external" / "locomo10.json"
CACHE = REPO / "benchmark_results" / "locomo_context_cache.jsonl"
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
EV_RE = re.compile(r"^D(\d+):(\d+)$")


def load_evidence_map() -> dict[str, dict]:
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for conv in raw:
        sid = conv.get("sample_id", "")
        conv_obj = conv.get("conversation", {})
        sessions = {}
        for k, v in conv_obj.items():
            if k.startswith("session_") and k[len("session_"):].split("_")[0].isdigit():
                if isinstance(v, list):
                    sessions[int(k[len("session_"):].split("_")[0])] = v
        for i, qa in enumerate(conv.get("qa", [])):
            qid = f"{sid}-qa-{i:03d}"
            out[qid] = {
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "category": qa.get("category"),
                "evidence": qa.get("evidence", []) or [],
                "sessions": sessions,
            }
    return out


def _turn_text(turn) -> str:
    if isinstance(turn, dict):
        return str(turn.get("text") or turn.get("dialogue") or turn.get("content") or "")
    if isinstance(turn, str):
        return turn
    return ""


def resolve_turn(sessions: dict[int, list], ev: str) -> tuple[bool, str, str]:
    """Locate an evidence id inside the raw conversation.

    Evidence ids are ``dia_id`` values (``Dn:m``) which are 1-based while the
    session lists are 0-based, so the turn is found by matching ``dia_id``
    rather than by indexing -- indexing off-by-one would silently compare the
    wrong turn's text.

    Returns ``(found, note, turn_text)``.
    """
    ev = ev.strip()
    m = EV_RE.match(ev)
    if not m:
        # The dataset packs several ids into one string for some questions
        # (e.g. "D8:6; D9:17" or "D9:1 D4:4 D4:6").
        parts = [p for p in re.split(r"[;\s]+", ev) if EV_RE.match(p)]
        if not parts:
            return False, f"unparseable evidence id {ev!r}", ""
        notes = []
        for p in parts:
            ok, note, txt = resolve_turn(sessions, p)
            notes.append(f"{p}: {note}")
            if ok:
                return True, "compound id; " + "; ".join(notes), txt
        return False, "compound id; " + "; ".join(notes), ""
    day, idx = int(m.group(1)), int(m.group(2))
    turns = sessions.get(day)
    if turns is None:
        return False, f"session {day} absent (present: {sorted(sessions)})", ""
    wanted = f"D{day}:{idx}"
    for t in turns:
        if isinstance(t, dict) and str(t.get("dia_id", "")).strip() == wanted:
            return True, f"session {day} dia_id {wanted}", _turn_text(t)
    # fall back to positional interpretation so the miss is still explained
    for pos in (idx, idx - 1):
        if 0 <= pos < len(turns):
            got = str(turns[pos].get("dia_id", "")) if isinstance(turns[pos], dict) else "?"
            return False, f"{wanted} not a dia_id; list[{pos}] is {got}", ""
    return False, f"session {day} has {len(turns)} turns, {wanted} out of range", ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify LoCoMo oracle-recall misses")
    ap.add_argument("--results", default="benchmark_results/locomo1540/locomo_final.json")
    ap.add_argument("--cache", default=str(CACHE.relative_to(REPO)))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show", type=int, default=8, help="print N examples per class")
    args = ap.parse_args()

    emap = load_evidence_map()
    cache: dict[str, str] = {}
    with open(REPO / args.cache, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            cache[d["qid"]] = d.get("context") or ""

    payload = json.loads((REPO / args.results).read_text(encoding="utf-8"))
    rows = payload["results"] if isinstance(payload, dict) else payload
    misses = [r for r in rows if not r.get("oracle_recall")]
    if args.limit:
        misses = misses[: args.limit]

    print(f"results : {args.results}")
    print(f"rows    : {len(rows)}   oracle misses: {len(misses)}")

    classes: dict[str, list[dict]] = defaultdict(list)
    for r in misses:
        qid = r["question_id"]
        meta = emap.get(qid)
        ctx = cache.get(qid, "")
        if meta is None:
            classes["unknown_qid"].append({"qid": qid})
            continue
        evs = meta["evidence"]
        if not evs:
            classes["no_evidence_annotation"].append(
                {"qid": qid, "cat": r["category"], "q": meta["question"]})
            continue
        resolved = [resolve_turn(meta["sessions"], e) for e in evs]
        found_ok = [e for e, (ok, _, _) in zip(evs, resolved) if ok]
        notes = [n for _, n, _ in resolved]
        # The context labels turns as [Dn:m ...], i.e. by dia_id.
        in_ctx_label = [e for e in evs if e.strip() in ctx]
        # Content probe: the exact source text of the dia_id-resolved turn.
        ctx_flat = re.sub(r"\s+", " ", ctx)
        in_ctx_content = []
        for e, (_, _, txt) in zip(evs, resolved):
            frag = re.sub(r"\s+", " ", txt).strip()
            if len(frag) >= 40 and frag[:60] in ctx_flat:
                in_ctx_content.append(e)

        entry = {
            "qid": qid,
            "cat": r["category"],
            "q": meta["question"],
            "evidence": evs,
            "resolve": notes,
            "label_in_ctx": in_ctx_label,
            "content_in_ctx": in_ctx_content,
            "n_ev": len(evs),
        }
        if in_ctx_label:
            classes["measured_wrong"].append(entry)
        elif not found_ok:
            classes["source_missing"].append(entry)
        elif in_ctx_content:
            classes["label_absent_content_present"].append(entry)
        else:
            classes["label_absent"].append(entry)

    print("\n=== classification of oracle misses ===")
    total = len(misses) or 1
    for k in sorted(classes, key=lambda x: -len(classes[x])):
        v = classes[k]
        print(f"  {k:32s} {len(v):5d}  ({100 * len(v) / total:5.1f}%)")

    print("\n=== per category x class ===")
    grid: dict[str, Counter] = defaultdict(Counter)
    for k, v in classes.items():
        for e in v:
            grid[CATEGORY_NAMES.get(e.get("cat", -1), str(e.get("cat")))][k] += 1
    for cat in ("multi-hop", "temporal", "open-domain", "single-hop"):
        if cat in grid:
            parts = ", ".join(f"{k}={c}" for k, c in grid[cat].most_common())
            print(f"  {cat:12s} {parts}")

    for k in ("label_absent", "label_absent_content_present", "source_missing", "measured_wrong"):
        v = classes.get(k) or []
        if not v:
            continue
        print(f"\n--- {k} (first {min(args.show, len(v))}) ---")
        for e in v[: args.show]:
            print(f"  [{e.get('qid')}] cat={e.get('cat')} ev={e.get('evidence')}")
            print(f"     Q: {str(e.get('q'))[:110]}")
            if "resolve" in e:
                print(f"     resolve: {e['resolve']}  label_in_ctx={e['label_in_ctx']}")

    out = REPO / "benchmark_results" / "_oracle_miss_anatomy.json"
    out.write_text(json.dumps({k: v for k, v in classes.items()}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nfull classification -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
