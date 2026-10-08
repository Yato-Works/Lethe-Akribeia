"""Quantify B2 (t_map collision) + B3 (cand_records substring cost/fp) on dev-250.

No LLM calls. Reports:
  - how many questions have duplicate (sid, content) turns (B2 exposure)
  - cand_records substring match: record count vs candidate turns, est. pair checks
  - false-positive probe: records matched only via short-content containment
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.compiler.ir_extractor import UniversalIRExtractor

with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8") as f:
    all_items = json.load(f)
with open(REPO / "benchmark_results" / "splits" / "lme_dev_qids.json", encoding="utf-8") as f:
    dev_qids = set(json.load(f))
dev_items = [it for it in all_items if it["question_id"] in dev_qids]

extractor = UniversalIRExtractor()
b2_exposed = 0
b2_dup_turns = 0
fp_probe_hits = 0
fp_examples: list[str] = []
t0 = time.perf_counter()
pair_checks = 0
for item in dev_items[:50]:
    all_turns = []
    all_records = []
    dates = item.get("haystack_dates", [])
    sids = item.get("haystack_session_ids", [])
    for s_idx, session in enumerate(item.get("haystack_sessions", [])):
        s_date = dates[s_idx] if s_idx < len(dates) else ""
        sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
        for turn in session:
            speaker = turn.get("role", "user")
            content = turn.get("content", "")
            header = f"[{sid} on {s_date}]" if s_date else f"[{sid}]"
            line = f"{header} {speaker.capitalize()}: {content}"
            all_turns.append((sid, s_date, speaker, content))
            for r in extractor.extract(content, default_source=speaker):
                r.raw_content = line
                all_records.append(r)
    # B2: duplicate (sid, content)
    seen_keys: set[tuple[str, str]] = set()
    dups = 0
    for t in all_turns:
        k = (t[0], t[3])
        if k in seen_keys:
            dups += 1
        seen_keys.add(k)
    if dups:
        b2_exposed += 1
        b2_dup_turns += dups
    # B3: substring pair-check cost (candidate_turns ~ full haystack proxy)
    cand_texts = {c[3] for c in all_turns}
    pair_checks += len(all_records) * len(cand_texts)
    # FP probe: record matched only because a SHORT turn content is substring of raw_content
    for r in all_records:
        for c in cand_texts:
            if len(c) < 40 and c and c in (r.raw_content or ""):
                fp_probe_hits += 1
                if len(fp_examples) < 5:
                    fp_examples.append(f"{item['question_id']}: turn={c[:50]!r}")
                break
elapsed = time.perf_counter() - t0
print(f"items scanned: 50/{len(dev_items)} in {elapsed:.1f}s")
print(f"B2 exposure: {b2_exposed}/50 questions have duplicate (sid,content) turns "
      f"({b2_dup_turns} dup turns)")
print(f"B3 pair checks (records x cand_texts, 50q): {pair_checks}")
print(f"B3 short-content FP probe hits: {fp_probe_hits}")
for e in fp_examples:
    print(f"  ex: {e}")
