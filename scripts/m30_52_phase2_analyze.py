#!/usr/bin/env python3
"""M30-52 Phase 2: Analyze the 8 remaining wrong cases to understand binding failures."""
import json, re, sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory_gen.recall.evidence_scorer import UniversalEvidenceScorer

STOP = set("the and for are was were did does has have had will would could should can may might "
           "must this that these those when where what which who why how not all any both each "
           "with from into about after before during between through over under".split())

scorer = UniversalEvidenceScorer()
MONTHS = {m.lower(): i for i, m in enumerate([
    "january","february","march","april","may","june","july",
    "august","september","october","november","december"], start=1)}
TURN_RE = re.compile(r"(?:\[SPEAKER:[^\]]+\]\s*)?\[(D\d+:\d+)\s+on\s+([^\]]+)\](.*)")

def parse_header_date(hdr):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", hdr or "")
    if m and m.group(2).lower() in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None

prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
baseline = {r["qid"]: r for r in prod["rows"]}
locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
sids = [c.get("sample_id", "") for c in locomo]
adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))

qid_to_conv = {}
conv_ids_needed = set()
for qid in baseline:
    m = re.match(r"lc_conv-(\d+)_(\d+)", qid)
    if not m: continue
    conv = f"conv-{m.group(1)}"
    conv_ids_needed.add(conv)
    qid_to_conv[qid] = f"{conv}-qa-{int(m.group(2)):03d}"

ctx_cache = {}
for idx, sid in enumerate(sids):
    if sid not in conv_ids_needed: continue
    turns, questions, ir_records = adapter.load_conversation(idx)
    for q in questions:
        cqid = q.question_id
        if cqid in qid_to_conv.values():
            pcc = adapter.compiler.compile(q.question, ir_records)
            ctx_cache[cqid] = (q.question, pcc.context_text, q.category)

WRONG_CASES = ["lc_conv-26_63", "lc_conv-41_20", "lc_conv-42_57", "lc_conv-47_43",
               "lc_conv-47_65", "lc_conv-48_12", "lc_conv-50_52", "lc_conv-41_10",
               "lc_conv-44_13", "lc_conv-47_48", "lc_conv-48_50", "lc_conv-49_59",
               "lc_conv-50_36"]

for qid in WRONG_CASES:
    if qid not in baseline: continue
    bp = baseline[qid]
    cqid = qid_to_conv.get(qid)
    if not cqid or cqid not in ctx_cache: continue
    question, ctx, cat = ctx_cache[cqid]
    gold = bp["gold"]
    raw = bp.get("predicted", "")

    correct = LoCoMoAdapter.score_binary(cat, gold, raw)
    if correct: continue  # already correct, skip

    print(f"\n{'='*70}")
    print(f"{qid} | gold={gold[:40]}")
    print(f"  Q: {question}")
    print(f"  Reader: {raw[:80]!r}")

    # Show all turns with relative expressions
    print(f"\n  Turns with relative expressions:")
    candidates = []
    for m in TURN_RE.finditer(ctx):
        dia, hdr, body = m.group(1), m.group(2), m.group(3)
        rels = re.findall(r"\b(yesterday|tomorrow|the next day|the day before|next month|last month|next week|last week|currently)\b", body, re.I)
        if not rels: continue
        base = parse_header_date(hdr)
        if not base: continue
        body_lower = body.lower()

        # Show what noun overlap scores
        qwords = set(w for w in re.findall(r"\b[a-z]{4,}\b", question.lower()) if w not in STOP and w not in {"when","what","which"})
        # Find question content words that are also known actors
        qkw = re.sub(r'\b(when|did|do|does|is|are|was|were|will|would|could|should|can|may|might)\b','',question,flags=re.I).lower()
        qkw = re.sub(r"[^a-z\s]", "", qkw).strip()
        words = re.findall(r"\b[a-z]{4,}\b", qkw)
        words = [w for w in words if w not in STOP]

        actor_match = sum(1 for a in scorer.KNOWN_ACTORS if f" {a} " in f" {body_lower} ")
        content_match = sum(1 for w in words if w in body_lower)
        score_v1 = actor_match * 2 + content_match

        # Extract explicit date
        date_match = re.search(r"\(?\s*(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\s*\)?", body)
        explicit = f"{date_match.group(1)} {date_match.group(2)}, {date_match.group(3)}" if date_match else None

        print(f"  {dia} | base={base.strftime('%Y-%m-%d')} | rel={rels[0]:12s} | actors={actor_match} | content={content_match} | v1score={score_v1:3d} | explicit={explicit}")
        print(f"       body: {body[:100]}...")
        candidates.append((dia, score_v1, rels[0], base, explicit, actor_match, content_match))

    # Show which turn v1 picks (highest content_match, then actor_match)
    candidates.sort(key=lambda c: (-c[1],))
    if candidates:
        best = candidates[0]
        print(f"\n  V1 picks: {best[0]} (score={best[1]}, rel={best[2]}) -> would resolve to: {best[4] or best[3]}")
        print(f"  CORRECT: {gold == (best[4] or str(best[3].strftime('%d %B, %Y')))}")
