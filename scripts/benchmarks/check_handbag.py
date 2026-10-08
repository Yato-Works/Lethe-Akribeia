import json
from artificial_memory.steroid.wide_slicer import WideSlicer
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor

with open("datasets/external/longmemeval_s_cleaned.json", encoding="utf-8") as f:
    items = json.load(f)

it = next(x for x in items if x["question_id"] == "7527f7e2")
extractor = UniversalIRExtractor()
wide_slicer = WideSlicer(per_channel_budget=12)

q = it["question"]
q_date = it.get("question_date", "")

all_records = []
for s_idx, session in enumerate(it["haystack_sessions"]):
    s_date = it["haystack_dates"][s_idx]
    sid = it["haystack_session_ids"][s_idx]
    for turn in session:
        line = f"[{sid} on {s_date}] {turn['role'].capitalize()}: {turn['content']}"
        recs = extractor.extract(turn["content"], default_source=turn["role"])
        for r in recs:
            r.raw_content = line
            r.time_scope = s_date
            all_records.append(r)

slice_res = wide_slicer.slice(q, all_records, reference_date_str=q_date)
cand_records = slice_res.candidate_records

print("Q:", q)
print("Answer sessions:", it["answer_session_ids"])
print("Records count:", len(cand_records))
found = [r for r in cand_records if any(sid in r.raw_content for sid in it["answer_session_ids"])]
print("Gold records found:", len(found))
for r in found:
    print("  ", r.raw_content[:100])
