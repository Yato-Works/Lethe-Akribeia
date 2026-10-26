import json
from artificial_memory_gen.steroid.wide_slicer import WideSlicer
from artificial_memory_gen.compiler.ir_extractor import UniversalIRExtractor

with open("datasets/external/longmemeval_s_cleaned.json", encoding="utf-8") as f:
    items = json.load(f)

it = next(x for x in items if x["question_id"] == "51a45a95")
extractor = UniversalIRExtractor()
wide_slicer = WideSlicer(per_channel_budget=12)

q = it["question"]
q_date = it.get("question_date", "")

all_records = []
dates = it["haystack_dates"]
sids = it["haystack_session_ids"]

for s_idx, session in enumerate(it["haystack_sessions"]):
    s_date = dates[s_idx] if s_idx < len(dates) else ""
    sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
    for turn in session:
        speaker = turn.get("role", "user")
        content = turn.get("content", "")
        header = f"[{sid} on {s_date}]" if s_date else f"[{sid}]"
        line = f"{header} {speaker.capitalize()}: {content}"
        recs = extractor.extract(content, default_source=speaker)
        for r in recs:
            r.raw_content = line
            r.time_scope = s_date
            all_records.append(r)

slice_res = wide_slicer.slice(q, all_records, reference_date_str=q_date)
cand_records = slice_res.candidate_records

print("Question:", q)
print("Top 10 candidate records from WideSlicer:")
for i, r in enumerate(cand_records[:10]):
    print(f"{i+1}. [{r.time_scope}] {r.raw_content[:100]} | ent={r.entity}, prop={r.property}, val={r.value}")
