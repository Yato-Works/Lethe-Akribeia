import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.compiler.samplers.per_session_sampler import PerSessionSampler

sampler = PerSessionSampler()
q = "How many items of clothing do I need to pick up or return from a store?"
terms = sampler.extract_query_terms(q)
print("Query terms:", sorted(list(terms)))

data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))
it = next(item for item in data if item["question_id"] == "0a995998")

for s, sid in zip(it["haystack_sessions"], it["haystack_session_ids"]):
    if sid == "answer_afa9873b_2":
        for idx, turn in enumerate(s):
            if turn["role"] == "user":
                sc, matched = sampler.score_turn(turn["content"], terms, speaker="user")
                print(f"[Turn {idx+1}] Score={sc:.1f} | Matched={matched} | Content: {turn['content'][:90]}")
