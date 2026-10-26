import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"

with open(DATASET_PATH, encoding="utf-8") as f:
    data = json.load(f)

target_qids = {'f35224e0', '8979f9ec', 'edced276', 'a3332713', '36b9f61e', 'a1cc6108', '157a136e', 'c18a7dc8'}

for it in data:
    if it['question_id'] in target_qids:
        print(f"=== QID: {it['question_id']} ===")
        print(f"Question: {it['question']}")
        print(f"Answer:   {it.get('answer')}")
        print()
