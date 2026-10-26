import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]

data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))
check_qids = ['b5ef892d', '1f2b8d4f', '2311e44b', 'a346bb18', '8e91e7d9', '6456829e', 'gpt4_15e38248', '4f54b7c9']

for it in data:
    qid = it['question_id']
    if qid in check_qids:
        print("=" * 80)
        print(f"[{qid}] Q: {it['question']}")
        print(f"       GT: {it['answer']}")
        print(f"       Gold SIDs: {it.get('answer_session_ids')}")
        for s, sid in zip(it['haystack_sessions'], it['haystack_session_ids']):
            if sid in it.get('answer_session_ids', []):
                print(f"  --- {sid} ---")
                for t in s:
                    if t['role'] == 'user':
                        print(f"    USER: {t['content'][:140]}")
