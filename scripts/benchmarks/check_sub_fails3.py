import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]

data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))

def inspect_q(qid):
    it = next(item for item in data if item["question_id"] == qid)
    print("=" * 80)
    print(f"[{qid}] Q: {it['question']}")
    print(f"       GT: {it['answer']}")
    print(f"       Gold SIDs: {it.get('answer_session_ids')}")
    for s, sid in zip(it['haystack_sessions'], it['haystack_session_ids']):
        if sid in it.get('answer_session_ids', []):
            for t in s:
                if t['role'] == 'user':
                    print(f"  [{sid}] USER: {t['content']}")

inspect_q('8e91e7d9')
inspect_q('b5ef892d')
inspect_q('2b8f3739')
