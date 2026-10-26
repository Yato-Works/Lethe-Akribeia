import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))
it = next(item for item in data if item['question_id'] == '8e91e7d9')

for s, sid in zip(it['haystack_sessions'], it['haystack_session_ids']):
    if sid in it.get('answer_session_ids', []):
        for t in s:
            c = t['content'].lower()
            if any(k in c for k in ['brother', 'sister', 'sibling']):
                print(f"[{sid}] {t['role']}: {t['content']}")
