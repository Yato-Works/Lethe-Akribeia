import json
data = json.load(open('datasets/external/longmemeval_s_cleaned.json', encoding='utf-8'))
it = next(item for item in data if item['question_id'] == '6456829e')
for s, sid in zip(it['haystack_sessions'], it['haystack_session_ids']):
    for t in s:
        if '26' in t['content'] and 'plant' in t['content'].lower():
            print(f"[{sid}] {t['role']}: {t['content'][:140]}")
