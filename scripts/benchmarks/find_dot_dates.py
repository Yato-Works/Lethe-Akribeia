import json
import re

dataset = json.load(open('datasets/external/locomo10.json', encoding='utf-8'))
for sample in dataset:
    for i, qa in enumerate(sample.get('qa', [])):
        ans = str(qa.get('answer', ''))
        # Look for unusual punctuation or glued words in dates
        if qa.get('category') == 2:
            if re.search(r'\d+[A-Za-z]+', ans) or re.search(r'[A-Za-z]+\.\d+', ans):
                qid = f"{sample['sample_id']}-qa-{i:03d}"
                print(f"  [{qid}] Q: {qa.get('question')}")
                print(f"      GT: '{ans}'")
