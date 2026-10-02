import json
dataset = json.load(open('datasets/external/locomo10.json', encoding='utf-8'))
conv42 = next(s for s in dataset if s['sample_id'] == 'conv-42')
for k, v in conv42['conversation'].items():
    if 'date_time' in k:
        print(f"{k}: '{v}'")
