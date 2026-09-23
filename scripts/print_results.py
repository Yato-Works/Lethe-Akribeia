#!/usr/bin/env python3
"""Print open-domain results."""
import json, sys
with open(sys.argv[1]) as f:
    for l in f:
        d = json.loads(l)
        qid = d.get('question_id', '').split('-')[-1]
        print(f"Q{qid}: pred='{d.get('prediction','')[:50]}' gt='{d.get('ground_truth','')[:50]}' correct={d.get('correct')}")
