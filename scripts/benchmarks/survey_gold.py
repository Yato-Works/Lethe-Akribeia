"""Survey answer_* sessions for a qid (generalized)."""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json",
              encoding="utf-8") as f:
        items = json.load(f)
    item = next(x for x in items if x["question_id"] == args.qid)
    print(f"QID: {args.qid}\nQ: {item['question']}\nGT: {item.get('answer')}")
    sessions = list(zip(item.get("haystack_session_ids", []),
                        item.get("haystack_dates", []),
                        item.get("haystack_sessions", [])))
    ans = [(sid, d, s) for sid, d, s in sessions if sid.startswith("answer_")]
    print(f"\nTotal sessions: {len(sessions)}, answer_* sessions: {len(ans)}")
    for sid, date, sess in ans[: args.limit]:
        print(f"\n=== {sid} ({date}) : {len(sess)} turns ===")
        for t in sess:
            print(f"  [{t.get('role')}]: {t.get('content', '')[:400]}")


if __name__ == "__main__":
    main()
