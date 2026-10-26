#!/usr/bin/env python3
"""M30-37 Phase 4: Generate Development (40) + Frozen Evaluation (120) sets.

New characters, cities, events, dates - independent from all prior sets.
Each category: Development 5, Frozen 15.

Key design: questions use DIFFERENT vocabulary than answer turns,
so retrieval must handle paraphrase (the identified failure mode).
"""
import json, random, re
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
random.seed(7777)

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}

# Completely new character set
CHARACTERS = [
    "Aiden", "Bella", "Caleb", "Diana", "Eli", "Fiona", "Gavin", "Hazel",
    "Ivan", "Julia", "Kevin", "Luna", "Marcus", "Nora", "Oscar", "Paula",
    "Quentin", "Rosa", "Silas", "Tara", "Ursula", "Victor", "Wendy", "Xavier",
]

CITIES = [
    "Northgate", "Southport", "Eastbrook", "Westfield", "Midtown",
    "Harborview", "Riverside", "Stonebridge", "Clearwater", "Sunnyvale",
]

EVENTS = [
    "poetry reading", "art auction", "jazz night", "wine tasting", "comedy show",
    "craft fair", "dance recital", "film screening", "photo exhibit", "garden tour",
]

# Question vocabulary that DIFFERS from turn vocabulary (paraphrase challenge)
QUESTION_TEMPLATES = {
    "direct_fact": [
        "On what date did {name} go to the {event}?",
        "When did {name} attend the {event}?",
        "What date was {name}'s {event}?",
    ],
    "paraphrase": [
        "When did {name} make the trip for the {event}?",
        "What day did {name} show up for the {event}?",
        "On which date was {name} present at the {event}?",
    ],
    "entity_disambiguation": [
        "On what date did {name} (not {distractor}) go to the {event}?",
        "When did {name} rather than {distractor} attend the {event}?",
    ],
    "multi_turn": [
        "How many days apart are the two {event}s {name} mentioned?",
        "What is the gap in days between {name}'s two {event}s?",
    ],
    "multi_session": [
        "Which session has {name}'s message about the {event}?",
        "In which session did {name} discuss the {event}?",
    ],
    "relative_date": [
        "If today is {ref}, what is the date {phrase}?",
        "Given today is {ref}, what date is {phrase}?",
    ],
    "answerability": [
        "On what date did {name} go to the {event}?",
        "What is {name}'s favorite hobby?",
    ],
    "granularity": [
        "In which year did {name} attend the {event}?",
        "In which month and year did {name} attend the {event}?",
    ],
}

# Turn vocabulary that DIFFERS from question vocabulary
TURN_TEMPLATES = [
    "Hey {other}, I finally made it to the {event} in {city} on {date}. Such a memorable time.",
    "Yes, the {event} in {city} happened on {date}. You should have been there.",
    "I was at the {event} in {city} on {date}. It exceeded all my expectations.",
    "On {date}, I showed up at the {event} in {city}. What an evening.",
    "The {event} in {city} took place on {date}. I'm still thinking about it.",
    "{date} was the day of the {event} in {city}. Great memories.",
    "I recall the {event} in {city} was on {date}. Time really flies.",
    "We went to the {event} in {city} on {date}. It was worth the trip.",
    "The {event} in {city} on {date} was fantastic. You missed out.",
    "I attended the {event} in {city} on {date}. Beautiful experience.",
]


def generate_conversation(num_sessions=3, turns_per_session=12):
    characters = random.sample(CHARACTERS, 2)
    city = random.choice(CITIES)
    event = random.choice(EVENTS)
    
    base_date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 300))
    
    conversation = {}
    all_turns = []
    
    for s in range(1, num_sessions + 1):
        session_key = f"session_{s}"
        session_date = base_date + timedelta(days=s * 22 + random.randint(0, 8))
        day = session_date.day
        session_dt_str = session_date.strftime(f"%I:%M %p on {day} %B, %Y")
        
        turns = []
        for t in range(1, turns_per_session + 1):
            speaker = characters[t % 2]
            other = [c for c in characters if c != speaker][0]
            month_name = MONTH_NAMES.get(session_date.month, "")
            date_str = f"{month_name} {day}, {session_date.year}"
            template = TURN_TEMPLATES[t % len(TURN_TEMPLATES)]
            text = template.format(other=other, event=event, city=city, date=date_str)
            
            dia_id = f"D{s}:{t}"
            turns.append({"dia_id": dia_id, "speaker": speaker, "text": text})
            all_turns.append({
                "dia_id": dia_id, "text": text, "session_date": session_date,
                "session_key": session_key, "speaker": speaker,
            })
        
        conversation[session_key] = turns
        conversation[f"{session_key}_date_time"] = session_dt_str
    
    return conversation, all_turns, city, event


def generate_question(category, conversation, all_turns, qid, city, event):
    answer_turn = random.choice(all_turns)
    answer_date = answer_turn["session_date"]
    y, mo, d = answer_date.year, answer_date.month, answer_date.day
    speaker = answer_turn["speaker"]
    
    if category == "direct_fact":
        q = f"On what date did {speaker} go to the {event}?"
        exp = f"{mo}/{d}/{y}"
    elif category == "paraphrase":
        q = f"When did {speaker} make the trip for the {event}?"
        exp = f"{mo}/{d}/{y}"
    elif category == "entity_disambiguation":
        other = random.choice([c for c in CHARACTERS if c != speaker])
        q = f"On what date did {speaker} (not {other}) go to the {event}?"
        exp = f"{mo}/{d}/{y}"
    elif category == "multi_turn":
        other_turn = random.choice(all_turns)
        delta = abs((answer_date - other_turn["session_date"]).days)
        q = f"How many days apart are the two events {speaker} mentioned?"
        exp = f"{delta} days"
    elif category == "multi_session":
        session_num = int(answer_turn["dia_id"].split(":")[0].replace("D", ""))
        q = f"Which session has {speaker}'s message about the {event}?"
        exp = f"Session {session_num}"
    elif category == "relative_date":
        phrase, days = random.choice([("yesterday", 1), ("tomorrow", -1), ("last month", 30), ("next month", -30), ("one year ago", 365)])
        target = answer_date - timedelta(days=days) if days > 0 else answer_date + timedelta(days=-days)
        q = f"If today is {mo}/{d}/{y}, what is the date {phrase}?"
        exp = f"{target.month}/{target.day}/{target.year}"
    elif category == "answerability":
        if random.random() < 0.5:
            q = f"On what date did {speaker} go to the {event}?"
            exp = f"{mo}/{d}/{y}"
        else:
            q = f"What is {speaker}'s favorite hobby?"
            exp = "I don't know."
    elif category == "granularity":
        if random.random() < 0.5:
            q = f"In which year did {speaker} attend the {event}?"
            exp = f"{y}"
        else:
            q = f"In which month and year did {speaker} attend the {event}?"
            exp = f"{mo}/{y}"
    
    return {
        "question_id": qid,
        "category": category,
        "question": q,
        "expected_answer": exp,
        "answer_turn": answer_turn["dia_id"],
        "answer_date": answer_date.isoformat(),
    }


def build_set(n_per_category, prefix, qid_start=0):
    categories = ["direct_fact", "paraphrase", "entity_disambiguation", "multi_turn",
                  "multi_session", "relative_date", "answerability", "granularity"]
    conversations = {}
    questions = []
    qid = qid_start
    
    for cat in categories:
        for i in range(n_per_category):
            conv_id = f"{prefix}_{cat}_{i}"
            conversation, all_turns, city, event = generate_conversation()
            conversations[conv_id] = conversation
            q = generate_question(cat, conversation, all_turns, f"{prefix}_q{qid:04d}", city, event)
            q["conversation_id"] = conv_id
            questions.append(q)
            qid += 1
    
    return conversations, questions


def main():
    print("=" * 78)
    print("M30-37 Phase 4: Generate Development + Frozen Evaluation Sets")
    print("=" * 78)
    print()
    
    dev_conv, dev_q = build_set(5, "dev", 0)
    frozen_conv, frozen_q = build_set(15, "frozen", 1000)
    
    print(f"Development: {len(dev_q)} questions, {len(dev_conv)} conversations")
    print(f"Frozen Evaluation: {len(frozen_q)} questions, {len(frozen_conv)} conversations")
    print()
    
    from collections import Counter
    print("Development categories:")
    for cat, count in sorted(Counter(q["category"] for q in dev_q).items()):
        print(f"  {cat}: {count}")
    print()
    print("Frozen categories:")
    for cat, count in sorted(Counter(q["category"] for q in frozen_q).items()):
        print(f"  {cat}: {count}")
    print()
    
    out_dir = PROJECT_ROOT / "results/m30_37"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    (out_dir / "dev_data.json").write_text(json.dumps({
        "metadata": {"total": len(dev_q), "seed": 7777},
        "conversations": dev_conv, "questions": dev_q,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    
    (out_dir / "frozen_data.json").write_text(json.dumps({
        "metadata": {"total": len(frozen_q), "seed": 7777},
        "conversations": frozen_conv, "questions": frozen_q,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"Saved: {out_dir / 'dev_data.json'}")
    print(f"Saved: {out_dir / 'frozen_data.json'}")
    print()
    print("FROZEN SET LOCKED - do not tune on this data")


if __name__ == "__main__":
    main()
