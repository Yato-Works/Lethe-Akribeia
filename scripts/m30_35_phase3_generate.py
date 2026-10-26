#!/usr/bin/env python3
"""M30-35 Phase 3: Generate new independent evaluation set (120 questions, 8 categories).

Uses entirely new fictional characters, cities, events, and dates.
Does NOT reuse any content from M30-33 or M30-34.
"""
import json, random, re, sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
random.seed(2025)

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}

# New fictional characters (not used in M30-33/34)
CHARACTERS = [
    "Liam", "Emma", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "William", "Mia", "James", "Charlotte", "Benjamin", "Amelia",
    "Lucas", "Harper", "Henry", "Evelyn", "Alexander",
]

# New cities
CITIES = [
    "Willow Creek", "Oak Valley", "Pine Ridge", "Cedar Falls", "Birchwood",
    "Maple Grove", "Elm Creek", "Aspen Hill", "Redwood", "Sycamore",
]

# New events
EVENTS = [
    "art exhibition", "book launch", "charity gala", "tech summit", "food festival",
    "film premiere", "science fair", "music recital", "sports tournament", "trade show",
]


def generate_conversation(conv_id, num_sessions=3, turns_per_session=12):
    """Generate synthetic conversation with explicit timestamps."""
    characters = random.sample(CHARACTERS, 2)
    city = random.choice(CITIES)
    event = random.choice(EVENTS)
    
    base_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))
    
    conversation = {}
    all_turns = []
    
    for s in range(1, num_sessions + 1):
        session_key = f"session_{s}"
        session_date = base_date + timedelta(days=s * 25 + random.randint(0, 10))
        day = session_date.day
        session_dt_str = session_date.strftime(f"%I:%M %p on {day} %B, %Y")
        
        turns = []
        for t in range(1, turns_per_session + 1):
            speaker = characters[t % 2]
            text = generate_turn(speaker, city, event, session_date, t, characters)
            dia_id = f"D{s}:{t}"
            turns.append({
                "dia_id": dia_id,
                "speaker": speaker,
                "text": text,
            })
            all_turns.append({
                "dia_id": dia_id,
                "text": text,
                "session_date": session_date,
                "session_key": session_key,
                "speaker": speaker,
            })
        
        conversation[session_key] = turns
        conversation[f"{session_key}_date_time"] = session_dt_str
    
    return conversation, all_turns


def generate_turn(speaker, city, event, session_date, turn_idx, characters):
    """Generate a turn with explicit date reference."""
    other = [c for c in characters if c != speaker][0]
    month_name = MONTH_NAMES.get(session_date.month, "")
    day = session_date.day
    year = session_date.year
    
    templates = [
        f"Hey {other}, I've been preparing for the {event} in {city} on {month_name} {day}, {year}.",
        f"Yes, the {event} in {city} is scheduled for {month_name} {day}, {year}. We should attend together.",
        f"I remember attending the {event} in {city} on {month_name} {day}, {year}. It was wonderful.",
        f"On {month_name} {day}, {year}, I visited {city} for the {event}. Great experience.",
        f"Last week, I went to {city} for the {event} on {month_name} {day}, {year}.",
        f"Next month, the {event} will be in {city} on {month_name} {day}, {year}.",
        f"A few years ago, we attended the {event} in {city} together.",
        f"This year, the {event} in {city} is on {month_name} {day}, {year}.",
        f"I can't wait for the {event} next month in {city}.",
        f"Remember the {event} we went to last year in {city}?",
        f"Tomorrow, I'm heading to {city} for the {event}.",
        f"One year ago, we were at the {event} in {city}.",
        f"Two years ago, I met {other} at the {event} in {city}.",
        f"Last week, I visited {city} for the {event}.",
        f"Next month, the {event} will take place in {city}.",
        f"Yesterday, I talked to {other} about the {event} in {city}.",
    ]
    
    return templates[turn_idx % len(templates)]


def generate_question(category, conversation, all_turns, qid):
    """Generate a question with deterministic expected answer."""
    
    answer_turn_idx = random.randint(0, len(all_turns) - 1)
    answer_turn = all_turns[answer_turn_idx]
    answer_date = answer_turn["session_date"]
    answer_year = answer_date.year
    answer_month = answer_date.month
    answer_day = answer_date.day
    answer_speaker = answer_turn["speaker"]
    
    if category == "direct_fact":
        question = f"On what date did {answer_speaker} attend the event?"
        expected = f"{answer_month}/{answer_day}/{answer_year}"
        
    elif category == "paraphrase":
        question = f"What was the date of {answer_speaker}'s visit to the city for the event?"
        expected = f"{answer_month}/{answer_day}/{answer_year}"
        
    elif category == "entity_disambiguation":
        other_chars = [c for c in CHARACTERS if c != answer_speaker]
        distractor = random.choice(other_chars)
        question = f"On what date did {answer_speaker} (not {distractor}) attend the event?"
        expected = f"{answer_month}/{answer_day}/{answer_year}"
        
    elif category == "multi_turn":
        other_idx = (answer_turn_idx + 5) % len(all_turns)
        other_turn = all_turns[other_idx]
        other_date = other_turn["session_date"]
        delta = abs((answer_date - other_date).days)
        question = f"How many days apart are the two events mentioned?"
        expected = f"{delta} days"
        
    elif category == "multi_session":
        session_num = int(answer_turn["dia_id"].split(":")[0].replace("D", ""))
        question = f"Which session contains the message about the event?"
        expected = f"Session {session_num}"
        
    elif category == "relative_date":
        rel_phrases = [
            ("yesterday", 1),
            ("tomorrow", -1),
            ("last month", 30),
            ("next month", -30),
            ("one year ago", 365),
        ]
        phrase, days = random.choice(rel_phrases)
        if days > 0:
            target_date = answer_date - timedelta(days=days)
        else:
            target_date = answer_date + timedelta(days=-days)
        question = f"If today is {answer_month}/{answer_day}/{answer_year}, what is the date {phrase}?"
        expected = f"{target_date.month}/{target_date.day}/{target_date.year}"
        
    elif category == "answerability":
        if random.random() < 0.5:
            question = f"On what date did {answer_speaker} attend the event?"
            expected = f"{answer_month}/{answer_day}/{answer_year}"
        else:
            question = f"What is {answer_speaker}'s favorite food?"
            expected = "I don't know."
            
    elif category == "granularity":
        if random.random() < 0.5:
            question = f"In which year did {answer_speaker} attend the event?"
            expected = f"{answer_year}"
        else:
            question = f"In which month and year did {answer_speaker} attend the event?"
            expected = f"{answer_month}/{answer_year}"
    
    return {
        "question_id": qid,
        "category": category,
        "question": question,
        "expected_answer": expected,
        "answer_turn": answer_turn["dia_id"],
        "answer_date": answer_date.isoformat(),
    }


def validate_data(questions, conversations):
    """Validate all questions and conversations."""
    errors = []
    
    for q in questions:
        conv_id = q["conversation_id"]
        if conv_id not in conversations:
            errors.append(f"{q['question_id']}: conversation {conv_id} not found")
            continue
        
        conv = conversations[conv_id]
        answer_dia_id = q["answer_turn"]
        
        found = False
        for key in conv:
            if key.endswith("_date_time") or not isinstance(conv[key], list):
                continue
            for turn in conv[key]:
                if turn["dia_id"] == answer_dia_id:
                    found = True
                    break
        if not found:
            errors.append(f"{q['question_id']}: answer turn {answer_dia_id} not found")
    
    return errors


def main():
    print("=" * 78)
    print("M30-35 Phase 3: New Independent Evaluation Set")
    print("=" * 78)
    print()
    
    conversations = {}
    all_questions = []
    
    categories = [
        "direct_fact",
        "paraphrase",
        "entity_disambiguation",
        "multi_turn",
        "multi_session",
        "relative_date",
        "answerability",
        "granularity",
    ]
    
    qid_counter = 0
    
    for cat in categories:
        for i in range(15):
            conv_id = f"conv_{cat}_{i}"
            conversation, all_turns = generate_conversation(conv_id)
            conversations[conv_id] = conversation
            
            qid = f"q_{qid_counter:04d}"
            qid_counter += 1
            
            question = generate_question(cat, conversation, all_turns, qid)
            question["conversation_id"] = conv_id
            all_questions.append(question)
    
    print(f"Generated {len(conversations)} conversations")
    print(f"Generated {len(all_questions)} questions")
    print()
    
    # Validate
    errors = validate_data(all_questions, conversations)
    if errors:
        print(f"VALIDATION ERRORS: {len(errors)}")
        for e in errors[:10]:
            print(f"  {e}")
        return
    
    print("Validation: ALL PASSED")
    print()
    
    # Category breakdown
    from collections import Counter
    cat_counts = Counter(q["category"] for q in all_questions)
    print("Category breakdown:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")
    print(f"  Total: {len(all_questions)}")
    print()
    
    # Save
    output = {
        "metadata": {
            "total_questions": len(all_questions),
            "categories": dict(cat_counts),
            "seed": 2025,
            "note": "New synthetic data, independent from M30-33/34",
        },
        "conversations": conversations,
        "questions": all_questions,
    }
    
    output_path = PROJECT_ROOT / "results/m30_35/test_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")
    print()
    print("DATASET FROZEN")


if __name__ == "__main__":
    main()
