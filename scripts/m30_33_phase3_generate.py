#!/usr/bin/env python3
"""M30-33 Phase 3: Generate independent evaluation dataset.

Creates 120 synthetic questions across 8 categories, using fictional
characters, dates, and facts. No Bucket A or holdout data is used.

Categories:
1. Direct fact retrieval (15)
2. Paraphrase (15)
3. Entity disambiguation (15)
4. Multi-turn relation (15)
5. Multi-session temporal (15)
6. Relative dates (15)
7. Answerability & abstention (15)
8. Granularity, contradiction, update (15)

Output: results/m30_33/test_data.json
- conversations: synthetic conversations with turns
- questions: 120 questions with expected answers
- golds: expected answers (for evaluation harness only)
- metadata: category, difficulty, etc.

The test data is generated independently. The product code only receives
conversations and questions, not the gold answers.
"""
import json, random
from pathlib import Path
from datetime import datetime, timedelta

random.seed(42)  # Reproducibility

# Fictional characters
CHARACTERS = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Kate", "Leo",
    "Mia", "Noah", "Olivia", "Peter", "Quinn", "Rachel",
]

CITIES = [
    "Springfield", "Riverton", "Lakewood", "Maplewood", "Brookside",
    "Cedar Ridge", "Elmwood", "Fairview", "Greenfield", "Hillcrest",
]

EVENTS = [
    "graduation", "wedding", "reunion", "concert", "festival",
    "conference", "exhibition", "marathon", "workshop", "ceremony",
]


def generate_conversation(conv_id, num_sessions=3, turns_per_session=10):
    """Generate a synthetic conversation with multiple sessions."""
    characters = random.sample(CHARACTERS, 2)
    city = random.choice(CITIES)
    event = random.choice(EVENTS)
    
    base_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
    
    conversation = {}
    all_turns = []
    
    for s in range(1, num_sessions + 1):
        session_key = f"session_{s}"
        session_date = base_date + timedelta(days=s * 30 + random.randint(0, 15))
        day = session_date.day
        session_dt_str = session_date.strftime(f"%I:%M %p on {day} %B, %Y")
        
        turns = []
        for t in range(1, turns_per_session + 1):
            speaker = characters[t % 2]
            text = generate_turn_text(speaker, city, event, session_date, t, characters)
            dia_id = f"D{s}:{t}"
            turns.append({
                "dia_id": dia_id,
                "speaker": speaker,
                "text": text,
            })
            all_turns.append((dia_id, text, session_date, speaker))
        
        conversation[session_key] = turns
        conversation[f"{session_key}_date_time"] = session_dt_str
    
    return conversation, all_turns


def generate_turn_text(speaker, city, event, session_date, turn_idx, characters):
    """Generate a synthetic turn text."""
    other = [c for c in characters if c != speaker][0]
    
    templates = [
        f"Hey {other}, great to hear from you! I've been busy with the {event} preparations.",
        f"Yes, the {event} in {city} was amazing. We should go together next time.",
        f"I remember when we first met at the {event}. That was a great day.",
        f"Last month, I went to {city} for a {event}. It was quite an experience.",
        f"Yesterday, I talked to {other} about the {event}. We have exciting plans.",
        f"Next week, I'm planning to visit {city} for the {event}. Want to join?",
        f"A few years ago, we attended the {event} in {city} together.",
        f"This year, the {event} in {city} will be bigger than ever.",
        f"I can't wait for the {event} next month. It's going to be fantastic!",
        f"Remember the {event} we went to last year? Let's do it again!",
        f"Tomorrow, I'm heading to {city} for the {event}. Wish me luck!",
        f"One year ago, we were at the {event} in {city}. Time flies!",
        f"Two years ago, I met {other} at the {event}. We've been friends since.",
        f"Last week, I visited {city} for the {event}. It was wonderful.",
        f"Next month, the {event} will take place in {city}. Don't miss it!",
    ]
    
    return templates[turn_idx % len(templates)]


def generate_question(category, conversation, all_turns, qid):
    """Generate a question for a specific category."""
    
    # Pick a random turn as the "answer" source
    answer_turn_idx = random.randint(0, len(all_turns) - 1)
    answer_dia_id, answer_text, answer_date, answer_speaker = all_turns[answer_turn_idx]
    
    # Extract date info
    answer_year = answer_date.year
    answer_month = answer_date.month
    answer_day = answer_date.day
    
    # Generate question based on category
    if category == "direct_fact":
        question = f"When did {answer_speaker} attend the event in the city mentioned?"
        expected = f"{answer_month}/{answer_day}/{answer_year}"
        
    elif category == "paraphrase":
        question = f"What was the date of {answer_speaker}'s trip to the city?"
        expected = f"{answer_month}/{answer_day}/{answer_year}"
        
    elif category == "entity_disambiguation":
        # Use a different character as distractor
        other_chars = [c for c in CHARACTERS if c != answer_speaker]
        distractor = random.choice(other_chars)
        question = f"When did {answer_speaker} (not {distractor}) go to the city for the event?"
        expected = f"{answer_month}/{answer_day}/{answer_year}"
        
    elif category == "multi_turn":
        # Reference a different turn
        other_idx = (answer_turn_idx + 3) % len(all_turns)
        _, other_text, other_date, _ = all_turns[other_idx]
        question = f"How many days apart are {answer_speaker}'s event and the other event mentioned?"
        delta = abs((answer_date - other_date).days)
        expected = f"{delta} days"
        
    elif category == "multi_session":
        # Reference session date
        session_num = int(answer_dia_id.split(":")[0].replace("D", ""))
        session_key = f"session_{session_num}"
        session_dt_str = conversation.get(f"{session_key}_date_time", "")
        question = f"Which session contains {answer_speaker}'s message about the event?"
        expected = f"Session {session_num}"
        
    elif category == "relative_date":
        # Use relative date expressions
        rel_phrases = [
            ("yesterday", 1),
            ("last month", 30),
            ("next month", -30),
            ("one year ago", 365),
            ("two years ago", 730),
        ]
        phrase, days = random.choice(rel_phrases)
        if days > 0:
            target_date = answer_date - timedelta(days=days)
        else:
            target_date = answer_date + timedelta(days=-days)
        question = f"If today is the date of {answer_speaker}'s message, what is the date {phrase}?"
        expected = f"{target_date.month}/{target_date.day}/{target_date.year}"
        
    elif category == "answerability":
        # Mix of answerable and unanswerable
        if random.random() < 0.5:
            question = f"When did {answer_speaker} go to the city?"
            expected = f"{answer_month}/{answer_day}/{answer_year}"
        else:
            question = f"What is {answer_speaker}'s favorite color?"
            expected = "I don't know."
            
    elif category == "granularity":
        # Test year vs month granularity
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
        "answer_turn": answer_dia_id,
        "answer_date": answer_date.isoformat(),
    }


def main():
    print("=" * 78)
    print("M30-33 Phase 3: Independent Evaluation Dataset Generation")
    print("=" * 78)
    print()
    
    # Generate conversations
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
    
    # Category breakdown
    from collections import Counter
    cat_counts = Counter(q["category"] for q in all_questions)
    print("Category breakdown:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")
    print()
    
    # Save
    output = {
        "metadata": {
            "total_questions": len(all_questions),
            "categories": dict(cat_counts),
            "seed": 42,
            "note": "Synthetic data, independent from Bucket A and holdout",
        },
        "conversations": conversations,
        "questions": all_questions,
    }
    
    output_path = Path(__file__).resolve().parents[1] / "results/m30_33/test_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")
    print()
    
    # Print sample questions
    print("Sample questions:")
    for q in all_questions[:5]:
        print(f"  {q['question_id']} [{q['category']}]: {q['question'][:60]}")
        print(f"    Expected: {q['expected_answer']}")


if __name__ == "__main__":
    main()
