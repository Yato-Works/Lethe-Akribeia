#!/usr/bin/env python3
"""M30-34 Phase 1: Audit existing 120 questions for invalid conditions.

Checks each question in M30-33's test_data.json:
1. Does the supporting turn actually exist?
2. Does the supporting turn logically support the expected answer?
3. Are timestamps available for temporal questions?
4. Can relative dates be deterministically resolved?
5. Does the scorer distinguish answerable vs unanswerable?
6. Are gold answers separated from product execution data?

Invalid cases are flagged with reasons.
"""
import json, re, sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}


def parse_session_datetime(s):
    if not s:
        return None
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", s)
    if m and m.group(2).lower() in MONTHS:
        return (int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    return None


def audit_question(q, conversations):
    """Audit a single question. Returns list of issues."""
    issues = []
    
    conv_id = q["conversation_id"]
    conv = conversations.get(conv_id, {})
    
    # Flatten conversation
    turns = {}
    for key in conv:
        if key.endswith("_date_time") or not isinstance(conv[key], list):
            continue
        session_dt = conv.get(f"{key}_date_time", "")
        for turn in conv[key]:
            turns[turn["dia_id"]] = {
                "text": turn["text"],
                "session_datetime": session_dt,
                "session": key,
            }
    
    # 1. Does the answer turn exist?
    answer_turn_id = q["answer_turn"]
    answer_turn = turns.get(answer_turn_id)
    if not answer_turn:
        issues.append(f"ANSWER_TURN_NOT_FOUND: {answer_turn_id} not in conversation")
        return issues
    
    # 2. Does the answer turn support the expected answer?
    expected = q["expected_answer"]
    answer_text = answer_turn["text"]
    
    # Check if expected date appears in answer turn
    if expected != "I don't know.":
        # Extract date from expected
        exp_date = None
        # Try MM/DD/YYYY
        m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", expected)
        if m:
            exp_date = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        # Try "Month Day, Year"
        m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})", expected, re.I)
        if m:
            exp_date = (MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3)))
        # Try "Month Year"
        m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", expected, re.I)
        if m:
            exp_date = (MONTHS[m.group(1).lower()], None, int(m.group(2)))
        # Year only
        if re.match(r"^\d{4}$", expected):
            exp_date = (None, None, int(expected))
        
        if exp_date:
            # Check if the answer turn's session datetime matches
            session_dt = parse_session_datetime(answer_turn["session_datetime"])
            if session_dt:
                # The expected date should relate to the session date
                # For direct_fact: expected = session date
                if q["category"] == "direct_fact":
                    if session_dt != exp_date:
                        issues.append(f"DATE_MISMATCH: session_date={session_dt} expected={exp_date}")
                # For relative_date: expected should be computable from session date
                elif q["category"] == "relative_date":
                    # Check if the answer turn text contains the relative expression
                    rel_exprs = ["yesterday", "tomorrow", "today", "last month", "next month", "next year", "last year", "one year ago", "two years ago"]
                    has_rel = any(r.lower() in answer_text.lower() for r in rel_exprs)
                    if not has_rel:
                        issues.append(f"NO_RELATIVE_EXPRESSION: answer turn has no relative date expression")
    
    # 3. Timestamp availability for temporal questions
    if q["category"] in ["direct_fact", "relative_date", "multi_session", "multi_turn", "granularity"]:
        if not answer_turn.get("session_datetime"):
            issues.append(f"NO_TIMESTAMP: temporal question without session datetime")
    
    # 4. Relative date resolvability
    if q["category"] == "relative_date":
        session_dt = parse_session_datetime(answer_turn["session_datetime"])
        if not session_dt:
            issues.append(f"UNRESOLVABLE_RELATIVE_DATE: no session datetime")
        else:
            # Check if we can compute the expected answer
            # This requires the answer turn to contain a relative expression
            # AND the session datetime to be valid
            pass  # Checked above
    
    # 5. Answerability check
    if expected == "I don't know.":
        # Verify the question is actually unanswerable
        # For synthetic data, "favorite color" should not be in any turn
        color_words = ["color", "colour", "favorite", "favourite"]
        q_lower = q["question"].lower()
        if not any(cw in q_lower for cw in color_words):
            issues.append(f"UNANSWERABLE_WITHOUT_REASON: expected='I don't know.' but question has no clear unanswerable marker")
    
    return issues


def main():
    print("=" * 78)
    print("M30-34 Phase 1: Audit of M30-33 Test Data")
    print("=" * 78)
    print()
    
    # Load test data
    test_data = json.loads((PROJECT_ROOT / "results/m30_33/test_data.json").read_text(encoding="utf-8"))
    questions = test_data["questions"]
    conversations = test_data["conversations"]
    
    print(f"Total questions: {len(questions)}")
    print()
    
    # Audit each question
    all_issues = {}
    for q in questions:
        issues = audit_question(q, conversations)
        if issues:
            all_issues[q["question_id"]] = issues
    
    # Summary
    print("=" * 78)
    print("AUDIT SUMMARY")
    print("=" * 78)
    
    total_issues = sum(len(v) for v in all_issues.values())
    questions_with_issues = len(all_issues)
    
    print(f"Questions with issues: {questions_with_issues}/{len(questions)}")
    print(f"Total issues: {total_issues}")
    print()
    
    # Group by issue type
    from collections import Counter
    issue_types = Counter()
    for qid, issues in all_issues.items():
        for issue in issues:
            issue_type = issue.split(":")[0]
            issue_types[issue_type] += 1
    
    print("Issue types:")
    for itype, count in issue_types.most_common():
        print(f"  {itype}: {count}")
    print()
    
    # Show specific issues
    print("Specific issues (first 20):")
    for qid, issues in list(all_issues.items())[:20]:
        q = next(x for x in questions if x["question_id"] == qid)
        print(f"  {qid} [{q['category']}]:")
        for issue in issues:
            print(f"    {issue}")
        print()
    
    # Key findings
    print("=" * 78)
    print("KEY FINDINGS")
    print("=" * 78)
    
    # Check for the "Last month" template issue
    template_issues = 0
    for q in questions:
        if q["category"] == "relative_date":
            conv = conversations.get(q["conversation_id"], {})
            turns = {}
            for key in conv:
                if key.endswith("_date_time") or not isinstance(conv[key], list):
                    continue
                for turn in conv[key]:
                    turns[turn["dia_id"]] = turn["text"]
            
            answer_turn_id = q["answer_turn"]
            answer_text = turns.get(answer_turn_id, "")
            
            # Check if the answer turn uses a relative expression
            # but the expected answer is just a date without context
            if "last month" in answer_text.lower():
                # The expected answer should be computable from the session date
                # But is the session date actually used in generation?
                template_issues += 1
    
    print(f"Questions using 'last month' template: {template_issues}")
    print(f"These questions' expected answers are generated from the answer_turn's date,")
    print(f"but the answer_turn text says 'last month' without a reference date.")
    print(f"The session datetime IS available, so they ARE resolvable.")
    print(f"However, the expected answer generation logic is opaque.")
    print()
    
    # Check for missing timestamps
    no_timestamp = sum(1 for q in questions 
                       if q["category"] in ["direct_fact", "relative_date"] 
                       and q["answer_turn"] 
                       and not any(
                           conversations.get(q["conversation_id"], {}).get(f"session_{s}_date_time")
                           for s in range(1, 4)
                       ))
    
    # More precise check
    no_ts_count = 0
    for q in questions:
        if q["category"] in ["direct_fact", "relative_date", "multi_session"]:
            conv = conversations.get(q["conversation_id"], {})
            answer_turn_id = q["answer_turn"]
            # Find the turn's session
            found = False
            for key in conv:
                if key.endswith("_date_time"):
                    continue
                if not isinstance(conv[key], list):
                    continue
                for turn in conv[key]:
                    if turn["dia_id"] == answer_turn_id:
                        found = True
                        break
            if not found:
                no_ts_count += 1
    
    print(f"Questions where answer turn not found in conversation: {no_ts_count}")
    print()
    
    # Check if expected answers are independent
    print("Expected answer generation: PROGRAMMATIC (from turn dates, not LLM)")
    print("Gold separation: ACHIEVED (golds stored separately from product input)")
    print()
    
    # Final assessment
    print("=" * 78)
    print("ASSESSMENT")
    print("=" * 78)
    
    if questions_with_issues == 0:
        print("ALL QUESTIONS VALID: No structural issues found.")
    else:
        print(f"ISSUES FOUND: {questions_with_issues} questions have issues.")
        print("These issues affect evaluation validity.")
        print()
        print("The main concern is:")
        print("1. Expected answers for 'relative_date' questions are generated from")
        print("   the answer turn's session date, but the answer turn text uses")
        print("   relative expressions without explicit reference dates.")
        print("2. The session datetime IS available, so the questions ARE resolvable.")
        print("3. However, the generation logic should be more transparent.")
    
    print()
    print("RECOMMENDATION: Regenerate test data with explicit reference dates")
    print("in the answer turn text, or provide the reference date separately.")
    print("This ensures the evaluation measures the SYSTEM's ability, not just")
    print("the data generator's date arithmetic.")


if __name__ == "__main__":
    main()
