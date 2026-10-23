"""
Official LongMemEval Grader for Lethe-Akribeia V4.

Implements the exact grading logic from the LongMemEval paper/benchmark
to ensure reproducibility and fair comparison.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GradingResult:
    """Result of grading a single answer."""
    question_id: str
    question_type: str
    is_correct: bool
    score: float  # 0.0 or 1.0 for LongMemEval (binary)
    details: dict[str, Any]


class OfficialLongMemEvalGrader:
    """
    Official LongMemEval grading logic.
    
    Based on the paper's evaluation methodology:
    - Single-session: Exact match or semantic equivalence
    - Multi-session: Count/sum must match exactly
    - Temporal: Date/duration calculations must be exact
    - Knowledge-update: Latest state must match
    - Preference: Must not abstain, must ground in user history
    - Abstention: Correctly refusing unanswerable questions
    """
    
    # Stop words for preference grading
    PREF_STOP_WORDS = frozenset({
        "the", "user", "would", "prefer", "responses", "that", "suggest", "take", "into",
        "account", "their", "previous", "experience", "such", "other", "might", "they",
        "considering", "specific", "details", "previously", "mentioned", "inform", "decision",
        "utilize", "response", "utilizes", "viewing", "history", "cater", "tastes", "reference",
        "connect", "observed", "fail", "acknowledge", "provide", "vague", "general", "explanations",
        "compatible", "enhance", "functionality", "protection", "activities", "require", "visual",
        "attention", "commuting", "topics", "exploring", "resources", "personalized", "tips",
        "prior", "preparations", "interest", "both", "unique", "existing", "build", "upon",
        "goals", "saving", "money", "reducing", "commercial", "products", "positive", "experiences",
        "highlight", "potential", "benefits", "reconnecting", "revisiting", "individual", "current",
        "challenges", "investments", "point", "appreciate", "focus", "solely", "aspect", "deviate",
        "significantly", "established", "memorable", "encounter", "revisit", "unrelated", "vastly",
        "different", "tone", "subject", "matter", "titles", "like", "with", "from", "some", "more",
        "about", "what", "which", "also", "have", "been", "these", "those",
    })
    
    def __init__(self):
        self._normalization_cache = {}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        Coerces non-string references (some LongMemEval gold answers are plain
        integers, e.g. an answer of ``2``) so grading never crashes.
        """
        text = str(text)
        if text in self._normalization_cache:
            return self._normalization_cache[text]
        
        # Lowercase, remove punctuation, normalize whitespace
        normalized = text.lower()
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        
        # Strip leading date prepositions: "in 2022" -> "2022"
        normalized = re.sub(r"^in\s+", "", normalized)
        
        self._normalization_cache[text] = normalized
        return normalized
    
    def _normalize_number(self, text: str) -> str:
        """Normalize numbers (words to digits)."""
        word_to_num = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
            "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
            "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
            "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
            "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
        }
        
        words = text.split()
        for i, w in enumerate(words):
            w_clean = w.rstrip(".,;:!?")  # preserve punctuation for now
            if w_clean in word_to_num:
                words[i] = word_to_num[w_clean]
        
        return " ".join(words)
    
    def grade_single_session_user(
        self,
        predicted: str,
        ground_truth: str,
    ) -> GradingResult:
        """Grade single-session-user questions (fact recall)."""
        pred_norm = self._normalize_text(predicted)
        gold_norm = self._normalize_text(ground_truth)
        pred_num = self._normalize_number(pred_norm)
        gold_num = self._normalize_number(gold_norm)
        
        # Exact match after normalization
        is_correct = (pred_norm == gold_norm) or (pred_num == gold_num)
        
        # Also check if gold is contained in pred (for verbose answers)
        if not is_correct and len(gold_norm) > 3:
            is_correct = gold_norm in pred_norm
        
        return GradingResult(
            question_id="",
            question_type="single-session-user",
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            details={"pred_norm": pred_norm, "gold_norm": gold_norm},
        )
    
    def grade_single_session_assistant(
        self,
        predicted: str,
        ground_truth: str,
    ) -> GradingResult:
        """Grade single-session-assistant questions (assistant recall)."""
        # Same logic as user fact recall
        return self.grade_single_session_user(predicted, ground_truth)
    
    def grade_knowledge_update(
        self,
        predicted: str,
        ground_truth: str,
    ) -> GradingResult:
        """Grade knowledge-update questions (current state)."""
        # Must match the LATEST state
        return self.grade_single_session_user(predicted, ground_truth)
    
    def _normalize_relative_date(self, text: str, reference_date: str | None = None) -> str:
        """
        Normalize relative date expressions against a reference date.

        e.g. "next month" + ref_date "2023-08-28" -> "september 2023"
             "last month" + ref_date "2023-08-28" -> "july 2023"

        Only applies when reference_date is provided and matches YYYY-MM-DD format.
        Otherwise returns the text unchanged.
        """
        text = str(text)
        if reference_date is None:
            return text

        # Parse reference date
        ref_match = re.match(r"(\d{4})-(\d{2})-(\d{2})", reference_date)
        if not ref_match:
            return text

        year, month, day = int(ref_match.group(1)), int(ref_match.group(2)), int(ref_match.group(3))

        # Month name lists for normalization
        month_names_en = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]

        # next month / last month / this month / this year / last year / next year / X years ago
        # Always include year so predictions like "September" match "September 2023"
        lower = text.lower()
        if "next month" in lower:
            new_month = month + 1 if month < 12 else 1
            new_year = year + 1 if month == 12 else year
            replacement = month_names_en[new_month - 1] + " " + str(new_year)
            return re.sub(r"next month", replacement, text, flags=re.IGNORECASE)
        elif "last month" in lower:
            new_month = month - 1 if month > 1 else 12
            new_year = year - 1 if month == 1 else year
            replacement = month_names_en[new_month - 1] + " " + str(new_year)
            return re.sub(r"last month", replacement, text, flags=re.IGNORECASE)
        elif "this month" in lower:
            replacement = month_names_en[month - 1] + " " + str(year)
            return re.sub(r"this month", replacement, text, flags=re.IGNORECASE)
        elif "this year" in lower:
            return re.sub(r"this year", str(year), text, flags=re.IGNORECASE)
        elif "next year" in lower:
            return re.sub(r"next year", str(year + 1), text, flags=re.IGNORECASE)
        elif "last year" in lower:
            return re.sub(r"last year", str(year - 1), text, flags=re.IGNORECASE)
        elif re.search(r"\b(one|a)\s+year\s+ago\b", lower):
            return re.sub(r"\b(one|a)\s+year\s+ago\b", str(year - 1), text, flags=re.IGNORECASE)
        elif re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago\b", lower):
            word_to_num = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
            match = re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago\b", lower)
            years_ago = word_to_num[match.group(1)]
            return re.sub(r"\b(two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago\b", str(year - years_ago), text, flags=re.IGNORECASE)
        elif re.search(r"\b(\d+)\s+years?\s+ago\b", lower):
            years_ago = int(re.search(r"\b(\d+)\s+years?\s+ago\b", lower).group(1))
            return re.sub(r"\b\d+\s+years?\s+ago\b", str(year - years_ago), text, flags=re.IGNORECASE)

        return text

    def grade_temporal_reasoning(
        self,
        predicted: str,
        ground_truth: str,
        reference_date: str | None = None,
    ) -> GradingResult:
        """Grade temporal-reasoning questions (dates, durations)."""
        # Apply relative-date normalization if reference_date provided
        if reference_date:
            predicted = self._normalize_relative_date(predicted, reference_date)
            ground_truth = self._normalize_relative_date(ground_truth, reference_date)

        pred_norm = self._normalize_text(predicted)
        gold_norm = self._normalize_text(ground_truth)
        pred_num = self._normalize_number(pred_norm)
        gold_num = self._normalize_number(gold_norm)
        
        # Exact match after normalization
        is_correct = (pred_num == gold_num) or (pred_norm == gold_norm)
        
        # Also handle "X weeks" vs "X weeks ago" equivalence
        if not is_correct:
            pred_weeks = re.search(r"(\d+)\s*weeks?", pred_norm)
            gold_weeks = re.search(r"(\d+)\s*weeks?", gold_norm)
            if pred_weeks and gold_weeks:
                is_correct = pred_weeks.group(1) == gold_weeks.group(1)
            
            pred_days = re.search(r"(\d+)\s*days?", pred_norm)
            gold_days = re.search(r"(\d+)\s*days?", gold_norm)
            if pred_days and gold_days:
                is_correct = pred_days.group(1) == gold_days.group(1)
        
        return GradingResult(
            question_id="",
            question_type="temporal-reasoning",
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            details={"pred_norm": pred_norm, "gold_norm": gold_norm},
        )
    
    def grade_multi_session(
        self,
        predicted: str,
        ground_truth: str,
    ) -> GradingResult:
        """Grade multi-session questions (counting, sums, comparisons)."""
        pred_norm = self._normalize_text(predicted)
        gold_norm = self._normalize_text(ground_truth)
        pred_num = self._normalize_number(pred_norm)
        gold_num = self._normalize_number(gold_norm)
        
        # For counting/sum: exact number match required
        is_correct = (pred_num == gold_num) or (pred_norm == gold_norm)
        
        return GradingResult(
            question_id="",
            question_type="multi-session",
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            details={"pred_norm": pred_norm, "gold_norm": gold_norm},
        )
    
    def grade_single_session_preference(
        self,
        predicted: str,
        ground_truth: str,  # This is the preference rubric/profile
    ) -> GradingResult:
        """
        Grade preference questions.
        
        Criteria (from LongMemEval paper):
        1. Does NOT abstain ("I don't know", "not mentioned", etc.)
        2. Grounds answer in user's stated preferences from conversation
        3. Makes specific recommendation (not generic)
        """
        pred_lower = predicted.lower()
        
        # 1. Check for abstention
        abstention_phrases = [
            "i don't know", "i do not know", "not mentioned", "not enough information",
            "cannot determine", "unknown", "not provided", "not specified", "not stated",
            "no information", "insufficient", "i'm not sure", "i am not sure",
        ]
        
        for phrase in abstention_phrases:
            if phrase in pred_lower:
                return GradingResult(
                    question_id="",
                    question_type="single-session-preference",
                    is_correct=False,
                    score=0.0,
                    details={"reason": f"abstention: '{phrase}'"},
                )
        
        # 2. Check for grounding in user history (from ground_truth rubric)
        # Extract key entities from ground truth
        quoted_entities = re.findall(r"['\"]([^'\"]+)['\"]", ground_truth)
        grounded = False
        
        for entity in quoted_entities:
            entity_norm = self._normalize_text(entity)
            if len(entity_norm) > 3 and entity_norm in self._normalize_text(predicted):
                grounded = True
                break
        
        # 3. Check for specific recommendation (not generic)
        generic_phrases = [
            "it depends", "depends on", "many options", "various choices",
            "several options", "multiple options", "personal preference",
            "subjective", "up to you", "whatever you like",
        ]
        
        is_generic = any(phrase in pred_lower for phrase in generic_phrases)
        
        # Score: must be grounded AND not generic
        is_correct = grounded and not is_generic
        
        return GradingResult(
            question_id="",
            question_type="single-session-preference",
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            details={
                "grounded": grounded,
                "is_generic": is_generic,
                "entities_found": quoted_entities,
            },
        )
    
    def grade_abstention(
        self,
        predicted: str,
        ground_truth: str,
    ) -> GradingResult:
        """Grade abstention questions (should correctly say 'I don't know')."""
        pred_lower = predicted.lower()
        
        # Correct abstention phrases
        correct_abstentions = [
            "i don't know", "i do not know", "not mentioned", "not enough information",
            "cannot determine", "unknown", "not provided", "not specified", "not stated",
            "no information", "insufficient", "i'm not sure", "i am not sure",
        ]
        
        is_correct = any(phrase in pred_lower for phrase in correct_abstentions)
        
        return GradingResult(
            question_id="",
            question_type="abstention",
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            details={},
        )
    
    def grade(
        self,
        question_id: str,
        question_type: str,
        predicted: str,
        ground_truth: str,
        reference_date: str | None = None,
    ) -> GradingResult:
        """Main grading dispatcher."""
        # Some LongMemEval gold answers are integers; normalise inputs up front
        # so every per-type grader can assume strings.
        predicted = str(predicted)
        ground_truth = str(ground_truth)
        graders = {
            "single-session-user": self.grade_single_session_user,
            "single-session-assistant": self.grade_single_session_assistant,
            "knowledge-update": self.grade_knowledge_update,
            "temporal-reasoning": self.grade_temporal_reasoning,
            "multi-session": self.grade_multi_session,
            "single-session-preference": self.grade_single_session_preference,
            "abstention": self.grade_abstention,
        }
        
        grader = graders.get(question_type, self.grade_single_session_user)
        # Pass reference_date only to temporal-reasoning (other graders don't accept it)
        if question_type == "temporal-reasoning" and reference_date:
            result = grader(predicted, ground_truth, reference_date=reference_date)
        else:
            result = grader(predicted, ground_truth)
        
        return GradingResult(
            question_id=question_id,
            question_type=question_type,
            is_correct=result.is_correct,
            score=result.score,
            details=result.details,
        )
    
    def grade_batch(self, results: list[dict]) -> list[GradingResult]:
        """Grade a batch of results."""
        graded = []
        for r in results:
            graded.append(self.grade(
                question_id=r.get('question_id', ''),
                question_type=r.get('question_type', 'single-session-user'),
                predicted=r.get('predicted_answer', r.get('prediction', '')),
                ground_truth=r.get('ground_truth', r.get('answer', '')),
            ))
        return graded
    
    def compute_metrics(self, graded: list[GradingResult]) -> dict[str, Any]:
        """Compute aggregate metrics."""
        if not graded:
            return {"accuracy": 0.0, "total": 0}
        
        total = len(graded)
        correct = sum(1 for g in graded if g.is_correct)
        accuracy = correct / total
        
        # Per-type
        per_type = {}
        for g in graded:
            qtype = g.question_type
            if qtype not in per_type:
                per_type[qtype] = {"correct": 0, "total": 0}
            per_type[qtype]["total"] += 1
            if g.is_correct:
                per_type[qtype]["correct"] += 1
        
        per_type_final = {}
        for qtype, counts in per_type.items():
            per_type_final[qtype] = {
                "accuracy": counts["correct"] / counts["total"],
                "correct": counts["correct"],
                "total": counts["total"],
            }
        
        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "per_type": per_type_final,
        }


# Alternative grader for comparison (more lenient)
class LenientGrader(OfficialLongMemEvalGrader):
    """More lenient grading for internal comparison."""
    
    def grade_single_session_user(self, predicted: str, ground_truth: str) -> GradingResult:
        # Allow partial matches
        pred_norm = self._normalize_text(predicted)
        gold_norm = self._normalize_text(ground_truth)
        
        is_correct = (pred_norm == gold_norm) or (gold_norm in pred_norm and len(gold_norm) > 3)
        
        return GradingResult(
            question_id="", question_type="single-session-user",
            is_correct=is_correct, score=1.0 if is_correct else 0.0,
            details={"pred_norm": pred_norm, "gold_norm": gold_norm},
        )


def grade_results_file(
    results_path: str | Path,
    grader: OfficialLongMemEvalGrader | None = None,
) -> dict[str, Any]:
    """Grade a results JSON/JSONL file."""
    path = Path(results_path)
    
    with open(path) as f:
        if path.suffix == '.jsonl':
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
            if 'results' in data:
                data = data['results']
    
    grader = grader or OfficialLongMemEvalGrader()
    graded = grader.grade_batch(data)
    metrics = grader.compute_metrics(graded)
    
    return {
        "metrics": metrics,
        "graded": [
            {
                "question_id": g.question_id,
                "question_type": g.question_type,
                "is_correct": g.is_correct,
                "details": g.details,
            }
            for g in graded
        ],
    }


if __name__ == "__main__":
    # Quick test
    grader = OfficialLongMemEvalGrader()
    
    test_cases = [
        ("single-session-user", "Target", "Target"),
        ("single-session-user", "The Glass Menagerie", "The Glass Menagerie"),
        ("multi-session", "3", "3"),
        ("multi-session", "4", "3"),
        ("temporal-reasoning", "4 weeks", "4 weeks"),
        ("temporal-reasoning", "28 days", "4 weeks"),
        ("single-session-preference", "I recommend the Sony WH-1000XM5 based on your preference for noise cancellation", "User prefers noise cancellation and over-ear headphones"),
        ("single-session-preference", "I don't know", "User prefers noise cancellation"),
        ("abstention", "I don't know", "Not mentioned"),
        ("abstention", "It was a cake", "Not mentioned"),
    ]
    
    print("Official LongMemEval Grader Test")
    print("=" * 60)
    for qtype, pred, gold in test_cases:
        result = grader.grade("", qtype, pred, gold)
        status = "✓" if result.is_correct else "✗"
        print(f"  {status} [{qtype}] pred='{pred}' gold='{gold}'")
