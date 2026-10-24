from __future__ import annotations

import math
import re
import time
from collections import Counter
from datetime import datetime

from artificial_memory.core.interfaces import MemoryStore, RecallEngine
from artificial_memory.core.models import (
    Memory,
    MemoryStatus,
    RecallEvent,
    RecallLevel,
    ResolutionLevel,
)
from artificial_memory.core.tokens import estimate_tokens


class BasicRecallEngine:
    """Basic recall engine for MVP - levels 0-2."""

    def __init__(
        self,
        store: MemoryStore,
        lexical: bool = False,
        lexical_candidate_limit: int = 1000,
    ):
        self.store = store
        # LEXICAL CHANNEL (off by default; byte-identical when off).
        #
        # Measured on the controlled arena: the runtime answered 13.5% of 200
        # planted-fact questions because the candidate pool was query-independent
        # (40 episodes + 20 semantic of 440 facts) and the ranking was unweighted
        # word overlap.  An IDF-weighted lexical ranker reaches the gold in the
        # top 20 for 80-100% of the retrieval-shaped categories.  See
        # design_docs/v4_world_class/V4_PROGRAM.md 4.12.
        self.lexical = lexical
        self.lexical_candidate_limit = lexical_candidate_limit

    @staticmethod
    def _stem(word: str) -> str:
        for suffix in ("ing", "ed", "es", "s", "e"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[: -len(suffix)]
        return word

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        return {cls._stem(w) for w in re.findall(r"[a-z0-9]{3,}", text.lower())}

    def _lexical_scores(self, query: str, memories: list[Memory]) -> dict[int, float]:
        """IDF-weighted overlap: a rare shared term counts, 'the user' does not."""
        docs = [self._terms(m.content) for m in memories]
        df: Counter = Counter()
        for doc in docs:
            df.update(doc)
        n = max(1, len(docs))
        query_terms = self._terms(query)
        scores: dict[int, float] = {}
        max_idf = sum(math.log((n + 1) / (df[t] + 1)) for t in query_terms) or 1.0
        for memory, doc in zip(memories, docs):
            shared = query_terms & doc
            raw = sum(math.log((n + 1) / (df[t] + 1)) for t in shared)
            scores[memory.id] = raw / max_idf
        return scores

    def recall(
        self,
        query: str,
        topic_id: int | None = None,
        level: RecallLevel = RecallLevel.CURRENT_ONLY,
        max_tokens: int = 4000,
    ) -> tuple[list[Memory], int]:
        """Recall memories based on query and level."""
        start_time = time.time()

        # Get candidate memories based on level
        candidates = self._get_candidates(query, topic_id, level)

        # Lexical channel: the level-based pool is query-independent and capped,
        # so on a large topic the answer may never be a candidate at all.  Pull
        # the topic's active memories in and let the IDF ranking decide.
        if self.lexical:
            seen = {m.id for m in candidates}
            for memory in self.store.get_memories(
                topic_id=topic_id,
                status=MemoryStatus.ACTIVE,
                limit=self.lexical_candidate_limit,
            ):
                if memory.id not in seen:
                    seen.add(memory.id)
                    candidates.append(memory)

        # Rank by relevance (simple scoring for MVP)
        ranked = self._rank_memories(query, candidates)

        # Select within token budget
        selected, total_tokens = self._select_within_budget(ranked, max_tokens)

        # Update access stats
        for memory in selected:
            memory.touch()
            self.store.update_memory(memory)

        # Log recall event
        latency_ms = int((time.time() - start_time) * 1000)
        recall_event = RecallEvent(
            query=query,
            topic_id=topic_id,
            recall_level=level,
            memories_retrieved=len(selected),
            tokens_returned=total_tokens,
            latency_ms=latency_ms,
        )
        self.store.log_recall(recall_event)

        return selected, total_tokens

    def _get_candidates(
        self,
        query: str,
        topic_id: int | None,
        level: RecallLevel
    ) -> list[Memory]:
        """Get candidate memories based on recall level."""
        candidates = []

        if level == RecallLevel.CURRENT_ONLY:
            # Level 0: Current state only
            memories = self.store.get_memories(
                topic_id=topic_id,
                is_current=True,
                status=MemoryStatus.ACTIVE,
                limit=50
            )
            candidates.extend(memories)

        elif level == RecallLevel.LONG_TERM_SUMMARY:
            # Level 1: Long-term summary (semantic + longterm resolutions)
            for res in [ResolutionLevel.SEMANTIC, ResolutionLevel.LONG_TERM, ResolutionLevel.DEEP_LONG_TERM]:
                memories = self.store.get_memories(
                    topic_id=topic_id,
                    resolution=res,
                    status=MemoryStatus.ACTIVE,
                    limit=30
                )
                candidates.extend(memories)

        elif level == RecallLevel.EPISODE:
            # Level 2: Episode memories
            memories = self.store.get_memories(
                topic_id=topic_id,
                resolution=ResolutionLevel.EPISODE,
                status=MemoryStatus.ACTIVE,
                limit=40
            )
            candidates.extend(memories)
            # Also include semantic for context
            semantic = self.store.get_memories(
                topic_id=topic_id,
                resolution=ResolutionLevel.SEMANTIC,
                status=MemoryStatus.ACTIVE,
                limit=20
            )
            candidates.extend(semantic)

        elif level == RecallLevel.LIGHT_COMPRESSION:
            # Level 3: Light compression
            memories = self.store.get_memories(
                topic_id=topic_id,
                resolution=ResolutionLevel.LIGHT,
                status=MemoryStatus.ACTIVE,
                limit=30
            )
            candidates.extend(memories)
            # Include episode and semantic
            for res in [ResolutionLevel.EPISODE, ResolutionLevel.SEMANTIC]:
                memories = self.store.get_memories(
                    topic_id=topic_id,
                    resolution=res,
                    status=MemoryStatus.ACTIVE,
                    limit=15
                )
                candidates.extend(memories)

        elif level == RecallLevel.RAW:
            # Level 4: Raw conversation (would need conversation store)
            # For MVP, return light + episode
            for res in [ResolutionLevel.LIGHT, ResolutionLevel.EPISODE, ResolutionLevel.SEMANTIC]:
                memories = self.store.get_memories(
                    topic_id=topic_id,
                    resolution=res,
                    status=MemoryStatus.ACTIVE,
                    limit=20
                )
                candidates.extend(memories)

        # Deduplicate by memory ID
        seen = set()
        unique = []
        for m in candidates:
            if m.id not in seen:
                seen.add(m.id)
                unique.append(m)

        return unique

    def _rank_memories(self, query: str, memories: list[Memory]) -> list[Memory]:
        """Simple relevance ranking for MVP."""
        query_words = set(query.lower().split())
        lexical = self._lexical_scores(query, memories) if self.lexical else {}

        scored = []
        for memory in memories:
            score = 0.0

            # Content overlap
            content_words = set(memory.content.lower().split())
            overlap = len(query_words & content_words)
            score += overlap * 0.5

            # Lexical channel: IDF-weighted, so a shared rare term ("atlas")
            # outweighs a shared common one ("the user").  Normalised to 0..1.
            if lexical:
                score += lexical.get(memory.id, 0.0) * 2.0

            # Importance boost
            score += memory.importance * 0.3

            # Confidence boost
            score += memory.confidence * 0.2

            # Recency boost (more recent = higher)
            if memory.updated_at:
                days_old = (datetime.now() - memory.updated_at).days
                recency_score = max(0, 1.0 - days_old / 365)
                score += recency_score * 0.2

            # Access count boost (frequently accessed = more relevant)
            if memory.access_count > 0:
                score += min(memory.access_count * 0.05, 0.3)

            scored.append((memory, score))

        scored.sort(key=lambda x: -x[1])
        return [m for m, _ in scored]

    def _select_within_budget(
        self,
        ranked_memories: list[Memory],
        max_tokens: int
    ) -> tuple[list[Memory], int]:
        """Select memories within token budget."""
        # Estimation is canonical (core.tokens: estimate_tokens).

        selected = []
        total_tokens = 0

        for memory in ranked_memories:
            mem_tokens = estimate_tokens(memory.content)
            if total_tokens + mem_tokens <= max_tokens:
                selected.append(memory)
                total_tokens += mem_tokens
            else:
                break

        return selected, total_tokens

    def expand_resolution(self, memory: Memory, target_resolution: ResolutionLevel) -> Memory | None:
        """Expand memory to higher resolution (lower resolution number)."""
        if target_resolution.value >= memory.resolution.value:
            return None  # Already at or below target resolution

        # Special case: expanding to RAW returns original content
        if target_resolution == ResolutionLevel.RAW:
            expanded = Memory(
                id=memory.id,
                topic_id=memory.topic_id,
                memory_type=memory.memory_type,
                content=memory.content,  # Original content is RAW
                resolution=ResolutionLevel.RAW,
                importance=memory.importance,
                confidence=memory.confidence,
                status=memory.status,
                valid_from=memory.valid_from,
                valid_until=memory.valid_until,
                is_current=memory.is_current,
                source_conversation_id=memory.source_conversation_id,
                source_message_id=memory.source_message_id,
                created_at=memory.created_at,
                updated_at=datetime.now(),
            )
            return expanded

        # Check if version exists
        version = self.store.get_memory_version(memory.id, target_resolution)
        if version:
            # Create expanded memory from version
            expanded = Memory(
                id=memory.id,
                topic_id=memory.topic_id,
                memory_type=memory.memory_type,
                content=version.content,
                resolution=target_resolution,
                importance=memory.importance,
                confidence=memory.confidence,
                status=memory.status,
                valid_from=memory.valid_from,
                valid_until=memory.valid_until,
                is_current=memory.is_current,
                source_conversation_id=memory.source_conversation_id,
                source_message_id=memory.source_message_id,
                created_at=memory.created_at,
                updated_at=datetime.now(),
            )
            return expanded

        return None

    def get_memory_provenance(self, memory: Memory) -> list[Memory]:
        """Get provenance chain: memory -> episode -> raw."""
        chain = [memory]

        # Get versions at lower resolutions (higher detail)
        for res in ResolutionLevel:
            if res.value < memory.resolution.value:
                version = self.store.get_memory_version(memory.id, res)
                if version:
                    chain.append(Memory(
                        id=memory.id,
                        topic_id=memory.topic_id,
                        memory_type=memory.memory_type,
                        content=version.content,
                        resolution=res,
                        importance=memory.importance,
                        confidence=memory.confidence,
                        status=memory.status,
                        valid_from=memory.valid_from,
                        valid_until=memory.valid_until,
                        is_current=memory.is_current,
                        source_conversation_id=memory.source_conversation_id,
                        source_message_id=memory.source_message_id,
                        created_at=memory.created_at,
                        updated_at=datetime.now(),
                    ))

        return chain

    def get_full_provenance(self, memory: Memory) -> dict:
        """Get full provenance including source conversation and messages."""
        result = {
            "memory": memory,
            "versions": self.get_memory_provenance(memory),
            "source_conversation": None,
            "source_messages": [],
        }

        # Get source conversation
        if memory.source_conversation_id:
            conv = self.store.get_conversation(memory.source_conversation_id)
            result["source_conversation"] = conv

            # Get source messages
            if conv:
                messages = self.store.get_messages(conv.id)
                result["source_messages"] = messages

        # If specific source message, highlight it
        if memory.source_message_id:
            for msg in result["source_messages"]:
                if msg.id == memory.source_message_id:
                    result["exact_source_message"] = msg
                    break

        return result

    def search_by_keywords(self, keywords: list[str], topic_id: int | None = None, limit: int = 20) -> list[Memory]:
        """Search memories by keywords."""
        all_memories = self.store.get_memories(topic_id=topic_id, limit=200)

        keyword_set = set(k.lower() for k in keywords)
        scored = []

        for memory in all_memories:
            content_lower = memory.content.lower()
            matches = sum(1 for kw in keyword_set if kw in content_lower)
            if matches > 0:
                score = matches * 0.5 + memory.importance * 0.3
                scored.append((memory, score))

        scored.sort(key=lambda x: -x[1])
        return [m for m, _ in scored[:limit]]

    def search_by_time_range(
        self,
        start: datetime,
        end: datetime,
        topic_id: int | None = None,
        limit: int = 20
    ) -> list[Memory]:
        """Search memories by time range."""
        all_memories = self.store.get_memories(topic_id=topic_id, limit=200)

        filtered = [
            m for m in all_memories
            if m.created_at >= start and m.created_at <= end
        ]

        filtered.sort(key=lambda m: m.created_at, reverse=True)
        return filtered[:limit]

    def search_associations(self, memory: Memory, max_depth: int = 2, min_strength: float = 0.3) -> list[Memory]:
        """Search associated memories via association graph."""
        visited = set()
        results = []
        current_level = [memory]

        for depth in range(max_depth):
            next_level = []
            for mem in current_level:
                if mem.id in visited:
                    continue
                visited.add(mem.id)

                related = self.store.get_related_memories(mem.id, min_strength)
                for related_mem, assoc in related:
                    if related_mem.id not in visited:
                        results.append(related_mem)
                        next_level.append(related_mem)

            current_level = next_level
            if not current_level:
                break

        return results


def create_recall_engine(store: MemoryStore) -> RecallEngine:
    """Factory function to create recall engine."""
    return BasicRecallEngine(store)
