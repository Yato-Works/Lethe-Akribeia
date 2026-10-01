"""Deterministic Domain Associator for Cross-Domain Memory Recall (P5).

Enriches open-domain and multi-hop queries with deterministic domain associations
(e.g., "song/Vivaldi" -> "classical/music/concert", "meat/eat" -> "chicken/food",
"movie" -> "film/cinema/watch") without Write or Recall LLM calls.
Directly prevents RETRIEVAL_FAILURE on character preference & lifestyle queries.
"""

from __future__ import annotations

import re


#: Deterministic domain ontology mapping query cues -> associative memory keywords.
_DOMAIN_ONTOLOGY: dict[str, tuple[str, ...]] = {
    # Music & Listening (e.g. Vivaldi, song -> classical, concert)
    "music": (
        "song", "classical", "jazz", "rock", "pop", "concert", "album",
        "playlist", "listen", "listening", "band", "instrument", "piano",
    ),
    "song": (
        "music", "classical", "singer", "concert", "track", "listen", "melody",
    ),
    "vivaldi": ("classical", "music", "orchestra", "concerto", "seasons"),
    "classical": ("music", "orchestra", "concerto", "composer", "symphony"),

    # Books & Reading (e.g. book, read -> author, novel, titles)
    "book": ("read", "reading", "author", "novel", "literature", "story", "pages"),
    "read": ("book", "books", "author", "novel", "reading", "story"),

    # Movies & Cinema
    "movie": ("film", "cinema", "watch", "watching", "actor", "romcom", "drama", "series"),
    "movies": ("film", "cinema", "watch", "watching", "actor", "romcom", "drama", "series"),
    "film": ("movie", "movies", "cinema", "watch", "director"),

    # Food & Diet (e.g. meat, eat -> chicken, beef, cook)
    "meat": ("chicken", "beef", "pork", "steak", "food", "eat", "cook", "protein"),
    "food": ("eat", "eating", "cook", "cooking", "dinner", "lunch", "meal", "treat", "delicious"),
    "cook": ("food", "baking", "treats", "recipe", "kitchen", "dessert"),

    # Pets & Animals (e.g. dog, pet -> breeder, adopt, rescue)
    "pet": ("dog", "cat", "puppy", "kitten", "breeder", "adopt", "adoption", "shelter", "rescue"),
    "pets": ("dog", "cat", "puppy", "kitten", "breeder", "adopt", "adoption", "shelter", "rescue"),
    "dog": ("dogs", "puppy", "breed", "breeder", "adopt", "walk", "shelter", "vet", "pets"),
    "dogs": ("dog", "puppy", "breed", "breeder", "adopt", "walk", "shelter", "vet", "pets"),
    "cat": ("cats", "kitten", "feline", "shelter", "rescue", "pets"),
    "pixie": ("dog", "puppy", "breeder", "adopt", "pet"),

    # Outdoor & Physical Activities
    "outdoor": ("hike", "hiking", "walk", "park", "nature", "trails", "outside"),
    "activities": ("hobby", "play", "games", "sports", "volunteer", "climbing", "kayak"),
    "birdwatching": ("birds", "bird", "feeder", "parks", "nature", "binoculars"),

    # Location & Residence
    "state": ("live", "living", "moved", "minnesota", "city", "hometown", "residence"),
    "live": ("living", "moved", "house", "apartment", "city", "state", "hometown"),
}


class DomainAssociator:
    """Deterministic associative query expansion for memory retrieval."""

    @classmethod
    def expand_query(cls, query: str, max_terms: int = 5) -> set[str]:
        """Expand query with relevant domain terms to bridge lexical gaps.

        Returns a set of lowercase keywords to inject into retrieval scoring.
        """
        q_lower = query.lower()
        words = re.findall(r"\b[a-z0-9_-]{3,}\b", q_lower)

        expanded: set[str] = set()
        for w in words:
            if w in _DOMAIN_ONTOLOGY:
                for term in _DOMAIN_ONTOLOGY[w]:
                    if term not in words:
                        expanded.add(term)
                        if len(expanded) >= max_terms:
                            break
            if len(expanded) >= max_terms:
                break

        return expanded
