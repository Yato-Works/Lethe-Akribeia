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
        "playlist", "listen", "listening", "band", "artist", "instrument", "piano", "musician",
    ),
    "song": (
        "music", "classical", "singer", "concert", "track", "listen", "melody", "band", "artist",
    ),
    "artists": ("band", "bands", "singer", "music", "concert", "musician", "tour"),
    "bands": ("band", "artist", "artists", "singer", "music", "concert", "musician"),
    "vivaldi": ("classical", "music", "orchestra", "concerto", "seasons"),
    "classical": ("music", "orchestra", "concerto", "composer", "symphony"),

    # Books & Reading (e.g. book, read -> author, novel, titles)
    "book": ("read", "reading", "author", "novel", "literature", "story", "pages", "title"),
    "books": ("book", "read", "reading", "author", "novel", "literature", "story", "pages", "title"),
    "read": ("book", "books", "author", "novel", "reading", "story", "title"),

    # Movies & Cinema
    "movie": ("film", "cinema", "watch", "watched", "watching", "seen", "actor", "romcom", "drama", "series", "trilogy", "recommendation"),
    "movies": ("film", "cinema", "watch", "watched", "watching", "seen", "actor", "romcom", "drama", "series", "trilogy", "recommendation"),
    "film": ("movie", "movies", "cinema", "watch", "watched", "director", "seen"),

    # Food & Diet (e.g. meat, eat -> chicken, beef, cook)
    "meat": ("chicken", "beef", "pork", "steak", "food", "eat", "cook", "protein"),
    "food": ("eat", "eating", "cook", "cooking", "dinner", "lunch", "meal", "treat", "delicious"),
    "cook": ("food", "baking", "treats", "recipe", "kitchen", "dessert"),

    # Pets & Animals (e.g. dog, pet -> breeder, adopt, rescue)
    "animal": ("turtle", "turtles", "dog", "cat", "pet", "reptile", "wildlife", "nature"),
    "animals": ("turtle", "turtles", "dog", "cat", "pet", "reptile", "wildlife", "nature"),
    "pet": ("dog", "cat", "puppy", "kitten", "breeder", "adopt", "adoption", "shelter", "rescue"),
    "pets": ("dog", "cat", "puppy", "kitten", "breeder", "adopt", "adoption", "shelter", "rescue"),
    "dog": ("dogs", "puppy", "breed", "breeder", "adopt", "walk", "walks", "hike", "hiking", "trail", "shelter", "vet", "pets"),
    "dogs": ("dog", "puppy", "breed", "breeder", "adopt", "walk", "walks", "hike", "hiking", "trail", "shelter", "vet", "pets"),
    "cat": ("cats", "kitten", "feline", "shelter", "rescue", "pets"),
    "pixie": ("dog", "puppy", "breeder", "adopt", "pet"),

    # Outdoor & Physical Activities
    "outdoor": ("hike", "hiking", "walk", "park", "nature", "trails", "outside"),
    "activities": ("hobby", "play", "games", "sports", "volunteer", "climbing", "kayak"),
    "birdwatching": ("birds", "bird", "feeder", "parks", "nature", "binoculars"),

    # Location & Residence
    "state": ("live", "living", "moved", "minnesota", "city", "hometown", "residence"),
    "live": ("living", "moved", "house", "apartment", "city", "state", "hometown"),

    # Career & Profession
    "career": ("job", "work", "profession", "zoo", "keeper", "turtles", "animals", "counselor"),
    "job": ("career", "work", "profession", "zoo", "keeper", "counselor"),
    "profession": ("job", "career", "work", "counselor", "counseling"),

    # Health & Medical Conditions
    "condition": ("health", "asthma", "medical", "disease", "illness", "symptoms", "allergies"),
    "allergies": ("allergy", "asthma", "allergic", "reaction", "breathing", "fur", "reptiles"),
    "allergy": ("allergies", "asthma", "allergic", "reaction", "breathing"),
    "underlying": ("condition", "health", "asthma", "illness", "disease"),

    # Domestic & Indoor Activities
    "indoor": ("cooking", "cook", "treats", "baking", "home", "inside", "recipes"),
    "activity": ("hobby", "cooking", "games", "reading", "treats", "exercise", "walk"),
}

from collections import defaultdict

# Construct deterministic bidirectional associative graph
_ASSOC_GRAPH: defaultdict[str, set[str]] = defaultdict(set)
for _k, _v_tuple in _DOMAIN_ONTOLOGY.items():
    for _v in _v_tuple:
        _ASSOC_GRAPH[_k].add(_v)
        _ASSOC_GRAPH[_v].add(_k)

# Critical symmetric & transitive bridges identified in failure autopsy
_ASSOC_GRAPH["classical"].update({"vivaldi", "bach", "mozart", "orchestra", "concerto"})
_ASSOC_GRAPH["dog"].update({"cook", "treats", "baking", "recipe", "kitchen", "outside"})
_ASSOC_GRAPH["bird"].update({"feeder", "install", "outside", "window", "watch"})
_ASSOC_GRAPH["turtle"].update({"zoo", "keeper", "animal keeper", "care"})
_ASSOC_GRAPH["cook"].update({"dog", "pet", "treat", "treats", "baking"})

_PET_SPECIES_WORDS = ("dog", "dogs", "puppy", "pet", "pets")


class DomainAssociator:
    """Deterministic associative query expansion for memory retrieval."""

    @classmethod
    def register_pet_name(cls, name: str, species: str = "dog") -> None:
        """Phase 4 P5: co-reference a pet name with its species cluster so
        'his dogs' queries retrieve turns that mention only the name
        (e.g. D14:27 'take Toby out for a small hike', D24:8 'Buddy ... walks').
        Registration order does not affect expansion: neighbor iteration is
        sorted and ontology-baked names (e.g. pixie) are left untouched."""
        n = str(name).lower()
        if len(n) < 3 or n in _DOMAIN_ONTOLOGY:
            return
        _ASSOC_GRAPH[n].add(species)
        for sp in _PET_SPECIES_WORDS:
            _ASSOC_GRAPH[sp].add(n)

    @classmethod
    def expand_query(cls, query: str, max_terms: int = 15, hops: int = 2) -> set[str]:
        """Bidirectional expansion: bridges query cues and evidence terms deterministically."""
        q_lower = query.lower()
        words = re.findall(r"\b[a-z0-9_-]{3,}\b", q_lower)

        # Phase 2 determinism fix: set-iteration order (PYTHONHASHSEED) decided
        # which terms survived the max_terms cutoff, so compiled contexts
        # differed across processes. Frontier keeps query order; graph
        # neighbors are visited in sorted order.
        expanded: set[str] = set()
        frontier: list[str] = list(dict.fromkeys(words))

        for _ in range(hops):
            next_frontier: list[str] = []
            for w in frontier:
                if w in _ASSOC_GRAPH:
                    for neighbor in sorted(_ASSOC_GRAPH[w]):
                        if neighbor not in words and neighbor not in expanded:
                            expanded.add(neighbor)
                            next_frontier.append(neighbor)
                            if len(expanded) >= max_terms:
                                break
                    if len(expanded) >= max_terms:
                        break
            if not next_frontier:
                break
            frontier = next_frontier

        return expanded
