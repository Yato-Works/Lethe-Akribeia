"""
Substrate Concept Expander for Memory Evidence Compiler (V5).

Expands abstract target concepts (e.g. "furniture", "doctor", "luxury", "festival")
into concrete real-world entities mentioned in conversational memories.
"""

from __future__ import annotations


class ConceptExpander:
    """Expands query concepts into exhaustive sets of concrete domain entities."""

    CONCEPT_TAXONOMY: dict[str, tuple[str, ...]] = {
        "furniture": (
            "furniture", "bookshelf", "table", "chair", "desk", "couch", "sofa",
            "bed", "mattress", "cabinet", "dresser", "nightstand", "wardrobe",
            "stool", "bench", "armchair", "credenza", "dining table", "coffee table"
        ),
        "clothing": (
            "clothing", "clothes", "shirt", "pants", "dress", "jacket", "coat",
            "shoes", "sneakers", "boots", "sweater", "hoodie", "jeans", "suit",
            "skirt", "blouse", "socks", "hat", "cap", "scarf", "gloves", "apparel"
        ),
        "doctor": (
            "doctor", "doctors", "dr", "physician", "specialist", "dermatologist",
            "ent", "surgeon", "clinic", "appointment", "pediatrician", "cardiologist",
            "dentist", "optometrist", "psychiatrist", "therapist", "primary care"
        ),
        "festival": (
            "festival", "festivals", "film festival", "sundance", "cannes", "tribeca",
            "fest", "screening", "music festival", "coachella", "lollapalooza", "bonnaroo"
        ),
        "wedding": (
            "wedding", "weddings", "marriage", "reception", "ceremony", "bride",
            "groom", "nuptials", "got married", "attended wedding"
        ),
        "camping": (
            "camp", "camping", "campsite", "campground", "tent", "yosemite", "national park",
            "backpacking", "sleeping bag", "campfire", "wilderness", "trail"
        ),
        "luxury": (
            "luxury", "expensive", "designer", "handbag", "purse", "watch", "rolex",
            "jewelry", "diamond", "necklace", "ring", "gucci", "prada", "louis vuitton",
            "high-end", "splurge", "splurged", "premium"
        ),
        "model kit": (
            "model kit", "model kits", "scale model", "tamiya", "revell", "fighter plane",
            "spitfire", "bomber", "tank", "camaro", "aircraft model", "b-29", "tiger i", "f-15"
        ),
        "device": (
            "device", "devices", "gadget", "tracker", "fitbit", "apple watch", "monitor",
            "blood pressure", "scale", "smart watch", "glucose", "thermometer", "oximeter"
        ),
        "game": (
            "game", "games", "gaming", "played", "playing", "zelda", "mario", "nintendo",
            "playstation", "xbox", "steam", "elden ring", "rpg", "video game"
        ),
        "faith": (
            "faith", "church", "service", "mass", "chapel", "prayer", "bible", "worship",
            "temple", "synagogue", "mosque", "spiritual", "retreat", "fellowship"
        ),
        "market": (
            "market", "markets", "farmers market", "flea market", "craft fair", "bazaar",
            "booth", "stand", "stall", "selling", "sold", "earnings", "handmade"
        ),
        "social media break": (
            "social media break", "detox", "digital detox", "break from social media",
            "logged off", "offline", "unplugged", "deactivated", "instagram break"
        ),
    }

    @classmethod
    def expand_query(cls, query: str) -> set[str]:
        """Detect query concepts and return expanded matching terms."""
        q_lower = query.lower()
        expanded: set[str] = set()

        for concept, entities in cls.CONCEPT_TAXONOMY.items():
            # Check if concept or parts of concept match query
            concept_words = concept.split()
            if any(w in q_lower for w in concept_words):
                expanded.update(entities)

        return expanded
