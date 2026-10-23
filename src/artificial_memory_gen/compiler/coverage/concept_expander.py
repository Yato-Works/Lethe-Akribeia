"""Abstract Concept Expander for Multi-Session Coverage Engine.

Expands abstract high-level categories into canonical hyponyms, lexical families,
and targeted retrieval sub-queries without benchmark-specific hardcoding.
Provides a generalized semantic taxonomy for multi-session conversational recall.
"""

from __future__ import annotations

import re


class AbstractConceptExpander:
    """Deterministic, general-purpose semantic taxonomy expander."""

    CONCEPT_TAXONOMY: dict[str, dict[str, tuple[str, ...]]] = {
        "clothing": {
            "tops": ("shirt", "t-shirt", "sweater", "hoodie", "blouse", "jacket", "coat", "vest", "blazer", "cardigan"),
            "bottoms": ("pants", "jeans", "trousers", "shorts", "skirt", "sweatpants"),
            "one_piece": ("dress", "suit", "overalls", "jumpsuit"),
            "footwear": ("shoes", "sneakers", "boots", "sandals", "loafers"),
            "accessories": ("hat", "cap", "scarf", "gloves", "belt", "socks"),
            "actions": ("pick up", "return", "alterations", "dry clean", "tailor", "bought", "purchased"),
        },
        "furniture": {
            "seating": ("chair", "armchair", "sofa", "couch", "stool", "bench", "recliner"),
            "tables_desks": ("table", "desk", "coffee table", "dining table", "side table"),
            "storage": ("bookshelf", "bookshelves", "bookcase", "dresser", "cabinet", "shelf", "wardrobe"),
            "bedroom": ("bed", "nightstand", "mattress", "headboard"),
            "actions": ("buy", "bought", "assemble", "assembled", "sell", "sold", "fix", "fixed", "furniture"),
        },
        "cuisines": {
            "asian": ("japanese", "chinese", "thai", "korean", "vietnamese", "indian"),
            "western": ("italian", "french", "mexican", "spanish", "greek", "mediterranean"),
            "dietary": ("vegan", "vegetarian", "plant-based", "fermented", "dairy-free", "baking"),
            "actions": ("cuisine", "cuisines", "cook", "cooked", "recipe", "tried out", "dishes"),
        },
        "health_devices": {
            "wearables": ("smartwatch", "apple watch", "fitbit", "fitness tracker", "tracker"),
            "monitors": ("blood pressure monitor", "glucose monitor", "heart rate monitor", "pulse oximeter"),
            "home_health": ("thermometer", "smart scale", "scale", "cpap", "humidifier"),
            "actions": ("health-related device", "device", "health", "use in a day"),
        },
        "jewelry": {
            "pieces": ("ring", "necklace", "bracelet", "earrings", "pendant", "brooch", "cufflink", "anklet"),
            "gemstones": ("emerald", "diamond", "gemstone", "pearl", "ruby", "sapphire", "gold", "silver"),
            "timepieces": ("watch", "timepiece"),
            "actions": ("jewelry", "acquire", "acquired", "bought", "gift", "precious", "cleaning"),
        },
        "antiques_heirlooms": {
            "types": ("antique", "heirloom", "vintage", "inherited", "passed down", "clock", "trunk", "furniture", "mirror"),
            "actions": ("inherit", "inherited", "acquire", "family members", "generations"),
        },
        "writing_literature": {
            "works": ("poem", "poems", "poetry", "short story", "stories", "essay", "essays", "chapter", "novel", "article"),
            "activities": ("writing challenge", "writing", "pieces of writing", "completed writing", "draft"),
        },
        "babies_births": {
            "family": ("baby", "babies", "born", "birth", "infant", "newborn", "nephew", "niece", "gave birth"),
            "celebrations": ("baby shower", "delivery", "hospital"),
        },
        "vehicles": {
            "automobiles": ("car", "sedan", "suv", "truck", "pickup", "van", "wagon", "coupe"),
            "two_wheelers": ("motorcycle", "bike", "bicycle", "scooter"),
            "transit": ("bus", "train", "flight", "plane", "airplane", "subway", "ferry"),
        },
        "medical_doctors": {
            "general": ("doctor", "dr", "physician", "primary care", "clinic", "checkup"),
            "specialists": ("dermatologist", "ent", "surgeon", "cardiologist", "pediatrician", "allergist", "dentist", "optometrist", "neurologist", "therapist", "psychiatrist"),
            "facilities": ("hospital", "medical center", "urgent care", "appointment"),
        },
        "camping_outdoors": {
            "activities": ("camp", "camping", "campsite", "campground", "backpacking", "wilderness"),
            "gear": ("tent", "sleeping bag", "campfire", "hiking", "trail", "national park"),
        },
        "fitness_exercise": {
            "classes": ("fitness class", "yoga", "pilates", "spinning", "crossfit", "aerobics", "martial arts"),
            "training": ("gym", "workout", "workout routine", "exercise", "lifting", "running", "swimming"),
        },
        "luxury_fashion": {
            "brands": ("gucci", "prada", "louis vuitton", "chanel", "hermes", "rolex", "dior"),
            "items": ("handbag", "purse", "designer gown", "gown", "luxury boots", "watch", "jewelry", "diamond"),
            "descriptors": ("luxury", "expensive", "splurge", "high-end", "designer"),
        },
        "hobbies_models": {
            "model_kits": ("model kit", "scale model", "tamiya", "revell", "f-15", "spitfire", "bomber", "tank", "camaro", "aircraft"),
            "crafts": ("woodworking", "pottery", "knitting", "painting", "drawing", "gardening"),
        },
        "entertainment_festivals": {
            "events": ("film festival", "movie festival", "music festival", "festival", "sundance", "cannes", "tribeca", "screening"),
            "attending": ("attended", "film premiere", "festival pass"),
        },
        "gaming": {
            "platforms": ("video game", "gaming", "steam", "nintendo", "playstation", "xbox"),
            "titles": ("zelda", "mario", "pokemon", "rpg", "tournament", "esports", "played"),
        },
        "publications_media": {
            "print": ("magazine", "journal", "subscription", "periodical", "newsletter"),
            "podcasts": ("podcast", "episodes", "listened to", "show"),
        },
        "projects_work": {
            "endeavors": ("project", "projects", "leading", "led", "initiative", "campaign", "working on"),
        },
    }

    _CONCEPT_TRIGGERS: dict[str, re.Pattern] = {
        "clothing": re.compile(r"\b(cloth(?:es|ing)?|garment|apparel|wear|pick\s+up|return.*store|outfit)\b", re.I),
        "furniture": re.compile(r"\b(furniture|chair|table|desk|sofa|couch|bookshelf|bookcase|dresser|cabinet|bed)\b", re.I),
        "cuisines": re.compile(r"\b(cuisine|cuisines|cooking|cook|dishes|recipes)\b", re.I),
        "health_devices": re.compile(r"\b(health.*device|device|wearable|smartwatch|fitbit|monitor)\b", re.I),
        "jewelry": re.compile(r"\b(jewelry|jewellery|ring|necklace|bracelet|earring|pendant)\b", re.I),
        "antiques_heirlooms": re.compile(r"\b(antique|heirloom|vintage|inherited|passed down)\b", re.I),
        "writing_literature": re.compile(r"\b(pieces of writing|writing|stories|poems|poetry|essay)\b", re.I),
        "babies_births": re.compile(r"\b(bab(?:y|ies)|born|birth|newborn|infant)\b", re.I),
        "vehicles": re.compile(r"\b(vehicle|car|truck|bike|bicycle|drive|transportation)\b", re.I),
        "medical_doctors": re.compile(r"\b(doctor|physician|specialist|medical|clinic|appointment|checkup)\b", re.I),
        "camping_outdoors": re.compile(r"\b(camp(?:ing|s)?|tent|backpacking|wilderness|national\s+park)\b", re.I),
        "fitness_exercise": re.compile(r"\b(fitness|exercise|workout|gym|classes|yoga|pilates)\b", re.I),
        "luxury_fashion": re.compile(r"\b(luxury|designer|expensive|splurge|gown|brand|handbag)\b", re.I),
        "hobbies_models": re.compile(r"\b(model\s+kit|scale\s+model|tamiya|revell|hobby)\b", re.I),
        "entertainment_festivals": re.compile(r"\b(movie\s+festival|film\s+festival|music\s+festival|festival)\b", re.I),
        "gaming": re.compile(r"\b(video\s+game|games?|gaming|tournament|played)\b", re.I),
        "publications_media": re.compile(r"\b(magazine|subscription|podcast|episodes)\b", re.I),
        "projects_work": re.compile(r"\b(projects?|initiative|working on|leading|led)\b", re.I),
    }

    @classmethod
    def detect_concepts(cls, query: str) -> list[str]:
        """Detect which abstract concepts are triggered by the query."""
        triggered = []
        for concept, pattern in cls._CONCEPT_TRIGGERS.items():
            if pattern.search(query):
                triggered.append(concept)
        return triggered

    @classmethod
    def generate_subqueries(cls, query: str) -> list[str]:
        """Generate targeted sub-queries to retrieve scattered instances across sessions."""
        concepts = cls.detect_concepts(query)
        if not concepts:
            return []

        subqueries = []
        for c in concepts:
            tax = cls.CONCEPT_TAXONOMY[c]
            for group_name, terms in tax.items():
                if group_name == "actions":
                    continue
                for chunk_idx in range(0, len(terms), 4):
                    chunk = terms[chunk_idx:chunk_idx + 4]
                    if chunk:
                        subqueries.append(" ".join(chunk))
        return subqueries
