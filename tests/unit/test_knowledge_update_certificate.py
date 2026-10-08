import pytest
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.compiler.certificates.knowledge_update_certificate import (
    KnowledgeUpdateCertificate,
    KnowledgeUpdateCertificateEngine,
)


def _make_record(content: str, sid: str, date: str, role: str = "user") -> StructuredIR:
    r = StructuredIR(
        entity="general",
        property="statement",
        value=content,
        raw_content=f"[{sid} on {date}] {role.capitalize()}: {content}",
        time_scope=date,
        source=role,
    )
    r.metadata = {"session_id": sid}
    return r


def test_supersedes_location_update():
    engine = KnowledgeUpdateCertificateEngine()
    records = [
        _make_record(
            "By the way, I need to take care of my old sneakers, I've been keeping them under my bed for storage, and they're starting to smell.",
            "sess_1", "2023/05/25 (Thu) 10:04"
        ),
        _make_record(
            "By the way, I recently cleaned out my closet and I'm looking forward to get rid of some of my old sneakers in a shoe rack in it, they're currently taking up space.",
            "sess_2", "2023/05/29 (Mon) 15:01"
        ),
    ]
    q = "Where do I currently keep my old sneakers?"
    cert = engine.generate(q, records)
    assert cert.has_certificate
    assert "shoe rack in my closet" in cert.latest_value.lower() or "shoe rack" in cert.latest_value.lower()
    assert cert.direct_answer_candidate is not None
    assert "shoe rack" in cert.direct_answer_candidate.lower()


def test_supersedes_travel_destination():
    engine = KnowledgeUpdateCertificateEngine()
    records = [
        _make_record(
            "I'm planning a birthday trip to Hawaii and I was wondering if you could recommend some good hiking trails on Kauai?",
            "sess_1", "2023/05/22 (Mon) 11:21"
        ),
        _make_record(
            "I'm planning a birthday trip to Hawaii in October and I was wondering if you could recommend some good snorkeling spots on the island? I'm actually planning to stay on Oahu, so the Hanauma Bay Nature Preserve and Shark's Cove sound perfect for me.",
            "sess_2", "2023/05/25 (Thu) 18:17"
        ),
    ]
    q = "Where am I planning to stay for my birthday trip to Hawaii?"
    cert = engine.generate(q, records)
    assert cert.has_certificate
    assert cert.latest_value.lower() == "oahu"
    assert cert.direct_answer_candidate == "Oahu"


def test_supersedes_count_update():
    engine = KnowledgeUpdateCertificateEngine()
    records = [
        _make_record(
            "have you tried any good Korean restaurants in your city lately? I've tried three different ones recently, and each has its own unique flavor and style.",
            "sess_1", "2023/08/11 (Fri) 09:09"
        ),
        _make_record(
            "have you tried any good Korean restaurants in my city lately? I've tried four different ones so far, and I'm always looking for new recommendations.",
            "sess_2", "2023/09/30 (Sat) 15:06"
        ),
    ]
    q = "How many Korean restaurants have I tried in my city?"
    cert = engine.generate(q, records)
    assert cert.has_certificate
    assert cert.direct_answer_candidate in ("4", "four")


def test_supersedes_rewards_stars():
    engine = KnowledgeUpdateCertificateEngine()
    records = [
        _make_record(
            "Wait, I apologize for the mistake earlier! To correct myself, I actually need 125 stars to reach the Gold level on the Starbucks Rewards app, not 400.",
            "sess_1", "2023/07/11 (Tue) 00:07"
        ),
        _make_record(
            "Actually, I need 120 stars to reach the gold level, not 300. Anyway, thanks for the tips on earning stars.",
            "sess_2", "2023/07/30 (Sun) 02:08"
        ),
    ]
    q = "How many stars do I need to reach the gold level on my Starbucks Rewards app?"
    cert = engine.generate(q, records)
    assert cert.has_certificate
    assert cert.direct_answer_candidate == "120"


def test_incremental_addition():
    engine = KnowledgeUpdateCertificateEngine()
    records = [
        _make_record(
            "By the way, I was thinking of organizing my pre-1920 American coins by denomination and mint mark. I have a total of 37 coins in that collection, and I think it would be cool to see them all displayed together.",
            "sess_1", "2023/05/27 (Sat) 18:09"
        ),
        _make_record(
            "By the way, I just added a new coin to my collection of pre-1920 American coins - a 1915-S Barber quarter.",
            "sess_2", "2023/05/29 (Mon) 00:54"
        ),
    ]
    q = "How many pre-1920 American coins do I have in my collection?"
    cert = engine.generate(q, records)
    assert cert.has_certificate
    assert cert.direct_answer_candidate == "38"
