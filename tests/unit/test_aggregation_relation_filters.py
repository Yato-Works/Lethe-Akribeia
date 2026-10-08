import pytest
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.compiler.certificates.aggregation_certificate import (
    AggregationCertificateEngine,
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


def test_exclusion_constraint():
    engine = AggregationCertificateEngine()
    records = [
        _make_record(
            "I'm actually working on a Master's thesis in AI and healthcare, and I've been using Trello to manage my project. I've created separate boards for my thesis, Data Mining project, and Database Systems project. It's been really helpful in staying organized and prioritizing tasks.",
            "sess_1", "2023/05/25 (Thu) 11:44"
        ),
        _make_record(
            "By the way, I've been learning a lot about data analysis in my Data Mining course, which has a group project that's keeping me pretty busy.",
            "sess_2", "2023/05/26 (Fri) 15:03"
        ),
        _make_record(
            "By the way, I've also been working on a group project for my Database Systems course, so I'm juggling multiple projects at the moment.",
            "sess_3", "2023/05/28 (Sun) 03:58"
        ),
    ]
    q = "How many projects have I been working on simultaneously, excluding my thesis?"
    cert = engine.generate(q, records)
    assert cert.has_certificate
    assert cert.total_count == 2
    assert cert.direct_answer_candidate in ("2", "two")


def test_temporal_month_scope_count():
    engine = AggregationCertificateEngine()
    records = [
        _make_record(
            "By the way, I finally went to see my primary care physician, Dr. Smith, on March 3rd, and he diagnosed me with bronchitis.",
            "sess_1", "2023/03/27 (Mon) 20:56"
        ),
        _make_record(
            "I'm scheduled to have an EMG test with Dr. Johnson on April 1st to rule out any nerve damage, so I'm a bit anxious about what to expect.",
            "sess_2", "2023/03/24 (Fri) 14:12"
        ),
        _make_record(
            "I had a follow-up appointment with my orthopedic surgeon, Dr. Thompson, on March 20th to check on my ACL recovery, and he cleared me to start physical therapy.",
            "sess_3", "2023/03/24 (Fri) 14:12"
        ),
    ]
    q = "How many doctor's appointments did I go to in March?"
    cert = engine.generate(q, records)
    assert cert.has_certificate
    assert cert.total_count == 2
    assert cert.direct_answer_candidate in ("2", "two")


def test_status_exclusion_camping():
    engine = AggregationCertificateEngine()
    records = [
        _make_record(
            "By the way, I've been loving the scenic drives and hikes in Utah lately - we had a 7-day family road trip there in February. We did a lot of driving and hiking, but not camping for this time.",
            "sess_1", "2023/04/29 (Sat) 10:31"
        ),
        _make_record(
            "By the way, I just got back from an amazing 5-day camping trip to Yellowstone National Park last month, and I'm still buzzing from the experience.",
            "sess_2", "2023/04/29 (Sat) 15:49"
        ),
        _make_record(
            "By the way, I just got back from a 3-day solo camping trip to Big Sur in early April and my current boots did okay, but I think I need something better.",
            "sess_3", "2023/04/29 (Sat) 20:02"
        ),
    ]
    q = "How many days did I spend on camping trips in the United States this year?"
    cert = engine.generate(q, records)
    assert cert.has_certificate
    assert cert.total_count == 8 or cert.difference_value == 8.0
    assert "8" in str(cert.direct_answer_candidate or cert.total_count)


def test_status_exclusion_weddings_attended():
    engine = AggregationCertificateEngine()
    records = [
        _make_record(
            "My cousin Emily's wedding in the city was really lovely, and I think part of what made it so special was the intimate, romantic atmosphere.",
            "sess_1", "2023/09/24 (Sun) 14:26"
        ),
        _make_record(
            "My cousin Rachel's wedding at the vineyard had a wine tasting, and it was so much fun.",
            "sess_1", "2023/09/24 (Sun) 14:26"
        ),
        _make_record(
            "By the way, I just got back from a friend's wedding last weekend, and it was amazing - the bride, Jen, looked stunning in her bohemian-inspired dress, and her husband, Tom, was clearly smitten with her.",
            "sess_2", "2023/10/15 (Sun) 19:23"
        ),
        _make_record(
            "I'm planning my own wedding and I was wondering if you could give me some tips on how to choose the perfect venue.",
            "sess_2", "2023/10/15 (Sun) 19:23"
        ),
    ]
    q = "How many weddings have I attended in this year?"
    cert = engine.generate(q, records)
    assert cert.has_certificate
    assert cert.total_count == 3
    assert cert.direct_answer_candidate in ("3", "three")
