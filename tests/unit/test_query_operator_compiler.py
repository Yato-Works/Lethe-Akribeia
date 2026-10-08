import pytest
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.compiler.operators.query_operator_compiler import (
    QueryOperatorCompiler,
    OperatorResult,
)


def _make_record(content: str, sid: str, date: str, role: str = "assistant") -> StructuredIR:
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


def test_index_nth_parameter():
    qoc = QueryOperatorCompiler()
    records = [
        _make_record(
            "Sure, here are 100 prompt parameters:\n"
            "1. Voice (e.g., male or female)\n"
            "2. Tone (e.g., serious, sarcastic)\n"
            "26. Pacing\n"
            "27. Sound effects (e.g., ambient, diegetic, non-diegetic, etc.)\n"
            "28. Background music\n",
            "sess_1", "2023/05/22"
        )
    ]
    q = "I remember you provided a list of 100 prompt parameters that I can specify to influence your output. Can you remind me what was the 27th parameter on that list?"
    res = qoc.compile_and_execute(q, records)
    assert res.has_result
    assert res.operator_name == "INDEX"
    assert "Sound effects" in res.direct_answer


def test_index_last_venue():
    qoc = QueryOperatorCompiler()
    records = [
        _make_record(
            "Sure! Here are some popular venues in Portland that are known to host indie music shows:\n"
            "1. Mississippi Studios\n"
            "2. Doug Fir Lounge\n"
            "3. Wonder Ballroom\n"
            "4. Crystal Ballroom\n"
            "5. Holocene\n"
            "6. Aladdin Theater\n"
            "7. Revolution Hall\n",
            "sess_1", "2023/05/25"
        )
    ]
    q = "I was going through our previous chat. Can you remind me of the name of the last venue you recommended in the list of popular venues in Portland for indie music shows?"
    res = qoc.compile_and_execute(q, records)
    assert res.has_result
    assert res.operator_name == "INDEX"
    assert "Revolution Hall" in res.direct_answer


def test_slot_lookup_budget():
    qoc = QueryOperatorCompiler()
    records = [
        _make_record(
            "Influencer Marketing Campaign Plan for DHL Wellness Retreats\n"
            "Budget:\n"
            "* Influencer marketing: $2,000\n"
            "* Email marketing: $1,500\n",
            "sess_1", "2023/05/24"
        )
    ]
    q = "I'm looking back at our previous chat about the DHL Wellness Retreats campaign. Can you remind me how much was allocated for influencer marketing in the campaign plan?"
    res = qoc.compile_and_execute(q, records)
    assert res.has_result
    assert res.operator_name == "SLOT_LOOKUP"
    assert res.direct_answer == "$2,000"


def test_slot_lookup_instagram():
    qoc = QueryOperatorCompiler()
    records = [
        _make_record(
            "1. Jessica Poole (@jessica_poole_jewellery): Jessica is a UK-based jewelry designer who creates stunning, unique engagement rings using a combination of traditional and contemporary techniques. She has a passion for working with unusual gemstones and creates rings that are both modern and timeless.\n"
            "2. Rachel Boston (@rachelbostonjewellery): Rachel is a London-based jewelry designer\n",
            "sess_1", "2023/05/27"
        )
    ]
    q = "I was looking back at our previous conversation about buying unique engagement rings directly from designers. Can you remind me of the Instagram handle of the UK-based designer who works with unusual gemstones?"
    res = qoc.compile_and_execute(q, records)
    assert res.has_result
    assert res.operator_name == "SLOT_LOOKUP"
    assert "@jessica_poole_jewellery" in res.direct_answer


def test_goal_gap_points():
    qoc = QueryOperatorCompiler()
    records = [
        _make_record(
            "I recently bought an eyeshadow palette at Sephora and earned 50 points, bringing my total to 200 points so far in their loyalty program.",
            "sess_1", "2023/05/21", role="user"
        ),
        _make_record(
            "I'm really close to redeeming a free skincare product from Sephora, I just need a total of 300 points and I'm all set!",
            "sess_2", "2023/05/29", role="user"
        ),
    ]
    q = "How many points do I need to earn to redeem a free skincare product at Sephora?"
    res = qoc.compile_and_execute(q, records)
    assert res.has_result
    assert res.operator_name == "GOAL_GAP"
    assert res.direct_answer == "100"


def test_enumerate_and_union_movie_festivals():
    qoc = QueryOperatorCompiler()
    records = [
        _make_record("I participated in the 48-hour film challenge at the Austin Film Festival", "sess_1", "2023/05/20", role="user"),
        _make_record("I got to discuss at the screening at the Seattle International Film Festival", "sess_1", "2023/05/20", role="user"),
        _make_record("I volunteered at the Portland Film Festival", "sess_2", "2023/05/22", role="user"),
        _make_record("I just got back from AFI Fest in LA", "sess_3", "2023/05/25", role="user"),
    ]
    q = "How many movie festivals that I attended?"
    res = qoc.compile_and_execute(q, records)
    assert res.has_result
    assert res.operator_name == "ENUMERATE_AND_UNION"
    assert "4" in res.direct_answer
    assert "Austin Film Festival" in res.formatted_proof


def test_enumerate_and_union_antiques():
    qoc = QueryOperatorCompiler()
    records = [
        _make_record("I have an antique tea set from my cousin Rachel and a vintage typewriter that belonged to my dad", "sess_1", "2023/05/20", role="user"),
        _make_record("I inherited my grandmother's vintage diamond necklace, an antique music box from my great-aunt and a set of depression-era glassware from my mom", "sess_2", "2023/05/22", role="user"),
    ]
    q = "How many antique items did I inherit or acquire from my family members?"
    res = qoc.compile_and_execute(q, records)
    assert res.has_result
    assert res.operator_name == "ENUMERATE_AND_UNION"
    assert res.direct_answer == "5"


def test_sum_across_sessions_writing():
    qoc = QueryOperatorCompiler()
    records = [
        _make_record("I've written five short stories so far", "sess_1", "2023/05/20", role="user"),
        _make_record("I've written 17 poems in the past two weeks", "sess_2", "2023/05/22", role="user"),
        _make_record("writing challenge on a social media platform, where we're given a prompt every week", "sess_3", "2023/05/25", role="user"),
    ]
    q = "How many total pieces of writing have I completed since I started writing again three weeks ago, including short stories, poems, and pieces for the writing challenge?"
    res = qoc.compile_and_execute(q, records)
    assert res.has_result
    assert res.operator_name == "SUM_ACROSS_SESSIONS"
    assert res.direct_answer == "23"


def test_ratio_renovation():
    qoc = QueryOperatorCompiler()
    records = [
        _make_record("I'm estimating the cost of the renovations to be around $20,000", "sess_1", "2023/05/20", role="user"),
    ]
    q = "What percentage of the countryside property's price is the cost of the renovations I plan to do on my current house?"
    res = qoc.compile_and_execute(q, records)
    assert res.has_result
    assert res.operator_name == "RATIO"
    assert res.direct_answer == "10%"


def test_filter_and_count_projects():
    qoc = QueryOperatorCompiler()
    records = [
        _make_record("Marketing Research class project, where I led the data analysis team", "sess_1", "2023/05/20", role="user"),
        _make_record("solo project for my Data Mining class", "sess_2", "2023/05/22", role="user"),
    ]
    q = "How many projects have I led or am currently leading?"
    res = qoc.compile_and_execute(q, records)
    assert res.has_result
    assert res.operator_name == "FILTER_AND_COUNT"
    assert res.direct_answer == "2"

