"""Entity identity: both directions.

The recall half is easy -- merge aggressively and recall goes up.  The reason this
module exists is the OTHER half: it must NOT merge what the evidence does not
support.  Every test here is one of those two obligations.

Eli's two stated dangers are tested explicitly:
  * Bob also owns a Bailey          -> must not join Melanie's Bailey
  * 'used to own' vs 'now owns'     -> two same-named mentions across a time
                                       change are DISTINCT entities
"""
import uuid

from artificial_memory.skills.autonomous_engines.entity_identity import (
    IdentityResolver,
    Mention,
    identity_key,
)


def m(turn: str, speaker: str, surface: str, text: str, owner: str = "") -> Mention:
    return Mention(turn_id=turn, speaker=speaker, surface=surface, text=text,
                   owner=owner, entity_type="pet")


# ------------------------------------------------------- ID generation is stable
def test_id_is_deterministic_and_surface_insensitive() -> None:
    a = identity_key("pet", "Melanie", "Bailey")
    b = identity_key("pet", "melanie ", " bailey")
    assert a == b


def test_id_separates_owner_and_type() -> None:
    mel = identity_key("pet", "Melanie", "Bailey")
    bob = identity_key("pet", "Bob", "Bailey")
    person = identity_key("person", "Melanie", "Bailey")
    assert mel != bob, "same name under a different owner is a different entity"
    assert mel != person, "a pet and a person sharing a name are different"


def test_id_is_a_pure_function_of_its_arguments() -> None:
    assert identity_key("pet", "Melanie", "Bailey") == \
        identity_key("pet", "Melanie", "Bailey")


# ------------------------------------------------------------- correct merges
def test_same_owner_same_name_across_turns_merges() -> None:
    # Fixture text must not itself contain a discontinuity cue, or the module is
    # right to refuse the merge and the test is wrong.  An earlier version of this
    # test used "We got another cat named Bailey too" and failed for exactly that
    # reason -- the module was correct and the fixture was not.
    r = IdentityResolver()
    r.add(m("D7:18", "Melanie", "Bailey",
            "Melanie: Luna and Oliver! They are sweet.", "Melanie"))
    ent = r.add(m("D13:4", "Melanie", "Bailey",
                  "Melanie: Bailey is settling in nicely.", "Melanie"))
    assert len(r.entities) == 1
    assert ent.turn_ids == ["D7:18", "D13:4"]


def test_merge_retains_provenance_for_every_item() -> None:
    r = IdentityResolver()
    r.add(m("D1:1", "Melanie", "Luna", "Melanie: My pets are Luna and Oliver.",
            "Melanie"))
    r.add(m("D5:2", "Melanie", "Bailey", "Melanie: Bailey is the third.",
            "Melanie"))
    for ent in r.entities.values():
        assert ent.turn_ids, "an entity with no provenance cannot be audited"
        for mention in ent.mentions:
            assert mention.text, "verbatim turn text is kept for audit"


# ------------------------------------------------------- Eli's danger 1: owner
def test_same_name_under_different_owners_does_not_merge() -> None:
    """Bob also owns a Bailey.  A name-only key would merge them."""
    r = IdentityResolver()
    mel = r.add(m("D1:1", "Melanie", "Bailey",
                  "Melanie: my cat Bailey is asleep.", "Melanie"))
    bob = r.add(m("D1:2", "Bob", "Bailey",
                  "Bob: my Bailey just knocked over a glass.", "Bob"))
    assert mel.key != bob.key
    assert len(r.entities) == 2
    assert mel.turn_ids == ["D1:1"]
    assert bob.turn_ids == ["D1:2"]


def test_unattributed_mention_never_joins_an_owned_one() -> None:
    r = IdentityResolver()
    r.add(m("D1:1", "Melanie", "Bailey", "Melanie: Bailey is mine.", "Melanie"))
    loose = r.add(m("D9:3", "Caroline", "Bailey",
                    "Caroline: I saw Bailey at the park.", ""))
    assert loose.turn_ids == ["D9:3"], "unknown owner must not inherit one"


# --------------------------------------------- Eli's danger 2: time discontinuity
def test_same_name_across_a_time_change_is_two_entities() -> None:
    """The Bailey that was given away and the Bailey adopted later are distinct."""
    r = IdentityResolver()
    old = r.add(m("D1:1", "Melanie", "Bailey",
                  "Melanie: I had to give Bailey away last year.", "Melanie"))
    new = r.add(m("D20:4", "Melanie", "Bailey",
                  "Melanie: I adopted another cat and named him Bailey.", "Melanie"))
    assert old.key != new.key
    assert len(r.entities) == 2


def test_explicit_continuity_assertion_overrides_a_generic_cue() -> None:
    r = IdentityResolver()
    r.add(m("D1:1", "Melanie", "Bailey",
            "Melanie: I used to have a Bailey.", "Melanie"))
    ent = r.add(m("D4:2", "Melanie", "Bailey",
                  "Melanie: it is the same Bailey, still around.", "Melanie"))
    assert ent.turn_ids == ["D1:1", "D4:2"], "'same ... still around' is continuity"


def test_superseded_entity_is_kept_not_deleted() -> None:
    """'What did Melanie used to have' must stay answerable."""
    r = IdentityResolver()
    r.add(m("D1:1", "Melanie", "Bailey",
            "Melanie: I gave Bailey away.", "Melanie"))
    r.add(m("D20:4", "Melanie", "Bailey",
            "Melanie: I adopted another Bailey.", "Melanie"))
    assert len(r.entities) == 2, "nothing is discarded, only separated"


# --------------------------------------------------- the counterweight to recall
def test_wrong_merge_audit_is_clean_on_a_correct_run() -> None:
    r = IdentityResolver()
    r.add(m("D1:1", "Melanie", "Bailey", "Melanie: Bailey.", "Melanie"))
    r.add(m("D1:2", "Bob", "Bailey", "Bob: my Bailey.", "Bob"))
    assert r.wrong_merge_audit() == []


def test_wrong_merge_audit_catches_a_forced_multi_owner_entity() -> None:
    """The audit must fire when a merge is wrong, or it is not a counterweight."""
    r = IdentityResolver()
    ent = r.add(m("D1:1", "Melanie", "Bailey", "Melanie: Bailey.", "Melanie"))
    # Force the illegal merge directly, bypassing add()'s guard, to prove the
    # audit detects it rather than merely never being exercised.
    ent.mentions.append(Mention(turn_id="D1:2", speaker="Bob", surface="Bailey",
                                text="Bob: my Bailey.", owner="Bob",
                                entity_type="pet"))
    findings = r.wrong_merge_audit()
    kinds = {f["kind"] for f in findings}
    assert "multi_owner" in kinds


def test_wrong_merge_audit_catches_a_merge_across_discontinuity() -> None:
    r = IdentityResolver()
    ent = r.add(m("D1:1", "Melanie", "Bailey", "Melanie: Bailey.", "Melanie"))
    ent.mentions.append(Mention(
        turn_id="D20:4", speaker="Melanie", surface="Bailey",
        text="Melanie: I adopted another Bailey.", owner="Melanie",
        entity_type="pet"))
    kinds = {f["kind"] for f in r.wrong_merge_audit()}
    assert "merged_across_discontinuity" in kinds
