"""Verify the two claims from the Cognee digest that would drive a Lethe build.

The digest calls UUID5 identity and the SHA-256 provenance chain its \"crown
jewels\" and tags both [DETERMINISTIC-FEASIBLE].  That tag is the entire reason
they are candidates for a NO-LLM project, so the claim is checked here rather
than trusted: both must be pure stdlib with no model, no network, no randomness.

UUID5 must be a pure function of the normalized surface -- same entity across
different surface spellings collapses to one identity, and a genuinely different
entity must not collide.  The chain must be order- and tamper-sensitive.
"""
import hashlib
import uuid

NS = uuid.NAMESPACE_OID


def identity_key(entity_type: str, values: list[str]) -> uuid.UUID:
    """Cognee's scheme: uuid5 over 'ClassName:v1|v2', lowercased, spaces to _."""
    norm = "|".join(v.strip().lower().replace(" ", "_") for v in values)
    return uuid.uuid5(NS, f"{entity_type}:{norm}")


def main() -> int:
    print("=== UUID5 entity identity (digest 'crown jewel') ===")
    a = identity_key("Pet", ["Melanie", "Bailey"])
    b = identity_key("Pet", ["Melanie", "bailey "])   # different surface form
    c = identity_key("Pet", ["Melanie", "Luna"])      # genuinely different entity
    print(f"  same entity, different surface -> same id : {a == b}")
    print(f"  different entity               -> differs : {a != c}")
    assert a == b, "surface variation must collapse"
    assert a != c, "distinct entities must not collide"
    assert uuid.uuid5(NS, f"Pet:melanie|bailey") == a, "must be a pure function"

    print("\n=== SHA-256 provenance chain ===")
    chain = []
    prev = b"\x00" * 32
    for entry in (b"turn D7:18", b"turn D13:4", b"turn D13:6"):
        prev = hashlib.sha256(prev + entry).digest()
        chain.append(prev)
    print(f"  3-entry chain built, head = {chain[-1].hex()[:24]}...")

    rebuilt = []
    prev = b"\x00" * 32
    for entry in (b"turn D7:18", b"turn D13:4", b"turn D13:6"):
        prev = hashlib.sha256(prev + entry).digest()
        rebuilt.append(prev)
    print(f"  same order        -> verifies : {chain == rebuilt}")

    reordered = []
    prev = b"\x00" * 32
    for entry in (b"turn D7:18", b"turn D13:6", b"turn D13:4"):
        prev = hashlib.sha256(prev + entry).digest()
        reordered.append(prev)
    print(f"  reordered         -> detected : {chain != reordered}")

    tampered = []
    prev = b"\x00" * 32
    for entry in (b"turn D7:18", b"turn D13:5", b"turn D13:6"):
        prev = hashlib.sha256(prev + entry).digest()
        tampered.append(prev)
    print(f"  single edit       -> detected : {chain != tampered}")
    assert chain != reordered and chain != tampered

    print("\nVERDICT: both are pure stdlib. No model, no network, no randomness.")
    print("The digest's [DETERMINISTIC-FEASIBLE] tag holds for both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
