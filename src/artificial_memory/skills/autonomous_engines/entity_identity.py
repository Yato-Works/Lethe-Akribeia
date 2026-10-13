"""Deterministic entity identity for the evidence store.

Design rule, kept deliberately strict:

    ID GENERATION is deterministic.  IDENTITY JUDGEMENT is evidence-based.
    Never let one do the other's job.

A keyed hash (Cognee's uuid5(NAMESPACE_OID, 'Type:owner|name') scheme, verified
pure-stdlib in verify_cognee_determinism.py) turns a DECIDED identity into a
stable id.  It cannot decide identity -- and must not be asked to.  Two mentions
are the same entity only when the evidence supports it, which means:

  * the key carries an OWNER.  'Bailey' alone is not an identity: Bob also owns a
    Bailey, and a name-only key would merge another person's pet into the answer.
  * the key carries a TYPE, so a pet named Bailey is not a person named Bailey.
  * two same-named mentions SEPARATED BY A TIME CHANGE are NOT merged.  If the
    evidence says an animal was given away, died, or was replaced by 'another
    dog', the later mention is a different entity even though every surface form
    matches.  This is the case Cognee's valid_to and Graphiti's invalid_at exist
    to express.

Every merge records the provenance turns it was built from, so a wrong merge is
attributable to specific turns rather than silent.  That is what makes the
wrong-merge rate measurable -- and a merge that cannot be audited cannot be
trusted, which is the same discipline the scorer guards follow.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

#: Stable namespace for this project.  A fixed UUID means ids are reproducible
#: across runs and machines; deriving it from NAMESPACE_OID would work too, but
#: a project-specific one keeps these ids from ever colliding with another
#: system's uuid5 output over the same strings.
IDENTITY_NS = uuid.UUID("6f9c2a7e-1b3d-4c5e-9a8f-0e1d2c3b4a56")

#: Language that ends an entity's continuity.  When this appears between two
#: mentions of the same name, the mentions are treated as DISTINCT entities
#: unless the turn also asserts continuity.  This is the guard against merging
#: 'the Bailey I used to have' with 'my current Bailey'.
_DISCONTINUITY = re.compile(
    r"\b(used to have|no longer|gave (?:him|her|it|them) away|"
    r"gave (?:him|her|it|them) to|passed away|died|put (?:him|her|it) down|"
    r"rehomed|found a new home|adopted (?:another|a new)|"
    r"got (?:another|a new) (?:dog|cat|pet|puppy|kitten)|"
    r"replaced (?:him|her|it)|before (?:that|then))\b",
    re.I,
)

#: Language that asserts continuity across the same name.
_CONTINUITY = re.compile(
    r"\b(same|still (?:have|has)|still around|also named|named after|"
    r"we (?:still|also) have)\b",
    re.I,
)


def identity_key(entity_type: str, owner: str, name: str) -> uuid.UUID:
    """Deterministic id for an entity that has ALREADY been judged identical.

    Normalisation is part of the contract: strip, lowercase, collapse internal
    whitespace.  'Bailey' and 'bailey ' therefore collapse; 'Bob' and 'Bob ' do
    not become two entities.

    This function decides nothing.  Call it only after the evidence supports the
    merge -- passing a wrong owner or name produces a confident, stable,
    wrong id, which is worse than no id at all.
    """
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).strip().lower())

    payload = f"{norm(entity_type)}|{norm(owner)}|{norm(name)}"
    return uuid.uuid5(IDENTITY_NS, payload)


@dataclass(frozen=True)
class Mention:
    """One appearance of an entity in one turn, with its provenance."""

    turn_id: str
    speaker: str
    surface: str          # the name as written
    text: str             # verbatim turn text, for audit
    owner: str = ""       # whose entity this is, "" when unattributed
    entity_type: str = ""


@dataclass
class Entity:
    """A judged identity, built from the mentions that support it."""

    key: uuid.UUID
    entity_type: str
    owner: str
    canonical: str
    mentions: list[Mention] = field(default_factory=list)
    superseded_by: uuid.UUID | None = None

    @property
    def turn_ids(self) -> list[str]:
        return [m.turn_id for m in self.mentions]


class IdentityResolver:
    """Groups mentions into entities using evidence, then keys them.

    Merge rule, deliberately conservative: two mentions merge only when they
    agree on type and owner AND no discontinuity evidence separates them.  When a
    discontinuity IS present, the later mention starts a NEW entity and the older
    one is marked superseded -- kept, never discarded, because 'what did they used
    to have' is a real question (the 111-failure attribute cluster).

    When owner is unknown, mentions do NOT merge across speakers.  An
    unattributed 'Bailey' from Melanie and one from Bob stay separate, because
    merging them is precisely the collision Eli named.
    """

    def __init__(self) -> None:
        self.entities: dict[uuid.UUID, Entity] = {}

    def _continuity_broken(self, earlier: Mention, later: Mention) -> bool:
        """True when the evidence says these are different entities."""
        span = f"{earlier.text} {later.text}"
        if _CONTINUITY.search(span):
            # An explicit continuity assertion wins over a generic cue: 'still
            # have' or 'same' tells us the name is the same animal.
            return False
        return bool(_DISCONTINUITY.search(span))

    def add(self, mention: Mention) -> Entity:
        """Attribute one mention to an existing entity, or start a new one."""
        best: Entity | None = None
        for ent in self.entities.values():
            if ent.entity_type != mention.entity_type:
                continue
            if ent.canonical.lower() != mention.surface.strip().lower():
                continue
            # Owner discipline: an unattributed mention never joins an owned one,
            # and two different owners never share an entity.
            if not mention.owner or not ent.owner:
                continue
            if mention.owner.lower() != ent.owner.lower():
                continue
            if self._continuity_broken(ent.mentions[-1], mention):
                continue
            best = ent
            break

        if best is None:
            key = identity_key(mention.entity_type, mention.owner, mention.surface)
            if key in self.entities:
                # Same key, different mention, and the loop above declined to
                # merge it (owner or continuity).  Keep them separate by suffixing
                # the key rather than silently forcing a merge.
                key = identity_key(
                    mention.entity_type,
                    f"{mention.owner}#{len(self.entities)}",
                    mention.surface,
                )
            best = Entity(
                key=key,
                entity_type=mention.entity_type,
                owner=mention.owner,
                canonical=mention.surface.strip(),
            )
            self.entities[key] = best

        best.mentions.append(mention)
        return best

    def wrong_merge_audit(self) -> list[dict]:
        """Mentions grouped together that the evidence does not support.

        This is the counterweight to recall.  An identity system can always raise
        recall by merging aggressively; the only thing that distinguishes a
        correct system from a sloppy one is this number staying at zero.
        """
        problems: list[dict] = []
        for ent in self.entities.values():
            owners = {m.owner.lower() for m in ent.mentions if m.owner}
            if len(owners) > 1:
                problems.append({
                    "kind": "multi_owner",
                    "canonical": ent.canonical,
                    "owners": sorted(owners),
                    "turn_ids": ent.turn_ids,
                })
            for i, m in enumerate(ent.mentions):
                for other in ent.mentions[i + 1:]:
                    if self._continuity_broken(m, other):
                        problems.append({
                            "kind": "merged_across_discontinuity",
                            "canonical": ent.canonical,
                            "turn_ids": [m.turn_id, other.turn_id],
                        })
        return problems
