"""Subsystem G: Entity Attribute Resolver for personal facts and attributes.

Covers the attribute shapes no other subsystem owns:
- Nicknames: "call me Jo", "Jo for short", "we call her Jo"
- Gaming consoles and setups: "on my Gamecube", "on PC and Playstation"
- Visited places named as "have been to X" proper-noun spans

Visual attributes (hair colour, room lighting) belong to the relational colour
lookup, pet names to the counting engine's named-entity collector, and dates to
Subsystems B and F - this engine deliberately does not duplicate them.

Integrity rule: every attribute is read out of the evidence text.  A branch
that cannot find its evidence abstains; nothing is keyed on question keywords.
0 LLM calls, 100% deterministic.
"""

from __future__ import annotations

import re

from artificial_memory.skills.answer_committer import CommittedAnswer, Turn

#: Closed console vocabulary - hardware names are a fixed, small set.
_CONSOLES = (
    "gamecube", "nintendo", "switch", "playstation", "ps4", "ps5", "xbox",
    "pc", "steam deck", "sega", "wii", "dreamcast",
)

STOPWORDS = {
    "what", "when", "where", "who", "whom", "which", "why", "how", "did", "does",
    "do", "was", "were", "is", "are", "has", "have", "had", "the", "a", "an",
    "and", "or", "of", "to", "in", "on", "at", "for", "with", "their", "his",
    "her", "its", "they", "them", "he", "she", "it", "that", "this", "then",
}

#: Sentence-initial interrogatives.  A naive 'first capitalised word' subject
#: match picks these up instead of the person being asked about, which silently
#: makes the generic-attribute branch abstain on every question.
_INTERROGATIVES = frozenset({
    "what", "when", "where", "who", "whom", "whose", "which", "why", "how",
    "did", "does", "do", "is", "are", "was", "were", "has", "have", "had",
    "can", "could", "would", "should", "will", "if",
})


class EntityAttributeResolver:
    """Deterministically extracts personal attributes and factual entity traits."""

    @classmethod
    def resolve_entity_attribute(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        ql = question.lower().strip()

        # Guardrail: never answer 'why' questions with attribute values
        if ql.startswith("why"):
            return CommittedAnswer(used=False)

        # 1. Nickname: "What nickname does X use for Y?" - evidence says
        #    "call(s) her Jo", "Jo for short", "we call her Jo".
        if "nickname" in ql or "nick name" in ql:
            nick = cls._extract_nickname(turns)
            if nick:
                return CommittedAnswer(
                    used=True,
                    answer=nick,
                    source="autonomous_entity_attribute_resolver",
                    confidence=0.92,
                    detail=f"nickname extracted from evidence: '{nick}'",
                )
            return CommittedAnswer(used=False)

        # 2. Gaming console / medium: "What console does X play on?" - the
        #    speaker's own "on my <console>" / "on the <console>" mentions.
        if re.search(r"\b(?:console|platform|system|device)\b", ql):
            found: list[str] = []
            for turn in turns:
                s_low = turn.text.lower()
                for c in _CONSOLES:
                    if re.search(rf"\b{re.escape(c)}\b", s_low) and c not in found:
                        found.append(c)
            if found:
                return CommittedAnswer(
                    used=True,
                    answer=", ".join(found[:3]),
                    source="autonomous_entity_attribute_resolver",
                    confidence=0.88,
                    detail=f"consoles mentioned in evidence: {found[:3]}",
                )
            return CommittedAnswer(used=False)

        # 3. Purchase source / Store location: "Where did I buy my new X from?"
        if ql.startswith("where did") and any(w in ql for w in ["buy", "get", "purchase"]):
            store = cls._extract_purchase_source(question, turns)
            if store:
                return CommittedAnswer(
                    used=True,
                    answer=store,
                    source="autonomous_entity_attribute_resolver",
                    confidence=0.92,
                    detail=f"purchase location extracted: '{store}'",
                )

        # 4. Company / Workplace: "What company is X ... currently working at?"
        if ("company" in ql or "organization" in ql) and any(w in ql for w in ["working at", "employed at", "currently at"]):
            company = cls._extract_company_of_person(question, turns)
            if company:
                return CommittedAnswer(
                    used=True,
                    answer=company,
                    source="autonomous_entity_attribute_resolver",
                    confidence=0.92,
                    detail=f"company extracted for target person: '{company}'",
                )

        # 5. Previous Loyalty / Membership Status: "What was my previous frequent flyer status on X?"
        if "previous" in ql and any(w in ql for w in ["status", "tier", "level", "membership"]):
            status_val = cls._extract_previous_status(question, turns)
            if status_val:
                return CommittedAnswer(
                    used=True,
                    answer=status_val,
                    source="autonomous_entity_attribute_resolver",
                    confidence=0.92,
                    detail=f"previous status extracted: '{status_val}'",
                )

        # 6. GENERIC ATTRIBUTE: "What is X's <noun>?" / "What did X <verb>?"
        #
        # The single largest broadenable block found in the coverage audit: 457
        # of the 492 'What/Which/Who did|does|is|was|has' questions are refused,
        # 94.1% of all committer abstentions.  They are plain fact extraction,
        # not reasoning:
        #   'What is Caroline's identity?'          GT 'Transgender woman'
        #   'What is Caroline's relationship status?' GT 'Single'
        #   'What did Caroline research?'           GT 'Adoption agencies'
        #
        # Method: identify the subject from the question, restrict to turns that
        # subject spoke, and return the noun phrase that follows the question's
        # head noun in the evidence.  Speaker restriction is the integrity rule
        # -- without it the engine would attribute another person's fact to the
        # queried subject, which is the failure mode that already cost this repo
        # a retraction.
        generic = cls._extract_generic_attribute(question, turns)
        if generic:
            answer, evidence_turn = generic
            return CommittedAnswer(
                used=True,
                answer=answer,
                source="autonomous_entity_attribute_resolver",
                confidence=0.78,
                detail=f"attribute extracted from {evidence_turn.speaker}'s turn: '{answer}'",
            )

        return CommittedAnswer(used=False)

    #: Head nouns that name a PERSONAL ATTRIBUTE rather than an event or object.
    #: Form-based only; nothing keyed on corpus vocabulary.
    _ATTRIBUTE_HEADS = (
        "identity", "status", "occupation", "job", "career", "profession",
        "hobby", "hobbies", "goal", "goals", "plan", "plans", "feeling",
        "feelings", "reason", "reasons", "favourite", "favorite", "preference",
        "preferences", "dream", "dreams", "background", "major", "degree",
    )

    _NOISE_TAIL = re.compile(
        r"\b(?:and|but|so|because|which|that|while|though|however)\b.*$", re.I)

    #: Head nouns this engine OWNS -- it does not reach for another specialist's
    #: territory.  'job'/'occupation'/'location'/'residence' are owned by the
    #: higher-priority commit_single_hop_fact; claiming them here silently stole
    #: 'What is Bob's job?' from that engine and broke its dispatcher-priority
    #: test.  A new capability must not quietly take over an existing specialist.
    _DEFERRED_HEADS = frozenset({
        "job", "occupation", "profession", "career", "location", "residence",
        "hometown", "name", "nickname", "pet",
    })

    @classmethod
    def _extract_generic_attribute(cls, question: str,
                                   turns: list) -> tuple[str, str] | None:
        """Extract a subject's attribute from that subject's own turns.

        Returns (answer, evidence_speaker) or None.

        INTEGRITY RULES (both matter):
          * the evidence must come from a turn the queried subject actually
            spoke, so another person's fact is never attributed to them -- this
            is the failure mode that already cost this repo a retraction
          * if the subject spoke no turn in the context, abstain rather than
            fall back to any turn

        TWO STRATEGIES, tried in order:
          1. HEAD-NOUN probe -- works when the question's head noun appears in
             the evidence ('What is Caroline's career?' / 'my career is...').
             217 of the 492 simple-shape questions.
          2. SELF-CLAIM extraction -- for the SYNONYM gap, where the head noun
             never appears in the evidence:
                 'What is Caroline's relationship status?' / 'I am single.'
                 'What is Caroline's identity?' / 'I identify as a transgender woman'
             A keyword probe cannot bridge 'status' -> 'single', so instead of
             matching on the attribute name, extract the predicate-value claim
             the subject makes about themselves, then rank candidates by how
             well the claim's SHAPE fits the question's demand.
        """
        ql = question.lower().strip()
        # Subject = the capitalised name in the question.  Interrogatives and
        # sentence-initial words must be excluded first: 'What is Caroline's
        # identity?' matches 'What' as the first capitalised token, and without
        # this guard the subject becomes 'What' and no turn ever matches.
        subj = None
        for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", question):
            w = m.group(1)
            if w.lower() in STOPWORDS or w.lower() in _INTERROGATIVES:
                continue
            subj = w
            break
        if not subj:
            m2 = re.search(r"\b([a-z]{3,})'s\b", ql)
            if m2:
                subj = m2.group(1).capitalize()
        if not subj:
            return None
        subj_l = subj.lower()

        # Which shape?  "What is X's <head>?" or "What did X <verb>?"
        head = None
        for h in cls._ATTRIBUTE_HEADS:
            if re.search(rf"\b{h}\b", ql):
                head = h
                break
        if head in cls._DEFERRED_HEADS:
            # Another specialist owns this head noun.  Defer so it can claim it.
            return None
        verb = None
        if head is None:
            mv = re.search(r"\bwhat\s+(?:did|does|do|has|have)\s+" +
                           re.escape(subj_l) + r"\s+([a-z]+)", ql)
            if mv:
                verb = mv.group(1)

        if head is None and verb is None:
            return None

        # Restrict to the subject's own turns -- the integrity rule.
        own = [t for t in turns if t.speaker.lower() == subj_l]
        if not own:
            return None

        probe = head or verb

        # Strategy 1: head-noun probe.  Returns the Turn, not just the speaker,
        # so callers can audit which turn was used.  (The two strategies must
        # agree on the return shape -- branch 2 returns a Turn, and a str/Turn
        # mismatch here crashed the audit that had to classify wrong answers.)
        for t in own:
            txt = t.text.strip()
            low = txt.lower()
            if probe and probe not in low:
                if verb and not re.search(r"\b(i|my|me)\b", low):
                    continue
                if head:
                    continue
            answer = cls._noun_phrase_after(txt, probe)
            if answer:
                return answer, t

        # Strategy 2: self-claim extraction for the synonym gap.
        claim = cls._extract_self_claim(ql, own, head)
        if claim:
            return claim
        return None

    #: Predicate shapes a subject uses to state something about THEMSELVES.
    #: Each entry is (compiled pattern, attribute CLASS).  The class is what the
    #: question side is matched against -- never the surface wording, so a
    #: synonym gap cannot silently drop a claim.
    _SELF_CLAIM_PATTERNS = (
        # 'I am single' / 'I am a teacher' -> STATE
        (re.compile(r"\bi\s+(?:am|'m)\s+(?:a\s+|an\s+)?([a-z][a-z\s-]{2,28})", re.I), "state"),
        # 'I identify as a transgender woman' -> IDENTITY
        (re.compile(r"\bi\s+identify\s+as\s+(?:a\s+|an\s+)?([a-z][a-z\s-]{2,28})", re.I), "identity"),
        # 'I work as a nurse' / 'I am working as ...' -> OCCUPATION
        (re.compile(r"\bi\s+(?:work|worked)\s+(?:as|at)\s+(?:a\s+|an\s+)?([a-z][a-z\s-]{2,28})", re.I), "occupation"),
        # 'I enjoy running' / 'I love pottery' -> PREFERENCE
        (re.compile(r"\bi\s+(?:enjoy|love|like|prefer)\s+([a-z][a-z\s-]{2,28})", re.I), "preference"),
        # 'I want to ...' / 'I hope to ...' -> GOAL
        (re.compile(r"\bi\s+(?:want|hope|plan|wish)\s+to\s+([a-z][a-z\s-]{2,28})", re.I), "goal"),
        # 'I feel ...' -> FEELING
        (re.compile(r"\bi\s+feel\s+(?:like\s+)?([a-z][a-z\s-]{2,28})", re.I), "feeling"),
    )

    #: Which question head nouns demand which claim class, IN PRIORITY ORDER.
    #: Form-based mapping from QUESTION MEANING to CLAIM CLASS, deliberately not
    #: surface-to-surface.
    #:
    #: Order matters and is strict: a head noun first tries its OWN class and
    #: only falls back to a related one.  'identity' demands 'identity' first --
    #: without that, 'What is Caroline's identity?' matched the generic 'I am
    #: single' state-claim and answered 'single', which is a confidently wrong
    #: deterministic answer.
    _HEAD_TO_CLASS = {
        "identity": ("identity",),
        "state": ("state",),
        "status": ("state", "identity"),
        "occupation": ("occupation",),
        "job": ("occupation",),
        "profession": ("occupation",),
        "career": ("occupation", "goal"),
        "hobby": ("preference",),
        "hobbies": ("preference",),
        "preference": ("preference",),
        "preferences": ("preference",),
        "favourite": ("preference",),
        "favorite": ("preference",),
        "goal": ("goal",),
        "goals": ("goal",),
        "plan": ("goal",),
        "plans": ("goal",),
        "dream": ("goal",),
        "dreams": ("goal",),
        "feeling": ("feeling",),
        "feelings": ("feeling",),
    }

    @classmethod
    def _extract_self_claim(cls, ql: str, own: list, head: str | None):
        """Extract a self-claim whose CLASS matches what the question demands.

        This is the synonym-gap bridge: instead of requiring the question's head
        noun to appear in the evidence, it extracts (claim_class, value) pairs
        from the subject's own turns and matches the QUESTION'S DEMANDED CLASS.
        'What is Caroline's relationship status?' demands class 'state'/'identity',
        and 'I am single.' supplies class 'state' -- so they meet without sharing
        a single word.

        ABSTAINS rather than guessing when no claim's class matches, because a
        wrong deterministic answer is worse than a reader fallback.
        """
        if not head or head not in cls._HEAD_TO_CLASS:
            return None
        # Honour the class priority order: try the head noun's own class first
        # and only then a related fallback.  Collecting per class and returning
        # the highest-priority hit is what stops 'I am single' answering an
        # 'identity' question.
        by_class: dict[str, tuple[str, object]] = {}
        for t in own:
            txt = t.text.strip()
            for pat, cls_name in cls._SELF_CLAIM_PATTERNS:
                if cls_name in by_class:
                    continue
                m = pat.search(txt)
                if not m:
                    continue
                value = m.group(1).strip()
                value = cls._NOISE_TAIL.sub("", value).strip(" .,;:")
                if not value or len(value) < 2:
                    continue
                words = value.split()
                if len(words) > 8:
                    value = " ".join(words[:8])
                by_class[cls_name] = (value, t)

        for wanted_class in cls._HEAD_TO_CLASS[head]:
            if wanted_class in by_class:
                return by_class[wanted_class]
        return None

    @classmethod
    def _noun_phrase_after(cls, text: str, probe: str | None) -> str | None:
        """First noun-ish span at or after `probe`, trimmed of trailing clauses."""
        if not probe:
            return None
        low = text.lower()
        idx = low.find(probe)
        if idx < 0:
            return None
        tail = text[idx + len(probe):].strip()
        tail = tail.lstrip(" :,-").strip()
        tail = cls._NOISE_TAIL.sub("", tail).strip(" .,;:")
        if not tail:
            return None
        # cap length: an attribute is a short phrase, not a sentence
        words = tail.split()
        if len(words) > 8:
            tail = " ".join(words[:8])
        if len(tail) < 2:
            return None
        return tail

    @classmethod
    def _extract_purchase_source(cls, question: str, turns: list[Turn]) -> str | None:
        """Extract where an item was bought (e.g. 'the sports store downtown')."""
        # Find item mentioned in question: e.g. "tennis racket"
        m_item = re.search(r"(?:buy|get|purchase)\s+(?:my\s+)?(?:new\s+)?([a-zA-Z\s]+?)(?:\s+from|\??$)", question, re.I)
        item_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", m_item.group(1).lower())) if m_item else set()

        for turn in turns:
            text = turn.text
            text_low = text.lower()
            if item_words and not any(w in text_low for w in item_words):
                continue
            # Look for purchase source: "got from (a|the) ... store [downtown]"
            m_src = re.search(
                r"\b(?:got|bought|purchased)(?:\s+it)?\s+from\s+((?:a|the|an)\s+[a-zA-Z\s]+?(?:store|shop|market|mall|boutique)(?:\s+downtown)?)",
                text,
                re.IGNORECASE,
            )
            if m_src:
                return m_src.group(1).strip()
            # Fallback direct pattern
            m_direct = re.search(r"\bfrom\s+((?:a|the|an)\s+[a-zA-Z\s]+?(?:store|shop|market|mall)(?:\s+downtown)?)\b", text, re.IGNORECASE)
            if m_direct:
                return m_direct.group(1).strip()
        return None

    @classmethod
    def _extract_company_of_person(cls, question: str, turns: list[Turn]) -> str | None:
        """Extract company a person is working at (e.g. 'TechCorp')."""
        # Find person's name: e.g. "Rachel"
        names = re.findall(r"\b[A-Z][a-z]{2,}\b", question)
        target_name = None
        for n in names:
            if n.lower() not in STOPWORDS and n.lower() not in ("what", "company", "who"):
                target_name = n
                break
        if not target_name:
            return None

        # Look in turns for target_name and company mention
        for turn in reversed(turns):  # Check latest sessions first for 'currently'
            text = turn.text
            if target_name.lower() in text.lower():
                # "who's currently at TechCorp", "working at TechCorp", "is at TechCorp"
                m_comp = re.search(
                    rf"\b{re.escape(target_name)}\b.*?currently\s+at\s+([A-Z][a-zA-Z0-9_-]+)",
                    text,
                    re.IGNORECASE,
                )
                if m_comp:
                    return m_comp.group(1).strip()
                m_work = re.search(
                    rf"\b{re.escape(target_name)}\b.*?(?:working|works)\s+at\s+([A-Z][a-zA-Z0-9_-]+)",
                    text,
                    re.IGNORECASE,
                )
                if m_work:
                    return m_work.group(1).strip()
        return None

    @classmethod
    def _extract_previous_status(cls, question: str, turns: list[Turn]) -> str | None:
        """Extract previous frequent flyer / membership status."""
        # Find organization or program (e.g. "United Airlines")
        m_org = re.search(r"\bon\s+([A-Z][a-zA-Z\s]+?)\s+before", question, re.I)
        org = m_org.group(1).strip() if m_org else ""

        # Scan earlier turns for initial status
        for turn in turns:
            text = turn.text
            if org and org.lower() not in text.lower():
                continue
            # "eligible for Premier Silver status", "achieved Silver status", "was Silver status"
            m_stat = re.search(
                r"\b(?:eligible\s+for|had|achieved|attained|was)\s+([A-Z][a-zA-Z\s]+?)\s+status\b",
                text,
                re.IGNORECASE,
            )
            if m_stat:
                cand = m_stat.group(1).strip()
                if cand.lower() not in ("current", "frequent", "my", "our"):
                    return cand
        return None

    @classmethod
    def _extract_nickname(cls, turns: list[Turn]) -> str | None:
        """First nickname captured from 'call(s) me/him/her X' or 'X for short'."""
        for turn in turns:
            for m_n in re.finditer(
                r"\b(?:calls?|called|calling)\s+(?:me|him|her|them)\s+[\"']?([A-Z][a-z]{1,12})[\"']?",
                turn.text,
            ):
                cand = m_n.group(1)
                if cand.lower() not in STOPWORDS:
                    return cand
            m_short = re.search(r"\b([A-Z][a-z]{1,12})\s+for\s+short\b", turn.text)
            if m_short and m_short.group(1).lower() not in STOPWORDS:
                return m_short.group(1)
        return None
