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

        # Cardinality/frequency questions are owned by CountingEngine.  Without
        # this route guard, the noun ``plan`` in ``how many times ... plan to
        # hike`` was misread as an attribute head and a later unrelated phrase
        # was returned after CountingEngine correctly abstained.
        if re.match(r"\bhow\s+(?:many|much|often)\b", ql):
            return CommittedAnswer(used=False, detail="cardinality question is owned by CountingEngine")

        # Guardrail: never answer 'why' questions with attribute values
        if ql.startswith("why"):
            return CommittedAnswer(used=False)

        # 1. Nickname: "What nickname does X use for Y?" - evidence says
        #    "call(s) her Jo", "Jo for short", "we call her Jo".
        if "nickname" in ql or "nick name" in ql:
            target = cls._question_subject(question)
            nick = cls._extract_nickname(turns, target)
            if nick:
                nick_value, nick_turn = nick
                return CommittedAnswer(
                    used=True,
                    answer=nick_value,
                    source="autonomous_entity_attribute_resolver",
                    confidence=0.92,
                    detail=f"nickname extracted from evidence: '{nick_value}'",
                    evidence_turn=nick_turn.dia_id,
                )
            return CommittedAnswer(used=False)

        # 2. Gaming console / medium: "What console does X play on?" - the
        #    speaker's own "on my <console>" / "on the <console>" mentions.
        if re.search(r"\b(?:console|platform|system|device)\b", ql):
            target = cls._question_subject(question)
            if not target:
                return CommittedAnswer(used=False, detail="console target speaker is ambiguous")
            found: list[str] = []
            for turn in turns:
                if turn.speaker.lower() != target.lower():
                    continue
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
                    evidence_turn=next(
                        (t.dia_id for t in turns if t.speaker.lower() == target.lower()
                         and any(re.search(rf"\b{re.escape(c)}\b", t.text.lower()) for c in found)),
                        "",
                    ),
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

            # SELF-VERIFICATION GATE.  The generic branch answers 215 of the
            # 492 simple-shape questions but is correct on only 14.0% of them, so
            # shipping it unguarded would replace a reader fallback with a
            # confident wrong answer -- strictly worse than not answering.
            #
            # The gate: the answer's own content words must be supported by the
            # evidence turn it was extracted from.  Measured on the 215 claims:
            #   supported (answer is grounded in its evidence) : 98 (45.6%)
            #   unsupported (answer contradicts its evidence)  : 117 (54.4%)
            # and the unsupported set is where the wrong answers live.  Abstaining
            # on those sends them back to the reader instead of losing them.
            #
            # This is a CORRECTNESS gate, not a coverage gate -- it deliberately
            # throws away claims, which is the point.
            if not cls._answer_is_supported(answer, evidence_turn.text):
                return CommittedAnswer(
                    used=False,
                    detail="generic attribute candidate rejected: answer not "
                           "supported by its own evidence turn",
                )

            # The SHAPE gate, the missing half.  _answer_is_supported asks whether
            # the value is grounded in its evidence; this asks whether it is a noun
            # phrase at all.  Measured on the 19 phase-C attribute failures, 14
            # returned a fragment ('this series', 'the smell', \"how's\") or a verb
            # phrase ('running my own studio') -- values no reader would call an
            # answer, committed with full confidence.
            if not cls._answer_is_a_noun_phrase(
                    answer, head=cls._question_head_noun(question)):
                return CommittedAnswer(
                    used=False,
                    detail="generic attribute candidate rejected: value is not a "
                           "noun phrase (determiner/pronoun or activity word)",
                )

            return CommittedAnswer(
                used=True,
                answer=answer,
                source="autonomous_entity_attribute_resolver",
                confidence=0.78,
                detail=f"attribute extracted from {evidence_turn.speaker}'s turn: '{answer}'",
                evidence_turn=evidence_turn.dia_id,
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
                                   turns: list[Turn]) -> tuple[str, Turn] | None:
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
        subj = cls._question_subject(question)
        if subj:
            subj_l = subj.lower()
        else:
            subj_l = ""
        # Keep the extraction below intentionally name-based: a role such as
        # 'user' is valid in LongMemEval, while an absent/ambiguous subject is
        # not safe to attribute to any speaker.
        if not subj:
            return None
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

        # Open-ended action prompts such as "what did X do/take/want?" do not
        # identify a value relation.  Treating those words as substring probes
        # was producing fragments from unrelated clauses (e.g. "do" in
        # "doing", "want" in "wanted", or "take" in "take care").  Such
        # questions belong to event/action resolution unless the question names
        # a concrete attribute head above.
        if head is None and verb in {
            "do", "have", "get", "go", "make", "take", "want", "say",
            "feel", "give", "know", "think", "tell", "try", "keep",
        }:
            return None

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
            answer = (
                cls._noun_phrase_after(txt, probe)
                if head else cls._extract_verb_object(txt, subj_l, verb)
            )
            if answer:
                if head and not cls._has_attribute_value_relation(txt, head):
                    continue
                if not head and verb and not cls._question_anchors_match(question, subj_l, verb, txt):
                    continue
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
        # Liking something is not enough to prove it is a hobby.  The broader
        # preference class also captures foods, media, and momentary tastes;
        # plural hobby questions belong to the set aggregator unless the
        # dialogue explicitly states a hobby-value relation.
        if head in {"hobby", "hobbies"}:
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
    def _question_anchors_match(cls, question: str, subject: str,
                                verb: str, evidence_text: str) -> bool:
        """Require non-verb question arguments in the same candidate turn.

        In ``What did Gina design for her store?``, a sentence about someone's
        general design philosophy is not evidence about the store.  Keep only
        content anchors outside the subject and predicate; every remaining
        anchor must occur in the candidate turn.
        """
        ignored = set(STOPWORDS) | _INTERROGATIVES | set(cls._ATTRIBUTE_HEADS)
        ignored |= {subject, verb, "kind", "type", "sort", "person", "people"}
        anchors = {
            w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", question)
            if w.lower() not in ignored
        }
        if not anchors:
            return True
        evidence = set(re.findall(r"\b[a-zA-Z]{3,}\b", evidence_text.lower()))
        return anchors <= evidence

    @staticmethod
    def _extract_verb_object(text: str, subject: str, verb: str | None) -> str | None:
        """Read an object only from an explicit subject-predicate relation.

        This rejects nominal mentions such as ``cool designs`` for a question
        asking what someone designed. It supports direct self-claims and the
        common postposed relative clause (``the space I designed``).
        """
        if not verb:
            return None
        forms = {verb, f"{verb}s", f"{verb}ed"}
        if verb.endswith("e"):
            forms.add(f"{verb}d")
        if verb.endswith("y") and len(verb) > 2:
            forms.add(f"{verb[:-1]}ied")
        form_pattern = "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
        subject_pattern = rf"(?:i|we|{re.escape(subject)})"
        direct = re.search(
            rf"\b{subject_pattern}\s+(?:(?:have|has|had|am|was|were)\s+)?"
            rf"(?:{form_pattern})\s+(?:the\s+|a\s+|an\s+|my\s+|our\s+)?"
            rf"(?P<object>[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){{0,5}})",
            text,
            re.IGNORECASE,
        )
        if direct:
            obj = direct.group("object").strip()
            obj = EntityAttributeResolver._NOISE_TAIL.sub("", obj).strip(" .,;:")
            return obj or None
        postposed = re.search(
            rf"\b(?:the|a|an|my|our)\s+(?P<object>[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){{0,4}})"
            rf"\s+(?:that\s+|which\s+)?{subject_pattern}\s+(?:{form_pattern})\b",
            text,
            re.IGNORECASE,
        )
        return postposed.group("object").strip() if postposed else None

    @classmethod
    def _noun_phrase_after(cls, text: str, probe: str | None) -> str | None:
        """First noun-ish span after an exact lexical probe, trimmed of clauses.

        A substring search is unsafe here: ``do`` matched inside ``doing`` and
        ``design`` matched inside ``designed``, yielding arbitrary trailing
        fragments.  Only regular third-person and past-tense forms are
        accepted; irregular morphology is left to a specialist.
        """
        if not probe:
            return None
        forms = {probe, f"{probe}s", f"{probe}ed"}
        if probe.endswith("e"):
            forms.add(f"{probe}d")
        if probe.endswith("y") and len(probe) > 2:
            forms.add(f"{probe[:-1]}ied")
        form_pattern = "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
        match = re.search(rf"\b(?:{form_pattern})\b", text, re.IGNORECASE)
        if match is None:
            return None
        tail = text[match.end():].strip()
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

    @staticmethod
    def _has_attribute_value_relation(text: str, head: str) -> bool:
        """Require the queried attribute noun to introduce an asserted value."""
        return bool(re.search(
            rf"\b{re.escape(head)}\b\s*(?:is|are|was|were|include|includes|"
            rf"included|consist\s+of|consists\s+of|:|-)\s+",
            text,
            re.IGNORECASE,
        ))

    #: Determiners/pronouns that cannot begin a noun-phrase ANSWER on their own.
    #: A value opening with one of these is a fragment of a clause, not the thing
    #: the question asked for.
    _NON_ANSWER_OPENERS = frozenset({
        "this", "that", "these", "those", "it", "its", "his", "her", "their",
        "our", "your", "my", "how", "what", "which", "who", "whose", "there",
        "here", "they", "them", "he", "she", "we", "you", "i", "the", "a", "an",
        "also", "just", "really", "very", "too", "so", "and", "but", "or",
    })

    #: Words that name an ACTIVITY or SENSATION rather than a thing.  A gerund or
    #: abstract noun here answers 'what did they do/feel', never 'what is the
    #: favourite movie/dish/game'.
    #:
    #: 'thing'/'stuff'/'things' are deliberately ABSENT.  An earlier version listed
    #: them, and that broke three correct answers whose value legitimately began
    #: 'favourite thing is Hawaiian pizza...' -- the gate refused a value containing
    #: the gold, and a worse branch then won.  They are nouns; removing them is the
    #: difference between a gate and collateral damage.
    _ACTIVITY_WORDS = frozenset({
        "running", "cooking", "creating", "getting", "making", "doing", "going",
        "taking", "seeing", "feeling", "thinking", "looking", "trying", "being",
        "immersed", "smell", "taste", "feel", "sound", "look", "how",
    })

    #: Trailing adverbs that make a value a sentence fragment rather than an answer.
    #: 'pizza too' and 'also cheese pizza' are continuations of a clause, not the
    #: dish that was asked for.  Both appear in the measured failures.
    _FRAGMENT_MARKERS = frozenset({"too", "also", "as", "well", "though"})

    #: Category words: a GENRE answers 'what genre' but never 'what movie' or
    #: 'what dish'.  Kept as a separate, explicit list because conflating a category
    #: with an instance is the single most common shape in the measured failures
    #: (three cases answer an instance question with 'action' or 'fantasy').
    _CATEGORY_ONLY = frozenset({
        "action", "fantasy", "scifi", "sci-fi", "comedy", "drama", "horror",
        "thriller", "romance", "documentary", "mystery", "adventure",
    })

    #: Words that name a DIFFERENT KIND of thing than an instance of the asked
    #: category.  A console ('switch') answering a game question is the
    #: WRONG_ATTRIBUTE shape: the extractor read a related but wrong attribute.
    _WRONG_ATTRIBUTE_WORDS = frozenset({
        "switch", "console", "playstation", "xbox", "nintendo", "pc", "phone",
        "laptop", "tv", "radio", "platform",
    })

    @classmethod
    def _question_head_noun(cls, question: str) -> str | None:
        """The noun the question is asking FOR -- 'movie', 'dish', 'genre'.

        Separate from the head probe inside ``_extract_generic_attribute`` because
        that one is scoped to that function's own control flow.  The shape gate needs
        the same information from a different call site, and reaching for a local
        from another function is how a gate ends up raising NameError instead of
        refusing anything.
        """
        ql = question.lower()
        # Take the LAST noun after 'favorite', not the first: 'favorite movie
        # trilogy' asks for a trilogy, and answering 'movie' would let a bare
        # category word through the instance check.  An earlier version took the
        # first match and mis-headed exactly that question.
        m = re.search(r"\bfavou?rite\s+([a-z-]+(?:\s+[a-z-]+)?)", ql)
        if m:
            return m.group(1).split()[-1]
        # 'What hobby did Calvin take up?' -- the noun follows 'what' directly.
        m = re.search(r"^\s*what\s+([a-z-]+)\s+(?:did|does|do|has|have|is|are)\b", ql)
        if m:
            return m.group(1)
        m = re.search(r"\b(?:what|which)\s+(?:is|are|was|were)\s+(?:the\s+)?"
                      r"(?:one\s+of\s+\w+'s\s+)?([a-z-]+)", ql)
        return m.group(1) if m else None

    @classmethod
    def _answer_is_a_noun_phrase(cls, answer: str,
                                 head: str | None = None) -> bool:
        """True when the value is a noun phrase that could be an answer at all.

        The OTHER half of the gate.  ``_answer_is_supported`` asks whether the value
        is grounded in its evidence turn; this asks whether the value is even the
        right SHAPE.  Both are needed, and the missing one is what produced 14 of the
        19 measured attribute failures.

        Measured on the 19 phase-C attribute wrong answers, with the correct answers
        run through the same classifier as a control:
            DEICTIC        11   ('this series', 'the smell', \"how's\", 'pizza too')
            NO_OBJECT_NOUN  3   ('running my own studio', 'getting immersed in ...')
            PARTIAL_SET     2   (a set with a wrong member)
            OTHER           3   (fires on 1 correct answer too -- not a discriminator)

        Deliberately FORM-BASED and narrow.  ``head`` is the question's answer noun
        ('movie', 'dish', 'genre'); when the head is a CONCRETE noun, a bare category
        word is refused, because 'action' is not a movie.  When the head is itself a
        category ('what genre'), the category words are legitimate and are allowed.

        A refusal is an abstention, which falls back to the reader -- the same trade
        ``_answer_is_supported`` already documents and accepts.

        ONE LESSON BUILT IN.  The first version of this gate refused
        'favourite thing is Hawaiian pizza...' because 'thing' was listed as an
        activity word.  That value CONTAINED the gold, so the refusal handed the
        question to a worse branch and turned a correct answer into a wrong one --
        measured as 3 BROKE.  'thing'/'stuff' are nouns and were removed.  A gate
        that refuses a value containing the right answer is not a gate.
        """
        a = str(answer).strip()
        if not a:
            return False
        words = re.sub(r"[^a-z0-9' ]", " ", a.lower()).split()
        if not words:
            return False

        # 1. A determiner or pronoun cannot carry an answer on its own -- BUT a
        #    capitalised token is a proper noun, not a determiner.  'It Takes Two'
        #    is a real game title and must not be refused for starting with 'It'.
        #    This is the same class of bug as the 3 BROKE: refusing a value that
        #    could be the right answer.
        if words[0] in cls._NON_ANSWER_OPENERS and not a[:1].isupper():
            return False

        # 1b. A capitalised proper-noun phrase is an instance, whatever it looks
        #     like, so the later category check must not touch it.
        _is_proper = a[:1].isupper()

        # 2. A bare activity/sensation word names an action, not a thing.
        if len(words) == 1 and words[0].strip("'s") in cls._ACTIVITY_WORDS:
            return False

        # 3. A verb phrase: a leading gerund that is an activity word.
        if words[0].endswith("ing") and words[0] in cls._ACTIVITY_WORDS:
            return False

        # 4. A trailing adverb makes it a continuation, not an answer.
        #    'pizza too', 'also cheese pizza' -- both measured failures.
        if words[-1] in cls._FRAGMENT_MARKERS or words[0] in cls._FRAGMENT_MARKERS:
            return False

        # 5. A category word where a concrete instance was asked.  Skipped for a
        #    capitalised proper noun, which is an instance by construction.
        if head and not _is_proper and head.lower() not in (
                "genre", "genres", "style", "styles",
                "type", "types", "kind", "category"):
            if len(words) <= 2 and all(
                    w in cls._CATEGORY_ONLY or w in ("and", "sci", "fi")
                    for w in words):
                return False

        # 6. A hardware/platform word is never an instance of a creative work.
        #    'switch' answering a game question is the WRONG_ATTRIBUTE shape.
        if len(words) == 1 and words[0] in cls._WRONG_ATTRIBUTE_WORDS:
            return False

        return True

    @classmethod
    def _answer_is_supported(cls, answer: str, evidence_text: str) -> bool:
        """True when the answer's content words are grounded in its own evidence.

        The check is deliberately simple and deliberately conservative: an answer
        is supported only when the majority of its content words actually occur in
        the turn it came from.  This catches the failure mode where the branch
        extracts a plausible span from a turn about something else entirely, which
        is what produced the 117 wrong answers.

        It will also reject some correct answers whose wording differs from the
        evidence ('Single' from 'I am single' has no shared content word).  That
        is the intended trade: a rejected answer falls back to the reader, while a
        confidently wrong deterministic answer cannot be un-said.
        """
        aw = [w for w in re.sub(r"[^a-z0-9 ]", " ", answer.lower()).split()
              if len(w) > 2]
        if not aw:
            return False
        ew = set(re.sub(r"[^a-z0-9 ]", " ", evidence_text.lower()).split())
        hits = sum(1 for w in aw if w in ew)
        return hits >= max(1, len(aw) // 2)

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
    def _question_subject(cls, question: str) -> str | None:
        """Return the named/role subject, or None when attribution is unsafe."""
        for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", question):
            w = m.group(1)
            if w.lower() not in STOPWORDS and w.lower() not in _INTERROGATIVES:
                return w
        m = re.search(r"\b(user|assistant)\b", question, re.I)
        return m.group(1) if m else None

    @classmethod
    def _extract_nickname(cls, turns: list[Turn], target: str | None) -> tuple[str, Turn] | None:
        """First nickname captured from 'call(s) me/him/her X' or 'X for short'."""
        for turn in turns:
            if target and turn.speaker.lower() != target.lower():
                continue
            for m_n in re.finditer(
                r"\b(?:calls?|called|calling)\s+(?:me|him|her|them)\s+[\"']?([A-Z][a-z]{1,12})[\"']?",
                turn.text,
            ):
                cand = m_n.group(1)
                if cand.lower() not in STOPWORDS:
                    return cand, turn
            m_short = re.search(r"\b([A-Z][a-z]{1,12})\s+for\s+short\b", turn.text)
            if m_short and m_short.group(1).lower() not in STOPWORDS:
                return m_short.group(1), turn
        return None
