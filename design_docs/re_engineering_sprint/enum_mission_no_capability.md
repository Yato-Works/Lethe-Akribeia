# Mission result — enumeration/counting expansion: NO GENERALISABLE CAPABILITY FOUND

Mission: expand a deterministic capability from 7 enumeration/counting failures.
Reported as instructed: the capability did not generalise, so nothing was built.

## What the 7 targets actually are

The mission warned against trusting an aggregate label, and the label does not
survive.  Verified per question against the frozen contexts:

```
gold really demands MULTIPLE items : 2/7
gold is a SINGLE item              : 5/7
```

Actual defects, one per question:

| cause | n | what is wrong |
|---|---:|---|
| `WRONG_EVENT_SCOPE` | 3 | a union IS produced, but from activities of a **different event** |
| `SINGLE_VALUE` | 3 | gold is one item; the enumeration label is wrong for that question |
| `CARDINALITY` | 1 | a "how many **people**" answered by counting sessions |

`lc_conv-41_124` and `lc_conv-50_72` are labelled enumeration but their gold is one
item.  `lc_conv-42_133` asks how many people attended a party and the engine answered
"ten times, 10" — it counted occurrences, not attendees.

## Why no union engine was built

The decisive line: **of the 3 questions where a union is even the right shape, all 3
already have a union — with the wrong scope.**

```
questions where a union is the right shape : 3
of those, union came from the WRONG EVENT   : 3
```

So a better union would fix SCOPE, not existence.  Scoping an item to the asked event
is event binding, which the previous mission measured as **83% semantic** — deciding
whether two mentions co-refer is judgement, not rules.  Building a third attempt at
that barrier would repeat an investment already shown not to close.

This is the third time this session has stopped at that same wall from a different
direction, which is itself the useful signal: it is the load-bearing constraint of a
deterministic engine on this corpus, not a local bug.

## Why no counting engine was built

Measured the generalisation surface rather than assuming it.  Across all 1,986
questions, `how many <noun>` appears 38 times, and the distribution is flat:

```
time 8 | month 3 | pet 3 | dog 2 | week 2 | video 2 | people 2 | year 2
then 1-2 each for: children, letter, hike, turtle, screenplay, tournament,
                  game, charity, day, ...
```

No noun exceeds 8, and the target noun — `people` — appears **twice**, one of which
is the failure being fixed.  A counting capability for one noun covers 2 questions
and would be the per-question special case the mission prohibits.

## What was NOT done, deliberately

- No engine change.  Every candidate either routes to the 83%-semantic barrier or
  covers 2 questions.
- No benchmark run, per the mission.
- The 7 remain open and are recorded with their true causes, so a future mission can
  pick the right one: the 3 `SINGLE_VALUE` cases are attribute/single-value
  extraction problems (adjacent to the shape gate just landed), and the `CARDINALITY`
  case is an occurrence-vs-cardinality confusion inside `CountingEngine`.

## Honest note on the mission's framing

The mission proposed "multi-turn set union, filtering by person/attribute,
deduplication, per-type counting" as candidates.  Measurement says the first three
already exist — `CountingEngine` unions and `entity_identity` dedupes — and their
failure is scope, not mechanism.  The fourth has no population.  So the candidates
were not wrong; they were **already built**, which is why building them again would
have produced motion rather than progress.
