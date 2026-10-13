# REJECTED — explicit-date precedence branch in `temporal_algebra_engine.py`

**Status:** rejected from the production path.  Experiment code archived, not deleted.
**Measured result:** FIXED 3 / BROKE 33 / net **−30** on 987 frozen questions, identical
contexts, identical repaired scorer, `git HEAD` engine vs the branch.
**Ruling (Eli):** adopt nothing, keep the experiment, complete the diagnosis, and give
the diagnosis a deadline.  Done — the diagnosis is below and it overturned the
hypothesis it set out to test.

## What was archived

```
design_docs/re_engineering_sprint/experiments/
  temporal_algebra_engine.EXPLICIT-DATE-REJECTED.py   the 670-line working copy
  temporal_rejected.py                                importable copy, for the diagnosis
```

The production engine was restored with `git restore` and is byte-identical to HEAD.
`tests/unit/test_explicit_date_precedence.py` is **kept** and marked `xfail` with the
reason recorded, so an accidental future fix is detected rather than silently passing.

A caveat found while archiving: the working tree already carried ~140 lines of
uncommitted changes from an earlier session before this branch was added (HEAD's
engine is 498 lines; the working copy was 638).  The A/B therefore compared "HEAD"
against "HEAD + prior changes + this branch".  That inflates the blast radius of what
is attributed to this branch, so the −30 is an upper bound on this branch's harm and
the true figure may be smaller.  It does not change the verdict — net-negative under
the most generous accounting is still net-negative.

## The diagnosis, and why the first hypothesis was wrong

The stated hypothesis was: *the branch takes a written-out date that belongs to a
different event, overriding a correct relative reading.*  That predicts BROKE cases
where the new answer is a **wrong date**.

Every one of the 33 BROKE cases has `new = ''`.  **The engine abstains.**

```
BROKE where the new branch's detail names the explicit-date rule : 0/33
BROKE where gold is contained in the new answer                 : 0/33
```

There is not one case of the hypothesised failure mode.  The branch did not produce
wrong dates — it produced *no* dates.

### The actual mechanism

The 33 BROKE are overwhelmingly **relative-anchor questions** (`'The week before 9 June
2023'`, `'The Friday before 14 August 2023'`) — the exact questions the earlier
branches exist to answer.  The new branch sits ahead of them and fires on any turn
containing a relative phrase plus a written-out date.  For those turns it returns the
in-text date, which is not what the gold asks for, so the answer scores wrong — or,
where the in-text date cannot be normalised, the branch returns nothing at all and
the whole engine abstains where it previously answered correctly.

**`lc_conv-26_58`, the branch's own poster child, is the clearest case:**

```
Q    : When did Melanie make a plate in pottery class?
gold : '24 August 2023'
OLD  : '24 August, 2023'   extracted explicit date
NEW  : ''                  abstains
```

The OLD engine already answered this correctly — via the ordinary explicit-date
branch at what is now line ~530.  The new branch did not fix a missing capability;
it **shadowed a working one and then failed to replace it**.  The 22-of-26
"parsing error" classification in `cause_of_offbyone.py` was therefore
mis-attributed: those turns were already being read correctly by the engine, and the
stored predictions that looked one day late came from the LLM reader, not from the
deterministic engine.

### The 3 FIXED are real

```
lc_conv-26_44  gold '13 August'          OLD ''  NEW '13 August, 2023'
lc_conv-30_30  gold '20 June, 2023'      OLD ''  NEW '20 June, 2023'
lc_conv-48_30  gold '21 February, 2023'  OLD ''  NEW '21 February, 2023'
```

All three are cases where the old engine **abstained** and the new branch commits
correctly.  The capability is genuinely missing somewhere — but not where the branch
put it, because the branch's position costs 33 to win 3.

### The 320 wrong→wrong

320 predictions changed with no verdict change.  Of those the branch mostly either
abstained where the old engine also failed, or produced a date for a question whose
gold is not a date at all (`gold 'pottery, camping, painting, swimming'`).  The second
group is the branch answering **non-date questions**, because it sits inside
`resolve_date` and fires on whatever turn that function happens to select.

## What a correct fix would have to satisfy

| condition | requirement |
|---|---|
| known regressions | all 33 BROKE ids reproduce as correct |
| known improvements | the 3 FIXED ids keep their behaviour |
| same conditions | same questions, frozen contexts, repaired scorer |
| effect | correct increases, wrong does not |
| generalisation | verified on ids not used to design the fix |

Plus the mechanism the diagnosis implies: an in-text date must be tied to the **asked
event**, and the branch must not fire at all on questions whose gold is not a date.
Those are two separate guards, and the rejected branch had neither.

**Deadline honoured:** the diagnosis is complete, the root cause is stated with its
evidence, and the remaining work is a *different, smaller* change that was not
attempted.  No further time is spent on this branch.  If the abstention cases behind
the 3 FIXED are revisited, they are a new hypothesis with their own pre-declared kill
criterion.

## The lesson, kept deliberately

Three times in this session a measurement overturned a confident prior:

1. the scorer guard removed a gain the previous session had reported;
2. the PPR spike lost to the retriever that already existed;
3. the explicit-date fix netted −30, and the failure mode was **not** the one written
   down before measuring.

In every case the pre-declared rule was what caught it.  The rule is worth more than
any of the three fixes.
