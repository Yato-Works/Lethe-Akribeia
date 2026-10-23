## Paired comparison: lme_dev250_v4_pipeline.json (A) vs lme_dev250_v4_R2.json (B)

paired on 250 shared questions (A-only 0, B-only 0)
- Accuracy: A 73.6% vs B 74.0% (delta +0.4 pp)
- Mean tokens: A 1329 vs B 1264 (delta -65)
- Mean latency: A 998ms vs B 948ms (delta -50ms)
- McNemar: both-correct 181, A-only 3, B-only 4, exact two-sided p = 1.0000

question_type                   n   acc_A   acc_B    d_pp   tok_A   tok_B   d_tok
------------------------------------------------------------------------------------
multi-session                  67   38.8%   40.3%   +1.5    1515    1393    -122
temporal-reasoning             66   77.3%   78.8%   +1.5    1240    1183     -57
knowledge-update               39   92.3%   92.3%   +0.0    1296    1264     -32
single-session-user            35   97.1%   94.3%   -2.9    1208    1160     -48
single-session-assistant       28   85.7%   85.7%   +0.0    1409    1389     -20
single-session-preference      15   86.7%   86.7%   +0.0    1110    1054     -56

### Flipped questions
- 2318644b [B-only] multi-session: A='$210' B='$270'
- 75499fd8 [A-only] single-session-user: A='Golden Retriever' B='I don't know.'
- b5ef892d [A-only] multi-session: A='7 + 5 = 12 days' B='7'
- gpt4_194be4b3 [B-only] multi-session: A='2' B='3'
- gpt4_2f8be40d [B-only] multi-session: A='4' B='3'
- gpt4_f2262a51 [A-only] multi-session: A='3' B='I don't know.'
- gpt4_fa19884d [B-only] temporal-reasoning: A='Jinsang' B='Bluegrass band featuring a banjo player'
