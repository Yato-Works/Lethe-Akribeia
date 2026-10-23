## Paired comparison: lme_dev250_v4_R0rerun.json (A) vs lme_dev250_v4_R2.json (B)

paired on 250 shared questions (A-only 0, B-only 0)
- Accuracy: A 74.0% vs B 74.8% (delta +0.8 pp)
- Mean tokens: A 1324 vs B 1259 (delta -65)
- Mean latency: A 1050ms vs B 962ms (delta -88ms)
- McNemar: both-correct 183, A-only 2, B-only 4, exact two-sided p = 0.6875

question_type                   n   acc_A   acc_B    d_pp   tok_A   tok_B   d_tok
------------------------------------------------------------------------------------
multi-session                  67   40.3%   41.8%   +1.5    1515    1393    -122
temporal-reasoning             66   77.3%   80.3%   +3.0    1221    1164     -57
knowledge-update               39   92.3%   92.3%   +0.0    1296    1264     -32
single-session-user            35   97.1%   94.3%   -2.9    1208    1160     -48
single-session-assistant       28   85.7%   85.7%   +0.0    1409    1389     -20
single-session-preference      15   86.7%   86.7%   +0.0    1110    1054     -56

### Flipped questions
- 4dfccbf7 [B-only] temporal-reasoning: A='70' B='70 days'
- 7405e8b1 [A-only] multi-session: A='Yes' B='No'
- 75499fd8 [A-only] single-session-user: A='Golden Retriever' B='I don't know.'
- gpt4_194be4b3 [B-only] multi-session: A='2' B='3'
- gpt4_2f8be40d [B-only] multi-session: A='4' B='3'
- gpt4_fa19884d [B-only] temporal-reasoning: A='Jinsang' B='Bluegrass band featuring a banjo player'
