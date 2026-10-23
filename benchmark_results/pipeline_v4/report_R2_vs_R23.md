## Paired comparison: lme_dev250_v4_R2.json (A) vs lme_dev250_v4_R23.json (B)

paired on 250 shared questions (A-only 0, B-only 0)
- Accuracy: A 74.4% vs B 74.4% (delta +0.0 pp)
- Mean tokens: A 1252 vs B 1267 (delta +15)
- Mean latency: A 927ms vs B 962ms (delta +35ms)
- McNemar: both-correct 184, A-only 2, B-only 2, exact two-sided p = 1.0000

question_type                   n   acc_A   acc_B    d_pp   tok_A   tok_B   d_tok
------------------------------------------------------------------------------------
multi-session                  67   41.8%   41.8%   +0.0    1386    1429     +43
temporal-reasoning             66   77.3%   77.3%   +0.0    1155    1156      +1
knowledge-update               39   92.3%   92.3%   +0.0    1264    1280     +17
single-session-user            35   94.3%   94.3%   +0.0    1153    1158      +4
single-session-assistant       28   85.7%   85.7%   +0.0    1389    1389      +0
single-session-preference      15   93.3%   93.3%   +0.0    1029    1029      +0

### Flipped questions
- 4dfccbf7 [B-only] temporal-reasoning: A='70' B='60 days'
- aae3761f [B-only] multi-session: A='14' B='15'
- gpt4_2f8be40d [A-only] multi-session: A='3' B='4'
- gpt4_fa19884d [A-only] temporal-reasoning: A='Bluegrass band featuring a banjo player' B='Jinsang'
