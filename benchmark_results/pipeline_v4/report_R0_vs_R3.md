## Paired comparison: lme_dev250_v4_R0rerun.json (A) vs lme_dev250_v4_R3.json (B)

paired on 250 shared questions (A-only 0, B-only 0)
- Accuracy: A 74.0% vs B 74.4% (delta +0.4 pp)
- Mean tokens: A 1324 vs B 1342 (delta +18)
- Mean latency: A 1050ms vs B 1082ms (delta +32ms)
- McNemar: both-correct 185, A-only 0, B-only 1, exact two-sided p = 1.0000

question_type                   n   acc_A   acc_B    d_pp   tok_A   tok_B   d_tok
------------------------------------------------------------------------------------
multi-session                  67   40.3%   41.8%   +1.5    1515    1568     +53
temporal-reasoning             66   77.3%   77.3%   +0.0    1221    1223      +1
knowledge-update               39   92.3%   92.3%   +0.0    1296    1313     +18
single-session-user            35   97.1%   97.1%   +0.0    1208    1212      +4
single-session-assistant       28   85.7%   85.7%   +0.0    1409    1409      +0
single-session-preference      15   86.7%   86.7%   +0.0    1110    1110      +0

### Flipped questions
- gpt4_f2262a51 [B-only] multi-session: A='4' B='3'
