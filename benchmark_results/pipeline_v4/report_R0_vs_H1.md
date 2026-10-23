## Paired comparison: lme_dev250_v4_R0rerun.json (A) vs lme_dev250_v4_H1.json (B)

paired on 250 shared questions (A-only 0, B-only 0)
- Accuracy: A 74.0% vs B 72.0% (delta -2.0 pp)
- Mean tokens: A 1324 vs B 1318 (delta -7)
- Mean latency: A 1050ms vs B 983ms (delta -66ms)
- McNemar: both-correct 178, A-only 7, B-only 2, exact two-sided p = 0.1797

question_type                   n   acc_A   acc_B    d_pp   tok_A   tok_B   d_tok
------------------------------------------------------------------------------------
multi-session                  67   40.3%   38.8%   -1.5    1515    1510      -5
temporal-reasoning             66   77.3%   74.2%   -3.0    1221    1219      -2
knowledge-update               39   92.3%   87.2%   -5.1    1296    1282     -14
single-session-user            35   97.1%   94.3%   -2.9    1208    1200      -8
single-session-assistant       28   85.7%   92.9%   +7.1    1409    1399     -10
single-session-preference      15   86.7%   80.0%   -6.7    1110    1104      -6

### Flipped questions
- 10e09553 [A-only] knowledge-update: A='7' B='9'
- 1c0ddc50 [A-only] single-session-preference: A='You might enjoy listening to more podcasts during your bike ' B='I don't know.'
- 3249768e [B-only] single-session-assistant: A='Triple Sec' B='Absinthe'
- 7405e8b1 [A-only] multi-session: A='Yes' B='No'
- b86304ba [A-only] single-session-user: A='The painting of a sunset is worth triple what you paid for i' B='I don't know.'
- c4ea545c [A-only] knowledge-update: A='Yes.' B='No'
- e982271f [B-only] single-session-assistant: A='The Old Church' B='Revolution Hall'
- eac54add [A-only] temporal-reasoning: A='Collaborating with an influencer who promoted my product to ' B='Launched my website and created a business plan outline.'
- gpt4_d6585ce8 [A-only] temporal-reasoning: A='Free outdoor concert series, Billie Eilish show in Philly' B='I don't know.'
