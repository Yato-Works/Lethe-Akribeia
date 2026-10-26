## Paired comparison: lme_dev250_v4_R0rerun.json (A) vs lme_dev250_v4_R1.json (B)

paired on 250 shared questions (A-only 0, B-only 0)
- Accuracy: A 74.0% vs B 71.2% (delta -2.8 pp)
- Mean tokens: A 1324 vs B 1324 (delta +0)
- Mean latency: A 1050ms vs B 1032ms (delta -17ms)
- McNemar: both-correct 174, A-only 11, B-only 4, exact two-sided p = 0.1185

question_type                   n   acc_A   acc_B    d_pp   tok_A   tok_B   d_tok
------------------------------------------------------------------------------------
multi-session                  67   40.3%   35.8%   -4.5    1515    1515      +0
temporal-reasoning             66   77.3%   77.3%   +0.0    1221    1221      +0
knowledge-update               39   92.3%   82.1%  -10.3    1296    1296      +0
single-session-user            35   97.1%   97.1%   +0.0    1208    1208      +0
single-session-assistant       28   85.7%   85.7%   +0.0    1409    1409      +0
single-session-preference      15   86.7%   86.7%   +0.0    1110    1110      +0

### Flipped questions
- 06db6396 [A-only] knowledge-update: A='5' B='9'
- 0bc8ad93 [B-only] temporal-reasoning: A='No.' B='No'
- 10e09553 [A-only] knowledge-update: A='7' B='9'
- 1f2b8d4f [A-only] multi-session: A='$750' B='$320'
- 2318644b [A-only] multi-session: A='$270' B='$250'
- 2e6d26dc [B-only] multi-session: A='4' B='5'
- 5a7937c8 [A-only] multi-session: A='3' B='1'
- 6aeb4375 [A-only] knowledge-update: A='4' B='7'
- a08a253f [B-only] multi-session: A='3' B='4'
- b6019101 [A-only] knowledge-update: A='5' B='4'
- dd2973ad [A-only] multi-session: A='2 AM' B='10 AM on Wednesday.'
- f35224e0 [B-only] multi-session: A='I don't know.' B='15 + 12 = 27'
- gpt4_372c3eed [A-only] multi-session: A='4 years' B='8'
- gpt4_59149c78 [A-only] temporal-reasoning: A='City Art Museum' B='I don't know.'
- gpt4_7fce9456 [A-only] multi-session: A='2' B='I don't know.'
