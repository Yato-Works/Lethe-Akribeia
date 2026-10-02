import sys
sys.path.insert(0, 'scripts')
import benchmarks.simulate_rules as sim

# test conv-50-qa-013 with 'Yes'
item50 = {'category': 3, 'answer': 'Yes; because he enjoys the rush of performing onstage to large crowds', 'am_prediction': 'Yes'}
ems50, _, _ = sim.official.eval_question_answering([item50], 'am_prediction', metric='f1')
print(f"conv-50-qa-013 with 'Yes': {ems50[0]}")

# test conv-42-qa-066 with conclusion only
item42 = {'category': 3, 'answer': 'an animalkeeper at a localzoo and workingwith turtles; as heknows a great dealabout turtles andhow to care for them,and he enjoys it.', 'am_prediction': 'an animalkeeper at a localzoo and workingwith turtles'}
ems42, _, _ = sim.official.eval_question_answering([item42], 'am_prediction', metric='f1')
print(f"conv-42-qa-066 conclusion only: {ems42[0]}")
