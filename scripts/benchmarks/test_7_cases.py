import sys
sys.path.insert(0, 'scripts')
import benchmarks.simulate_rules as sim

cases = [
    (1, 'conv-30-qa-023', 'How did Gina promote her clothes store?',
     'worked with an artist to make unique fashion pieces, made limited-edition sweatshirts, got some new offers and promotions for online store, developed a video pr',
     'worked with an artist to make unique fashion pieces, made limited-edition sweatshirts, got some new offers and promotions for online store, developed a video pr'),
    
    (3, 'conv-42-qa-066', 'What alternative career might Nate consider after gaming?',
     'an animalkeeper at a localzoo and workingwith turtles; as heknows a great dealabout turtles andhow to care for them,and he enjoys it.',
     'an animalkeeper at a localzoo and workingwith turtles'),
    
    (1, 'conv-26-qa-076', 'When did Melanie go on a hike after the roadtrip?',
     '19 October 2023',
     '19 October 2023'),
    
    (3, 'conv-41-qa-017', "What might John's degree be in?",
     'Political science, Public administration, Public affairs',
     'Political science, Public administration, Public affairs'),
    
    (3, 'conv-50-qa-013', 'Would Calvin enjoy performing at the Hollywood Bowl?',
     'Yes; because he enjoys the rush of performing onstage to large crowds',
     'Yes'),
    
    (4, 'conv-26-qa-124', 'What pets does Melanie have?',
     'Two cats and a dog',
     'Two cats and a dog'),
    
    (4, 'conv-26-qa-088', 'What is Caroline excited about in the adoption process?',
     'creating a family for kids who need one',
     'creating a family for kids who need one'),
]

for cat, qid, q, gt, pred in cases:
    item = {'category': cat, 'answer': gt, 'am_prediction': pred}
    ems, _, _ = sim.official.eval_question_answering([item], 'am_prediction', metric='f1')
    print(f"[{qid}] Cat {cat}: F1={ems[0]} | Pred='{pred[:40]}...'")
