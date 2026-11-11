"""Offline checks that execution feedback cannot leak task goals or metrics."""
import ast
import copy
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
trees = [ast.parse((root/'checks'/name).read_text()) for name in
         ['run_closed_loop_baseline.py','run_closed_loop_feedback.py']]
prompts = []
for tree in trees:
    assignment = next(n for n in tree.body if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='SYSTEM' for t in n.targets))
    prompts.append(ast.literal_eval(assignment.value))
assert prompts[0] == prompts[1], 'System prompt changed'
function = next(n for n in trees[1].body if isinstance(n,ast.FunctionDef) and n.name=='execution_feedback')
namespace = {'np':np}
exec(compile(ast.Module(body=[function]),'<feedback function>','exec'),namespace)
feedback = namespace['execution_feedback']
row = {'action':'MOVE_FORWARD','before':{'position':[10.0,2.0,30.0]},
       'after':{'position':[10.0,2.0,30.0]},
       'metrics':{'collisions':{'is_collision':True},'distance_to_goal':0.1,'success':1,'spl':1}}
expected = feedback(row)
assert expected['forward_execution']=='blocked'
changed = copy.deepcopy(row)
changed['metrics'].update(distance_to_goal=999,success=0,spl=0)
changed.update(goal_position=[999,1,4],reference_path=[[10,2,30],[999,1,4]],optimal_action='STOP')
assert feedback(changed)==expected, 'Ground-truth task leakage'
for key in ['before','after']:
    changed[key]['position']=[x+1234 for x in changed[key]['position']]
assert feedback(changed)==expected, 'Absolute position leaked'
assert set(expected)=={'previous_action','observed_displacement_m','collision_detected','forward_execution'}
base=json.loads((root/'data/baseline/closed_loop_v1/manifest.json').read_text())
new=json.loads((root/'data/baseline/closed_loop_feedback_v2/manifest.json').read_text())
assert {k for k in base if base[k]!=new[k]}=={'version','model_inputs'}
print('FEEDBACK_CONTRACT_READY: identical system/task settings; no goals/distances/reference leakage')
