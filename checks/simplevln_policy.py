"""Use the author's pinned navigation agent through a small standard-library HTTP shim.

Only the policy class is loaded, avoiding replacement of the validated Habitat environment.
Random invalid-action fallback is recorded and converted to an explicit failed STOP.
"""
import ast
import base64
import hashlib
import os
import random
import re
import time
import urllib.request
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tmp/habitat_setup/navigation_source_review/eval/vln/eval.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='72efa80174a832cec519eb7bb73ba2f635848b47af4dcaafaa03078e0d8d3a1f'
SYSTEM='You are a visual language navigation model.'
_agent=None
_last_call=None
_fallback=False


def invalid_random(a,b):
    global _fallback
    _fallback=True
    return 0


def openai_shim(base_url,**_):
    def create(**payload):
        global _last_call
        payload['max_tokens']=payload.pop('max_completion_tokens')
        payload['stream']=False
        payload['seed']=7
        started=time.perf_counter()
        request=urllib.request.Request(base_url+'/chat/completions',data=json.dumps(payload).encode(),
                                       headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=120) as response:result=json.load(response)
        content=payload['messages'][1]['content']
        _last_call={'raw':result['choices'][0]['message']['content'],
                    'finish_reason':result['choices'][0].get('finish_reason'),
                    'usage':result.get('usage'),'model_request_s':time.perf_counter()-started,
                    'image_count':sum(item['type']=='image_url' for item in content),
                    'request_text':[item['text'] for item in content if item['type']=='text'],
                    'request_image_sha256':[hashlib.sha256(base64.b64decode(item['image_url']['url'].split(',',1)[1])).hexdigest() for item in content if item['type']=='image_url']}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=_last_call['raw']))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


tree=ast.parse(SOURCE.read_text())
nodes=[node for node in tree.body if (isinstance(node,ast.FunctionDef) and node.name=='encode_image_base64')
       or (isinstance(node,ast.ClassDef) and node.name=='SimpleVLN_Agent')]
namespace=dict(Agent=object,OpenAI=openai_shim,os=os,re=re,BytesIO=BytesIO,base64=base64,Image=Image,
               SYSTEM_PROMPT=SYSTEM,random=SimpleNamespace(randint=invalid_random))
exec(compile(ast.Module(body=nodes),str(SOURCE),'exec'),namespace)
AuthorAgent=namespace['SimpleVLN_Agent']


def request_action(rgb,instruction,history,endpoint,model,feedback):
    global _agent,_last_call,_fallback
    started=time.perf_counter()
    if not history:
        _agent=AuthorAgent(endpoint,model,str(ROOT/'tmp/habitat_setup/simplevln_agent_runtime'),25,15,200,1.0,False)
    _last_call=None;_fallback=False
    action=_agent.act({'rgb':rgb,'instruction':{'text':instruction}}, {}, 'diagnostic')['action']
    error='invalid_action_random_fallback_rejected' if _fallback else None
    if _last_call and _last_call['finish_reason']=='length':error='truncated_model_output'
    if error:
        action=0
        _agent.pending_action_list=[]
    call=_last_call or {'raw':None,'finish_reason':None,'usage':None,'model_request_s':0.0,
                        'image_count':0,'request_text':[],'request_image_sha256':[]}
    return dict(call,action=['STOP','MOVE_FORWARD','TURN_LEFT','TURN_RIGHT'][action],error=error,
                model_called=_last_call is not None,queued_actions_remaining=list(_agent.pending_action_list),
                visual_history_window=min(len(history)+1,200),decision_s=time.perf_counter()-started)
