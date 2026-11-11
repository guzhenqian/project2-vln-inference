"""Bounded visual history and model-authored route progress; no task truth inputs."""
import base64
import io
import json
import re
import time
import urllib.request

import imageio

SYSTEM = '''You navigate an indoor environment following a human route instruction. Each call chooses ONE action. Images are labeled in chronological order; the last image is your current forward-facing camera. Earlier images let you recognize places you passed and changes in the scene. Your previous memory is a fallible note, not ground truth: update or correct it from visual evidence.
MOVE_FORWARD moves 0.25m; TURN_LEFT and TURN_RIGHT rotate 15 degrees in place. Turning once is a small change, not a full right-angle turn. STOP ends the task. Identify the ordered landmarks and the final destination in the instruction. Track which landmarks you have actually passed. Choose STOP when you have reached the described destination; do not keep walking merely because more floor is visible. Do not STOP simply because the current view is blocked: first turn to find a walkable route. Do not mistake a commanded movement for successful movement. Use the execution feedback. Avoid repeating a blocked forward command in the same direction. Use visible doorways and floor, not walls, furniture or closed doors. You have no goal coordinates, map or success-distance signal.
Return one JSON object with exactly these keys: "action" (STOP, MOVE_FORWARD, TURN_LEFT or TURN_RIGHT), "memory" (at most 35 words summarizing observed route progress and the next landmark), "reason" (at most 25 words of visual evidence for this action). No markdown. Do not invent observed landmarks.'''

_frames = {}
_memory = ''


def request_action(rgb, instruction, history, endpoint, model, feedback):
    global _frames, _memory
    started = time.perf_counter()
    step = len(history)
    if not history:
        _frames, _memory = {}, ''
    _frames[step] = rgb.copy()
    indices = sorted(set([0, max(0, step-12), max(0, step-4), step]))
    previous_memory = _memory
    text = ('Route instruction: {}\nExecuted actions (oldest to newest): {}\n'
            'Previous action execution feedback: {}\nYour previous route-progress note: {}\n'
            'Use the ordered camera observations below; only the last is current. '
            'Choose one action and update the progress note.').format(
                instruction, ', '.join(history[-20:]) or '(start)',
                json.dumps(feedback, sort_keys=True), previous_memory or '(none)')
    content = [{'type':'text','text':text}]
    for index in indices:
        content.append({'type':'text','text':'Camera BEFORE action {}{}:'.format(index+1,' (CURRENT)' if index==step else ' (PAST)')})
        stream = io.BytesIO()
        imageio.imwrite(stream, _frames[index], format='png')
        url = 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')
        content.append({'type':'image_url','image_url':{'url':url}})
    # Keep the start and the last twelve observations only.
    _frames = {index:frame for index,frame in _frames.items() if index==0 or index>=step-12}
    payload = {'model':model,'messages':[{'role':'system','content':SYSTEM},
               {'role':'user','content':content}], 'temperature':0,'max_tokens':160,'seed':7,'stream':False}
    request = urllib.request.Request(endpoint+'/chat/completions',data=json.dumps(payload).encode(),
                                     headers={'Content-Type':'application/json'})
    call_started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    call_s = time.perf_counter()-call_started
    raw = result['choices'][0]['message']['content']
    normalized = raw.strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',normalized,flags=re.DOTALL)
    if fence:
        normalized = fence.group(1).strip()
    error = None
    try:
        parsed = json.loads(normalized)
        assert isinstance(parsed,dict) and set(parsed)=={'action','memory','reason'}
        assert parsed['action'] in ['STOP','MOVE_FORWARD','TURN_LEFT','TURN_RIGHT']
        assert isinstance(parsed['memory'],str) and isinstance(parsed['reason'],str)
        assert len(parsed['memory'])<=800 and len(parsed['reason'])<=600
        action = parsed['action']
        _memory = parsed['memory']
        reason = parsed['reason']
    except (ValueError,TypeError,AssertionError,KeyError):
        action, reason, error = 'STOP',None,'invalid_model_action'
    return {'action':action,'raw':raw,'error':error,'usage':result.get('usage'),
            'markdown_fence_removed':bool(fence),'execution_feedback_input':feedback,
            'visual_history_steps':indices,'memory_input':previous_memory,'memory_output':_memory,
            'reason':reason,'user_text':text,'finish_reason':result['choices'][0].get('finish_reason'),
            'model_request_s':call_s,'decision_s':time.perf_counter()-started}
