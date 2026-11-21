"""Opt-in host timestamps for FINISH delivery and CounterEvent blocking.

Independent diagnostic only: no active service imports this file. Set
PROJECT2_LATENCY_SCHEDULE_TRACE=1 and call install() in a fresh service module.
No CUDA synchronization, GPU reads, or tensor copies are added. Events append
to bounded in-memory buffers; FINISH responses flush after response dispatch.
The diagnostic HTTP times include flush overhead and must not enter primary
speed tables. Intended first for UniExecutor, where model and scheduler share
a process. MP worker-only traces need an explicit flush() or graceful exit.
"""
import atexit
import asyncio
import functools
import hashlib
import json
import os
from pathlib import Path
import time


SOURCE_SHA256 = {
    'engine_loop.py': 'ed2a1c26751785c9f94a8c12202d82bdb30b96e741f73506e5f1058543d50f41',
    'model_agent.py': '7844f9e2a6f05dc7b1356860e6f669ca3b57268632066837b37f9f85df404a27',
}
_INSTALLED = False
_EVENTS = []
_DROPPED = 0
_INDEX = 0
_LIMIT = 100000


def _record(name, **fields):
    global _DROPPED
    if len(_EVENTS) >= _LIMIT:
        _DROPPED += 1
        return
    try:
        task = asyncio.current_task()
    except RuntimeError:
        # Direct construction and synchronous diagnostic callbacks may run
        # outside an event loop; observing them must not break construction.
        task = None
    _EVENTS.append(dict(event=name, time_ns=time.perf_counter_ns(),
                        task=task.get_name() if task else None, **fields))


def flush(reason='explicit'):
    global _EVENTS, _DROPPED, _INDEX
    directory = os.environ.get('PROJECT2_STAGE2_RUN_DIR')
    if not directory or not _EVENTS:
        return
    data = dict(scope='Host diagnostic timestamps; not kernel timing or primary HTTP speed evidence',
                pid=os.getpid(), reason=reason, source_sha256=SOURCE_SHA256,
                dropped_events=_DROPPED, events=_EVENTS)
    path = Path(directory)/f'schedule_trace_{os.getpid()}_{_INDEX:04d}.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False)
        stream.write('\n')
    _INDEX += 1
    _EVENTS = []
    _DROPPED = 0


def install():
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get('PROJECT2_LATENCY_SCHEDULE_TRACE') != '1':
        return False
    from lmdeploy.pytorch.engine import engine_loop, model_agent
    for module in (engine_loop, model_agent):
        path = Path(module.__file__)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != SOURCE_SHA256[path.name]:
            raise RuntimeError(f'Unexpected scheduler source: {path.name} {actual}')
    Loop, Event, Agent = engine_loop.EngineLoop, engine_loop.CounterEvent, model_agent.BaseModelAgent
    original_init = Loop.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.forward_event._latency_schedule_observed = True
        _record('engine_loop_created', loop=id(self), event_id=id(self.forward_event),
                executor=type(self.executor).__name__)

    Loop.__init__ = init
    for method in ('set', 'clear'):
        original = getattr(Event, method)

        @functools.wraps(original)
        def mutate(self, *args, _original=original, _method=method, **kwargs):
            observed = getattr(self, '_latency_schedule_observed', False)
            if observed:
                _record('counter_'+_method+'_before', event_id=id(self), counter=self._counter, is_set=self.is_set())
            result = _original(self, *args, **kwargs)
            if observed:
                _record('counter_'+_method+'_after', event_id=id(self), counter=self._counter, is_set=self.is_set())
            return result

        setattr(Event, method, mutate)
    original_wait = Event.wait

    @functools.wraps(original_wait)
    async def wait(self):
        observed = getattr(self, '_latency_schedule_observed', False)
        if observed:
            _record('counter_wait_begin', event_id=id(self), counter=self._counter, is_set=self.is_set())
        result = await original_wait(self)
        if observed:
            _record('counter_wait_end', event_id=id(self), counter=self._counter, is_set=self.is_set())
        return result

    Event.wait = wait
    original_make = Loop._make_infer_outputs

    @functools.wraps(original_make)
    def make(self, *args, **kwargs):
        outputs = original_make(self, *args, **kwargs)
        for sid, out in outputs.items():
            _record('cpu_output_ready', session_id=sid, finish=bool(out.finish),
                    output_tokens=len(out.token_ids), event_id=id(self.forward_event),
                    counter=self.forward_event._counter, is_set=self.forward_event.is_set())
        return outputs

    Loop._make_infer_outputs = make
    original_send = Loop._send_resp

    @functools.wraps(original_send)
    def send(self, out):
        _record('engine_response_dispatch', session_id=out.session_id, finish=bool(out.finish),
                output_tokens=len(out.token_ids), event_id=id(self.forward_event),
                counter=self.forward_event._counter, is_set=self.forward_event.is_set())
        result = original_send(self, out)
        if out.finish:
            flush('finish_response_dispatched')
        return result

    Loop._send_resp = send
    original_step = Agent._async_step_background

    @functools.wraps(original_step)
    async def step(self, *args, **kwargs):
        inputs = kwargs.get('inputs', args[0] if args else None)
        count = kwargs.get('loop_count', args[1] if len(args) > 1 else None)
        is_decoding = inputs.is_decoding if inputs is not None else None
        _record('model_block_begin', agent=id(self), loop_count=count, is_decoding=is_decoding)
        try:
            result = await original_step(self, *args, **kwargs)
        except BaseException as error:
            _record('model_block_error', agent=id(self), error_type=type(error).__name__)
            raise
        _record('model_block_end', agent=id(self), loop_count=count, is_decoding=is_decoding)
        return result

    Agent._async_step_background = step
    atexit.register(flush, 'process_exit')
    _INSTALLED = True
    return True
