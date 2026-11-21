"""Tiny CPU checks of schedule instrumentation and native method contracts."""
import argparse
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PROJECT2_LATENCY_SCHEDULE_TRACE'] = '1'

import numpy as np
import torch
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.engine import ResponseType
from lmdeploy.pytorch.engine.engine_loop import EngineLoop, EngineLoopConfig
from lmdeploy.pytorch.engine.model_agent import BaseModelAgent, BatchedOutputs
from lmdeploy.pytorch.messages import MessageStatus
import navigation_latency_schedule_probe_20261120 as probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    original_step_signature = inspect.signature(BaseModelAgent._async_step_background)
    original_make_signature = inspect.signature(EngineLoop._make_infer_outputs)
    assert probe.install() and probe.install()
    assert inspect.signature(BaseModelAgent._async_step_background) == original_step_signature
    assert inspect.signature(EngineLoop._make_infer_outputs) == original_make_signature
    original_step_signature.bind(None, inputs=SimpleNamespace(is_decoding=True), loop_count=4)
    original_step_signature.bind(None, SimpleNamespace(is_decoding=True), 1)
    outputs_sent = []

    class SequenceStrategy:
        def update_running(self, running, batched_outputs, model_inputs):
            running[0].status = MessageStatus.STOPPED
            running[0].generated_ids = np.array([7, 8], dtype=np.int64)

    with tempfile.TemporaryDirectory(prefix='latency_schedule_cpu_') as temp:
        os.environ['PROJECT2_STAGE2_RUN_DIR'] = temp
        try:
            # Real EngineLoop construction outside any running asyncio loop.
            loop = EngineLoop(req_manager=SimpleNamespace(response=outputs_sent.append),
                              scheduler=SimpleNamespace(), executor=SimpleNamespace(),
                              seq_strategy=SequenceStrategy(), inputs_maker=None,
                              config=EngineLoopConfig(role=EngineRole.Hybrid))
            assert probe._EVENTS[-1]['event'] == 'engine_loop_created'
            assert probe._EVENTS[-1]['task'] is None
            event = loop.forward_event

            async def event_check():
                event.clear()
                event.clear()
                waiter = asyncio.create_task(event.wait(), name='cpu_probe_waiter')
                await asyncio.sleep(0)
                event.set()
                await asyncio.sleep(0)
                assert not waiter.done() and event._counter == 1
                event.set()
                assert await waiter is True
                assert event._counter == 0 and event.is_set()

            asyncio.run(event_check())
            msg = SimpleNamespace(num_token_ids=1, status=MessageStatus.RUNNING,
                                  generated_ids=np.array([7]), session_id=1, resp_cache=False,
                                  resp=SimpleNamespace(type=ResponseType.SUCCESS, data=None),
                                  sampling_param=SimpleNamespace(num_logprobs=0),
                                  engine_events=[], return_logits=False, routed_experts=None)
            batch = BatchedOutputs(next_token_ids=torch.tensor([8]), stopped=torch.tensor([True]),
                                   new_token_timestamp=123)
            result = loop._make_infer_outputs(batch, [msg], SimpleNamespace(is_decoding=True))
            assert isinstance(result, dict) and set(result) == {1}
            assert result[1].finish and list(result[1].token_ids) == [7, 8]
            loop._send_resp(result[1])
            assert len(outputs_sent) == 1 and outputs_sent[0].type == ResponseType.FINISH
            traces = [json.loads(p.read_text()) for p in Path(temp).glob('schedule_trace_*.json')]
            assert len(traces) == 1 and traces[0]['dropped_events'] == 0
            events = traces[0]['events']
            names = {e['event'] for e in events}
            assert {'counter_wait_begin', 'counter_wait_end', 'cpu_output_ready',
                    'engine_response_dispatch', 'engine_loop_created'} <= names
            ready = next(e for e in events if e['event'] == 'cpu_output_ready')
            dispatch = next(e for e in events if e['event'] == 'engine_response_dispatch')
            assert ready['finish'] and dispatch['finish']
            assert ready['time_ns'] <= dispatch['time_ns']
            assert not torch.cuda.is_initialized()
            report = dict(passed=True, scope='CPU fake dependencies with actual EngineLoop, CounterEvent, InferOutput path',
                          cuda_initialized=False, trace_events=len(events),
                          checks=['outside-loop construction does not raise', 'double install is idempotent',
                                  'native method signatures retained', 'positional and keyword block args bind',
                                  'CounterEvent two-clear/two-set wait semantics retained',
                                  'actual _make_infer_outputs returns dict of InferOutput',
                                  'FINISH dispatch reaches request manager and writes expected trace'],
                          model_step_signature=str(original_step_signature),
                          make_outputs_signature=str(original_make_signature),
                          probe_sha256=hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest(),
                          source_sha256=probe.SOURCE_SHA256,
                          limits=['No model forward or GPU test; no inference timing conclusion.'])
        finally:
            probe.flush('cpu_probe_cleanup')
            os.environ.pop('PROJECT2_STAGE2_RUN_DIR', None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
