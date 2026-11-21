"""Optional, low-volume CPU instrumentation for the new latency experiment.

Call install() at the service module top level so spawn children install it too.
Enable recording with PROJECT2_LATENCY_PROFILE=1. The run directory is the
existing PROJECT2_STAGE2_RUN_DIR. No CUDA synchronization, tensor copies, input
changes, or inference algorithm changes are introduced by this module.

Events from different processes share perf_counter's system monotonic clock.
The MP receive interval includes block waiting; it is not pure memcpy time.
"""
import functools
import json
import os
from pathlib import Path
import time


_INSTALLED = False
_LARGE = 1 << 20


def _log(event, start, end, **extra):
    if os.environ.get('PROJECT2_LATENCY_PROFILE') != '1':
        return
    directory = os.environ.get('PROJECT2_STAGE2_RUN_DIR')
    if not directory:
        return
    record = dict(event=event, start=start, end=end,
                  wall_ms=(end-start)*1000, pid=os.getpid(), **extra)
    with (Path(directory)/f'latency_frontend_{os.getpid()}.jsonl').open(
            'a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False)+'\n')


class _ObservedPickle:
    """Rebind just the importing module's reference, not global pickle."""
    def __init__(self, original, label):
        self.original = original
        self.label = label

    def __getattr__(self, name):
        return getattr(self.original, name)

    def dumps(self, value, *args, **kwargs):
        start = time.perf_counter()
        result = self.original.dumps(value, *args, **kwargs)
        end = time.perf_counter()
        if len(result) >= _LARGE:
            _log(self.label+'_serialize', start, end, bytes=len(result))
        return result

    def loads(self, value, *args, **kwargs):
        start = time.perf_counter()
        result = self.original.loads(value, *args, **kwargs)
        end = time.perf_counter()
        if len(value) >= _LARGE:
            _log(self.label+'_deserialize', start, end, bytes=len(value))
        return result


def install():
    global _INSTALLED
    if _INSTALLED or os.environ.get('PROJECT2_LATENCY_PROFILE') != '1':
        return
    _INSTALLED = True
    from lmdeploy.serve.vl_async_engine import VLAsyncEngine
    from lmdeploy.vl.engine import ImageEncoder
    from lmdeploy.pytorch.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from lmdeploy.pytorch.engine.mp_engine import zmq_rpc
    from lmdeploy.pytorch.engine.executor import mp_executor

    for cls, name, label in (
        (VLAsyncEngine, '_get_prompt_input', 'frontend_prompt'),
        (ImageEncoder, 'preprocess', 'frontend_preprocess'),
        (ImageEncoder, 'wrap_for_pytorch', 'frontend_wrap'),
    ):
        original = getattr(cls, name)

        @functools.wraps(original)
        async def wrapper(self, *args, _original=original, _label=label, **kwargs):
            start = time.perf_counter()
            result = await _original(self, *args, **kwargs)
            _log(_label, start, time.perf_counter())
            return result

        setattr(cls, name, wrapper)

    original_prepare = Qwen3VLForConditionalGeneration.prepare_inputs_for_generation

    @functools.wraps(original_prepare)
    def prepare(self, *args, **kwargs):
        start = time.perf_counter()
        result = original_prepare(self, *args, **kwargs)
        end = time.perf_counter()
        pixels = result.get('pixel_values')
        if pixels is not None:
            _log('worker_prepare_inputs', start, end,
                 pixel_shape=list(pixels.shape), pixel_dtype=str(pixels.dtype),
                 pixel_bytes=pixels.numel()*pixels.element_size(),
                 input_tokens=result['input_ids'].numel())
        return result

    Qwen3VLForConditionalGeneration.prepare_inputs_for_generation = prepare
    zmq_rpc.pickle = _ObservedPickle(zmq_rpc.pickle, 'zmq_rpc')
    mp_executor.pickle = _ObservedPickle(mp_executor.pickle, 'mp_executor')

    original_pack = mp_executor.SharedBuffer.pack_data

    @functools.wraps(original_pack)
    def pack(self, *args, **kwargs):
        start = time.perf_counter()
        blocks = 0
        for block in original_pack(self, *args, **kwargs):
            blocks += 1
            yield block
        if blocks > 1:
            _log('mp_pack_and_send_blocks', start, time.perf_counter(), blocks=blocks)

    mp_executor.SharedBuffer.pack_data = pack
    original_receive = mp_executor.SharedBuffer._receive_step1

    @functools.wraps(original_receive)
    def receive(self, dumped_data, is_receiver, remain_size):
        start = time.perf_counter()
        total_bytes = len(dumped_data)+remain_size
        result = original_receive(self, dumped_data, is_receiver, remain_size)
        if is_receiver and total_bytes >= _LARGE:
            _log('mp_receive_remaining_blocks', start, time.perf_counter(),
                 bytes=total_bytes, includes_block_wait=True)
        return result

    mp_executor.SharedBuffer._receive_step1 = receive
