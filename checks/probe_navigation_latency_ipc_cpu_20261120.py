"""CPU-only old/new MP transport interoperability and content probe.

Uses the actual SharedBuffer and Notifier, including the 32-block barrier.
No model, image processor, or CUDA device is loaded. Run this while GPU timing
is idle, since the old transport intentionally consumes significant CPU memory
bandwidth. Each old/new pairing runs in fresh processes.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.pop('PROJECT2_LATENCY_PROFILE', None)
os.environ.pop('PROJECT2_LATENCY_LINEAR_IPC', None)

import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from lmdeploy.pytorch.engine.executor.mp_executor import Notifier, SharedBuffer, SHARED_BLOCK_SIZE
from navigation_latency_ipc_20261120 import install, SOURCE_SHA256


def signature(value):
    if isinstance(value, bytes):
        return dict(type='bytes', size=len(value), sha256=hashlib.sha256(value).hexdigest())
    if isinstance(value, torch.Tensor):
        assert value.device.type == 'cpu'
        a = value.numpy()
        return dict(type='tensor', shape=list(a.shape), dtype=str(a.dtype),
                    sha256=hashlib.sha256(memoryview(a)).hexdigest())
    if isinstance(value, np.ndarray):
        return dict(type='numpy', shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(memoryview(value)).hexdigest())
    if isinstance(value, dict):
        return {k: signature(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [signature(v) for v in value]
    return value


def enable(mode):
    if mode == 'new':
        os.environ['PROJECT2_LATENCY_LINEAR_IPC'] = '1'
        assert install()
        assert install()
    else:
        assert not install()
    assert not torch.cuda.is_initialized()


def receiver(mode, notifier, name, count, output, async_mode):
    enable(mode)
    buffer = SharedBuffer(0, notifier, name=name)
    try:
        async def run():
            results = []
            for _ in range(count):
                started = time.perf_counter()
                value = await buffer.receive_async() if async_mode else buffer.receive()
                elapsed = time.perf_counter()-started
                results.append(dict(signature=signature(value), wall_s=elapsed))
            return results
        rows = asyncio.run(run())
        output.put(dict(rows=rows, buffer_id=buffer._buf_id,
                        notifier_id=notifier._event_id, cuda_initialized=torch.cuda.is_initialized()))
    finally:
        buffer.close()


def run_case(sender_mode, receiver_mode, async_mode):
    enable(sender_mode)
    ctx = multiprocessing.get_context('spawn')
    notifier = Notifier(1, ctx)
    buffer = SharedBuffer(-1, notifier)
    output = ctx.Queue()
    b = SHARED_BLOCK_SIZE
    sizes = [0, 1, b-64, b-1, b, b+1, 2*b-1, 32*b+17, 63*b]
    count = len(sizes)+4
    child = ctx.Process(target=receiver, args=(receiver_mode, notifier, buffer.name(), count, output, async_mode))
    child.start()
    sent = []

    async def send(value, mask=1, label=None):
        expected = signature(value) if mask else None
        started = time.perf_counter()
        if async_mode:
            await buffer.send_async(value, receiver_mask=mask)
        else:
            buffer.send(value, receiver_mask=mask)
        sent.append(dict(label=label, signature=expected, wall_s=time.perf_counter()-started))

    async def run():
        for n in sizes:
            await send(os.urandom(n), label=f'bytes_{n}')
        rng = np.random.default_rng(20261120)
        # Match the actual nine-image FP32 pixel shape and metadata size.
        pixels = rng.standard_normal((10800, 1536), dtype=np.float32)
        await send(dict(pixel_values=pixels, image_grid_thw=np.tile([1, 30, 40], (9, 1)),
                        input_ids=list(range(2815))), label='numpy_nine_image_structure')
        await send(dict(pixel_values=torch.from_numpy(pixels),
                        image_grid_thw=torch.tensor([[1, 30, 40]]*9)), label='tensor_nine_image_structure')
        # A non-target receiver must still consume all blocks and barriers.
        await send(os.urandom(33*b+5), mask=0, label='masked_33_blocks')
        await send(dict(after_mask=True, payload=b'transport-still-aligned'), label='after_mask')

    try:
        asyncio.run(run())
        received = output.get(timeout=60)
        child.join(60)
        assert child.exitcode == 0, child.exitcode
        assert len(received['rows']) == len(sent) == count
        for a, r in zip(sent, received['rows']):
            assert a['signature'] == r['signature'], a['label']
            a['receive_wall_s'] = r['wall_s']
        assert buffer._buf_id == received['buffer_id']
        assert notifier._event_id == received['notifier_id']
        assert not received['cuda_initialized'] and not torch.cuda.is_initialized()
        return dict(passed=True, sender=sender_mode, receiver=receiver_mode,
                    async_mode=async_mode, source_sha256=SOURCE_SHA256,
                    count=count, final_buffer_id=buffer._buf_id,
                    final_notifier_id=notifier._event_id, cases=sent,
                    timing_scope='CPU transport probe only; includes waits and is not inference latency')
    finally:
        if child.is_alive():
            child.terminate()
            child.join(10)
        buffer.close()
        notifier.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pair', nargs=2, choices=['old', 'new'])
    p.add_argument('--async-mode', action='store_true')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    assert not args.output.exists(), args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.pair:
        receipt_dir = args.output.with_name(args.output.stem+'_receipts')
        receipt_dir.mkdir(exist_ok=False)
        os.environ['PROJECT2_STAGE2_RUN_DIR'] = str(receipt_dir)
        result = run_case(*args.pair, args.async_mode)
        receipts = [json.loads(p.read_text()) for p in sorted(receipt_dir.glob('linear_ipc_*.json'))]
        expected_sha = hashlib.sha256(Path(__file__).with_name('navigation_latency_ipc_20261120.py').read_bytes()).hexdigest()
        assert len([r for r in receipts if r['kind'] == 'install']) == args.pair.count('new')
        for kind, needed in [('first_large_pack', args.pair[0] == 'new'),
                             ('first_large_receive', args.pair[1] == 'new')]:
            matches = [r for r in receipts if r['kind'] == kind]
            assert len(matches) == int(needed), (kind, matches)
            if needed:
                assert matches[0]['completed'] and matches[0]['payload_bytes'] >= SHARED_BLOCK_SIZE
        for r in receipts:
            assert r['patch_module_sha256'] == expected_sha
            assert r['original_mp_source_sha256'] == SOURCE_SHA256
            assert r['installed_pack_function']['module'] == 'navigation_latency_ipc_20261120'
            assert r['installed_receive_function']['module'] == 'navigation_latency_ipc_20261120'
        result['patch_module_sha256'] = expected_sha
        result['execution_receipts'] = receipts
    else:
        results = []
        for mode in (False, True):
            for sender, receiver in [('old', 'new'), ('new', 'old'), ('new', 'new')]:
                target = args.output.with_name(f'{args.output.stem}_{sender}_{receiver}_{"async" if mode else "sync"}.json')
                cmd = [sys.executable, __file__, '--pair', sender, receiver, '--output', str(target)]
                if mode:
                    cmd.append('--async-mode')
                subprocess.run(cmd, check=True, timeout=240)
                results.append(json.loads(target.read_text()))
        result = dict(passed=all(r['passed'] for r in results), pairs=results,
                      total_messages=sum(r['count'] for r in results),
                      source_sha256=SOURCE_SHA256, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
