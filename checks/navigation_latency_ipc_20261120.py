"""Opt-in linear-copy transport for LMDeploy 0.11.1 MP SharedBuffer.

The wire format, notifier/barrier boundaries, buffers, pickle protocol, and
receiver masks are unchanged. Only redundant immutable-bytes copies change.
Install before optional frontend instrumentation, in each spawn process.
"""
import hashlib
import json
import os
from pathlib import Path
import time


SOURCE_SHA256 = '36d4e3d6100ed469f0381d71d3789e9ce34feaba012823c5c53d464d85fda314'
_INSTALLED = False
_INSTALL_RECEIPT = False
_PACK_RECEIPT = False
_RECEIVE_RECEIPT = False
_MODULE_SHA256 = None
_MP_MODULE = None
_LARGE_PAYLOAD = 1 << 20


def _function_identity(fn):
    result = dict(module=fn.__module__, name=fn.__name__, qualname=fn.__qualname__,
                  code_filename=fn.__code__.co_filename)
    wrapped = getattr(fn, '__wrapped__', None)
    if wrapped is not None:
        result['wrapped'] = _function_identity(wrapped)
    return result


def _write_receipt(kind, **extra):
    directory = os.environ.get('PROJECT2_STAGE2_RUN_DIR')
    if not directory:
        return False
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    record = dict(kind=kind, pid=os.getpid(), time_ns=time.time_ns(),
                  patch_module=str(Path(__file__).resolve()), patch_module_sha256=_MODULE_SHA256,
                  original_mp_source_sha256=SOURCE_SHA256, enabled_env='1',
                  installed_pack_function=_function_identity(_MP_MODULE.SharedBuffer.pack_data),
                  installed_receive_function=_function_identity(_MP_MODULE.SharedBuffer._receive_step1),
                  **extra)
    path = directory/f'linear_ipc_{kind}_{os.getpid()}.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return True


def _ensure_install_receipt():
    global _INSTALL_RECEIPT
    if not _INSTALL_RECEIPT:
        _INSTALL_RECEIPT = _write_receipt('install', scope='process-local transport patch installed')


def install():
    """Return whether the opt-in patch is installed; reject unknown source."""
    global _INSTALLED, _MODULE_SHA256, _MP_MODULE
    if _INSTALLED:
        return True
    if os.environ.get('PROJECT2_LATENCY_LINEAR_IPC') != '1':
        return False
    from lmdeploy.pytorch.engine.executor import mp_executor as mp

    actual = hashlib.sha256(Path(mp.__file__).read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f'Unexpected MP executor source SHA256: {actual}')
    if (mp.SHARED_BLOCK_SIZE, mp.HEAD_SIZE, mp.NUM_SHARED_BLOCK) != (1 << 20, 8, 32):
        raise RuntimeError('Unexpected MP transport constants')
    for name in ('pack_data', '_receive_step1'):
        fn = getattr(mp.SharedBuffer, name)
        if (fn.__module__ != mp.__name__ or fn.__qualname__ != 'SharedBuffer.'+name
                or Path(fn.__code__.co_filename).resolve() != Path(mp.__file__).resolve()):
            raise RuntimeError('Install IPC patch before any SharedBuffer wrapper: '+name)
    _MODULE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _MP_MODULE = mp

    def pack_data(self, data, receiver_mask):
        global _PACK_RECEIPT
        if not _INSTALL_RECEIPT:
            _ensure_install_receipt()
        dumped_data = mp.pickle.dumps(data, protocol=mp.pickle.HIGHEST_PROTOCOL)
        data_size = len(dumped_data)
        head = mp.struct.pack('II', data_size, receiver_mask)
        with memoryview(dumped_data) as payload:
            for offset in range(0, data_size, mp.SHARED_BLOCK_SIZE):
                with self.acquire_buf() as buf:
                    pac_size = min(data_size-offset, mp.SHARED_BLOCK_SIZE)
                    buf[:mp.HEAD_SIZE] = head
                    buf[mp.HEAD_SIZE:mp.HEAD_SIZE+pac_size] = payload[offset:offset+pac_size]
                    # The caller notifies exactly once per yielded block, as before.
                    yield buf
        if not _PACK_RECEIPT and data_size >= _LARGE_PAYLOAD:
            _PACK_RECEIPT = _write_receipt('first_large_pack', path='send', payload_bytes=data_size,
                                         blocks=mp.get_num_packages(data_size), receiver_mask=receiver_mask,
                                         completed=True,
                                         implementation='memoryview offset slices; unchanged block notifier protocol')

    def receive_step1(self, dumped_data, is_receiver, remain_size):
        global _RECEIVE_RECEIPT
        if not _INSTALL_RECEIPT:
            _ensure_install_receipt()
        position = len(dumped_data)
        data_size = position+remain_size
        first_block_bytes = position
        assembled = bytearray(position+remain_size) if is_receiver else None
        if is_receiver:
            assembled[:position] = dumped_data
        while remain_size > 0:
            with self.notifier.wait(), self.acquire_buf() as buf:
                pac_size = min(remain_size, mp.SHARED_BLOCK_SIZE)
                remain_size -= pac_size
                if not is_receiver:
                    continue
                assembled[position:position+pac_size] = buf[mp.HEAD_SIZE:mp.HEAD_SIZE+pac_size]
                position += pac_size
        if not is_receiver:
            return None
        result = mp.pickle.loads(assembled)
        if not _RECEIVE_RECEIPT and data_size >= _LARGE_PAYLOAD:
            _RECEIVE_RECEIPT = _write_receipt('first_large_receive', path='receive', payload_bytes=data_size,
                                            first_block_bytes=first_block_bytes,
                                            remaining_bytes=data_size-first_block_bytes,
                                            blocks=mp.get_num_packages(data_size), completed=True,
                                            implementation='single bytearray allocation; offset copies and pickle.loads')
        return result

    mp.SharedBuffer.pack_data = pack_data
    mp.SharedBuffer._receive_step1 = receive_step1
    _INSTALLED = True
    _ensure_install_receipt()
    return True
