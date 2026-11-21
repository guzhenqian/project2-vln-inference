"""Opt-in HTTP construction bridge to LMDeploy's native in-process engine.

PytorchEngineConfig.enable_mp_engine=False is a native engine path, but the
0.11.1 HTTP helper forces it True. This process-local experiment changes that
one option immediately before delegating to the original construction method.
It does not copy the HTTP server or alter model mathematics, preprocessing,
weights, generation, KV layout, or graph execution. The private bridge is
source-guarded and must be treated as an experiment, not a general HTTP API.

Tradeoff: removing ZMQ also removes process isolation. Frontend and engine
coroutines share an event loop, so host work could hurt responsiveness. The
entry is restricted to the project's single-GPU UniExecutor configuration.
"""
from dataclasses import asdict
import functools
import hashlib
import json
import os
from pathlib import Path


_SOURCE_SHA256 = {
    'serve/async_engine.py': 'f9a12e14b1fa172ce260c29b7ff968a53faa852e4bafbd790f50370c190a99a9',
    'serve/openai/api_server.py': '11a39253b54a880a6ec37ab71b80a314f3c3f9bffdc356e123b157d1083c885c',
    'pytorch/engine/engine.py': '80c6703b623df59dc24971f515bca6d444d8b39bd266b45abf4542e5edf4720c',
}
_INSTALLED = False


def _class_name(value):
    cls = type(value)
    return cls.__module__+'.'+cls.__qualname__


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda x: getattr(x, 'value', str(x))))


def install():
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get('PROJECT2_LATENCY_INPROCESS') != '1':
        return False
    import lmdeploy
    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.serve.async_engine import AsyncEngine
    from lmdeploy.pytorch.engine import Engine
    from lmdeploy.pytorch.engine.executor.uni_executor import UniExecutor

    base = Path(lmdeploy.__file__).parent
    for relative, expected in _SOURCE_SHA256.items():
        actual = hashlib.sha256((base/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Unexpected LMDeploy source {relative}: {actual}')
    original = AsyncEngine._build_pytorch
    if Path(original.__code__.co_filename).resolve() != (base/'serve/async_engine.py').resolve():
        raise RuntimeError('Unexpected wrapper around AsyncEngine._build_pytorch')

    @functools.wraps(original)
    def build(self, model_path, backend_config=None, speculative_config=None, **kwargs):
        if not isinstance(backend_config, PytorchEngineConfig):
            raise ValueError('Explicit PytorchEngineConfig is required')
        required = dict(tp=1, dp=1, max_batch_size=1,
                        distributed_executor_backend='uni', device_type='cuda',
                        empty_init=False, enable_return_routed_experts=False)
        for name, expected in required.items():
            actual = getattr(backend_config, name)
            if actual != expected:
                raise ValueError(f'In-process experiment requires {name}={expected!r}, got {actual!r}')
        if speculative_config is not None:
            raise ValueError('Speculative decoding is outside this experiment')
        directory = os.environ.get('PROJECT2_STAGE2_RUN_DIR')
        if not directory or not Path(directory).is_dir():
            raise ValueError('Existing PROJECT2_STAGE2_RUN_DIR is required')
        receipt = Path(directory)/'effective_inprocess_engine.json'
        if receipt.exists():
            raise FileExistsError(receipt)
        before = _jsonable(asdict(backend_config))
        # This is the sole inference-setting mutation made by this bridge.
        backend_config.enable_mp_engine = False
        delegated = _jsonable(asdict(backend_config))
        engine = original(self, model_path=model_path, backend_config=backend_config,
                          speculative_config=speculative_config, **kwargs)
        if type(engine) is not Engine or type(engine.executor) is not UniExecutor:
            raise RuntimeError('The returned engine/executor does not match the native in-process path')
        if engine.engine_config.enable_mp_engine:
            raise RuntimeError('Native engine still reports enable_mp_engine=True')
        agent = engine.executor.model_agent
        graph_or_model = agent.patched_model
        model = getattr(graph_or_model, 'model', graph_or_model)
        data = dict(scope='Observed after native engine initialization; not merely requested launch configuration',
                    pid=os.getpid(), native_engine_observed=True,
                    engine_class=_class_name(engine), executor_class=_class_name(engine.executor),
                    model_agent_class=_class_name(agent), graph_or_model_class=_class_name(graph_or_model),
                    model_class=_class_name(model), builder_input_config=before,
                    delegated_config=delegated, effective_engine_config=_jsonable(asdict(engine.engine_config)),
                    only_bridge_setting_changed='enable_mp_engine', source_sha256=_SOURCE_SHA256,
                    limitations=['Single-process lifecycle and output behavior still require end-to-end validation.',
                                 'This is a guarded private HTTP construction bridge to a native engine path.'])
        receipt.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        return engine

    AsyncEngine._build_pytorch = build
    _INSTALLED = True
    return True
