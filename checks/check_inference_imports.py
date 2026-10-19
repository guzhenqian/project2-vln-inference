"""项目2：检查推理依赖能否导入；不下载或加载模型。"""

import sys
from importlib.metadata import version


print("Python path:", sys.executable)
expected_versions = {
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "triton": "3.1.0",
    "lmdeploy": "0.11.1",
    "transformers": "4.57.6",
}

for package, expected in expected_versions.items():
    actual = version(package)
    print(f"{package}: {actual}")
    if actual != expected:
        raise SystemExit(f"FAIL: {package} expected {expected}, found {actual}")

# import 会真正加载 Python 模块，可暴露缺失依赖和部分版本冲突。
import lmdeploy
from lmdeploy import PytorchEngineConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# 仅构造单卡配置，不创建推理引擎，也不申请模型权重。
engine_config = PytorchEngineConfig(tp=1, session_len=2048)
print("LMDeploy import: OK")
print("Qwen3-VL model class:", Qwen3VLForConditionalGeneration.__name__)
print("Processor class:", AutoProcessor.__name__)
print("Tensor parallel size:", engine_config.tp)
print("PASS: inference imports are ready; model inference is not tested yet.")
