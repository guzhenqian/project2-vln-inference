"""项目2：检查老师原始业务模块能否导入及接口能否注册，不启动服务。"""

import sys
from importlib.metadata import version
from pathlib import Path


root = Path(__file__).resolve().parents[1]
source_dir = root / "Qwen3VL-main/Qwen3VL-main"
sys.path.insert(0, str(source_dir))

print("Python path:", sys.executable)
print("PyTorch:", version("torch"))
print("LMDeploy:", version("lmdeploy"))

import cv2
import fastapi
import app.main as main_module


expected_source = source_dir / "app/main.py"
if Path(main_module.__file__).resolve() != expected_source.resolve():
    raise SystemExit("FAIL: imported an unexpected app.main module")

print("Business source:", main_module.__file__)
print("OpenCV:", cv2.__version__)
print("FastAPI:", fastapi.__version__)

# 生成接口说明，检查业务模块导入和请求/响应模型能否正确注册。
# 此处不会执行应用启动生命周期，也不会加载第二份模型权重。
schema = main_module.app.openapi()
required_paths = ["/health", "/v1/chat/completions", "/vln/task/create", "/vln/frame/sync"]
for path in required_paths:
    if path not in schema["paths"]:
        raise SystemExit(f"FAIL: missing business endpoint: {path}")
    print("Route registered:", path)

print("PASS: business modules import and routes are registered.")
