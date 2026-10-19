"""检查老师的 Gradio 页面构建，不启动服务器，不发送模型请求。"""

import os
import sys
import importlib.util
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
root = Path(__file__).resolve().parents[1]
source = root / "Qwen3VL-main/Qwen3VL-main"

import cv2
import gradio

# app/__init__.py 会连带导入整个业务后端；页面是独立文件，可直接加载。
spec = importlib.util.spec_from_file_location("teacher_vln_webui", source / "app/vln_webui.py")
vln_webui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vln_webui)

print("Python path:", sys.executable)
print("Gradio:", gradio.__version__)
print("OpenCV:", cv2.__version__)
print("Business WebUI source:", vln_webui.__file__)
if Path(vln_webui.__file__).resolve() != source / "app/vln_webui.py":
    raise SystemExit("FAIL: unexpected WebUI source")
demo = vln_webui.create_ui("http://127.0.0.1:20000")
if not isinstance(demo, gradio.Blocks):
    raise SystemExit("FAIL: expected Gradio Blocks")
print("PASS: teacher WebUI imports and builds; browser access and video are not tested yet.")
