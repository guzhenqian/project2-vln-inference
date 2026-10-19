"""项目2：使用老师的渲染器检查中文叠加，不调用模型。"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "Qwen3VL-main/Qwen3VL-main"))
import cv2
from PIL import ImageFont
from app.vln import Waypoint
from app.vln.visualizer import Visualizer

visualizer = Visualizer()
font = visualizer._pil_font
if not isinstance(font, ImageFont.FreeTypeFont):
    raise SystemExit("FAIL: no Chinese font was loaded")
frame = cv2.imread(str(root / "data/tum_rgbd/sample_rgb.png"))
if frame is None:
    raise SystemExit("FAIL: sample image could not be read")
rendered = visualizer.render_frame(
    frame=frame,
    waypoints=[Waypoint(dx=0.0, dy=0.0, dtheta=0.0)],
    environment="办公桌：显示器、键盘和鼠标。",
    action="保持停止，无法确认地面路径。",
    linear_vel=0.0,
    angular_vel=0.0,
)
output = root / "logs/chinese_overlay_preview.png"
output.parent.mkdir(exist_ok=True)
if not cv2.imwrite(str(output), rendered):
    raise SystemExit("FAIL: preview could not be saved")
print("Loaded font:", font.path)
print("Saved preview:", output)
print("RENDER_OK: compare the Chinese text in the saved preview; no model was called.")
