"""
VLN WebUI - 视觉语言导航可视化界面

功能：
1. 输入视频 URL 或本地路径
2. 实时显示带标注的输出画面
3. 发布/切换导航任务
4. 显示导航状态和历史

启动方式：
    python -m app.vln_webui --port 7860

访问地址：
    http://localhost:7860
"""

import argparse
import base64
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2
import gradio as gr
import numpy as np
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLNWebUI:
    """VLN WebUI 控制器"""

    def __init__(self, api_url: str = "http://localhost:20000"):
        self.api_url = api_url.rstrip("/")
        self.current_task_id: Optional[str] = None
        self.current_instruction: str = ""
        self.is_running: bool = False
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.task_history: list = []
        self.latest_result: dict = {}
        self._stop_event = threading.Event()

    def check_health(self) -> Tuple[bool, str]:
        """检查服务状态"""
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=5)
            data = resp.json()
            if data.get("status") == "healthy":
                return True, f"✅ 服务正常 | 模型: {data.get('model')} | GPU: {data.get('gpu_count')}"
            return False, f"⚠️ 服务状态: {data.get('status')}"
        except Exception as e:
            return False, f"❌ 服务不可用: {str(e)}"

    def create_task(self, instruction: str) -> Tuple[bool, str]:
        """创建导航任务"""
        if not instruction.strip():
            return False, "❌ 请输入导航指令"

        try:
            # 停止当前任务
            if self.current_task_id:
                self._stop_task(self.current_task_id)

            resp = requests.post(
                f"{self.api_url}/vln/task/create",
                json={
                    "instruction": instruction,
                    "config": {
                        "output_fps": 5.0,
                        "keyframe_threshold": 0.2,
                        "enable_visualization": True,
                    },
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self.current_task_id = data["task_id"]
            self.current_instruction = instruction
            self.task_history.append({
                "task_id": self.current_task_id,
                "instruction": instruction,
                "time": time.strftime("%H:%M:%S"),
            })
            return True, f"✅ 任务创建成功\nID: {self.current_task_id}\n指令: {instruction}"
        except Exception as e:
            return False, f"❌ 创建任务失败: {str(e)}"

    def _stop_task(self, task_id: str):
        """停止任务"""
        try:
            requests.post(f"{self.api_url}/vln/task/{task_id}/stop", timeout=5)
        except:
            pass

    def load_video(self, video_input: str) -> Tuple[bool, str, Optional[np.ndarray]]:
        """加载视频"""
        if not video_input.strip():
            return False, "❌ 请输入视频路径或 URL", None

        video_path = video_input.strip()

        # 检查是否为 URL
        if video_path.startswith("http://") or video_path.startswith("https://"):
            # 下载视频
            try:
                logger.info(f"下载视频: {video_path}")
                resp = requests.get(video_path, stream=True, timeout=60)
                resp.raise_for_status()

                # 保存到临时文件
                parsed = urlparse(video_path)
                ext = Path(parsed.path).suffix or ".mp4"
                temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                for chunk in resp.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file.close()
                video_path = temp_file.name
                logger.info(f"视频已下载: {video_path}")
            except Exception as e:
                return False, f"❌ 下载视频失败: {str(e)}", None

        # 检查文件是否存在
        if not os.path.exists(video_path):
            return False, f"❌ 视频文件不存在: {video_path}", None

        # 打开视频
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            return False, f"❌ 无法打开视频: {video_path}", None

        # 获取视频信息
        total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 读取第一帧预览
        ret, frame = self.video_cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开头

        self.frame_count = 0
        info = f"✅ 视频加载成功\n分辨率: {width}x{height}\n帧数: {total_frames}\nFPS: {fps:.1f}"
        return True, info, frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """处理单帧"""
        if not self.current_task_id:
            return frame, {"error": "无活动任务"}

        try:
            # BGR -> RGB -> BGR for encoding
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode()

            resp = requests.post(
                f"{self.api_url}/vln/frame",
                json={
                    "task_id": self.current_task_id,
                    "frame": frame_base64,
                    "timestamp": time.time(),
                },
                timeout=10,
            )
            # 异步流水线可能已结束任务，而上一条帧响应仍是历史结果。
            # 后端拒绝已结束任务的后续帧时，查询真实状态并结束页面送帧。
            if resp.status_code == 400:
                task_resp = requests.get(
                    f"{self.api_url}/vln/task/{self.current_task_id}", timeout=5,
                )
                task_resp.raise_for_status()
                task_state = task_resp.json()
                if task_state.get("status") == "completed":
                    self.stop_video()
                    return frame, {
                        "terminal_status": "completed",
                        "environment": task_state.get("current_environment", ""),
                        "action": task_state.get("current_action", ""),
                    }
            if not resp.ok:
                raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:1000]}")
            resp.raise_for_status()
            result = resp.json()
            self.latest_result = result
            if result.get("reached_goal") is True:
                self.stop_video()
                result["terminal_status"] = "completed"

            # 解码可视化帧
            if result.get("visualized_frame"):
                vis_data = base64.b64decode(result["visualized_frame"])
                vis_frame = cv2.imdecode(np.frombuffer(vis_data, np.uint8), cv2.IMREAD_COLOR)
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                return vis_frame, result

            return frame, result

        except Exception as e:
            logger.error(f"处理帧失败: {e}")
            # 首次失败即暂停，让用户看到错误，避免循环发送失败请求。
            self.stop_video()
            return frame, {"error": str(e)}

    def run_video(self, skip_frames: int = 2):
        """运行视频处理（生成器）"""
        if not self.video_cap or not self.video_cap.isOpened():
            yield None, "❌ 请先加载视频"
            return

        if not self.current_task_id:
            yield None, "❌ 请先创建导航任务"
            return

        self.is_running = True
        self._stop_event.clear()
        frame_idx = 0

        while self.is_running and not self._stop_event.is_set():
            ret, frame = self.video_cap.read()
            if not ret:
                # 视频结束，循环播放
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            # 跳帧
            if frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vis_frame, result = self.process_frame(frame_rgb)

            self.frame_count = frame_idx
            frame_idx += 1

            # 构建状态信息
            status = self._build_status(result)
            yield vis_frame, status

            time.sleep(0.1)  # 控制帧率

        self.is_running = False

    def stop_video(self):
        """停止视频处理"""
        self._stop_event.set()
        self.is_running = False

    def _build_status(self, result: dict) -> str:
        """构建状态显示"""
        lines = []
        lines.append(f"**当前任务**: {self.current_instruction}")
        lines.append(f"**任务 ID**: {self.current_task_id}")
        lines.append(f"**帧号**: {self.frame_count}")
        lines.append("")

        if result.get("terminal_status") == "completed":
            lines.append("**任务状态**: 业务层已标记结束，页面已停止送帧。")
            lines.append("此状态不代表真实机器人已到达目标；结束原因仍需核查。")
            lines.append(f"**最后环境**: {result.get('environment', '')}")
            lines.append(f"**最后动作**: {result.get('action', '')}")
        elif "error" in result:
            lines.append(f"**错误**: {result['error']}")
            lines.append("已停止送帧；排查后请重新发布任务。")
        else:
            lines.append(f"**环境**: {result.get('environment', 'N/A')}")
            lines.append(f"**动作**: {result.get('action', 'N/A')}")
            lines.append(f"**线速度**: {result.get('linear_vel', 0):.3f} m/s")
            lines.append(f"**角速度**: {result.get('angular_vel', 0):.3f} rad/s")
            lines.append(f"**航点数**: {len(result.get('waypoints', []))}")
            lines.append(f"**推理时间**: {result.get('inference_time', 0):.3f}s")

        return "\n".join(lines)

    def get_task_history(self) -> str:
        """获取任务历史"""
        if not self.task_history:
            return "暂无任务历史"

        lines = []
        for i, task in enumerate(reversed(self.task_history[-10:])):
            lines.append(f"{i+1}. [{task['time']}] {task['instruction'][:30]}...")
        return "\n".join(lines)


def create_ui(api_url: str = "http://localhost:20000"):
    """创建 Gradio UI"""
    controller = VLNWebUI(api_url)

    with gr.Blocks(title="VLN WebUI - 视觉语言导航", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 VLN WebUI - 视觉语言导航系统")
        gr.Markdown("基于 Qwen3-VL 的机器狗视觉语言导航可视化界面")

        with gr.Row():
            # 左侧：视频显示
            with gr.Column(scale=2):
                video_output = gr.Image(
                    label="导航画面",
                    height=480,
                )
                with gr.Row():
                    start_btn = gr.Button("▶️ 开始", variant="primary")
                    stop_btn = gr.Button("⏹️ 停止", variant="stop")

            # 右侧：控制面板
            with gr.Column(scale=1):
                # 服务状态
                with gr.Group():
                    gr.Markdown("### 📡 服务状态")
                    status_text = gr.Textbox(
                        label="",
                        value="点击检查状态",
                        interactive=False,
                        lines=1,
                    )
                    check_btn = gr.Button("🔄 检查服务", size="sm")

                # 视频输入
                with gr.Group():
                    gr.Markdown("### 📹 视频输入")
                    video_input = gr.Textbox(
                        label="视频路径或 URL",
                        placeholder="输入本地路径或 http:// 开头的 URL",
                        lines=1,
                    )
                    load_btn = gr.Button("📂 加载视频")
                    video_info = gr.Textbox(
                        label="视频信息",
                        interactive=False,
                        lines=3,
                    )

                # 任务控制
                with gr.Group():
                    gr.Markdown("### 🎯 导航任务")
                    instruction_input = gr.Textbox(
                        label="导航指令",
                        placeholder="例如：向前走，找到门",
                        lines=2,
                    )
                    task_btn = gr.Button("🚀 发布任务", variant="primary")
                    task_status = gr.Textbox(
                        label="任务状态",
                        interactive=False,
                        lines=3,
                    )

                # 导航状态
                with gr.Group():
                    gr.Markdown("### 📊 导航状态")
                    nav_status = gr.Markdown("等待开始...")

        # 任务历史
        with gr.Accordion("📜 任务历史", open=False):
            history_text = gr.Textbox(
                label="",
                interactive=False,
                lines=5,
            )
            refresh_history_btn = gr.Button("🔄 刷新历史", size="sm")

        # 事件绑定
        def check_service():
            ok, msg = controller.check_health()
            return msg

        def load_video(video_path):
            ok, info, preview = controller.load_video(video_path)
            return info, preview

        def create_task(instruction):
            ok, msg = controller.create_task(instruction)
            history = controller.get_task_history()
            return msg, history

        def start_processing():
            for frame, status in controller.run_video(skip_frames=2):
                yield frame, status

        def stop_processing():
            controller.stop_video()
            return "已停止"

        def refresh_history():
            return controller.get_task_history()

        check_btn.click(check_service, outputs=status_text)
        load_btn.click(load_video, inputs=video_input, outputs=[video_info, video_output])
        task_btn.click(create_task, inputs=instruction_input, outputs=[task_status, history_text])
        start_btn.click(start_processing, outputs=[video_output, nav_status])
        stop_btn.click(stop_processing, outputs=nav_status)
        refresh_history_btn.click(refresh_history, outputs=history_text)

        # 启动时检查服务
        demo.load(check_service, outputs=status_text)

    return demo


def main():
    parser = argparse.ArgumentParser(description="VLN WebUI")
    parser.add_argument("--port", type=int, default=7860, help="WebUI 端口")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--api-url", default="http://localhost:20000", help="VLN API 地址")
    parser.add_argument("--share", action="store_true", help="创建公共链接")

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           VLN WebUI - 视觉语言导航可视化界面              ║
╠══════════════════════════════════════════════════════════╣
║  API 地址: {args.api_url:<44} ║
║  WebUI 地址: http://{args.host}:{args.port:<28} ║
╚══════════════════════════════════════════════════════════╝
    """)

    demo = create_ui(args.api_url)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
