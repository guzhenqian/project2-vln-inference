"""
航点解析器

解析 VLM 输出为航点序列
"""

import json
import logging
import math
import re
from typing import Dict, List, Optional, Tuple

from . import Waypoint

logger = logging.getLogger(__name__)


class WaypointParser:
    """航点解析器"""

    def __init__(
        self,
        max_dx: float = 1.0,
        max_dy: float = 0.5,
        max_dtheta: float = 1.0,
    ) -> None:
        """
        初始化解析器

        Args:
            max_dx: 最大前进距离限制
            max_dy: 最大横向距离限制
            max_dtheta: 最大旋转角度限制
        """
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.max_dtheta = max_dtheta

    def parse(self, vlm_output: str) -> Dict:
        """
        解析 VLM 输出

        Args:
            vlm_output: VLM 生成的文本

        Returns:
            解析结果字典，包含:
            - waypoints: 航点列表
            - environment: 环境描述
            - action: 动作描述
            - reasoning: 推理过程
            - progress: 进度
            - reached_goal: 是否到达目标
            - raw_output: 原始输出
            - parse_status/parse_error: 结构校验结果（供诊断区分停止与解析失败）
        """
        result = {
            "waypoints": [],
            "environment": "",
            "action": "",
            "reasoning": "",
            "progress": 0.0,
            "reached_goal": False,
            "raw_output": vlm_output,
        }

        # 只接受结构有效的JSON；不从自然语言或否定句里猜测运动/到达。
        json_data = self._extract_json(vlm_output)
        if isinstance(json_data, dict):
            try:
                result.update(self._parse_json_response(json_data))
                result.update(parse_status="valid_json", parse_error="")
            except (ValueError, TypeError, OverflowError) as error:
                result.update(self._parse_text_response(vlm_output))
                result["parse_error"] = str(error)
        else:
            result.update(self._parse_text_response(vlm_output))

        # 验证和限制航点
        result["waypoints"] = self._validate_waypoints(result["waypoints"])

        return result

    def _extract_json(self, text: str) -> Optional[dict]:
        """从文本中提取 JSON"""
        # 尝试直接解析
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 块
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"\{[\s\S]*\}",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if "```" in pattern else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue

        return None

    def _parse_json_response(self, data: dict) -> Dict:
        """解析 JSON 格式的响应"""
        if type(data.get("reached_goal")) is not bool:
            raise ValueError("reached_goal must be a JSON boolean")
        if not isinstance(data.get("waypoints"), list):
            raise ValueError("waypoints must be a JSON array")
        progress = self._finite_number(data.get("progress", 0.0), "progress")
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be in [0, 1]")
        result = {
            "environment": str(data.get("environment", "")),
            "action": str(data.get("action", "")),
            "reasoning": str(data.get("reasoning", "")),
            "progress": progress,
            "reached_goal": data["reached_goal"],
            "waypoints": [],
        }

        # 解析航点
        for wp in data["waypoints"]:
            if not isinstance(wp, dict) or not all(key in wp for key in ("dx", "dy", "dtheta")):
                raise ValueError("Each waypoint must contain dx, dy and dtheta")
            result["waypoints"].append(Waypoint(
                dx=self._finite_number(wp["dx"], "dx"),
                dy=self._finite_number(wp["dy"], "dy"),
                dtheta=self._finite_number(wp["dtheta"], "dtheta"),
                confidence=self._finite_number(wp.get("confidence", 1.0), "confidence"),
            ))
        if result["reached_goal"] and any(
            wp.dx != 0 or wp.dy != 0 or wp.dtheta != 0 for wp in result["waypoints"]
        ):
            raise ValueError("reached_goal=true conflicts with nonzero waypoints")

        return result

    @staticmethod
    def _finite_number(value, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite JSON number")
        return float(value)

    def _parse_text_response(self, text: str) -> Dict:
        """无有效结构化结果时保持停止；原始文本由parse的raw_output完整保留。"""
        return {
            "environment": "",
            "action": "停止",
            "reasoning": "未获得有效的结构化导航结果，保持停止；原始回答保留在诊断中。",
            "progress": 0.0,
            "reached_goal": False,
            "waypoints": [],
            "parse_status": "fallback_stop",
            "parse_error": "Expected a valid navigation JSON object",
        }

    def _validate_waypoints(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """验证和限制航点"""
        validated = []
        for wp in waypoints[:5]:  # 最多5个航点
            validated_wp = Waypoint(
                dx=max(-self.max_dx, min(self.max_dx, wp.dx)),
                dy=max(-self.max_dy, min(self.max_dy, wp.dy)),
                dtheta=max(-self.max_dtheta, min(self.max_dtheta, wp.dtheta)),
                confidence=max(0.0, min(1.0, wp.confidence)),
            )
            validated.append(validated_wp)
        return validated

    def waypoints_to_velocity(
        self,
        waypoints: List[Waypoint],
        dt: float = 0.2,
    ) -> Tuple[float, float]:
        """
        将航点转换为速度命令（用于直接控制）

        Args:
            waypoints: 航点列表
            dt: 时间步长

        Returns:
            (linear_velocity, angular_velocity)
        """
        if not waypoints:
            return 0.0, 0.0

        # 使用第一个航点计算速度
        wp = waypoints[0]
        v = wp.dx / dt  # 线速度 m/s
        w = wp.dtheta / dt  # 角速度 rad/s

        # 限制速度范围
        v = max(-1.0, min(1.5, v))
        w = max(-1.0, min(1.0, w))

        return v, w
