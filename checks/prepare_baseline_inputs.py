"""项目2：从既有TUM视频固定5张测量输入，保存来源和SHA-256；不调用模型。"""

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2


INDICES = [0, 30, 60, 90, 120]  # 从0开始，按视频解码顺序选择。
INSTRUCTION = "观察当前桌面场景，寻找键盘。如果画面不能确认机器人可通行的地面路径，请保持停止并说明原因。"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, manifest):
    if [case["video_frame_index"] for case in manifest["cases"]] != INDICES:
        raise ValueError("Unexpected frame selection")
    for case in manifest["cases"]:
        path = root / case["image_path"]
        if sha256(path) != case["image_sha256"]:
            raise ValueError(f"Image checksum differs: {path}")
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (480, 640):
            raise ValueError(f"Image cannot be decoded as 640x480: {path}")
        print(f"Image OK: {case['case_id']}, 640x480, SHA256={case['image_sha256'][:12]}...", flush=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Path(__file__).resolve().parents[1]
    video = root / "data/tum_rgbd/desk_clip_150.avi"
    provenance = json.loads((root / "data/tum_rgbd/desk_clip_150_source.json").read_text(encoding="utf-8"))
    video_hash = sha256(video)
    if video_hash != provenance["video_sha256"]:
        raise ValueError("Source video checksum differs from its provenance record")
    if len(provenance["frames"]) != 150:
        raise ValueError("Source timestamp record is incomplete")
    model = json.loads((root / "setup/model-manifest.json").read_text(encoding="utf-8"))
    parent = root / "data/baseline"
    parent.mkdir(exist_ok=True)
    destination = parent / "v1"
    manifest_path = destination / "manifest.json"

    if destination.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_video_sha256"] != video_hash or manifest["instruction"] != INSTRUCTION:
            raise ValueError("Existing input set differs; share this output before changing it")
        if manifest["model"] != {"repo_id": model["repo_id"], "revision": model["revision"]}:
            raise ValueError("Existing model reference differs")
        verify(root, manifest)
        print("Existing input set verified; files were reused.")
    else:
        cases = []
        # 完整生成并验证后再放入v1，避免把中途失败的输出当作固定样本。
        with tempfile.TemporaryDirectory(prefix="prepare_inputs_", dir=parent) as temporary:
            staging = Path(temporary) / "v1"
            staging.mkdir()
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise ValueError("Source video could not be opened")
            try:
                index = 0
                while index <= INDICES[-1]:
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError(f"Video ended before frame {index}")
                    if index in INDICES:
                        if image.shape[:2] != (480, 640):
                            raise ValueError("Unexpected source frame size")
                        name = f"frame_{index:03d}.png"
                        path = staging / name
                        if not cv2.imwrite(str(path), image):
                            raise ValueError(f"Could not write {name}")
                        source_frame = provenance["frames"][index]
                        cases.append({
                            "case_id": f"tum_desk_{index:03d}",
                            "video_frame_index": index,
                            "image_path": (destination / name).relative_to(root).as_posix(),
                            "image_sha256": sha256(path),
                            "width": 640, "height": 480,
                            "source_rgb_timestamp": source_frame["timestamp"],
                            "source_archive_member": source_frame["archive_member"],
                            "source_rgb_png_sha256": source_frame["sha256"],
                        })
                    index += 1
            finally:
                capture.release()
            manifest = {
                "schema_version": 1,
                "purpose": "fixed_inputs_for_initial_baseline_not_measurement_results",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_video": video.relative_to(root).as_posix(),
                "source_video_sha256": video_hash,
                "source_archive_url": provenance["archive_url"],
                "image_encoding": "PNG decoded from an existing lossy MJPG AVI; not original RGB PNG bytes",
                "model": {"repo_id": model["repo_id"], "revision": model["revision"]},
                "instruction": INSTRUCTION,
                "planned_request_scope": "independent single-frame navigation requests; not streaming-video throughput",
                "existing_navigation_generation": {"max_tokens": 512, "temperature": 0.3},
                "quality_labels": "none; environment/action/waypoint consistency requires manual review",
                "cases": cases,
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            staging.rename(destination)
        verify(root, manifest)
    print("Manifest:", manifest_path)
    print("Manifest SHA256:", sha256(manifest_path))
    print("PASS: 5 fixed baseline inputs are ready; no model inference or timing was performed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"FAIL: {error}; share this output")
