"""项目2：从 TUM 官方 fr1/desk 序列选取最早 150 张 RGB 帧，生成加载测试视频。"""

import argparse
import hashlib
import heapq
import json
import math
import statistics
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

import cv2
import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_video(path, expected_count):
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError("Cannot reopen the generated video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (480, 640):
                raise ValueError("Unexpected decoded frame size")
            count += 1
    finally:
        capture.release()
    if count != expected_count or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Video decode check failed: frames={count}, fps={fps}")
    return fps


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data/tum_rgbd"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "desk_clip_150.avi"
    source_path = out_dir / "desk_clip_150_source.json"
    target_count = 150
    if video_path.exists() or source_path.exists():
        if not video_path.is_file() or not source_path.is_file():
            raise ValueError("Video/provenance pair is incomplete; share this output")
        record = json.loads(source_path.read_text(encoding="utf-8"))
        if digest(video_path) != record["video_sha256"]:
            raise ValueError("Existing video checksum differs; share this output")
        fps = verify_video(video_path, target_count)
        print("Existing video verified:", video_path)
        print(f"PASS: {target_count} decoded frames, 640x480, FPS={fps:.3f}")
        return

    sequence = "rgbd_dataset_freiburg1_desk"
    url = f"https://cvg.cit.tum.de/rgbd/dataset/freiburg1/{sequence}.tgz"
    print("Source:", url, flush=True)
    print("Reading the full archive stream to select the earliest 150 RGB timestamps.", flush=True)
    print("Download: hundreds of MB; the archive itself is not saved.", flush=True)

    class ProgressReader:
        def __init__(self, response):
            self.response = response
            self.count = 0
            self.next_report = 16 * 1024 * 1024

        def read(self, size=-1):
            data = self.response.read(size)
            self.count += len(data)
            if self.count > 1024**3:
                raise ValueError("Archive exceeded the expected download size")
            if self.count >= self.next_report:
                print(f"Downloaded: {self.count / 1024**2:.0f} MiB", flush=True)
                self.next_report += 16 * 1024 * 1024
            return data

    # 包内顺序不代表时间顺序；最大堆只保留时间最早的 150 张图片。
    selected = []
    rgb_count = 0
    with urlopen(url, timeout=45) as response:
        reader = ProgressReader(response)
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or len(parts) != 3 or parts[:2] != (sequence, "rgb"):
                    continue
                if not parts[2].endswith(".png"):
                    continue
                timestamp = float(PurePosixPath(parts[2]).stem)
                if not math.isfinite(timestamp):
                    raise ValueError("Invalid RGB timestamp")
                rgb_count += 1
                if len(selected) == target_count and timestamp >= -selected[0][0]:
                    continue
                if member.size > 10 * 1024 * 1024:
                    raise ValueError("Unexpectedly large RGB frame")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Cannot read RGB member")
                raw = stream.read()
                entry = (-timestamp, member.name, raw)
                if len(selected) < target_count:
                    heapq.heappush(selected, entry)
                else:
                    heapq.heapreplace(selected, entry)
    if len(selected) != target_count:
        raise ValueError(f"Expected 150 RGB frames; found {len(selected)}")
    selected.sort(key=lambda item: -item[0])
    timestamps = [-item[0] for item in selected]
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    if min(gaps) <= 0 or max(gaps) > 0.15:
        raise ValueError("Selected frames have duplicate timestamps or large gaps; share this output")
    fps = 1.0 / statistics.median(gaps)

    # MJPG 是有损编码；原始帧名、哈希和时间戳另存，以免混淆数据来源。
    frame_records = []
    with tempfile.TemporaryDirectory(prefix="prepare_video_", dir=out_dir) as temp_dir:
        temp_video = Path(temp_dir) / video_path.name
        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
        if not writer.isOpened():
            raise ValueError("MJPG VideoWriter could not be opened")
        try:
            for negative_timestamp, name, raw in selected:
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if frame is None or frame.shape[:2] != (480, 640):
                    raise ValueError(f"Unexpected RGB image: {name}")
                writer.write(frame)
                frame_records.append({"archive_member": name, "timestamp": -negative_timestamp,
                                      "sha256": hashlib.sha256(raw).hexdigest()})
        finally:
            writer.release()
        decoded_fps = verify_video(temp_video, target_count)
        record = {
            "purpose": "video_loading_functional_check_not_navigation_benchmark",
            "archive_url": url,
            "official_page": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download",
            "sequence": sequence,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_rgb_count": rgb_count,
            "downloaded_bytes": reader.count,
            "selection": "earliest 150 RGB timestamps in the archive",
            "encoding": "lossy MJPG in AVI; constant FPS from median source timestamp interval",
            "width": 640, "height": 480, "frame_count": target_count,
            "video_fps": decoded_fps, "source_span_s": timestamps[-1] - timestamps[0],
            "max_source_gap_s": max(gaps),
            "video_sha256": digest(temp_video), "frames": frame_records,
        }
        temp_source = Path(temp_dir) / source_path.name
        temp_source.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_video.replace(video_path)
        temp_source.replace(source_path)
    print("Saved video:", video_path)
    print("Saved source record:", source_path)
    print(f"PASS: {target_count} decoded frames, 640x480, FPS={decoded_fps:.3f}")
    print("Paste this Linux path into the WebUI video input:")
    print(video_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"FAIL: {error}; share this output")
