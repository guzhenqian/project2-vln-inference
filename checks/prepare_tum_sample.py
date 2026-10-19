"""项目2：从老师提到的 TUM RGB-D 官方序列中读取一张彩色帧。"""

import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

from PIL import Image


root = Path(__file__).resolve().parents[1]
out_dir = root / "data/tum_rgbd"
out_dir.mkdir(parents=True, exist_ok=True)
image_path = out_dir / "sample_rgb.png"
source_path = out_dir / "sample_source.json"
url = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
sequence = "rgbd_dataset_freiburg1_desk"

if image_path.is_file() or source_path.is_file():
    if not image_path.is_file() or not source_path.is_file():
        raise SystemExit("FAIL: sample or provenance is missing; share this output")
    record = json.loads(source_path.read_text(encoding="utf-8"))
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit("FAIL: existing sample checksum differs; share this output")
    print("Existing sample verified:", image_path)
    raise SystemExit(0)

print("Source: TUM RGB-D official fr1/desk sequence")
print("Reading the archive stream until the first RGB frame; please wait...", flush=True)
with urlopen(url, timeout=30) as response:
    # r|gz 按顺序读取压缩包，找到彩色帧后关闭连接，不保存完整数据集。
    with tarfile.open(fileobj=response, mode="r|gz") as archive:
        for index, member in enumerate(archive):
            if index < 5:
                print("Archive entry:", member.name, flush=True)
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) != 3:
                continue
            if parts[0] != sequence or parts[1] != "rgb" or not parts[2].endswith(".png"):
                continue
            if member.size > 10 * 1024 * 1024:
                raise SystemExit("FAIL: RGB frame is unexpectedly large")
            # 只读取所选文件内容，保存到指定文件名，不按压缩包路径解压。
            stream = archive.extractfile(member)
            if stream is None:
                raise SystemExit("FAIL: could not read RGB frame")
            raw = stream.read()
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                mode = image.mode
                image.verify()
            image_path.write_bytes(raw)
            record = {
                "dataset": "TUM RGB-D SLAM Dataset and Benchmark",
                "sequence": sequence,
                "official_page": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download",
                "archive_url": url,
                "archive_member": member.name,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "width": width,
                "height": height,
                "mode": mode,
                "purpose": "single_image_functional_check",
            }
            source_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Frame:", member.name)
            print("Image size:", width, "x", height)
            print("Saved image:", image_path)
            print("Saved source record:", source_path)
            print("PASS: one TUM RGB frame is ready.")
            break
        else:
            raise SystemExit("FAIL: no expected RGB frame was found")
