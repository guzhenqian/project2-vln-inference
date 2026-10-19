"""项目2：获取并查看老师推荐的 R2R 公开标注，不安装模拟器。"""

import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


root = Path(__file__).resolve().parents[1]
data_dir = root / "data/r2r/annotations"
data_dir.mkdir(parents=True, exist_ok=True)
path = data_dir / "R2R_val_seen.json"

# 链接来自作者仓库 tasks/R2R/data/download.sh，已读取核对过内容。
source = "https://www.dropbox.com/s/8ye4gqce7v8yzdm/R2R_val_seen.json?dl=1"
expected_sha256 = "755633db9d5e64d5ca71d329c5046bb722781a09f0d1d501d7e8d094c83652ec"
if path.is_file():
    raw = path.read_bytes()
    print("Using local annotations:", path)
else:
    print("Downloading public R2R annotations (about 293 KB)...", flush=True)
    try:
        with urlopen(source, timeout=30) as response:
            raw = response.read(5_000_001)
    except (URLError, TimeoutError) as error:
        raise SystemExit(f"FAIL: annotation download did not complete: {error}") from error
    if len(raw) > 5_000_000:
        raise SystemExit("FAIL: response exceeds the expected annotation file size")

if hashlib.sha256(raw).hexdigest() != expected_sha256:
    raise SystemExit("FAIL: annotation content changed or is incomplete; share this output")

episodes = json.loads(raw)
required_fields = {"distance", "scan", "path_id", "path", "heading", "instructions"}
if not episodes or any(not required_fields.issubset(item) for item in episodes):
    raise SystemExit("FAIL: unexpected R2R annotation structure")

if not path.is_file():
    path.write_bytes(raw)
provenance = {
    "dataset": "Room-to-Room (R2R)",
    "split": "val_seen",
    "source": source,
    "source_download_script": "https://github.com/peteanderson80/Matterport3DSimulator/blob/master/tasks/R2R/data/download.sh",
    "sha256": expected_sha256,
    "bytes": len(raw),
    "note": "Annotations only. Matterport3D scene images are not included.",
}
(data_dir / "source.json").write_text(
    json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
)

first = episodes[0]
print("Split: val_seen")
print("Navigation routes:", len(episodes))
print("Navigation instructions:", sum(len(item["instructions"]) for item in episodes))
print("Scene IDs:", len({item["scan"] for item in episodes}))
print("First route ID:", first["path_id"])
print("First scene ID:", first["scan"])
print("Viewpoints on first route:", len(first["path"]))
print("First instruction excerpt:", " ".join(first["instructions"][0].split()[:12]) + " ...")
print("Scene images: not included; access must be requested separately.")
print("PASS: R2R annotations are readable. No image inference was performed.")
