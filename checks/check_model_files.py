"""项目2：检查已下载模型的文件、配置与权重校验值。仅使用标准库。"""

import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "setup/model-manifest.json").read_text(encoding="utf-8"))
model_dir = root / "models/Qwen3-VL-2B-Instruct"

print("Model:", manifest["repo_id"])
print("Expected revision:", manifest["revision"])
print("Local directory:", model_dir)

errors = []
for name, expected_size in manifest["required_files"].items():
    path = model_dir / name
    if not path.is_file():
        errors.append(f"Missing file: {name}")
    elif path.stat().st_size != expected_size:
        errors.append(f"Unexpected file size: {name}")
    else:
        print("File size OK:", name)

if errors:
    for error in errors:
        print("FAIL:", error)
    raise SystemExit(1)

config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
if config.get("model_type") != manifest["model_type"]:
    raise SystemExit("FAIL: unexpected model_type in config.json")
if manifest["architecture"] not in config.get("architectures", []):
    raise SystemExit("FAIL: unexpected model architecture in config.json")
print("Model configuration: OK")

# 顺序读取权重计算 SHA-256，核对是否与官方仓库公布的内容相同。
# 每次只读 8 MiB，不会把整个约 4.26 GB 文件一次放进内存。
print("Checking weight SHA-256; reading the local weight file...", flush=True)
digest = hashlib.sha256()
with (model_dir / "model.safetensors").open("rb") as weight_file:
    while chunk := weight_file.read(8 * 1024 * 1024):
        digest.update(chunk)

if digest.hexdigest() != manifest["weight_sha256"]:
    raise SystemExit("FAIL: weight SHA-256 mismatch; share this output before continuing")

print("Weight SHA-256: OK")
print("PASS: model files are complete and match the expected weights.")
