"""项目2：向模型后端或老师的业务服务发送 TUM 图片，保存真实问答结果。"""

import argparse
import base64
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description="Send one TUM image to a running project2 chat API.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:23333",
        help="LMDeploy: http://127.0.0.1:23333; business API: http://127.0.0.1:20000",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    image_path = root / "data/tum_rgbd/sample_rgb.png"
    if not image_path.is_file():
        raise SystemExit("FAIL: sample image is missing; share this output")

    image_bytes = image_path.read_bytes()
    # base64 把图片字节编码成可放入 JSON 的文本；它不会改变图像内容。
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = "请用中文列出图中三个明显物体，并说明它们在画面中的大致位置。看不清的细节请说明不确定。"
    payload = {
        "model": "Qwen3-VL-2B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_base64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": False,
    }
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    via_business_api = args.base_url.rstrip("/") == "http://127.0.0.1:20000"
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = datetime.now(timezone.utc)
    record = {
        "purpose": "image_functional_check_not_performance_baseline",
        "started_at_utc": started_at.isoformat(),
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "url": url,
        "service_layer": "business_api" if via_business_api else "direct_or_other_api",
        "request": payload,
        "visual_quality_review": "pending_human_comparison_with_source_image",
    }
    print("API:", url)
    print("Image:", image_path)
    print("Prompt:", prompt)
    print("Sending image to the running model; please wait...", flush=True)
    start = time.perf_counter()
    body = None
    try:
        with urlopen(request, timeout=180) as response:
            record["http_status"] = response.status
            body = json.loads(response.read().decode("utf-8"))
            record["response"] = body
    except HTTPError as error:
        record["http_status"] = error.code
        record["error_body"] = error.read().decode("utf-8", errors="replace")
    except (URLError, TimeoutError, ValueError) as error:
        record["error"] = str(error)
    finally:
        record["client_elapsed_s"] = time.perf_counter() - start
        log_dir = root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"image_smoke_{started_at.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        log_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved request and response:", log_path)

    if body is None:
        print("HTTP:", record.get("http_status", "no response"))
        print("Error:", record.get("error_body", record.get("error", "unknown error")))
        raise SystemExit("FAIL: image request did not complete; share this output")

    choices = body.get("choices") or []
    text = (choices[0].get("message") or {}).get("content", "") if choices else ""
    print("HTTP:", record["http_status"])
    print("Model reply:\n" + str(text))
    print("Usage:", body.get("usage"))
    print("Finish reason:", choices[0].get("finish_reason") if choices else None)
    if via_business_api:
        print("Note: the original business adapter hardcodes finish_reason=stop; this does not verify answer completeness.")
    if not isinstance(text, str) or not text.strip():
        raise SystemExit("FAIL: no text answer was returned; share this output")

    print("REQUEST_OK: image API returned text. Compare the answer with the image.")


if __name__ == "__main__":
    main()
