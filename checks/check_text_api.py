"""项目2：向已运行的 LMDeploy 发一条短文本请求，检查实际生成链路。"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


root = Path(__file__).resolve().parents[1]
url = "http://127.0.0.1:23333/v1/chat/completions"
payload = {
    "model": "Qwen3-VL-2B-Instruct",
    "messages": [{"role": "user", "content": "请只回复：你好"}],
    "max_tokens": 16,
    "temperature": 0.0,
    "stream": False,
}

print("API:", url)
print("Prompt:", payload["messages"][0]["content"])
print("Waiting for a real model response...", flush=True)
started_at = datetime.now(timezone.utc)
start = time.perf_counter()
request = Request(
    url,
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urlopen(request, timeout=60) as response:
        status = response.status
        body = json.loads(response.read().decode("utf-8"))
except HTTPError as error:
    print("HTTP error:", error.code)
    print(error.read().decode("utf-8", errors="replace"))
    raise SystemExit(1)
except (URLError, TimeoutError) as error:
    raise SystemExit(f"FAIL: request did not complete: {error}") from error

elapsed = time.perf_counter() - start
log_dir = root / "logs"
log_dir.mkdir(exist_ok=True)
log_path = log_dir / f"text_smoke_{started_at.strftime('%Y%m%dT%H%M%S%fZ')}.json"
log_path.write_text(
    json.dumps(
        {
            "purpose": "functional_smoke_check_not_performance_baseline",
            "started_at_utc": started_at.isoformat(),
            "url": url,
            "request": payload,
            "http_status": status,
            "client_elapsed_s": elapsed,
            "response": body,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

choices = body.get("choices") or []
text = (choices[0].get("message") or {}).get("content", "") if choices else ""
print("HTTP:", status)
print("Model reply:", text)
print("Usage:", body.get("usage"))
print("Saved request and response:", log_path)
if not isinstance(text, str) or "你好" not in text:
    raise SystemExit("FAIL: the expected greeting was not found; share this output")

print("PASS: real text generation returned the expected greeting.")
