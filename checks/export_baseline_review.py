"""Export verified diagnostic results with images for human review; no automatic quality verdict."""
import argparse
import hashlib
import html
import json
import statistics
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    b = json.loads(args.log.read_text(encoding="utf-8"))
    m = json.loads(args.manifest.read_text(encoding="utf-8"))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if b["status"] != "complete" or b["selection_sha256"] != digest(args.manifest):
        raise ValueError("Incomplete run or mismatched manifest")
    cases = {c["case_id"]: c for c in m["cases"]}
    out = args.manifest.resolve().parent
    for c in cases.values():
        image = (out / c["image"]).resolve()
        if not image.is_relative_to(out) or digest(image) != c["sha256"]:
            raise ValueError("Image hash/path mismatch")
    rows = []
    for a in b["attempts"]:
        t = a["diagnostic"]["trace"]
        rows.append(dict(case_id=a["case_id"], repeat=a["repeat_index"],
                         client_elapsed_s=a["client_elapsed_s"], model_response=t["model_response"],
                         parsed=t["parsed"], derived_velocity=t.get("derived_velocity"),
                         timings_s=t.get("timings_s"), observation=a["observation"],
                         human_review=dict(reviewer="", grounding="pending", instruction="pending",
                                           consistency="pending", evidence="")))
    summary = dict(requests=len(rows), valid_json=sum(r["parsed"]["parse_status"] == "valid_json" for r in rows),
                   nonzero_raw_waypoints=sum(r["observation"]["raw_waypoint_state"] == "nonzero" for r in rows),
                   raw_stop_actions=sum(r["observation"]["raw_action"] == "停止" for r in rows),
                   client_median_s=statistics.median(r["client_elapsed_s"] for r in rows),
                   stats_delta=b["stats_delta"], quality="pending_manual_review")
    safe = dict(manifest_sha256=digest(args.manifest), measurement_sha256=digest(args.log),
                measurement_file=args.log.name, expected_profile=b["expected_profile"],
                prompt_sha256=b["expected_prompt_sha256"], generation=b["expected_generation"],
                summary=summary, attempts=rows)
    (out / "results.json").write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
    esc = html.escape
    page = '<!doctype html><meta charset="utf-8"><title>基线图像与模型结果</title><style>body{font:17px sans-serif;max-width:1100px;margin:30px auto;line-height:1.6}article{border-top:1px solid #aaa;padding:20px 0}img{max-width:100%;width:640px}pre{white-space:pre-wrap;background:#f3f4f5;padding:15px}</style>'
    page += '<h1>单帧诊断基线 · 待人工复核</h1><p>12张图来自两段视频；不是12个独立环境。速度为代码换算，无实机运动。数量、结构和耗时不等于导航质量。此轮无预热，耗时仅作请求记录。</p><p>指令：' + esc(m["instruction"]) + '</p>'
    page += '<pre>' + esc(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>'
    for r in rows:
        c = cases[r["case_id"]]
        page += f'<article><h2>{esc(r["case_id"])} · 第{r["repeat"]}次</h2><img src="{esc(c["image"])}"><p>可见依据：{esc(c["visible_evidence"])}</p><p>草案判据：{esc(c["review_criteria"])}</p>'
        page += '<h3>模型原始回答</h3><pre>' + esc(r["model_response"]["text"]) + '</pre>'
        page += '<details><summary>解析结果与换算速度</summary><pre>' + esc(json.dumps(dict(parsed=r["parsed"],velocity=r["derived_velocity"]),ensure_ascii=False,indent=2)) + '</pre></details><p>人工复核：待定。请指出具体句子、图中依据及建议判据。</p></article>'
    (out / "results.html").write_text(page, encoding="utf-8")
    bundle = ROOT / "tmp" / f"baseline_{m['version']}_results.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
        for name in ["results.html","results.json","index.html","manifest.json","README.txt"]+[c["image"] for c in cases.values()]:
            z.write(out / name, m["version"] + "/" + name)
    print(json.dumps(summary,ensure_ascii=False))
    print(bundle)

if __name__ == "__main__":
    main()
