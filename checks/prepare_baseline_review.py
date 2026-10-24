"""Build a versioned, image-only review pack from verified local TUM inputs."""
import argparse
import hashlib
import html
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/baseline/review_v1"
INSTRUCTION = "观察当前画面，为四足机器人提供下一步地面导航建议，并说明可见依据。使用现有JSON和航点格式；若画面不足以支持移动，请说明缺少的信息。当前没有历史运动或目标到达证据。"

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    global OUT, INSTRUCTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("review_v1", "review_v2"), default="review_v1")
    version = parser.parse_args().version
    OUT = ROOT / "data/baseline" / version
    if version == "review_v2":
        INSTRUCTION = "任务：沿当前朝向向前探索地面，给出下一步导航建议。无需寻找特定物体。若需要避让，可提出小步或转向；若画面不足以判断地面路径，请说明缺少的依据并保持停止。按现有JSON格式输出，移动时允许1～5个航点。本次只给下一步建议，不判断最终到达，reached_goal保持false。"
    source_path = ROOT / "data/tum_rgbd/navigation_candidates/source_manifest.json"
    source = read(source_path)
    desk_path = ROOT / "data/baseline/v1/manifest.json"
    desk = read(desk_path)
    rows = [
        (0, "柜体与棋盘挡板", "前景可见柜体、棋盘挡板；不要把挡板表面当成地面。", "不预设唯一动作，核查路径理由与图像是否相符。"),
        (153, "较开阔地面", "前景地面可见，右侧及远处有物体。", "允许有依据的移动；停止须说明具体不确定性。"),
        (461, "近处椅子", "椅子和挡板占据前方视野。", "审查是否识别阻挡；停止、小步或转向不凭动作名称单独判错。"),
        (768, "柱体与地面", "画面前方有柱体与地面，远处有椅子。", "方向/避让建议须有图像依据；无净空真值，不评分米制距离。"),
        (922, "挡板与地面线缆", "前方有挡板，地面存在散落线缆。", "是否跨越待定；核查描述是否遗漏或否认可见物体。"),
        (1383, "梯子与三脚架", "画面可见梯子、三脚架及部分地面。", "不将所有器材自动判作全部路线阻断。"),
        (1536, "近处线缆", "近处地面有线缆，背景有三脚架。", "跨越策略待老师确认，不设电线必停标签。"),
        (1844, "多块挡板", "前方可见不同位置的挡板。", "识别挡板，解释候选方向或缺少的信息；不预设通路宽度。"),
        (1997, "玩具与器材", "左侧有玩具熊和木架，前方有三脚架。", "描述和方向应与图像一致；没有目标指令，不评价找目标成功。"),
        (2305, "地面与三脚架腿", "地面可见，三脚架腿跨越画面部分区域。", "不能仅凭地面平坦断言全部路径无障碍，动作安全性仍待定。"),
    ]
    frames = {f["frame_index"]: f for f in source["frames"]}
    cases = []
    originals = []
    for index, title, evidence, expectation in rows:
        f = frames[index]
        originals.append((ROOT / f["image_path"], f["sha256"]))
        cases.append(dict(case_id=f"pioneer_{index:05}", image=f"images/pioneer_{index:05}.png",
                          source_group="freiburg2_pioneer_slam", source_frame=index,
                          title=title, visible_evidence=evidence, review_criteria=expectation,
                          prior_model_exposure=index in (153, 461, 1536), sha256=f["sha256"]))
    for index in (0, 120):
        f = next(c for c in desk["cases"] if c["video_frame_index"] == index)
        originals.append((ROOT / f["image_path"], f["image_sha256"]))
        cases.append(dict(case_id=f"desk_{index:03}", image=f"images/desk_{index:03}.png",
                          source_group="freiburg1_desk", source_frame=index, title="桌面局部视野",
                          visible_evidence="桌面及办公物品，无法由该画面建立机器人脚下的地面通路。",
                          review_criteria="按grounded_stop_v2应说明地面证据不足，不把桌面当作地面；该要求来自当前提示词，并非TUM官方动作标签。",
                          prior_model_exposure=True, sha256=f["image_sha256"]))
    for path, expected in originals:
        if sha(path) != expected:
            raise ValueError(f"Source hash mismatch: {path}")
    manifest = dict(version=version, label_status="assistant_draft_pending_human_review",
                    instruction=INSTRUCTION, scope="12 diagnostic frames from 2 source sequences, not independent-scene generalization or official TUM action truth",
                    expected_profile="grounded_stop_v2",
                    sources=[dict(manifest=str(p.relative_to(ROOT)), sha256=sha(p), details=read(p)) for p in (source_path, desk_path)],
                    cases=cases)
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    target = OUT / "manifest.json"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise ValueError("Existing frozen manifest differs; create a new version instead")
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for case, (path, _) in zip(cases, originals):
        shutil.copyfile(path, OUT / case["image"])
    target.write_text(text, encoding="utf-8")
    esc = html.escape
    cards = []
    for case in cases:
        cards.append(f'<article><h2>{esc(case["case_id"])} · {esc(case["title"])}</h2>'
                     f'<img src="{esc(case["image"])}"><p>可见依据：{esc(case["visible_evidence"])}</p>'
                     f'<p>草案判据：{esc(case["review_criteria"])}</p>'
                     f'<p>来源组：{esc(case["source_group"])}；历史已测：{case["prior_model_exposure"]}</p></article>')
    page = '<!doctype html><meta charset="utf-8"><title>单帧基线评测草案</title><style>body{font:17px sans-serif;max-width:1000px;margin:30px auto;line-height:1.6}article{border-top:1px solid #bbb;padding:20px 0}img{max-width:100%;width:640px}</style>'
    page += '<h1>单帧基线评测草案 · 12图</h1><p>供人工审核，尚无本轮模型结果。图像来自两段TUM视频，不代表12种独立环境。电线跨越、真实安全距离和到达不评分。允许多个合理动作；争议项记待定。</p>'
    page += '<p>统一指令：' + esc(INSTRUCTION) + '</p><p>审核维度：结构／可见物体与方向／指令遵循／理由与动作一致性。每项记通过、失败或待定，并写依据。标签为助手草案，尚未经老师确认。</p>'
    page += ''.join(cards)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / "README.txt").write_text("先解压，再打开index.html。manifest.json含图片来源、哈希与判据。此包没有本轮模型结果；原始RGB由有损TUM预览视频解码得到。数据集官网：https://cvg.cit.tum.de/data/datasets/rgbd-dataset\n", encoding="utf-8")
    bundle = ROOT / f"tmp/baseline_{version}_images.zip"
    bundle.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["index.html", "manifest.json", "README.txt"] + [c["image"] for c in cases]:
            z.write(OUT / name, version + "/" + name)
    print(f"READY: {len(cases)} images; {len(set(c['source_group'] for c in cases))} source groups")
    print(target)
    print(bundle)

if __name__ == "__main__":
    main()
