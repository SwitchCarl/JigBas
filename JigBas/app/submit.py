# ==============================================================
# 官方提交格式推理 — 在赛方指定数据集上验证模型（最终系统）
#
# 比赛提交要求（Context.txt）：单个 JSON
#   {"result": {"results": [{"id":"识别音频名字","content":"推理文本",
#                             "label":"识别标签","cer":"xx"}, ...],
#                "final_cer":"xx", "duration":"t"}}
# 字段为中文键（唤醒音频名字/唤醒文本名字/识别音频名字/识别文本名字），
# id = 测试音频名字；拒识样本 label 为空；duration = batch=1 推理全部
# 音频的总秒数（对应效率指标 10%）。
#
# 本脚本在测试集上逐条跑最终系统（SX 提取 + sim_sx 门控 + SC-scx）：
#   sim_sx >= 阈值 → 接受，content = 转写文本，label = 参考标签（若有）
#   sim_sx <  阈值 → 拒识，content / label 均为空
# 逐条计算 cer（正样本），全部正样本聚合为 final_cer（编辑距离 / 总字数，
# 与 evaluate.compute_metrics 同口径），duration = 逐条推理耗时总和。
#
# 用法：
#   python submit.py --test <目录或json> --out result.json
#   python submit.py --test D:/testset/ --out result.json --limit 20
#
# 测试集支持三种形态：
#   目录   → 自动探测 test.json / test_manifest.jsonl / dev_manifest.jsonl
#   单 JSON → list 或 {"results":[...]} 或 {"result":{"results":[...]}}
#   .jsonl → 逐行读取
# 条目键名兼容中文（唤醒音频名字…）与英文（wake_audio…），两者等价。
#
# 说明：若测试集提供参考文本（识别文本名字），label/cer/final_cer 自动
# 计算；不提供则 label 为空、cer/final_cer 留空（由赛方判定）。
# ==============================================================

import argparse
import json
import os
import sys
import time

# 直接运行本脚本时把项目根加入 sys.path（python app/submit.py）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.paths import REPO_ROOT
from app.demo import (load_final_engine, recognize_sx, SX_DEFAULT_THRESHOLD,
                      SC_GRAY_ZONE)
from app.evaluate import normalize_text, edit_distance

_PROGRESS_ENABLED = False

# 官方测试集中文键 → 内部英文键（兼容两套命名）
KEY_MAP = {
    "唤醒音频名字": "wake_audio",
    "唤醒文本名字": "wake_text",
    "识别音频名字": "rec_audio",
    "识别文本名字": "rec_text",
}

# 目录模式下自动探测的测试集文件名
PROBE_FILES = ("test.json", "test_manifest.jsonl", "dev_manifest.jsonl")


def emit_progress(**kw):
    """--progress 开启时输出结构化进度行（供 UI / 脚本解析）"""
    if _PROGRESS_ENABLED:
        print(f"[PROGRESS] {json.dumps(kw, ensure_ascii=False)}", flush=True)


# ---------------------------------------------------------------
# 测试集加载
# ---------------------------------------------------------------
def load_entries(test):
    """
    加载测试集为内部条目列表 [{id, wake_audio, wake_text, rec_audio, rec_text}]，
    返回 (entries, base_dir)。
    base_dir = 音频相对路径的解析根（目录模式 = 目录本身；文件模式 = 文件所在目录）。
    """
    if os.path.isdir(test):
        path = next((os.path.join(test, f) for f in PROBE_FILES
                     if os.path.isfile(os.path.join(test, f))), None)
        if path is None:
            raise FileNotFoundError(
                f"目录 {test} 下未找到测试集文件（自动探测 {', '.join(PROBE_FILES)}）")
        base_dir = os.path.abspath(test)
    else:
        path = test
        if not os.path.isfile(path):
            raise FileNotFoundError(f"测试集不存在: {path}")
        base_dir = os.path.dirname(os.path.abspath(path))

    def _norm(item):
        row = {KEY_MAP.get(k, k): v for k, v in item.items()}
        if not row.get("rec_audio"):
            row["rec_audio"] = row.get("id", "")
        if not row.get("id"):
            row["id"] = row["rec_audio"]
        return row

    rows = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(_norm(json.loads(line)))
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            inner = data.get("result", data)
            if isinstance(inner, dict):
                data = inner.get("results", [])
            else:
                data = inner
        if not isinstance(data, list):
            raise ValueError(
                f"无法解析测试集 {path}：需要 list 或 {{'results':[...]}} 结构")
        rows = [_norm(x) for x in data]
    return rows, base_dir


# ---------------------------------------------------------------
# 推理主流程
# ---------------------------------------------------------------
def run_submit(args, log=print):
    """逐条跑最终系统并写官方提交 JSON，返回输出路径"""
    entries, base_dir = load_entries(args.test)
    if args.limit > 0:
        entries = entries[: args.limit]
    n = len(entries)
    if n == 0:
        raise ValueError("测试集为空")

    import torch
    from lib.models import ModelHub

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"[提交] 测试集 {args.test}，样本 {n} 条，设备 {device}")

    hub = ModelHub(device=device)
    hub._load_wespeaker(log)
    sx, sc_model, sc_kwargs = load_final_engine(
        device=device, sx_checkpoint=args.sx_checkpoint,
        sc_checkpoint=args.sc_checkpoint, log=log)

    def _abs(base, p):
        if not p or os.path.isabs(p):
            return p
        return os.path.join(base, p)

    results, err_chars, ref_chars = [], 0, 0
    duration = 0.0
    t_start = time.time()
    for i, row in enumerate(entries, 1):
        out = recognize_sx(
            hub, sx, sc_model, sc_kwargs,
            _abs(base_dir, row.get("wake_audio")),
            _abs(base_dir, row.get("rec_audio")),
            args.threshold, gray=SC_GRAY_ZONE,
            log=lambda m: None)  # 逐条日志关闭，统一由这里汇总
        duration += out["elapsed"]
        accepted = bool(out["accepted"]) and not out["error"]
        content = out["text"] if accepted else ""
        ref = row.get("rec_text") or ""
        label = ref if (ref and accepted) else ""
        per_cer = ""
        if ref:
            ref_n = normalize_text(ref)
            d, L = edit_distance(ref_n, normalize_text(content)), len(ref_n)
            per_cer = f"{d / L:.4f}" if L else ""
            err_chars += d
            ref_chars += L
        results.append({
            "id": str(row["id"]),
            "content": content or "",
            "label": label or "",
            "cer": per_cer,
        })
        if i % 10 == 0 or i == n:
            emit_progress(phase="submit", done=i, total=n,
                          elapsed=round(time.time() - t_start, 1))
            log(f"[提交] {i}/{n}")

    final_cer = f"{err_chars / ref_chars:.4f}" if ref_chars else ""
    payload = {"result": {
        "results": results,
        "final_cer": final_cer,
        "duration": f"{duration:.2f}",
    }}
    out_path = args.out or os.path.join(REPO_ROOT, "result.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"[提交] 完成：final_cer={final_cer}，duration={duration:.2f}s，"
        f"输出 {out_path}")
    emit_progress(phase="done", done=1, total=1,
                  final_cer=final_cer, duration=round(duration, 2),
                  output=os.path.basename(out_path))
    return out_path


def build_parser():
    ap = argparse.ArgumentParser(description="官方提交格式推理（最终系统）")
    ap.add_argument("--test", required=True,
                    help="测试集目录或 JSON/JSONL 文件（自动探测/解析）")
    ap.add_argument("--out", default=None,
                    help="输出 JSON 路径（默认 <项目根>/result.json）")
    ap.add_argument("--threshold", type=float, default=SX_DEFAULT_THRESHOLD,
                    help=f"sim_sx 接受阈值（默认 {SX_DEFAULT_THRESHOLD}）")
    ap.add_argument("--sx-checkpoint", default=None,
                    help="SX 提取器权重（默认最终系统 checkpoint）")
    ap.add_argument("--sc-checkpoint", default=None,
                    help="SC-scx ASR 权重（默认最终系统 checkpoint）")
    ap.add_argument("--device", default=None, help="cpu / cuda:0（默认自动）")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（调试用）")
    ap.add_argument("--progress", action="store_true",
                    help="输出 [PROGRESS] 结构化进度行")
    return ap


def main():
    global _PROGRESS_ENABLED
    args = build_parser().parse_args()
    _PROGRESS_ENABLED = args.progress
    run_submit(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
