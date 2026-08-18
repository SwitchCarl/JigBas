# ==============================================================
# Release 打包脚本 — 生成可交付的 submit/ 目录
#
# 交付物 = 运行时（lib/ + app/ + JigBas.py + requirements.txt）+ 说明文档。
# 训练 / 造数据工具（tools/）不默认打包——它们保留在主分支供复现，但
# 实际交付不需要（比赛只收官方提交 JSON）。
# 权重也不默认打包：体积大且路径固定在仓库外（Temp/Datasets），
# 默认只在 README 写明位置；需要随包带权重时用 --include-weights。
#
# 用法：
#   python package_release.py                    # 生成 <项目根>/submit/
#   python package_release.py --include-weights  # 连 Models/ 一起打包
#   python package_release.py --out D:/release --tag v1.0-final
#
# 收尾（手动）：git tag v1.0-final；可选把 submit/ 压缩后挂到 Release。
# ==============================================================

import argparse
import os
import shutil
import sys
import time

# 直接运行本脚本时把项目根加入 sys.path（python tools/package_release.py）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.paths import REPO_ROOT, MODELS_DIR, DATASETS_ROOT

# 参与交付的包 / 文件（tools/ 训练工具不交付）
PACKAGE_DIRS = ["lib", "app"]
PACKAGE_FILES = ["JigBas.py", "requirements.txt"]

# 最终系统权重（README 注明位置；实际在仓库外 DATASETS_ROOT 下，相对推导可随迁移）
FINAL_SX_CKPT = os.path.join(
    DATASETS_ROOT, "20260811_1914_sxtrain", "checkpoints",
    "sx_20260812_235228", "step_6000.pt")
FINAL_SC_CKPT = os.path.join(
    DATASETS_ROOT, "20260813_0105_scx8k", "checkpoints",
    "sc_20260813_cont4000", "step_4000.pt")


def _copy_tree(src, dst):
    """复制目录（跳过 __pycache__ / *.pyc），目标已存在则覆盖"""
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _write_readme(out, weights_note, tag):
    ver = tag or time.strftime("%Y%m%d")
    readme = f"""# JigBas — 抗干扰语音指令识别系统（交付包 {ver}）

目标说话人识别 + 拒识 + 指令转写系统。
最终系统综合分 **0.7813**（CER 37.48% / RR 93.75%）。
管线：rec 音频 → SX 提取目标人声 → wespeaker 声纹门控（灰区滑窗精判，
sim_sx ≥ 0.30 接受）→ SC-scx 在提取音频上转写。

## 安装

    pip install -r requirements.txt

推荐 Python ≥ 3.10 + CUDA（torch 2.x）。声纹模型（wespeaker chinese）首次
加载自动下载；ASR 权重见下方「权重位置」。

## 快速开始

    # 环境自检
    python JigBas.py verify

    # 单条演示（sx = 最终系统）
    python JigBas.py demo --mode sx --wake wake.wav --rec rec.wav

    # 官方提交格式推理（最终系统）—— 在赛方测试集上验证模型
    python JigBas.py submit --test <测试集目录或json> --out result.json

    # 自有数据集评估（--final 一键最终系统，输出 5 配置消融表）
    python JigBas.py eval --dataset latest --final

    # 无参数进入交互菜单
    python JigBas.py

## 权重位置（未随包附带，需自行放置）

| 模型 | 用途 | 默认路径 |
|---|---|---|
| SX 提取器 | 抽出目标人声 | `{FINAL_SX_CKPT}` |
| SC-scx ASR | 提取音频上转写 | `{FINAL_SC_CKPT}` |
| Paraformer | 基线 ASR / SC 底座 | `Models/speech_paraformer-.../`（本包） |

权重默认在仓库外 `{DATASETS_ROOT}`。若测试机路径不同，
用 `--sx-checkpoint` / `--sc-checkpoint` 显式指定（demo / eval --final /
submit 均支持）。

## 目录结构

    JigBas.py          总入口（子命令 + 交互菜单）
    app/               运行时：demo（单条）/ ui（界面）/ evaluate（评估）/ submit（官方提交）
    lib/               纯库：models / datasets / sc_model / sx_model / sc_data / paths
    requirements.txt   依赖清单

训练 / 数据集构建工具（tools/）不在交付包内，保留在主分支供复现。
"""
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def _write_tag(out, tag):
    ver = tag or time.strftime("%Y%m%d")
    with open(os.path.join(out, "VERSION.txt"), "w", encoding="utf-8") as f:
        f.write(ver + "\n")


def main():
    ap = argparse.ArgumentParser(description="生成可交付的 submit/ 目录")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "submit"),
                    help="输出目录（默认 <项目根>/submit/）")
    ap.add_argument("--include-weights", action="store_true",
                    help="同时复制 Models/（权重体积大，默认不复制）")
    ap.add_argument("--tag", default=None,
                    help="版本号写入 README 与 VERSION.txt（如 v1.0-final）")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    for d in PACKAGE_DIRS:
        _copy_tree(os.path.join(REPO_ROOT, d), os.path.join(out, d))
    for f in PACKAGE_FILES:
        src = os.path.join(REPO_ROOT, f)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(out, f))

    weights_note = "已包含 Models/" if args.include_weights else "未包含（见 README 权重说明）"
    if args.include_weights and os.path.isdir(MODELS_DIR):
        _copy_tree(MODELS_DIR, os.path.join(out, "Models"))

    _write_readme(out, weights_note, args.tag)
    _write_tag(out, args.tag)

    print(f"[打包] 输出目录: {out}")
    for d in PACKAGE_DIRS:
        print(f"  - {d}/")
    for f in PACKAGE_FILES:
        print(f"  - {f}")
    print(f"  - README.md  VERSION.txt")
    print(f"  权重: {weights_note}")
    print("[打包] 完成。可 git tag v1.0-final 标记版本，再把 submit/ 压缩上传 Release。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
