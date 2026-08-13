# ==============================================================
# 生成 SC 微调用的 SX 提取版数据集（scx8k）
#
# 镜像源数据集的 train/rec/*.wav → 用 SX 提取器抽出目标人声波形，
# manifest / 元数据原样复制（rec_audio 相对路径不变），
# wake 嵌入缓存（spk_emb/）硬链接复用（同盘，几乎零成本）。
#
# 用法：
#   python build_scx_dataset.py                       # 使用默认路径（train8k→scx8k）
#   python build_scx_dataset.py --src <目录> --dst <目录> --ckpt <权重>
#   python build_scx_dataset.py --batch 32 --limit 100   # 调大批量 / 调试
#
# 默认路径与最终系统一致（20260810_0045_train8k → 20260813_0105_scx8k，
# SX = 阶段C 纯分离模型 step_6000）。
# ==============================================================

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import soundfile as sf

# 直接运行本脚本时把项目根加入 sys.path（python tools/build_scx_dataset.py）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.sc_data import read_wav, emb_path
from lib.sx_model import sx_load, SAMPLE_RATE

# 默认路径（与最终系统一致）
DEFAULT_SRC = r"E:/Desktop/Jigbas/Temp/Datasets/20260810_0045_train8k"
DEFAULT_DST = r"E:/Desktop/Jigbas/Temp/Datasets/20260813_0105_scx8k"
DEFAULT_CKPT = (r"E:/Desktop/Jigbas/Temp/Datasets/20260811_1914_sxtrain/"
                r"checkpoints/sx_20260812_235228/step_6000.pt")
BATCH = 16


def build_scx(args, log=print):
    """源数据集 train 波形 → SX 提取 → 目标数据集，返回目标目录"""
    src, dst = args.src, args.dst
    if not os.path.isdir(src):
        raise FileNotFoundError(f"源数据集目录不存在: {src}")
    rows = [json.loads(l) for l in open(
        os.path.join(src, "train_manifest.jsonl"), encoding="utf-8")]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"{src}/train_manifest.jsonl 为空")
    log(f"[提取] 样本 {len(rows)} 条，批量 {args.batch}")

    os.makedirs(os.path.join(dst, "train", "rec"), exist_ok=True)
    m = sx_load(args.ckpt, device=args.device, log=log)
    done = 0
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        wavs, lens = [], []
        for r in chunk:
            w = read_wav(os.path.join(src, r["rec_audio"]))
            wavs.append(w)
            lens.append(len(w))
        mx = max(lens)
        mix = torch.zeros(len(chunk), mx)
        for j, w in enumerate(wavs):
            mix[j, :len(w)] = torch.from_numpy(w)
        emb = torch.from_numpy(np.stack([
            np.load(emb_path(src, "train", r["id"])).astype(np.float32)
            for r in chunk]))
        with torch.no_grad():
            est = m.separate(mix.to(args.device), emb.to(args.device)) \
                   .cpu().numpy()
        for j, r in enumerate(chunk):
            sf.write(os.path.join(dst, r["rec_audio"]), est[j, :lens[j]],
                     SAMPLE_RATE, subtype="PCM_16")
        done += len(chunk)
        if done % 800 < args.batch or done >= len(rows):
            log(f"[提取] {done}/{len(rows)}")

    # manifest 原样复制（rec_audio 相对路径不变），元数据复制，嵌入缓存硬链接
    shutil.copy(os.path.join(src, "train_manifest.jsonl"),
                os.path.join(dst, "train_manifest.jsonl"))
    if os.path.isfile(os.path.join(src, "metadata.json")):
        shutil.copy(os.path.join(src, "metadata.json"),
                    os.path.join(dst, "metadata.json"))
    os.makedirs(os.path.join(dst, "spk_emb"), exist_ok=True)
    src_emb = os.path.join(src, "spk_emb", "train")
    dst_emb = os.path.join(dst, "spk_emb", "train")
    os.makedirs(dst_emb, exist_ok=True)
    if os.path.isdir(src_emb):
        for f in os.listdir(src_emb):
            s, d = os.path.join(src_emb, f), os.path.join(dst_emb, f)
            if not os.path.exists(d):
                os.link(s, d)
    log(f"[完成] {dst}（{done} 条提取波形 + manifest + 嵌入缓存）")
    return dst


def build_parser():
    ap = argparse.ArgumentParser(description="用 SX 提取器生成 SC 微调数据集（scx8k）")
    ap.add_argument("--src", default=DEFAULT_SRC, help="源数据集目录（train8k）")
    ap.add_argument("--dst", default=DEFAULT_DST, help="目标数据集目录（scx8k）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="SX 提取器权重路径")
    ap.add_argument("--batch", type=int, default=BATCH, help="提取批大小")
    ap.add_argument("--device", default=None, help="cpu / cuda:0（默认自动）")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（调试用）")
    return ap


def main():
    args = build_parser().parse_args()
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    build_scx(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
