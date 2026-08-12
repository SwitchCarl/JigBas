# ==============================================================
# SX-Extractor 训练脚本（第三周 阶段 C）
#
# 从零训练 VoiceFilter 式掩码提取网络（sx_model.SXExtractor）：
#   输入  = rec_audio（混合波形，含重叠/噪声）
#   条件  = wake 说话人嵌入（复用 SC 的 spk_emb 缓存，缺失时自动补提）
#   监督  = clean_audio（--save-clean 生成的对齐干净轨；拒识=全零）
#   损失  = 波形域（waveform_loss：正样本负 SNR + 拒识 log 能量抑制，
#           破除幅度 L1 的"全抑制"退化解）；SI-SDR / 幅度 L1 仅监控
#
# 用法：
#   # 过拟合验证：前 32 条反复训练 300 步
#   python sx_train.py --dataset sxtrain --overfit 32 --steps 300
#   # 正式训练
#   python sx_train.py --dataset sxtrain --epochs 30 --batch-size 16
#   # 断点续训（checkpoint 含优化器状态）
#   python sx_train.py --dataset sxtrain --init-from <ckpt> --steps N
# ==============================================================

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets as ds
from sc_data import read_wav, emb_path
from sx_model import (SXExtractor, mag_l1_loss, waveform_loss, si_sdr,
                      DEFAULT_CONFIG, SAMPLE_RATE)

_PROGRESS_ENABLED = False


def emit_progress(**kw):
    if _PROGRESS_ENABLED:
        print(f"[PROGRESS] {json.dumps(kw, ensure_ascii=False)}", flush=True)


# ---------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------
class SXDataset:
    """
    重叠目标提取数据集。要求 manifest 带 clean_audio 字段（--save-clean）。
    每行 → {mix, ref, spk_emb, id, type}
      mix: rec_audio 混合波形 (samples,) float32
      ref: clean_audio 对齐干净轨；拒识样本为等长全零
    """

    def __init__(self, dataset, split, limit=0):
        entry = ds.resolve_dataset(dataset)
        self.ds_path = entry["path"]
        self.split = split
        manifest = os.path.join(self.ds_path, f"{split}_manifest.jsonl")
        self.rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
        if self.rows and "clean_audio" not in self.rows[0]:
            raise ValueError(
                f"数据集 {entry['name']} 的 manifest 缺少 clean_audio 字段，"
                f"请用 build_dataset.py --save-clean 重新生成")
        if limit > 0:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        mix = read_wav(os.path.join(self.ds_path, row["rec_audio"]))
        ca = row["clean_audio"]
        if not ca:
            ref = np.zeros_like(mix)              # 拒识：监督全零
        elif ca == row["rec_audio"]:
            ref = mix.copy()                      # 无重叠：rec 即干净轨
        else:
            ref = read_wav(os.path.join(self.ds_path, ca))
        # 生成时已保证逐点等长；防御性截断到公共长度
        if ref.shape[0] != mix.shape[0]:
            n = min(ref.shape[0], mix.shape[0])
            mix, ref = mix[:n], ref[:n]
        emb = np.load(emb_path(self.ds_path, self.split, row["id"]))
        return {
            "id": row["id"],
            "type": row["type"],
            # 重叠正样本标记：训练监控只对这类样本算 SNR/SI-SDR——
            # 无重叠样本 ref==mix，恒等输出即满分，混进指标会造成
            # "snr 30-60dB"的假象（v3 训练实际学到了纯恒等的教训）
            "overlap": bool(ca and ca != row["rec_audio"]),
            "mix": mix,                           # (samples,) float32
            "ref": ref,                           # (samples,) float32
            "spk_emb": emb.astype(np.float32),    # (256,)
        }


def collate_fn(batch):
    """补零对齐：(B, max_samples) 的 mix/ref + 长度 + 嵌入"""
    n = len(batch)
    lengths = torch.tensor([b["mix"].shape[0] for b in batch],
                           dtype=torch.int64)
    max_len = int(lengths.max())
    mix = torch.zeros(n, max_len, dtype=torch.float32)
    ref = torch.zeros(n, max_len, dtype=torch.float32)
    for i, b in enumerate(batch):
        L = b["mix"].shape[0]
        mix[i, :L] = torch.from_numpy(b["mix"])
        ref[i, :L] = torch.from_numpy(b["ref"])
    spk_emb = torch.from_numpy(np.stack([b["spk_emb"] for b in batch]))
    return {
        "mix": mix, "ref": ref, "lengths": lengths, "spk_emb": spk_emb,
        "ids": [b["id"] for b in batch],
        "types": [b["type"] for b in batch],
        "overlaps": [b["overlap"] for b in batch],
    }


def ensure_embeddings(dataset, split, limit, log=print):
    """检查所用样本的 wake 嵌入缓存，缺失时自动补提（wespeaker CPU）"""
    from sc_data import extract_embeddings
    entry = ds.resolve_dataset(dataset)
    manifest = os.path.join(entry["path"], f"{split}_manifest.jsonl")
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    if limit > 0:
        rows = rows[:limit]
    missing = sum(1 for r in rows
                  if not os.path.isfile(emb_path(entry["path"], split, r["id"])))
    if not missing:
        return
    log(f"[数据] {missing}/{len(rows)} 条缺 wake 嵌入缓存，自动补提"
        f"（wespeaker CPU，约 0.4s/条）...")
    extract_embeddings(dataset, split, limit=limit, log=log)


# ---------------------------------------------------------------
# 训练
# ---------------------------------------------------------------
def train(args, log=print):
    from sx_model import sx_load  # noqa: F401  (init_from 走下方手动加载)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 嵌入缓存（过拟合只补前 N 条；全量训练首次会自动补提全部，较久）
    limit = args.overfit if args.overfit > 0 else args.limit
    ensure_embeddings(args.dataset, "train", limit, log)

    model = SXExtractor(**{**DEFAULT_CONFIG, "mask_bias": args.mask_bias}).to(device)
    optim_state = None
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        optim_state = ckpt.get("optimizer") if isinstance(ckpt, dict) else None
        log(f"[训练] 从 checkpoint 继续: {args.init_from}"
            f"（{'含' if optim_state else '无'}优化器状态）")
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"[训练] SXExtractor 参数量 {n_param:.2f}M（{device}）")
    model.train()

    dataset = SXDataset(args.dataset, "train", limit=limit)
    log(f"[训练] 样本数: {len(dataset)}（batch_size={args.batch_size}）")
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=not args.overfit,  # 过拟合固定顺序便于观察
                        collate_fn=collate_fn, num_workers=0,
                        drop_last=False)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    if optim_state is not None:
        optim.load_state_dict(optim_state)
        log("[训练] 优化器状态已恢复（断点续训）")

    # checkpoint 目录：<数据集>/checkpoints/sx_<时间>/
    entry = ds.resolve_dataset(args.dataset)
    run_dir = args.output or os.path.join(
        entry["path"], "checkpoints", f"sx_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in vars(args).items()}, f,
                  ensure_ascii=False, indent=2, default=str)
    log(f"[训练] checkpoint 目录: {run_dir}")

    total_steps = args.steps if args.steps > 0 else args.epochs * len(loader)
    # --steps 模式：epoch 数按步数反推，保证能跑满 total_steps
    n_epochs = (args.epochs if args.steps <= 0
                else max(1, -(-args.steps // len(loader))))
    step = 0
    t0 = time.time()
    done = False
    loss = None
    for epoch in range(n_epochs):
        if done:
            break
        for batch in loader:
            step += 1
            mix = batch["mix"].to(device)
            ref = batch["ref"].to(device)
            lengths = batch["lengths"].to(device)
            spk_emb = batch["spk_emb"].to(device)

            est_mag, _, phase = model(mix, spk_emb)
            est_wav = model.istft(est_mag, phase, mix.shape[-1])
            loss = waveform_loss(est_wav, ref, lengths, batch["types"],
                                 rej_weight=args.rej_weight,
                                 rej_clamp=args.rej_clamp)
            if args.mag_aux > 0:
                # 幅度域辅助损失（第五轮探针）：波形 -SNR 经 ISTFT 回传
                # 的梯度相位敏感、对掩码选择性学习信号弱；幅度 L1 给掩码
                # 直接的逐频 bin 监督（拒识 ref=0 与抑制方向一致）
                ref_mag = model.stft(ref).abs()
                loss = loss + args.mag_aux * mag_l1_loss(
                    est_mag, ref_mag, lengths)
            if not torch.isfinite(loss):
                # 安全网：非有限 loss 跳过该步（正常不应触发，触发即记录）
                log(f"[训练] [警告] step {step} loss 非有限"
                    f"（{loss.item()}），跳过本步")
                optim.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            optim.zero_grad(set_to_none=True)

            if step % args.log_every == 0 or step == 1:
                # 监控：SNR/SI-SDR 只在"重叠正样本"上计算（无重叠样本
                # ref==mix，恒等输出即满分，会淹没问题——见 SXDataset）
                snr_txt = sdr_txt = l1_txt = rej_txt = "n/a"
                pos = [i for i, (t, ov) in enumerate(
                    zip(batch["types"], batch["overlaps"]))
                    if t == "positive" and ov]
                rej = [i for i, t in enumerate(batch["types"])
                       if t != "positive"]
                with torch.no_grad():
                    if pos:
                        idx = torch.tensor(pos, device=device)
                        snr_txt = f"{-float(waveform_loss(est_wav[idx], ref[idx], lengths[idx], ['positive']*len(pos))):.2f}dB"
                        sdr = si_sdr(est_wav[idx], ref[idx], lengths[idx]).mean()
                        sdr_txt = f"{float(sdr):.2f}dB"
                    if rej:
                        idx = torch.tensor(rej, device=device)
                        r = est_wav[idx].pow(2).mean(-1).sqrt()
                        rej_txt = f"{float(r.mean()):.4f}"
                    ref_mag = model.stft(ref).abs()
                    l1_txt = f"{float(mag_l1_loss(est_mag, ref_mag, lengths)):.4f}"
                log(f"[训练] step {step}/{total_steps} "
                    f"loss={loss.item():.2f} ov_snr={snr_txt} "
                    f"ov_si_sdr={sdr_txt} rej_rms={rej_txt} mag_l1={l1_txt} "
                    f"({time.time()-t0:.0f}s)")
                emit_progress(phase="train", done=step, total=total_steps,
                              loss=round(loss.item(), 3),
                              si_sdr=None if sdr_txt == "n/a"
                              else round(float(sdr), 2))
            if step % args.save_every == 0 or step == total_steps:
                ckpt = os.path.join(run_dir, f"step_{step}.pt")
                torch.save({"state_dict": model.state_dict(),
                            "optimizer": optim.state_dict(),
                            "config": DEFAULT_CONFIG, "step": step}, ckpt)
                log(f"[训练] 已保存: {ckpt}")
            if args.steps > 0 and step >= args.steps:
                done = True
                break
        if args.steps > 0:
            continue
        log(f"[训练] epoch {epoch+1}/{args.epochs} 完成")

    emit_progress(phase="done", done=1, total=1, output=run_dir,
                  final_loss=round(loss.item(), 4) if loss is not None else None)
    log(f"[训练] 结束，最终 loss={loss.item():.4f}，"
        f"用时 {time.time()-t0:.0f}s")
    return run_dir


def build_parser():
    ap = argparse.ArgumentParser(description="SX-Extractor 提取前端训练")
    ap.add_argument("--dataset", default="latest",
                    help="训练数据集（latest / 文件夹名 / 别名 / 时间前缀）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-bias", type=float, default=0.0,
                    help="掩码头偏置初始化（消融结论：+2 会杀死分离学习）")
    ap.add_argument("--rej-weight", type=float, default=1.0,
                    help="拒识抑制项权重（消融结论：0.3 会杀死分离学习）")
    ap.add_argument("--rej-clamp", type=float, default=-40.0,
                    help="拒识抑制项下限 dB（-40 防 sigmoid 深饱和死锁）")
    ap.add_argument("--mag-aux", type=float, default=0.0,
                    help="幅度 L1 辅助损失权重（0=关闭；给掩码逐bin监督）")
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--overfit", type=int, default=0,
                    help="过拟合测试：只取前 N 条训练")
    ap.add_argument("--steps", type=int, default=0,
                    help="固定训练步数（>0 时覆盖 --epochs）")
    ap.add_argument("--limit", type=int, default=0, help="仅使用前 N 条（调试用）")
    ap.add_argument("--init-from", default=None,
                    help="从指定 SX checkpoint 继续训练")
    ap.add_argument("--output", default=None,
                    help="checkpoint 目录（默认 <数据集>/checkpoints/sx_<时间>）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--progress", action="store_true")
    return ap


def main():
    global _PROGRESS_ENABLED
    args = build_parser().parse_args()
    _PROGRESS_ENABLED = args.progress
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
