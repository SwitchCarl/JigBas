# ==============================================================
# SC-Paraformer 训练脚本（第二周 阶段 E）
#
# 从 model.pt 全量微调 SCParaformer（FiLM 说话人条件）：
#   正样本学习目标文本，拒识样本学习空目标（只输出 EOS）。
#
# 用法：
#   # 过拟合测试（阶段 E2）：前 50 条反复训练 200 步，关 specaug
#   python sc_train.py --dataset baseline --overfit 50 --steps 200 --no-specaug
#   # 正式训练（阶段 E3）
#   python sc_train.py --dataset baseline --epochs 10 --batch-size 8
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
from sc_data import SCDataset, collate_fn

_PROGRESS_ENABLED = False


def emit_progress(**kw):
    if _PROGRESS_ENABLED:
        print(f"[PROGRESS] {json.dumps(kw, ensure_ascii=False)}", flush=True)


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def run_step(model, frontend, batch, device):
    """单步前向：波形 → frontend（CPU）→ 特征上 GPU → forward(spk_emb)"""
    with torch.no_grad():
        feats, feats_lens = frontend(batch["speech"], batch["speech_lengths"])
    loss, stats, _ = model(feats.to(device), feats_lens.to(device),
                           batch["text"].to(device),
                           batch["text_lengths"].to(device),
                           spk_emb=batch["spk_emb"].to(device))
    return loss, stats


def train(args, log=print):
    from sc_model import build_sc_model

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log(f"[训练] 构建 SC 模型（{device}）...")
    model, kwargs = build_sc_model(device=device,
                                   sc_checkpoint=args.init_from)
    tokenizer, frontend = kwargs["tokenizer"], kwargs["frontend"]
    if args.no_specaug:
        model.specaug = None
        log("[训练] specaug 已关闭（过拟合/调试用）")
    model.train()

    # 数据
    limit = args.overfit if args.overfit > 0 else args.limit
    dataset = SCDataset(args.dataset, "train", tokenizer, limit=limit)
    log(f"[训练] 样本数: {len(dataset)}（batch_size={args.batch_size}）")
    sampler = None
    shuffle = not args.overfit  # 过拟合固定顺序便于观察
    if args.rej_balance:
        # 拒识过采样：正:拒 抽到 1:1，让"闭嘴"梯度不被正样本淹没（A 方案）
        types = [r["type"] for r in dataset.rows]
        n_pos = sum(1 for t in types if t == "positive")
        n_rej = len(types) - n_pos
        w_rej = n_pos / max(n_rej, 1)
        weights = [1.0 if t == "positive" else w_rej for t in types]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(dataset), replacement=True)
        shuffle = False  # sampler 与 shuffle 互斥，采样器自带随机
        log(f"[训练] 拒识平衡采样: 正 {n_pos} / 拒 {n_rej}，拒识权重 {w_rej:.2f}")
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=shuffle, sampler=sampler,
                        collate_fn=collate_fn, num_workers=0,
                        drop_last=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    # checkpoint 目录：<数据集>/checkpoints/sc_<时间>/
    entry = ds.resolve_dataset(args.dataset)
    run_dir = args.output or os.path.join(
        entry["path"], "checkpoints", f"sc_{time.strftime('%Y%m%d_%H%M%S')}")
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
    for epoch in range(n_epochs):
        if done:
            break
        for batch in loader:
            step += 1
            loss, stats = run_step(model, frontend, batch, device)
            if not torch.isfinite(loss):
                # 安全网：非有限 loss 跳过该步（正常不应触发，触发即记录）
                log(f"[训练] [警告] step {step} loss 非有限（{loss.item()}），跳过本步")
                optim.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            optim.zero_grad(set_to_none=True)

            if step % args.log_every == 0 or step == 1:
                msg = (f"[训练] step {step}/{total_steps} "
                       f"loss={loss.item():.4f} "
                       f"att={float(stats['loss_att']):.4f} "
                       f"pre={float(stats['loss_pre']):.4f} "
                       f"acc={float(stats['acc']):.4f} "
                       f"({time.time()-t0:.0f}s)")
                log(msg)
                emit_progress(phase="train", done=step, total=total_steps,
                              loss=round(loss.item(), 4),
                              loss_att=round(float(stats["loss_att"]), 4),
                              loss_pre=round(float(stats["loss_pre"]), 4),
                              acc=round(float(stats["acc"]), 4))
            if step % args.save_every == 0 or step == total_steps:
                ckpt = os.path.join(run_dir, f"step_{step}.pt")
                torch.save(model.state_dict(), ckpt)
                log(f"[训练] 已保存: {ckpt}")
            if args.steps > 0 and step >= args.steps:
                done = True
                break
        if args.steps > 0:
            continue
        log(f"[训练] epoch {epoch+1}/{args.epochs} 完成")

    emit_progress(phase="done", done=1, total=1, output=run_dir,
                  final_loss=round(loss.item(), 4))
    log(f"[训练] 结束，最终 loss={loss.item():.4f}，用时 {time.time()-t0:.0f}s")
    return run_dir


def build_parser():
    ap = argparse.ArgumentParser(description="SC-Paraformer 微调训练")
    ap.add_argument("--dataset", default="latest",
                    help="训练数据集（latest / 文件夹名 / 别名 / 时间前缀）")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--overfit", type=int, default=0,
                    help="过拟合测试：只取前 N 条训练")
    ap.add_argument("--steps", type=int, default=0,
                    help="固定训练步数（>0 时覆盖 --epochs）")
    ap.add_argument("--limit", type=int, default=0, help="仅使用前 N 条（调试用）")
    ap.add_argument("--no-specaug", action="store_true", help="关闭 specaug")
    ap.add_argument("--rej-balance", action="store_true",
                    help="拒识样本过采样至与正样本 1:1（WeightedRandomSampler）")
    ap.add_argument("--init-from", default=None,
                    help="从指定 SC checkpoint 继续训练")
    ap.add_argument("--output", default=None,
                    help="checkpoint 目录（默认 <数据集>/checkpoints/sc_<时间>）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
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
