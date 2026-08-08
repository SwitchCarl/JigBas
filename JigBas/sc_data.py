# ==============================================================
# SC-Paraformer 数据管线（第二周）
#
# 为说话人条件 ASR 训练提供数据：
#   1. 预提取 wake 音频的 ECAPA 说话人嵌入，缓存到
#      <数据集>/spk_emb/<split>/<样本id>.npy（256 维 float32）
#   2. SCDataset + collate_fn：波形 / 文本 id（拒识为空）/ 嵌入 组成 batch
#
# 用法：
#   python sc_data.py --dataset latest --extract-emb          # 预提取嵌入（train+dev）
#   python sc_data.py --dataset latest --split dev --extract-emb
#   python sc_data.py --dataset latest --smoke                # 单 batch 冒烟测试
#   （均支持 --progress 输出 [PROGRESS] 结构化行）
# ==============================================================

import argparse
import json
import os
import sys

import numpy as np

import datasets as ds

EMB_DIR = "spk_emb"          # 数据集文件夹内的嵌入缓存子目录
IGNORE_ID = -1               # 与 funasr 训练一致：padding 用 -1
SAMPLE_RATE = 16000

_PROGRESS_ENABLED = False


def emit_progress(**kw):
    """--progress 开启时输出结构化进度行（供 UI / 训练脚本解析）"""
    if _PROGRESS_ENABLED:
        print(f"[PROGRESS] {json.dumps(kw, ensure_ascii=False)}", flush=True)


# ---------------------------------------------------------------
# B1. 说话人嵌入预提取
# ---------------------------------------------------------------
def emb_path(ds_path, split, sample_id):
    return os.path.join(ds_path, EMB_DIR, split, f"{sample_id}.npy")


def extract_embeddings(dataset, split, hub=None, limit=0, log=print):
    """
    遍历 <split>_manifest.jsonl，为每条样本的 wake 音频提取 ECAPA 嵌入并缓存。
    已存在的 .npy 跳过（幂等，可中断重跑）。返回 (新增数, 跳过数)。
    """
    entry = ds.resolve_dataset(dataset)
    manifest = os.path.join(entry["path"], f"{split}_manifest.jsonl")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"manifest 不存在: {manifest}")
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    if limit > 0:
        rows = rows[:limit]

    if hub is None:
        from models import ModelHub
        hub = ModelHub()
        # 只加载 wespeaker（提取嵌入用不到 ASR 模型，省下加载时间）
        hub._load_wespeaker(log)
        if hub.status["wespeaker"] != "就绪":
            raise RuntimeError("wespeaker 加载失败")

    from models import extract_embedding

    n_new = n_skip = 0
    total = len(rows)
    for i, row in enumerate(rows, 1):
        out = emb_path(entry["path"], split, row["id"])
        if os.path.isfile(out):
            n_skip += 1
        else:
            wav = os.path.join(entry["path"], row["wake_audio"])
            emb = extract_embedding(hub, wav).astype(np.float32).ravel()
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.save(out, emb)
            n_new += 1
        if i % 20 == 0 or i == total:
            emit_progress(phase="extract_emb", done=i, total=total,
                          split=split, new=n_new, skipped=n_skip)
            log(f"[嵌入] {split} {i}/{total}（新增 {n_new}，跳过 {n_skip}）")
    return n_new, n_skip


# ---------------------------------------------------------------
# B2. Dataset + collate
# ---------------------------------------------------------------
def read_wav(path):
    """读单声道 16k 波形为 float32；多声道取均值，采样率不符则报错"""
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"采样率 {sr} != {SAMPLE_RATE}: {path}")
    return wav.mean(axis=1)  # (samples,)


class SCDataset:
    """
    说话人条件 ASR 数据集。
    每行 manifest → {speech, text_ids, spk_emb, id, type}
      正样本: text_ids = tokenizer.encode(rec_text)
      拒识样本: text_ids = []（空目标，解码侧只会输出 EOS）
    """

    def __init__(self, dataset, split, tokenizer, limit=0):
        entry = ds.resolve_dataset(dataset)
        self.ds_path = entry["path"]
        self.split = split
        manifest = os.path.join(self.ds_path, f"{split}_manifest.jsonl")
        self.rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
        if limit > 0:
            self.rows = self.rows[:limit]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        speech = read_wav(os.path.join(self.ds_path, row["rec_audio"]))
        text = row.get("rec_text") or ""
        text_ids = self.tokenizer.encode(text) if text else []
        emb = np.load(emb_path(self.ds_path, self.split, row["id"]))
        return {
            "id": row["id"],
            "type": row["type"],
            "speech": speech,                       # (samples,) float32
            "text_ids": np.asarray(text_ids, dtype=np.int64),
            "spk_emb": emb.astype(np.float32),      # (256,)
        }


def collate_fn(batch):
    """
    补零对齐为训练 batch：
      speech        (B, max_samples)  float32
      speech_lengths(B,)             int64
      text          (B, max_tokens)   int64，padding=IGNORE_ID
      text_lengths  (B,)             int64（空文本为 0）
      spk_emb       (B, 256)          float32
    """
    import torch

    speeches = [torch.from_numpy(b["speech"]) for b in batch]
    speech_lengths = torch.tensor([s.shape[0] for s in speeches], dtype=torch.int64)
    speech = torch.zeros(len(batch), int(speech_lengths.max()), dtype=torch.float32)
    for i, s in enumerate(speeches):
        speech[i, : s.shape[0]] = s

    text_lengths = torch.tensor([len(b["text_ids"]) for b in batch], dtype=torch.int64)
    max_tokens = max(1, int(text_lengths.max()))  # 全空批也保留 1 列
    text = torch.full((len(batch), max_tokens), IGNORE_ID, dtype=torch.int64)
    for i, b in enumerate(batch):
        if len(b["text_ids"]):
            text[i, : len(b["text_ids"])] = torch.from_numpy(b["text_ids"])

    spk_emb = torch.from_numpy(np.stack([b["spk_emb"] for b in batch]))

    return {
        "speech": speech,
        "speech_lengths": speech_lengths,
        "text": text,
        "text_lengths": text_lengths,
        "spk_emb": spk_emb,
        "ids": [b["id"] for b in batch],
        "types": [b["type"] for b in batch],
    }


# ---------------------------------------------------------------
# B3. 冒烟测试：混批过 collate + 模型 frontend，打印形状
# ---------------------------------------------------------------
def smoke_test(dataset, batch_size=8, log=print):
    import torch
    from funasr.auto.auto_model import AutoModel
    from models import FUNASR_MODEL_DIR

    log(f"[冒烟] 加载模型组件（build_model: {os.path.basename(FUNASR_MODEL_DIR)}）...")
    model, kwargs = AutoModel.build_model(model=FUNASR_MODEL_DIR)
    tokenizer = kwargs["tokenizer"]

    # 取前 batch_size//2 条正样本 + 前 batch_size//2 条拒识样本，强制混批
    entry = ds.resolve_dataset(dataset)
    rows = [json.loads(l) for l in open(os.path.join(
        entry["path"], "dev_manifest.jsonl"), encoding="utf-8")]
    pos = [r for r in rows if r["type"] == "positive"][: batch_size // 2]
    rej = [r for r in rows if r["type"] == "rejection"][: batch_size // 2]
    data = SCDataset(dataset, "dev", tokenizer)
    by_id = {r["id"]: i for i, r in enumerate(data.rows)}
    batch = collate_fn([data[by_id[r["id"]]] for r in pos + rej])
    frontend = kwargs["frontend"]  # Paraformer 不内嵌 frontend，特征提取由数据侧完成

    log("[冒烟] batch 内容: " + ", ".join(
        f"{i}({t})" for i, t in zip(batch["ids"], batch["types"])))
    log(f"[冒烟] speech         {tuple(batch['speech'].shape)}  dtype={batch['speech'].dtype}")
    log(f"[冒烟] speech_lengths {batch['speech_lengths'].tolist()}")
    log(f"[冒烟] text           {tuple(batch['text'].shape)}  dtype={batch['text'].dtype}")
    log(f"[冒烟] text_lengths   {batch['text_lengths'].tolist()}")
    log(f"[冒烟] spk_emb        {tuple(batch['spk_emb'].shape)}  dtype={batch['spk_emb'].dtype}")

    # 注意：空文本经 tokenizer.encode 会得到 [<unk>]（seg_dict 路径），
    # 所以拒识样本必须绕过 tokenizer 直接给 []（SCDataset 已处理）
    empty = tokenizer.encode("")
    log(f"[冒烟] tokenizer.encode('') = {empty}（<unk>；拒识样本不走这里，直接 []）")
    first_pos = next(b for b in (data[by_id[r['id']]] for r in pos)
                     if len(b["text_ids"]))
    log(f"[冒烟] 首条正样本回读: '{tokenizer.decode(first_pos['text_ids'].tolist())}'")

    # 过 frontend：确认波形→560 维特征正常（训练循环同样先做这步再 model.forward）
    with torch.no_grad():
        feats, feats_lens = frontend(batch["speech"], batch["speech_lengths"])
    log(f"[冒烟] frontend 输出  {tuple(feats.shape)}（期望最后一维 560）")
    assert feats.shape[-1] == 560, "frontend 输出维度异常"
    assert torch.isfinite(feats).all(), "frontend 输出含 NaN/Inf"
    log("[冒烟] 全部通过")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="SC-Paraformer 数据管线（嵌入缓存 / 冒烟测试）")
    ap.add_argument("--dataset", default="latest",
                    help="目标数据集（latest / 文件夹名 / 别名 / 时间前缀）")
    ap.add_argument("--split", default="train", help="嵌入提取的划分（默认 train）")
    ap.add_argument("--extract-emb", action="store_true",
                    help="预提取 wake 说话人嵌入到 <数据集>/spk_emb/")
    ap.add_argument("--both-splits", action="store_true",
                    help="--extract-emb 时同时处理 train 和 dev")
    ap.add_argument("--smoke", action="store_true", help="单 batch 冒烟测试")
    ap.add_argument("--batch-size", type=int, default=8, help="冒烟测试批大小")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（调试用）")
    ap.add_argument("--progress", action="store_true",
                    help="输出 [PROGRESS] 结构化进度行")
    return ap


def main():
    global _PROGRESS_ENABLED
    args = build_parser().parse_args()
    _PROGRESS_ENABLED = args.progress

    if args.extract_emb:
        splits = ["train", "dev"] if args.both_splits else [args.split]
        for split in splits:
            n_new, n_skip = extract_embeddings(
                args.dataset, split, limit=args.limit)
            print(f"[完成] {split}: 新增 {n_new}，跳过 {n_skip}")
    if args.smoke:
        smoke_test(args.dataset, batch_size=args.batch_size)
    if not args.extract_emb and not args.smoke:
        build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
