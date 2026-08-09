# ==============================================================
# JigBas 评估脚本 — 基线指标：CER / 拒识率 / 推理耗时
#
# 对指定数据集（Temp/Datasets/<时间>_<别名>/）的 manifest 批量推理：
#   1. 每条样本只跑一次完整流水线（声纹比对 + 正样本 ASR）
#   2. 声纹阈值扫描无需重复推理（ASR 结果与阈值无关）
#   3. 输出各阈值下的 CER、拒识率(RR)、误接受率(FAR)、误拒识率(FRR)
#      及耗时统计，结果写入该数据集文件夹 evals/<评估时间>.json
#
# 用法：
#   python evaluate.py --dataset latest               # 评估最近构建的数据集
#   python evaluate.py --dataset baseline --limit 50  # 别名/时间前缀定位
#   python evaluate.py --dataset ... --progress       # [PROGRESS] 行供 UI 解析
#   python evaluate.py --dataset baseline --sc-checkpoint latest
#       # SC-Paraformer 评估（第二周）：输出空=拒识，无需阈值扫描
# ==============================================================

import argparse
import json
import os
import re
import sys
import time

import datasets as ds
from models import ModelHub, extract_embedding, cosine_similarity

_PROGRESS_ENABLED = False


def emit_progress(**kw):
    """--progress 开启时输出结构化进度行（供 UI 解析，普通日志不受影响）"""
    if _PROGRESS_ENABLED:
        print(f"[PROGRESS] {json.dumps(kw, ensure_ascii=False)}", flush=True)


# ---------------------------------------------------------------
# CER：字级编辑距离（去除空白与标点，中英文按字符计）
# ---------------------------------------------------------------
_PUNCT_RE = re.compile(r"[\s，。！？、；：""''（）《》〈〉【】…—·,.!?;:'\"()\[\]<>-]")


def normalize_text(s):
    return _PUNCT_RE.sub("", s or "")


def edit_distance(ref, hyp):
    """标准 Levenshtein，O(min(len)) 空间"""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    ref, hyp = normalize_text(ref), normalize_text(hyp)
    if not ref:
        return 0, 0
    return edit_distance(ref, hyp), len(ref)


# ---------------------------------------------------------------
# 批量推理：每条样本记录 sim 与（正样本的）ASR 文本
# ---------------------------------------------------------------
def run_inference(hub, rows, data_root, log=print):
    """返回 samples: [{id,type,sim,ref,hyp_asr,elapsed,duration}]"""
    samples = []
    n = len(rows)
    t_start = time.time()
    for i, row in enumerate(rows, 1):
        wake_path = os.path.join(data_root, row["wake_audio"])
        rec_path = os.path.join(data_root, row["rec_audio"])
        t0 = time.time()
        rec = {
            "id": row["id"],
            "type": row["type"],
            "sim": None,
            "ref": row.get("rec_text", ""),
            "hyp_asr": None,       # 正样本的 ASR 原文（阈值无关）
            "elapsed": 0.0,
            "duration": row.get("duration", 0.0),
            "error": None,
        }
        try:
            emb_wake = extract_embedding(hub, wake_path)
            emb_rec = extract_embedding(hub, rec_path)
            rec["sim"] = cosine_similarity(emb_wake, emb_rec)

            # 只对正样本做 ASR：拒识样本的文本不影响任何指标
            if row["type"] == "positive":
                asr_result = hub.funasr.generate(input=rec_path)
                if asr_result and len(asr_result) > 0:
                    rec["hyp_asr"] = (asr_result[0].get("text", "")
                                      if isinstance(asr_result[0], dict)
                                      else str(asr_result[0]))
        except Exception as e:
            rec["error"] = str(e)
            log(f"[错误] {row['id']}: {e}")
        rec["elapsed"] = time.time() - t0
        samples.append(rec)

        if i % 10 == 0 or i == n:
            emit_progress(phase="infer", done=i, total=n,
                          elapsed=round(time.time() - t_start, 1))
        if i % 20 == 0 or i == n:
            log(f"[推理] {i}/{n}")
    return samples


# ---------------------------------------------------------------
# 指标计算（给定阈值）
# ---------------------------------------------------------------
def compute_metrics(samples, threshold):
    """返回该阈值下的指标 dict"""
    n_pos = n_rej = 0
    err_chars = ref_chars = 0
    false_accept = 0   # 拒识样本被接受
    false_reject = 0   # 正样本被拒识
    pos_elapsed = rej_elapsed = 0.0

    for s in samples:
        if s["error"] or s["sim"] is None:
            # 推理失败：正样本按全错、拒识样本按误接受计，避免高估
            if s["type"] == "positive":
                n_pos += 1
                d, L = cer(s["ref"], "")
                err_chars += d
                ref_chars += L
                false_reject += 1
            else:
                n_rej += 1
                false_accept += 1
            continue

        accepted = s["sim"] >= threshold
        if s["type"] == "positive":
            n_pos += 1
            pos_elapsed += s["elapsed"]
            hyp = s["hyp_asr"] if accepted else ""
            d, L = cer(s["ref"], hyp or "")
            err_chars += d
            ref_chars += L
            if not accepted:
                false_reject += 1
        else:
            n_rej += 1
            rej_elapsed += s["elapsed"]
            if accepted:
                false_accept += 1

    cer_val = err_chars / ref_chars if ref_chars else 0.0
    rr = (n_rej - false_accept) / n_rej if n_rej else 0.0
    far = false_accept / n_rej if n_rej else 0.0
    frr = false_reject / n_pos if n_pos else 0.0
    # 比赛权重：CER 40% + RR 40%（效率另计）；综合分越高越好
    score = 0.5 * (1 - cer_val) + 0.5 * rr
    return {
        "threshold": round(threshold, 3),
        "cer": round(cer_val, 4),
        "rr": round(rr, 4),
        "far": round(far, 4),
        "frr": round(frr, 4),
        "score": round(score, 4),
        "n_positive": n_pos,
        "n_rejection": n_rej,
        "ref_chars": ref_chars,
        "err_chars": err_chars,
        "avg_time_positive": round(pos_elapsed / n_pos, 3) if n_pos else 0.0,
        "avg_time_rejection": round(rej_elapsed / n_rej, 3) if n_rej else 0.0,
    }


def parse_sweep(spec):
    """'0.3:0.75:0.05' -> [0.30, 0.35, ..., 0.75]"""
    lo, hi, step = (float(x) for x in spec.split(":"))
    out, v = [], lo
    while v <= hi + 1e-9:
        out.append(round(v, 3))
        v += step
    return out


# ---------------------------------------------------------------
# SC-Paraformer 评估（第二周）：输出空文本即拒识，无需阈值
# ---------------------------------------------------------------
SC_THRESHOLD = 0.5  # sim 合成值（1.0=非空接受 / 0.0=空拒识）的判定阈值


def resolve_sc_checkpoint(dataset, spec):
    """'latest' → 数据集 checkpoints/ 下步数最大的 step_*.pt；否则按路径使用"""
    if spec and spec != "latest":
        return spec
    entry = ds.resolve_dataset(dataset)
    import glob
    ckpts = glob.glob(os.path.join(
        entry["path"], "checkpoints", "sc_*", "step_*.pt"))

    def step_of(p):
        m = re.search(r"step_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    ckpts = sorted(ckpts, key=lambda p: (step_of(p), p))
    if not ckpts:
        raise FileNotFoundError(
            f"数据集 {entry['name']} 下没有 SC checkpoint，请先训练")
    return ckpts[-1]


def run_sc_inference(model, frontend, tokenizer, rows, entry, split,
                     batch_size=8, log=print):
    """
    SC 批量推理：贪心解码，输出非空=接受（sim=1.0），空=拒识（sim=0.0）。
    复用 compute_metrics 的阈值判定（threshold=0.5），口径与基线一致。
    """
    import torch
    from sc_data import SCDataset, collate_fn, emb_path

    # 缺嵌入缓存时自动补提（dev 200 条约 1 分钟）
    missing = [r for r in rows
               if not os.path.isfile(emb_path(entry["path"], split, r["id"]))]
    if missing:
        log(f"[评估] 缺少 {len(missing)} 条 wake 嵌入缓存，先补提...")
        import sc_data
        sc_data.extract_embeddings(entry["name"], split)

    dataset = SCDataset(entry["name"], split, tokenizer,
                        limit=len(rows) if rows else 0)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    from sc_model import sc_greedy_decode
    device = next(model.parameters()).device
    ref_by_id = {r["id"]: r.get("rec_text", "") for r in dataset.rows}
    dur_by_id = {r["id"]: r.get("duration", 0.0) for r in dataset.rows}
    samples, n_done = [], 0
    t_start = time.time()
    for batch in loader:
        t0 = time.time()
        hyps = sc_greedy_decode(model, frontend, tokenizer, batch, device)
        per = (time.time() - t0) / max(1, len(hyps))
        for sid, typ, hyp in zip(batch["ids"], batch["types"], hyps):
            hyp = hyp or ""
            samples.append({
                "id": sid, "type": typ,
                "sim": 1.0 if normalize_text(hyp) else 0.0,
                "ref": ref_by_id.get(sid, ""),
                "hyp_asr": hyp,
                "elapsed": per,
                "duration": dur_by_id.get(sid, 0.0),
                "error": None,
            })
        n_done += len(hyps)
        if n_done % 40 < batch_size or n_done >= len(rows):
            emit_progress(phase="infer", done=min(n_done, len(rows)),
                          total=len(rows),
                          elapsed=round(time.time() - t_start, 1))
            log(f"[推理] {min(n_done, len(rows))}/{len(rows)}")
    return samples


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="JigBas 基线评估（CER / 拒识率 / 耗时）")
    ap.add_argument("--dataset", default="latest",
                    help="目标数据集（latest / 文件夹名 / 别名 / 时间前缀）")
    ap.add_argument("--split", default="dev", help="评估的划分（默认 dev）")
    ap.add_argument("--manifest", default=None,
                    help="显式指定 manifest（覆盖 --dataset/--split）")
    ap.add_argument("--data-root", default=None, help="音频相对路径的根目录")
    ap.add_argument("--threshold", type=float, default=None,
                    help="单阈值评估（默认扫描 --sweep）")
    ap.add_argument("--sweep", default="0.30:0.75:0.05",
                    help="阈值扫描范围 lo:hi:step（默认 0.30:0.75:0.05）")
    ap.add_argument("--sc-checkpoint", default=None,
                    help="SC-Paraformer 评估：checkpoint 路径或 latest（数据集下最新）。"
                         "设置后走 SC 通路：输出空=拒识，不做阈值扫描")
    ap.add_argument("--sc-batch-size", type=int, default=8, help="SC 推理批大小")
    ap.add_argument("--device", default=None, help="cpu / cuda:0（默认自动）")
    ap.add_argument("--limit", type=int, default=0, help="仅评估前 N 条（调试用）")
    ap.add_argument("--output", default=None,
                    help="结果 JSON 路径（默认 <数据集>/evals/eval_<时间>.json）")
    ap.add_argument("--detail", action="store_true",
                    help="额外写出逐样本明细 jsonl")
    ap.add_argument("--progress", action="store_true",
                    help="输出 [PROGRESS] 结构化进度行（供 UI 解析）")
    return ap


def run(args):
    """执行评估，返回结果 JSON 路径（供菜单 / UI 复用）"""
    global _PROGRESS_ENABLED
    _PROGRESS_ENABLED = getattr(args, "progress", False)

    # 定位数据集与 manifest
    entry = ds.resolve_dataset(args.dataset)
    data_root = args.data_root or entry["path"]
    manifest = args.manifest or os.path.join(
        entry["path"], f"{args.split}_manifest.jsonl")
    if not os.path.isfile(manifest):
        print(f"[评估] manifest 不存在: {manifest}")
        sys.exit(1)
    print(f"[评估] 数据集: {entry['name']}  split: {args.split}")

    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    if args.limit > 0:
        rows = rows[: args.limit]
    n_pos = sum(1 for r in rows if r["type"] == "positive")
    print(f"[评估] 样本 {len(rows)} 条（正样本 {n_pos}，拒识 {len(rows)-n_pos}）")

    # ---- SC-Paraformer 通路：输出空=拒识，无阈值 ----
    if getattr(args, "sc_checkpoint", None):
        from sc_model import build_sc_model
        ckpt = resolve_sc_checkpoint(args.dataset, args.sc_checkpoint)
        device = args.device or ("cuda:0" if __import__("torch").cuda.is_available()
                                 else "cpu")
        print(f"[评估] SC-Paraformer checkpoint: {ckpt}")
        model, kwargs = build_sc_model(device=device, sc_checkpoint=ckpt)
        model.eval()
        t0 = time.time()
        samples = run_sc_inference(model, kwargs["frontend"], kwargs["tokenizer"],
                                   rows, entry, args.split,
                                   batch_size=args.sc_batch_size)
        total_infer = time.time() - t0
        best = compute_metrics(samples, SC_THRESHOLD)
        print(f"\n[SC] CER {best['cer']:.2%} | 拒识率 {best['rr']:.2%} "
              f"| FAR {best['far']:.2%} | FRR {best['frr']:.2%} "
              f"| 综合分 {best['score']:.4f}")
        total_audio = sum(s["duration"] for s in samples)
        print(f"[耗时] 总推理 {total_infer:.1f}s，音频总时长 {total_audio:.1f}s，"
              f"整体 RTF {total_infer/max(total_audio,1e-9):.3f}")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = args.output or os.path.join(
            entry["path"], ds.EVALS_DIR, f"eval_{stamp}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "method": "sc-paraformer",
            "checkpoint": ckpt,
            "dataset": entry["name"],
            "split": args.split,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest": os.path.abspath(manifest),
            "device": device,
            "n_samples": len(rows),
            "n_errors": 0,
            "total_infer_seconds": round(total_infer, 2),
            "total_audio_seconds": round(total_audio, 2),
            "rtf": round(total_infer / max(total_audio, 1e-9), 4),
            "best_threshold": SC_THRESHOLD,
            "metrics": [best],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[输出] 指标已保存: {out_path}")
        if args.detail:
            detail_path = os.path.splitext(out_path)[0] + "_detail.jsonl"
            with open(detail_path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"[输出] 明细已保存: {detail_path}")
        emit_progress(phase="done", done=1, total=1,
                      output=os.path.basename(out_path),
                      best_threshold=SC_THRESHOLD,
                      cer=best["cer"], rr=best["rr"], rtf=payload["rtf"])
        return out_path

    # ---- 基线通路：声纹相似度 + 阈值扫描 ----
    hub = ModelHub(device=args.device)
    print(f"[评估] 推理设备: {hub.device}")
    hub.load()
    if not hub.ready():
        print("[评估] 模型加载失败，退出")
        sys.exit(1)

    t0 = time.time()
    samples = run_inference(hub, rows, data_root)
    total_infer = time.time() - t0

    thresholds = ([args.threshold] if args.threshold is not None
                  else parse_sweep(args.sweep))
    results = [compute_metrics(samples, t) for t in thresholds]

    # 打印阈值扫描表
    print()
    print(f"{'阈值':>6} {'CER':>8} {'拒识率':>8} {'FAR':>8} {'FRR':>8} {'综合分':>8}")
    print("-" * 54)
    for m in results:
        print(f"{m['threshold']:>6.2f} {m['cer']:>8.2%} {m['rr']:>8.2%} "
              f"{m['far']:>8.2%} {m['frr']:>8.2%} {m['score']:>8.4f}")
    best = max(results, key=lambda m: m["score"])
    print("-" * 54)
    print(f"[最优] 阈值 {best['threshold']:.2f}: CER {best['cer']:.2%}, "
          f"拒识率 {best['rr']:.2%}")

    total_audio = sum(s["duration"] for s in samples)
    ok = [s for s in samples if not s["error"]]
    print(f"\n[耗时] 总推理 {total_infer:.1f}s，音频总时长 {total_audio:.1f}s，"
          f"整体 RTF {total_infer/max(total_audio,1e-9):.3f}")
    print(f"[耗时] 单条均值 {sum(s['elapsed'] for s in ok)/max(1,len(ok)):.3f}s")

    # 结果写入数据集文件夹 evals/<评估时间>.json
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output or os.path.join(
        entry["path"], ds.EVALS_DIR, f"eval_{stamp}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "dataset": entry["name"],
        "split": args.split,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": os.path.abspath(manifest),
        "device": hub.device,
        "n_samples": len(rows),
        "n_errors": len(samples) - len(ok),
        "total_infer_seconds": round(total_infer, 2),
        "total_audio_seconds": round(total_audio, 2),
        "rtf": round(total_infer / max(total_audio, 1e-9), 4),
        "best_threshold": best["threshold"],
        "metrics": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[输出] 指标已保存: {out_path}")

    if args.detail:
        detail_path = os.path.splitext(out_path)[0] + "_detail.jsonl"
        with open(detail_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[输出] 明细已保存: {detail_path}")

    emit_progress(phase="done", done=1, total=1,
                  output=os.path.basename(out_path),
                  best_threshold=best["threshold"],
                  cer=best["cer"], rr=best["rr"], rtf=payload["rtf"])
    return out_path


def main():
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
