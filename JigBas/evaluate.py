# ==============================================================
# JigBas 评估脚本 — 基线指标：CER / 拒识率 / 推理耗时
#
# 对 build_dataset.py 生成的 manifest 批量推理：
#   1. 每条样本只跑一次完整流水线（声纹比对 + 正样本 ASR）
#   2. 声纹阈值扫描无需重复推理（ASR 结果与阈值无关）
#   3. 输出各阈值下的 CER、拒识率(RR)、误接受率(FAR)、误拒识率(FRR)
#      及耗时统计，结果写入 Records/
#
# 用法：
#   python evaluate.py --manifest Dataset/dev_manifest.jsonl --data-root Dataset
#   python evaluate.py --manifest ... --threshold 0.5          # 单阈值
#   python evaluate.py --manifest ... --limit 50               # 调试用小样本
# ==============================================================

import argparse
import json
import os
import re
import sys
import time

from core import ModelHub, extract_embedding, cosine_similarity

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RECORDS_DIR = os.path.join(PROJECT_ROOT, "Records")


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
# 主流程
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="JigBas 基线评估（CER / 拒识率 / 耗时）")
    ap.add_argument("--manifest", required=True, help="manifest jsonl 路径")
    ap.add_argument("--data-root", default=".", help="音频相对路径的根目录")
    ap.add_argument("--threshold", type=float, default=None,
                    help="单阈值评估（默认扫描 --sweep）")
    ap.add_argument("--sweep", default="0.30:0.75:0.05",
                    help="阈值扫描范围 lo:hi:step（默认 0.30:0.75:0.05）")
    ap.add_argument("--device", default=None, help="cpu / cuda:0（默认自动）")
    ap.add_argument("--limit", type=int, default=0, help="仅评估前 N 条（调试用）")
    ap.add_argument("--output", default=None,
                    help="结果 JSON 输出路径（默认 Records/eval_<时间戳>.json）")
    ap.add_argument("--detail", action="store_true",
                    help="额外写出逐样本明细 jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    if args.limit > 0:
        rows = rows[: args.limit]
    n_pos = sum(1 for r in rows if r["type"] == "positive")
    print(f"[评估] 样本 {len(rows)} 条（正样本 {n_pos}，拒识 {len(rows)-n_pos}）")

    hub = ModelHub(device=args.device)
    print(f"[评估] 推理设备: {hub.device}")
    hub.load()
    if not hub.ready():
        print("[评估] 模型加载失败，退出")
        sys.exit(1)

    t0 = time.time()
    samples = run_inference(hub, rows, args.data_root)
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

    # 写 Records
    os.makedirs(RECORDS_DIR, exist_ok=True)
    out_path = args.output or os.path.join(
        RECORDS_DIR, time.strftime("eval_%Y%m%d_%H%M%S.json"))
    payload = {
        "manifest": os.path.abspath(args.manifest),
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


if __name__ == "__main__":
    main()
