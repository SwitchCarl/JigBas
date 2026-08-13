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
#   python evaluate.py --dataset baseline --final
#       # 最终系统一键评估（第四周定版）：SX 提取 + 声纹门控 + SC-scx，
#       # 自动填入最终 checkpoint 与灰区；sx-gate/sx-asr 行即最终系统
# ==============================================================

import argparse
import json
import os
import re
import sys
import time

# 直接运行本脚本时把项目根加入 sys.path（python app/evaluate.py）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import datasets as ds
from lib.models import (ModelHub, extract_embedding, extract_embedding_pcm,
                        cosine_similarity)
from lib.paths import REPO_ROOT

_PROGRESS_ENABLED = False

# 最终系统（第 4 周定版，综合分 0.7813）默认权重：--final 一键评估时自动填入。
# SX 提取器 = 阶段 C 纯分离模型（未微调）；SC-scx = 在提取音频上微调
# 至绝对 8000 步（过拟合前峰值）。路径与 demo.py 的 DEFAULT_SX_CKPT /
# DEFAULT_SCX_CKPT 保持一致。
FINAL_SX_CKPT = os.path.normpath(os.path.join(
    REPO_ROOT, "..", "..", "Temp", "Datasets", "20260811_1914_sxtrain",
    "checkpoints", "sx_20260812_235228", "step_6000.pt"))
FINAL_SC_CKPT = os.path.normpath(os.path.join(
    REPO_ROOT, "..", "..", "Temp", "Datasets", "20260813_0105_scx8k",
    "checkpoints", "sc_20260813_cont4000", "step_4000.pt"))
# 最终系统门控口径：两级门控灰区 0.05:0.60，滑窗精判 0.7*max+0.3*full
FINAL_GATE_REFINE = "0.05:0.60"


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
    from lib.sc_data import SCDataset, collate_fn, emb_path

    # 缺嵌入缓存时自动补提（dev 200 条约 1 分钟）
    missing = [r for r in rows
               if not os.path.isfile(emb_path(entry["path"], split, r["id"]))]
    if missing:
        log(f"[评估] 缺少 {len(missing)} 条 wake 嵌入缓存，先补提...")
        from lib import sc_data
        sc_data.extract_embeddings(entry["name"], split)

    dataset = SCDataset(entry["name"], split, tokenizer,
                        limit=len(rows) if rows else 0)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    from lib.sc_model import sc_greedy_decode
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


def segment_sims_pcm(hub, pcm, emb_wake, win_s=2.5, hop_s=1.25,
                     min_s=1.0, sr=16000):
    """滑窗分段提取嵌入并与 wake 算 sim（灰区精判用，内存波形版）。

    重叠场景下整段嵌入被干扰说话人平均拉偏；目标说话人在无重叠窗内
    是清晰的，分段后取 max 能恢复真实相似度。过短/提取失败的窗跳过。
    """
    win, hop, minlen = int(win_s * sr), int(hop_s * sr), int(min_s * sr)
    sims, st = [], 0
    while st + minlen <= len(pcm):
        seg = pcm[st:st + win]
        try:
            emb = extract_embedding_pcm(hub, seg, sr)
            sims.append(cosine_similarity(emb_wake, emb))
        except Exception:
            pass  # 纯噪声窗等提取失败直接跳过
        if st + win >= len(pcm):
            break
        st += hop
    return sims


def segment_sims(hub, wav_path, emb_wake, win_s=2.5, hop_s=1.25,
                 min_s=1.0, sr=16000):
    """segment_sims_pcm 的文件版包装（demo.py 用）"""
    import soundfile as sf
    pcm, fs = sf.read(wav_path, dtype="float32")
    assert fs == sr, f"采样率 {fs} != {sr}"
    return segment_sims_pcm(hub, pcm, emb_wake, win_s, hop_s, min_s, sr)


def add_hybrid_sims(samples, rows, entry, split, data_root, log=print,
                    refine=None, win_s=2.5, hop_s=1.25):
    """
    混合系统（B 方案）：逐样本计算 wespeaker 余弦相似度，回填 sim 与 elapsed。
    回填后 compute_metrics 的阈值判定即为"声纹门控 + SC 识别文本"：
      接受 = sim >= 阈值；正样本的识别文本来自 SC 贪心解码（已全部算好）。
    wake 嵌入优先复用 SC 缓存（spk_emb/<split>/<id>.npy，与注册嵌入同源），
    缺失时才在线提取；rec 嵌入始终在线提取（这是真实部署的推理成本）。
    refine=(lo,hi) 时启用两级门控：整段 sim 落在灰区 (lo,hi) 内的样本
    追加滑窗分段精判，取各窗 sim 的 max 作为最终 sim（抗重叠污染）。
    """
    import numpy as np
    from lib.sc_data import emb_path

    hub = ModelHub()
    hub._load_wespeaker(log)  # 只加载声纹模型（CPU），不动 funasr
    row_by_id = {r["id"]: r for r in rows}
    n = len(samples)
    n_refined = 0
    for i, s in enumerate(samples, 1):
        t0 = time.time()
        cache = emb_path(entry["path"], split, s["id"])
        if os.path.isfile(cache):
            emb_wake = np.load(cache)
        else:
            emb_wake = extract_embedding(
                hub, os.path.join(data_root, row_by_id[s["id"]]["wake_audio"]))
        rec_path = os.path.join(data_root, row_by_id[s["id"]]["rec_audio"])
        emb_rec = extract_embedding(hub, rec_path)
        s["sim"] = cosine_similarity(emb_wake, emb_rec)
        if refine and refine[0] < s["sim"] < refine[1]:
            seg = segment_sims(hub, rec_path, emb_wake,
                               win_s=win_s, hop_s=hop_s)
            n_refined += 1
            if seg:
                s["seg_sims"] = [round(x, 4) for x in seg]  # 供离线聚合方式消融
                s["sim_full"] = round(s["sim"], 4)
                # 0.7*max + 0.3*full：局部最优窗为主、整段全局证据兜底
                # （离线消融 6 种聚合方式中综合分最高，抑制单窗侥幸高分）
                s["sim"] = 0.7 * max(seg) + 0.3 * s["sim"]
                s["refined"] = True
        s["elapsed"] += time.time() - t0
        if i % 50 == 0 or i == n:
            log(f"[混合] 相似度 {i}/{n}")
    if refine:
        log(f"[混合] 灰区 {refine[0]}:{refine[1]} 分段精判 {n_refined}/{n} 条")
    return samples


# ---------------------------------------------------------------
# 阶段 D：SX 提取前端接入（第三周）
# ---------------------------------------------------------------
def run_sx_artifacts(samples, rows, entry, split, data_root,
                     sc_model, sc_kwargs, args, log=print):
    """
    SX 提取前端产物：逐样本用 wake 嵌入条件提取目标人声，回填：
      sim_raw/hyp_raw  原混合通路的门控 sim 与 SC 文本（改名列存）
      sx_rms           提取波形 RMS（能量门控依据，明细留档供离线调阈值）
      sim_sx           提取波形上的声纹 sim（RMS < 能量门控 → 直接 0，白送拒识；
                       否则过 wespeaker，整段落灰区同样分段精判）
      hyp_sx           SC 在提取波形上的重解码文本（全样本都解码，
                       保证 "原始门控+SX-ASR" 消融不受能量门控污染）
    """
    import numpy as np
    import torch
    from lib.sc_data import emb_path, read_wav
    from lib.sc_model import sc_greedy_decode
    from lib.sx_model import sx_load

    device = next(sc_model.parameters()).device
    sx = sx_load(args.sx_checkpoint, device=device, log=log)
    hub = ModelHub()
    hub._load_wespeaker(log)
    row_by_id = {r["id"]: r for r in rows}
    refine = None
    if args.gate_refine:
        refine = tuple(float(x) for x in args.gate_refine.split(":"))

    n = len(samples)
    pcms, embs = [], []
    n_gated = 0
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        row = row_by_id[s["id"]]
        pcm = read_wav(os.path.join(data_root, row["rec_audio"]))
        cache = emb_path(entry["path"], split, s["id"])
        if os.path.isfile(cache):
            emb_wake = np.load(cache)
        else:
            emb_wake = extract_embedding(
                hub, os.path.join(data_root, row["wake_audio"]))
        emb_wake = np.asarray(emb_wake, dtype="float32").ravel()
        with torch.no_grad():
            out = sx.separate(
                torch.from_numpy(pcm).unsqueeze(0).to(device),
                torch.from_numpy(emb_wake).unsqueeze(0).to(device))
        sx_pcm = out.squeeze(0).cpu().numpy().astype("float32")
        rms = float(np.sqrt(np.mean(np.square(sx_pcm)) + 1e-12))

        s["sim_raw"], s["hyp_raw"] = s["sim"], s["hyp_asr"]
        s["sx_rms"] = round(rms, 5)
        if rms < args.sx_energy_gate:
            s["sim_sx"] = 0.0     # 能量门控：目标缺席 → 白送拒识
            n_gated += 1
        else:
            sim = cosine_similarity(
                emb_wake, extract_embedding_pcm(hub, sx_pcm))
            if refine and refine[0] < sim < refine[1]:
                seg = segment_sims_pcm(hub, sx_pcm, emb_wake,
                                       win_s=args.gate_win,
                                       hop_s=args.gate_hop)
                if seg:
                    sim = 0.7 * max(seg) + 0.3 * sim  # 与原始通路同聚合
                    s["refined_sx"] = True
            s["sim_sx"] = sim
        pcms.append(sx_pcm)
        embs.append(emb_wake)
        if i % 50 == 0 or i == n:
            log(f"[SX] 提取+门控 {i}/{n}（{time.time()-t0:.0f}s）")
    log(f"[SX] 能量门控（<{args.sx_energy_gate}）判拒 {n_gated}/{n} 条")

    # SC 在提取波形上重解码（全样本，批处理）
    bs = args.sc_batch_size
    for k in range(0, n, bs):
        grp = list(range(k, min(k + bs, n)))
        Lm = max(len(pcms[i]) for i in grp)
        speech = torch.zeros(len(grp), Lm)
        for j, i in enumerate(grp):
            speech[j, : len(pcms[i])] = torch.from_numpy(pcms[i])
        batch = {
            "speech": speech,
            "speech_lengths": torch.tensor([len(pcms[i]) for i in grp]),
            "spk_emb": torch.from_numpy(np.stack(embs[k:k + bs])),
        }
        hyps = sc_greedy_decode(sc_model, sc_kwargs["frontend"],
                                sc_kwargs["tokenizer"], batch, device)
        for i, h in zip(grp, hyps):
            samples[i]["hyp_sx"] = h or ""
        if (k + bs) % 40 < bs or k + bs >= n:
            log(f"[SX] SC 重解码 {min(k + bs, n)}/{n}")
    return samples


def config_samples(samples, gate, asr):
    """按消融配置选 sim/hyp 来源，生成 compute_metrics 用的样本副本"""
    out = []
    for s in samples:
        c = dict(s)
        if gate == "sx":
            c["sim"] = s["sim_sx"]
        elif gate == "comb":
            # 联合门控（阶段D 离线扫描最优）：原始/提取两条 wespeaker 声纹
            # 通路互补，平均后判决——FRR 34.56%→32.35%、CER 43.34→41.34%，
            # RR 保持 93.75%（综合分 0.7520→0.7621）。零重训成本。
            # ⚠️ dev 集调参，报告需如实标注过拟合风险。
            c["sim"] = 0.5 * s["sim_raw"] + 0.5 * s["sim_sx"]
        else:
            c["sim"] = s["sim_raw"]
        c["hyp_asr"] = s["hyp_sx"] if asr == "sx" else s["hyp_raw"]
        out.append(c)
    return out


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
    ap.add_argument("--sc-hybrid", action="store_true",
                    help="混合系统：wespeaker 相似度门控 + SC 识别文本，"
                         "恢复阈值扫描（需同时给 --sc-checkpoint）")
    ap.add_argument("--gate-refine", default=None, metavar="LO:HI",
                    help="两级门控灰区（如 0.15:0.45）：整段 sim 落灰区的样本"
                         "做滑窗分段精判（取 0.7*max+0.3*full），"
                         "抗重叠说话人污染嵌入；仅 --sc-hybrid 下生效")
    ap.add_argument("--gate-win", type=float, default=2.5, help="分段窗长（秒）")
    ap.add_argument("--gate-hop", type=float, default=1.25, help="分段步长（秒）")
    ap.add_argument("--sx-checkpoint", default=None,
                    help="SX 提取前端权重（第三周阶段 D）：需配合 --sc-hybrid，"
                         "逐样本先提目标人声再分别跑门控/ASR 消融（4 配置）")
    ap.add_argument("--sx-energy-gate", type=float, default=0.0,
                    help="SX 提取波形的能量门控：RMS 低于此值直接判拒。"
                         "⚠️ 已证伪并默认关闭（默认 0.0）：正/拒样本 RMS 分布重叠，"
                         "且该硬截止会误杀音量偏低的正样本（rms≈0.006 时 sim_sx 仍可高达 0.6）。"
                         "保留参数仅供复现阶段D旧结果")
    ap.add_argument("--final", action="store_true",
                    help="最终系统一键评估（第四周定版）：自动填入 SX/SC-scx 最终"
                         "checkpoint、灰区 0.05:0.60 与混合通路（能量门控默认关闭）。"
                         "输出 5 配置消融表，sx-gate/sx-asr 行即最终系统"
                         "（预期综合分 ≈0.7813）；可用 --sc-checkpoint/"
                         "--sx-checkpoint/--gate-refine 显式覆盖")
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

    # --final：最终系统一键评估，填默认 checkpoint / 灰区 / 混合通路
    # （仅当用户未显式给出时；显式参数优先级更高）
    if getattr(args, "final", False):
        args.sc_checkpoint = args.sc_checkpoint or FINAL_SC_CKPT
        args.sx_checkpoint = args.sx_checkpoint or FINAL_SX_CKPT
        args.gate_refine = args.gate_refine or FINAL_GATE_REFINE
        args.sc_hybrid = True

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
        from lib.sc_model import build_sc_model
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

        if args.sc_hybrid:
            # 混合系统：真实声纹相似度替换合成 sim，恢复阈值扫描
            refine = None
            if args.gate_refine:
                refine = tuple(float(x) for x in args.gate_refine.split(":"))
            t_emb = time.time()
            add_hybrid_sims(samples, rows, entry, args.split, data_root,
                            refine=refine, win_s=args.gate_win,
                            hop_s=args.gate_hop)
            total_infer += time.time() - t_emb
            thresholds = ([args.threshold] if args.threshold is not None
                          else parse_sweep(args.sweep))

            def print_sweep(tag, results):
                print()
                print(f"[{tag}]")
                print(f"{'阈值':>6} {'CER':>8} {'拒识率':>8} {'FAR':>8} "
                      f"{'FRR':>8} {'综合分':>8}")
                print("-" * 54)
                for m in results:
                    print(f"{m['threshold']:>6.2f} {m['cer']:>8.2%} "
                          f"{m['rr']:>8.2%} {m['far']:>8.2%} "
                          f"{m['frr']:>8.2%} {m['score']:>8.4f}")
                b = max(results, key=lambda m: m["score"])
                print("-" * 54)
                print(f"[最优] {tag} 阈值 {b['threshold']:.2f}: "
                      f"CER {b['cer']:.2%}, 拒识率 {b['rr']:.2%}, "
                      f"综合分 {b['score']:.4f}")
                return b

            if args.sx_checkpoint:
                # 阶段 D：SX 提取前端，4 种门控×ASR 消融配置一次跑完
                t_sx = time.time()
                samples = run_sx_artifacts(samples, rows, entry, args.split,
                                           data_root, model, kwargs, args)
                total_infer += time.time() - t_sx
                configs = [("raw-gate/raw-asr(混合v2基线)", "raw", "raw"),
                           ("sx-gate/raw-asr", "sx", "raw"),
                           ("raw-gate/sx-asr", "raw", "sx"),
                           ("sx-gate/sx-asr", "sx", "sx"),
                           ("comb-gate/sx-asr(联合门控)", "comb", "sx")]
                config_metrics, config_best = {}, {}
                for name, g, a in configs:
                    res = [compute_metrics(config_samples(samples, g, a), t)
                           for t in thresholds]
                    config_metrics[name] = res
                    config_best[name] = print_sweep(name, res)
                # 全局最优配置决定 best_threshold（明细 payload 用）
                best_name = max(config_best,
                                key=lambda k: config_best[k]["score"])
                best = config_best[best_name]
                print(f"\n[阶段D] 最优配置: {best_name}")
                metrics = config_metrics
                method = ("sc-hybrid-sx-refine" if refine
                          else "sc-hybrid-sx")
                best_threshold = best["threshold"]
            else:
                results = [compute_metrics(samples, t) for t in thresholds]
                best = print_sweep("混合系统", results)
                metrics = results
                method = "sc-hybrid-refine" if refine else "sc-hybrid"
                best_threshold = best["threshold"]
        else:
            best = compute_metrics(samples, SC_THRESHOLD)
            print(f"\n[SC] CER {best['cer']:.2%} | 拒识率 {best['rr']:.2%} "
                  f"| FAR {best['far']:.2%} | FRR {best['frr']:.2%} "
                  f"| 综合分 {best['score']:.4f}")
            metrics, method = [best], "sc-paraformer"
            best_threshold = SC_THRESHOLD

        total_audio = sum(s["duration"] for s in samples)
        print(f"[耗时] 总推理 {total_infer:.1f}s，音频总时长 {total_audio:.1f}s，"
              f"整体 RTF {total_infer/max(total_audio,1e-9):.3f}")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = args.output or os.path.join(
            entry["path"], ds.EVALS_DIR, f"eval_{stamp}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "method": method,
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
            "best_threshold": best_threshold,
            "metrics": metrics,
        }
        if args.sc_hybrid and args.sx_checkpoint:
            payload["sx_checkpoint"] = args.sx_checkpoint
            payload["sx_energy_gate"] = args.sx_energy_gate
            payload["best_config"] = best_name
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
                      best_threshold=best_threshold,
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
