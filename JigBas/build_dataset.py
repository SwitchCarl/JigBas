# ==============================================================
# JigBas 数据集构建脚本（第 1 周：数据构建）
#
# 功能：
#   1. 扫描干净中文语料（按说话人分目录组织）与噪声库
#   2. 数据增强（核心混音基于 Lhotse）：
#      - 加噪：Cut.mix(snr=...)，SNR -5 ~ 5 dB
#      - 双人重叠：Cut.mix(snr=...)，重叠率 0 ~ 100%
#      - 变速：Recording.perturb_speed（Kaldi 风格，含变调）
#      - 混响：Recording.reverb_rir（真实 RIR 或合成 RIR）
#   3. 生成训练三元组 <唤醒音频, 识别音频, 目标文本>
#      - 正样本：唤醒与识别音频来自同一说话人，文本为识别标签
#      - 拒识样本：识别音频来自不同说话人，文本标注为空
#
# 语料目录约定（AISHELL / aidatatang 等均满足）：
#   clean_dir/
#     <speakerA>/xxx.wav [xxx.txt]     # 单句文本可放同名 .txt
#     <speakerB>/...
#     **/transcript*.txt               # 或转写文件: "utt_id 文 本 ..."
#   若 clean_dir 下找不到转写，会顺带查找 clean_dir 的上级目录
#   （兼容 AISHELL-1 的 wav/train + ../transcript 布局），也可用
#   --transcript 显式指定。
#
# 用法：
#   python build_dataset.py --alias baseline --num-train 2000 --num-dev 200
#   python build_dataset.py --stats --dataset latest
#   python build_dataset.py --progress   # 输出 [PROGRESS] 结构化行（供 UI 解析）
#
# 输出：<数据集根>/<创建时间>_<别名>/（含 metadata.json，构建后自动生成）
# ==============================================================

import argparse
import io
import json
import math
import os
import re
import sys
import time

import numpy as np
import soundfile as sf

import datasets as ds

SAMPLE_RATE = 16000

# 默认路径均为相对项目根（JigBas/JigBas）的相对路径，运行时经 ds.resolve_path 解析
DEFAULT_CLEAN_DIR = os.path.join("..", "Datasets", "data_aishell", "wav", "train")
DEFAULT_NOISE_DIR = os.path.join("..", "Datasets", "musan", "noise")
DEFAULT_RIR_DIR = os.path.join("..", "Datasets", "sim_rir_8k")
DEFAULT_TRANSCRIPT = os.path.join("..", "Datasets", "data_aishell",
                                  "transcript", "aishell_transcript_v0.8.txt")

_PROGRESS_ENABLED = False


def emit_progress(**kw):
    """--progress 开启时输出结构化进度行（供 UI 解析，普通日志不受影响）"""
    if _PROGRESS_ENABLED:
        print(f"[PROGRESS] {json.dumps(kw, ensure_ascii=False)}", flush=True)

# ---------------------------------------------------------------
# 基础音频工具
# ---------------------------------------------------------------
def load_audio(path, sr=SAMPLE_RATE):
    """读取音频并重采样到 16kHz 单声道，返回 float32 numpy"""
    data, orig_sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if orig_sr != sr:
        import librosa
        data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
    return data


def save_audio(path, data, sr=SAMPLE_RATE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, data, sr)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def peak_normalize(x, peak=0.9):
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (peak / m) if m > peak else x


def trim_silence(x, top_db=25):
    import librosa
    trimmed, _ = librosa.effects.trim(x, top_db=top_db)
    return trimmed if trimmed.size > 0 else x


# ---------------------------------------------------------------
# Lhotse 封装：内存 Recording / 混音 / 变速 / 混响
# ---------------------------------------------------------------
def rec_from_array(x, rid="mem"):
    """numpy 单声道数组 -> 内存中的 Lhotse Recording（16kHz）"""
    from lhotse import Recording
    buf = io.BytesIO()
    sf.write(buf, np.asarray(x, dtype=np.float32), SAMPLE_RATE,
             format="WAV", subtype="FLOAT")
    return Recording.from_bytes(buf.getvalue(), rid)


def rec_from_file(path):
    """音频文件 -> Lhotse Recording（重采样到 16kHz）"""
    from lhotse import Recording
    rec = Recording.from_file(path)
    if rec.sampling_rate != SAMPLE_RATE:
        rec = resample_rec(rec)
    return rec


def resample_rec(rec):
    return rec.resample(SAMPLE_RATE)


def mono_cut(rec):
    """Recording -> MonoCut（多声道时取第 0 声道转内存 Recording）"""
    if rec.num_channels == 1:
        return rec.to_cut()
    return rec_from_array(rec.load_audio()[0], rec.id + "_ch0").to_cut()


def lhotse_mix(target, other, snr_db, ref_id="t", oth_id="o"):
    """
    用 Lhotse Cut.mix 按 SNR/SIR 混合两条等长单声道音频。
    snr_db > 0 表示 target（参考）比 other 响 snr_db 分贝。
    返回 float32 numpy 数组。
    """
    t_cut = rec_from_array(target, ref_id).to_cut()
    o_cut = rec_from_array(other, oth_id).to_cut()
    mixed = t_cut.mix(o_cut, offset_other_by=0.0, snr=snr_db,
                      allow_padding=True)
    return mixed.load_audio()[0].astype(np.float32)


def lhotse_speed_perturb(x, rng, lo=0.9, hi=1.1):
    """变速扰动（Lhotse perturb_speed，Kaldi 风格，音调随速度变化）"""
    # 因子保留两位小数：保证 16000*factor 为整数且与 16000 有大公约数，
    # 否则 torchaudio 多相重采样滤波器会膨胀到 GB 级，直接撑爆内存/崩溃
    factor = round(float(rng.uniform(lo, hi)), 2)
    if abs(factor - 1.0) < 1e-3:
        return x, 1.0
    rec = rec_from_array(x, "sp").perturb_speed(factor, affix_id=False)
    return rec.load_audio()[0].astype(np.float32), round(factor, 3)


def make_synthetic_rir(rng, rt60=None):
    """无 RIR 库时合成指数衰减白噪声冲激响应"""
    rt60 = rt60 or rng.uniform(0.2, 0.6)
    length = int(SAMPLE_RATE * rt60)
    t = np.arange(length) / SAMPLE_RATE
    decay = np.exp(-6.91 * t / rt60)
    rir = rng.standard_normal(length).astype(np.float32) * decay
    return rir / (np.max(np.abs(rir)) + 1e-8)


def lhotse_reverb(x, rng, rir_paths=None):
    """加混响（Lhotse reverb_rir）：优先随机选 RIR 文件，否则合成"""
    if rir_paths and rng.random() < 0.7:
        rir_rec = rec_from_file(rng.choice(rir_paths))
        if rms(rir_rec.load_audio()) < 1e-6:
            rir_rec = rec_from_array(make_synthetic_rir(rng), "rir")
    else:
        rir_rec = rec_from_array(make_synthetic_rir(rng), "rir")
    wet = rec_from_array(x, "rev").reverb_rir(rir_rec, affix_id=False)
    return peak_normalize(wet.load_audio()[0].astype(np.float32), 0.95)


# ---------------------------------------------------------------
# 增强算子（基于 Lhotse）
# ---------------------------------------------------------------
def add_noise(x, noise, snr_db, rng):
    """按目标 SNR 叠加噪声（噪声过短时循环，过长时随机裁剪）"""
    if noise.size < x.size:
        reps = math.ceil(x.size / max(1, noise.size))
        noise = np.tile(noise, reps)
    start = rng.integers(0, noise.size - x.size + 1) if noise.size > x.size else 0
    noise = noise[start : start + x.size]
    if rms(noise) < 1e-6:
        return x
    return lhotse_mix(x, noise, snr_db)


def mix_overlap(target, interf, overlap, sir_db, rng):
    """
    双人重叠混音（Lhotse Cut.mix）。
    overlap ∈ (0,1] 为干扰语音与目标语音的重叠比例；
    sir_db = 20*log10(rms_target / rms_interf)。
    返回 (混合音频, 实际重叠率)
    """
    d_t = len(target)
    if len(interf) < d_t:
        interf = np.tile(interf, math.ceil(d_t / max(1, len(interf))))
    interf = interf[: max(d_t, len(interf))]

    if overlap >= 1.0:
        offset = 0
    else:
        max_shift = int(d_t * (1 - overlap))
        offset = int(rng.integers(-max_shift, max_shift + 1)) if max_shift > 0 else 0

    # Lhotse mix 不支持负偏移：将两条音频按各自起始位置补零对齐到同一时间轴
    off_t = max(0, -offset)
    off_i = max(0, offset)
    total = max(off_t + d_t, off_i + len(interf))
    ref = np.zeros(total, dtype=np.float32)
    ref[off_t : off_t + d_t] = target
    oth = np.zeros(total, dtype=np.float32)
    oth[off_i : off_i + len(interf)] = interf

    mix = lhotse_mix(ref, oth, sir_db)

    ov = (min(d_t, offset + len(interf)) - max(0, offset)) / d_t
    return peak_normalize(mix, 0.95), round(float(max(0.0, min(1.0, ov))), 3)


# ---------------------------------------------------------------
# 语料扫描与转写读取
# ---------------------------------------------------------------
def _parse_transcript_line(line, table):
    parts = line.strip().split(maxsplit=1)
    if len(parts) == 2 and re.match(r"^[\w\-.]+$", parts[0]):
        table[parts[0]] = parts[1].replace(" ", "")


def load_transcripts(clean_dir, transcript_file=None):
    """
    收集转写表：--transcript 指定文件 > clean_dir 递归 transcript*.txt/*.trn
    > clean_dir 上级目录（兼容 AISHELL-1 的 wav/train + ../transcript 布局）
    """
    table = {}

    def _load_file(path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                _parse_transcript_line(line, table)

    if transcript_file:
        _load_file(transcript_file)
        return table

    search_dirs = [clean_dir]
    # 向上追溯两级祖先目录（兼容 AISHELL-1 的 data_aishell/wav/train
    # + data_aishell/transcript 布局：转写在祖父目录的子目录里）
    d = os.path.normpath(clean_dir)
    for _ in range(2):
        d = os.path.dirname(d)
        if d and d not in search_dirs:
            search_dirs.append(d)
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                low = fn.lower()
                if ("transcript" in low and low.endswith(".txt")) or low.endswith(".trn"):
                    _load_file(os.path.join(root, fn))
    return table


def scan_corpus(clean_dir, transcript_table):
    """
    扫描干净语料 -> {speaker: [{"path", "text", "dur"}]}
    说话人 = 音频相对 clean_dir 的第一级目录名；
    文本优先级：全局转写表 > 同名 .txt > 空串
    """
    audio_ext = (".wav", ".flac", ".mp3")
    speakers = {}
    for root, _, files in os.walk(clean_dir):
        for fn in files:
            if not fn.lower().endswith(audio_ext):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, clean_dir)
            parts = rel.split(os.sep)
            if len(parts) < 2:
                continue  # 根目录下的音频无法归属说话人，跳过
            speaker = parts[0]
            stem = os.path.splitext(fn)[0]
            text = transcript_table.get(stem, "")
            sidecar = os.path.join(root, stem + ".txt")
            if not text and os.path.isfile(sidecar):
                with open(sidecar, encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip().replace(" ", "")
            try:
                dur = sf.info(path).duration
            except Exception:
                continue
            if dur < 0.5:
                continue  # 过短音频不可用
            speakers.setdefault(speaker, []).append(
                {"path": path, "text": text, "dur": dur})
    return speakers


def scan_audio_dir(d):
    out = []
    if not d or not os.path.isdir(d):
        return out
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith((".wav", ".flac", ".mp3")):
                out.append(os.path.join(root, fn))
    return out


# ---------------------------------------------------------------
# 三元组生成
# ---------------------------------------------------------------
def pick_wake_utterance(utts, rng):
    """唤醒音频偏好 1~3.5s 的短句；无合适句时取最短"""
    short = [u for u in utts if 1.0 <= u["dur"] <= 3.5]
    if short:
        return rng.choice(short)
    return min(utts, key=lambda u: u["dur"])


def pick_rec_utterance(utts, wake, rng):
    candidates = [u for u in utts if u["path"] != wake["path"]]
    return rng.choice(candidates) if candidates else wake


def augment_wake(x, rng, noise_paths, rir_paths):
    """唤醒音频做轻度信道扰动：语速 ±5% / 混响 / 较高 SNR 底噪"""
    if rng.random() < 0.3:
        x, _ = lhotse_speed_perturb(x, rng, 0.95, 1.05)
    if rng.random() < 0.3:
        x = lhotse_reverb(x, rng, rir_paths)
    if noise_paths and rng.random() < 0.5:
        noise = load_audio(rng.choice(noise_paths))
        x = add_noise(x, noise, rng.uniform(10, 25), rng)
    return peak_normalize(x, 0.95)


def augment_rec(x, rng, noise_paths, rir_paths):
    """
    识别音频做主增强：变速 / 混响 / 加噪（SNR -5~5dB）
    返回 (音频, meta)
    """
    meta = {"speed": 1.0, "reverb": False, "snr_db": None}
    if rng.random() < 0.3:
        x, meta["speed"] = lhotse_speed_perturb(x, rng)
    if rng.random() < 0.4:
        x = lhotse_reverb(x, rng, rir_paths)
        meta["reverb"] = True
    if noise_paths and rng.random() < 0.9:
        noise = load_audio(rng.choice(noise_paths))
        snr = round(rng.uniform(-5, 5), 1)
        x = add_noise(x, noise, snr, rng)
        meta["snr_db"] = snr
    return peak_normalize(x, 0.95), meta


def generate_sample(idx, split, kind, speakers, speaker_ids, rng,
                    noise_paths, rir_paths, out_dir, overlap_prob, trim):
    """生成单条三元组样本，写出 wav 并返回 manifest 记录"""
    target_spk = rng.choice(speaker_ids)
    wake_utt = pick_wake_utterance(speakers[target_spk], rng)

    if kind == "positive":
        rec_spk = target_spk
        rec_utt = pick_rec_utterance(speakers[rec_spk], wake_utt, rng)
        rec_text = rec_utt["text"]
    else:  # rejection：识别音频来自不同说话人，文本为空
        others = [s for s in speaker_ids if s != target_spk]
        rec_spk = rng.choice(others)
        rec_utt = rng.choice(speakers[rec_spk])
        rec_text = ""

    wake = load_audio(wake_utt["path"])
    rec = load_audio(rec_utt["path"])
    if trim:
        wake, rec = trim_silence(wake), trim_silence(rec)
        if wake.size < SAMPLE_RATE // 2 or rec.size < SAMPLE_RATE // 2:
            return None

    wake = augment_wake(wake, rng, noise_paths, rir_paths)
    rec, meta = augment_rec(rec, rng, noise_paths, rir_paths)

    # 双人重叠：干扰人随机选取（≠ 目标说话人）
    meta["overlap_ratio"] = 0.0
    meta["sir_db"] = None
    if rng.random() < overlap_prob:
        others = [s for s in speaker_ids if s != target_spk]
        if others:
            interf_spk = rng.choice(others)
            interf = load_audio(rng.choice(speakers[interf_spk])["path"])
            if trim:
                interf = trim_silence(interf)
            if interf.size >= SAMPLE_RATE // 2:
                sir = round(rng.uniform(-5, 5), 1)
                rec, ov = mix_overlap(rec, interf, rng.uniform(0.1, 1.0), sir, rng)
                meta["overlap_ratio"], meta["sir_db"] = ov, sir

    sid = f"{split}_{idx:06d}"
    wake_rel = os.path.join(split, "wake", sid + ".wav")
    rec_rel = os.path.join(split, "rec", sid + ".wav")
    save_audio(os.path.join(out_dir, wake_rel), wake)
    save_audio(os.path.join(out_dir, rec_rel), rec)

    return {
        "id": sid,
        "type": kind,                        # positive / rejection
        "wake_audio": wake_rel.replace(os.sep, "/"),
        "wake_text": wake_utt["text"],
        "rec_audio": rec_rel.replace(os.sep, "/"),
        "rec_text": rec_text,                # 拒识样本为空串
        "target_speaker": target_spk,
        "rec_speaker": rec_spk,
        "duration": round(len(rec) / SAMPLE_RATE, 2),
        **meta,
    }


# ---------------------------------------------------------------
# 数据集构建主流程
# ---------------------------------------------------------------
def build(args):
    # 脚本内混用了 numpy Generator 风格 API（integers/standard_normal），
    # 必须用 np.random.default_rng，不能用 random.Random
    rng = np.random.default_rng(args.seed)

    # 相对路径 → 绝对路径
    clean_dir = ds.resolve_path(args.clean_dir)
    noise_dir = ds.resolve_path(args.noise_dir)
    rir_dir = ds.resolve_path(args.rir_dir)
    transcript = ds.resolve_path(args.transcript) if args.transcript else None

    # 输出目录：--output 显式指定；默认 <数据集根>/<时间>_<别名>/
    if args.output:
        out_dir = ds.resolve_path(args.output)
        folder_name = os.path.basename(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir, folder_name = ds.new_dataset_folder(args.alias)
    args.output = out_dir  # generate_sample / stats 均通过 args.output 取路径

    print(f"[输出] 数据集目录: {out_dir}")
    emit_progress(phase="scan", done=0, total=0)

    print(f"[扫描] 干净语料: {clean_dir}")
    transcripts = load_transcripts(clean_dir, transcript)
    speakers = scan_corpus(clean_dir, transcripts)
    # 至少需要 2 个说话人才能构造拒识 / 重叠样本
    speakers = {s: u for s, u in speakers.items() if u}
    if len(speakers) < 2:
        print(f"[错误] 有效说话人不足（{len(speakers)} 个），"
              f"请检查 --clean-dir 是否按 <speaker>/<utt>.wav 组织")
        sys.exit(1)

    n_utts = sum(len(u) for u in speakers.values())
    n_hits = sum(1 for u in speakers.values() for x in u if x['text'])
    print(f"[扫描] 说话人 {len(speakers)} 个，有效音频 {n_utts} 条，"
          f"转写命中 {n_hits} 条")
    if n_hits == 0:
        print("[错误] 转写命中 0 条，生成的正样本将没有文本标注，数据集不可用。")
        print("       请用 --transcript 显式指定转写文件后重试。")
        sys.exit(1)

    noise_paths = scan_audio_dir(noise_dir)
    rir_paths = scan_audio_dir(rir_dir)
    print(f"[扫描] 噪声 {len(noise_paths)} 条，RIR {len(rir_paths)} 条")
    if not noise_paths:
        print("[警告] 未提供噪声库，将跳过加噪增强")

    # 按说话人划分 train / dev，保证验证集说话人不可见
    speaker_ids = sorted(speakers)
    rng.shuffle(speaker_ids)
    n_dev = max(1, int(len(speaker_ids) * args.dev_speaker_ratio))
    split_map = {
        "dev": speaker_ids[:n_dev],
        "train": speaker_ids[n_dev:] or speaker_ids[:1],
    }
    print(f"[划分] train 说话人 {len(split_map['train'])}，dev 说话人 {n_dev}")

    for split, count in (("train", args.num_train), ("dev", args.num_dev)):
        if count <= 0:
            continue
        spk_ids = split_map[split]
        manifest_path = os.path.join(out_dir, f"{split}_manifest.jsonl")
        made, idx = 0, 0
        with open(manifest_path, "w", encoding="utf-8") as mf:
            while made < count:
                kind = "rejection" if rng.random() < args.reject_ratio else "positive"
                try:
                    rec = generate_sample(
                        idx, split, kind, speakers, spk_ids, rng,
                        noise_paths, rir_paths, out_dir,
                        args.overlap_prob, args.trim)
                except Exception as e:
                    print(f"[跳过] {split} #{idx} 生成失败: {e}")
                    idx += 1
                    continue
                idx += 1
                if rec is None:
                    continue
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                made += 1
                if made % 100 == 0:
                    print(f"[{split}] {made}/{count}")
                    emit_progress(phase=split, done=made, total=count)
        print(f"[完成] {split}: {made} 条 -> {manifest_path}")

    # 自动生成元数据（条目数 / 时长 / SNR / 重叠分布 + 构建参数）
    splits = stats_dict(out_dir)
    meta = {
        "name": folder_name,
        "alias": args.alias,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "clean_dir": args.clean_dir, "transcript": args.transcript,
            "noise_dir": args.noise_dir, "rir_dir": args.rir_dir,
            "num_train": args.num_train, "num_dev": args.num_dev,
            "reject_ratio": args.reject_ratio,
            "overlap_prob": args.overlap_prob,
            "dev_speaker_ratio": args.dev_speaker_ratio,
            "trim": args.trim, "seed": args.seed,
        },
        "splits": splits,
    }
    ds.write_metadata(out_dir, meta)
    ds.update_latest(folder_name)
    emit_progress(phase="done", done=1, total=1, folder=folder_name)

    print(f"\n数据集已生成: {out_dir}")
    print(f"元数据: {os.path.join(out_dir, ds.METADATA_FILE)}")
    print_stats(out_dir)
    print("下一步：python evaluate.py --dataset", folder_name)


# ---------------------------------------------------------------
# 统计
# ---------------------------------------------------------------
def _split_stats(rows):
    pos = [r for r in rows if r["type"] == "positive"]
    ovl = [r for r in rows if r["overlap_ratio"] > 0]
    snr = [r["snr_db"] for r in rows if r["snr_db"] is not None]
    dur = sum(r["duration"] for r in rows)
    d = {
        "total": len(rows),
        "positive": len(pos),
        "rejection": len(rows) - len(pos),
        "duration_h": round(dur / 3600, 3),
        "avg_duration_s": round(dur / max(1, len(rows)), 1),
        "overlap_count": len(ovl),
        "overlap_pct": round(len(ovl) / max(1, len(rows)), 3),
        "missing_text": sum(1 for r in pos if not r["rec_text"]),
    }
    if snr:
        d["snr_min"] = min(snr)
        d["snr_max"] = max(snr)
        d["low_snr_pct"] = round(sum(1 for s in snr if s < 0) / len(snr), 3)
    return d


def stats_dict(folder):
    """读取 folder 下 train/dev manifest，返回 {split: 统计}（供元数据与打印共用）"""
    out = {}
    for split in ("train", "dev"):
        path = os.path.join(folder, f"{split}_manifest.jsonl")
        if not os.path.isfile(path):
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        out[split] = _split_stats(rows)
    return out


def print_stats(folder):
    for split, d in stats_dict(folder).items():
        print(f"== {split}: 共 {d['total']} 条 (正样本 {d['positive']}, "
              f"拒识 {d['rejection']})")
        print(f"   识别音频总时长 {d['duration_h']} h, 平均 {d['avg_duration_s']} s")
        print(f"   含重叠语音 {d['overlap_count']} 条 ({d['overlap_pct']:.0%})")
        if "snr_min" in d:
            print(f"   SNR 范围 [{d['snr_min']}, {d['snr_max']}] dB, "
                  f"低信噪比(<0dB)占比 {d['low_snr_pct']:.0%}")
        if d["missing_text"]:
            print(f"   [注意] {d['missing_text']} 条正样本缺少转写文本")


def stats(args):
    entry = ds.resolve_dataset(args.dataset)
    print(f"[统计] 数据集: {entry['path']}")
    print_stats(entry["path"])


# ---------------------------------------------------------------
# 入口
# ---------------------------------------------------------------
def main():
    global _PROGRESS_ENABLED
    ap = argparse.ArgumentParser(description="JigBas 三元组数据集构建（Lhotse 混音）")
    ap.add_argument("--alias", default="run", help="数据集别名（目录名 <时间>_<别名>）")
    ap.add_argument("--clean-dir", default=DEFAULT_CLEAN_DIR,
                    help="干净语料根目录（按说话人分目录，相对项目根）")
    ap.add_argument("--transcript", default=DEFAULT_TRANSCRIPT,
                    help="全局转写文件（utt_id 文本），留空自动探测")
    ap.add_argument("--noise-dir", default=DEFAULT_NOISE_DIR, help="噪声库目录")
    ap.add_argument("--rir-dir", default=DEFAULT_RIR_DIR, help="房间冲激响应目录")
    ap.add_argument("--output", default=None,
                    help="显式指定完整输出目录（默认自动生成 <时间>_<别名>）")
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-dev", type=int, default=200)
    ap.add_argument("--reject-ratio", type=float, default=0.3,
                    help="拒识样本占比（默认 0.3）")
    ap.add_argument("--overlap-prob", type=float, default=0.4,
                    help="识别音频叠加双人重叠的概率（默认 0.4）")
    ap.add_argument("--dev-speaker-ratio", type=float, default=0.1,
                    help="dev 集说话人占比，按说话人划分（默认 0.1）")
    ap.add_argument("--no-trim", dest="trim", action="store_false",
                    help="不裁剪首尾静音（默认裁剪）")
    ap.set_defaults(trim=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress", action="store_true",
                    help="输出 [PROGRESS] 结构化进度行（供 UI 解析）")
    ap.add_argument("--stats", action="store_true", help="仅统计数据集")
    ap.add_argument("--dataset", default="latest",
                    help="--stats 的目标数据集（latest / 别名 / 时间前缀）")
    args = ap.parse_args()
    _PROGRESS_ENABLED = args.progress

    if args.stats:
        stats(args)
        return
    build(args)


if __name__ == "__main__":
    main()
