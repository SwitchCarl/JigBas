# ==============================================================
# JigBas 基础演示 — 单条语音的目标说话人识别
#
# 三种模式（--mode 选择）：
#   baseline（旧）：唤醒音频声纹比对 → 余弦相似度过阈值才 Paraformer ASR
#   sc（新，第二周）：两级声纹门控（灰区滑窗分段精判）+ SC-Paraformer 识别
#   sx（最终，第四周）：SX 提取目标人声 → 提取波形上声纹门控 + SC-scx 识别
# 用法：
#   python demo.py --wake wake.wav --rec rec.wav --threshold 0.30
#   python demo.py --mode sc --wake wake.wav --rec rec.wav
#   python demo.py --mode sx --wake wake.wav --rec rec.wav
# 不带参数时进入交互式问答（供 JigBas.py 菜单调用），每轮可选模式。
# UI「基础演示」模块走 demo.recognize / recognize_sc / recognize_sx 三通路。
# ==============================================================

import argparse
import os
import sys
import time

# 直接运行本脚本时把项目根加入 sys.path（python app/demo.py）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.models import (ModelHub, extract_embedding, extract_embedding_pcm,
                        cosine_similarity)
from lib.paths import REPO_ROOT

DEFAULT_THRESHOLD = 0.30     # 第 1 周基线最优阈值（见 Records 交接文档）
SC_DEFAULT_THRESHOLD = 0.32  # SC 混合系统最优阈值（第二周门控消融，见 8.10 记录）
SC_GRAY_ZONE = (0.05, 0.60)  # 两级门控灰区：落灰区则滑窗分段精判
# SC 混合系统的默认 ASR 权重（SC-v2，train8k 10 轮）；可用 --sc-checkpoint 覆盖
DEFAULT_SC_CKPT = os.path.normpath(os.path.join(
    REPO_ROOT, "..", "..", "Temp", "Datasets", "20260810_0045_train8k",
    "checkpoints", "sc_20260810_073412", "step_10000.pt"))

# 最终系统（第 4 周定版，综合分 0.7813）默认权重与阈值；可用
# --sx-checkpoint / --sc-checkpoint 覆盖。详见 JigBas全项目详解.md §12.3。
SX_DEFAULT_THRESHOLD = 0.30   # 最终系统最优阈值（sim_sx >= 0.30 接受）
# SX 提取器（阶段 C 纯分离模型，未微调）
DEFAULT_SX_CKPT = os.path.normpath(os.path.join(
    REPO_ROOT, "..", "..", "Temp", "Datasets", "20260811_1914_sxtrain",
    "checkpoints", "sx_20260812_235228", "step_6000.pt"))
# SC-scx ASR（在提取音频上微调，绝对 8000 步为过拟合前峰值）
DEFAULT_SCX_CKPT = os.path.normpath(os.path.join(
    REPO_ROOT, "..", "..", "Temp", "Datasets", "20260813_0105_scx8k",
    "checkpoints", "sc_20260813_cont4000", "step_4000.pt"))


# ---------------------------------------------------------------
# 识别流水线：唤醒音频声纹比对 → 目标说话人才转写
# ---------------------------------------------------------------
def recognize(hub, wake_path, rec_path, threshold, on_stage=None, log=print):
    """
    完整识别流程：
      1. 提取唤醒音频声纹嵌入（on_stage("wake")）
      2. 提取识别音频声纹嵌入（on_stage("rec")）
      3. 余弦相似度 >= 阈值 → ASR 转写（on_stage("asr")），否则拒识

    返回 dict:
      similarity: float | None   声纹余弦相似度
      accepted:   bool           是否判为目标说话人
      text:       str | None     转写文本（拒识或失败时为 None）
      error:      str | None     错误信息（无错误为 None）
      elapsed:    float          总耗时（秒）
    """
    t0 = time.time()
    result = {"similarity": None, "accepted": False, "text": None,
              "error": None, "elapsed": 0.0}

    def stage(name):
        if on_stage:
            on_stage(name)

    try:
        log(f"[识别] 唤醒音频: {wake_path}")
        log(f"[识别] 识别音频: {rec_path}")

        stage("wake")
        log("[识别] 提取唤醒音频声纹...")
        emb_wake = extract_embedding(hub, wake_path)

        stage("rec")
        log("[识别] 提取识别音频声纹...")
        emb_rec = extract_embedding(hub, rec_path)

        sim = cosine_similarity(emb_wake, emb_rec)
        result["similarity"] = sim
        log(f"[识别] 余弦相似度: {sim:.4f} (阈值 {threshold:.2f})")

        if sim >= threshold:
            result["accepted"] = True
            stage("asr")
            log("[识别] 判为目标说话人，执行 ASR 转写...")
            asr_result = hub.funasr.generate(input=rec_path)
            text = ""
            if asr_result and len(asr_result) > 0:
                text = (asr_result[0].get("text", "")
                        if isinstance(asr_result[0], dict) else str(asr_result[0]))
            result["text"] = text
        else:
            log("[识别] 判为非目标说话人，拒识（输出空）")
    except Exception as e:
        result["error"] = str(e)
        log(f"[识别] 失败: {e}")

    result["elapsed"] = time.time() - t0
    log(f"[识别] 完成，总耗时 {result['elapsed']:.2f}s")
    return result


# ---------------------------------------------------------------
# SC 模式（第二周混合系统）：两级门控 + SC-Paraformer 识别
# ---------------------------------------------------------------
def recognize_sc(hub, sc_model, sc_kwargs, wake_path, rec_path, threshold,
                 gray=SC_GRAY_ZONE, on_stage=None, log=print):
    """
    SC 混合识别流程（与 evaluate.py --sc-hybrid --gate-refine 同口径）：
      1. 提取唤醒/识别音频声纹嵌入（on_stage("wake"/"rec")）
      2. 整段 sim 落灰区 → 滑窗分段精判，sim = 0.7*max(窗)+0.3*整段
      3. sim >= 阈值 → SC-Paraformer 贪心解码（on_stage("asr")），否则拒识
    返回 dict 字段与 recognize() 一致，另加 refined（是否触发分段精判）。
    """
    import numpy as np
    import soundfile as sf
    import torch
    from app.evaluate import segment_sims
    from lib.sc_model import sc_greedy_decode

    t0 = time.time()
    result = {"similarity": None, "accepted": False, "text": None,
              "error": None, "elapsed": 0.0, "refined": False}

    def stage(name):
        if on_stage:
            on_stage(name)

    try:
        log(f"[识别] 唤醒音频: {wake_path}")
        log(f"[识别] 识别音频: {rec_path}")

        stage("wake")
        log("[识别] 提取唤醒音频声纹...")
        emb_wake = extract_embedding(hub, wake_path)

        stage("rec")
        log("[识别] 提取识别音频声纹...")
        emb_rec = extract_embedding(hub, rec_path)

        sim = cosine_similarity(emb_wake, emb_rec)
        if gray and gray[0] < sim < gray[1]:
            log(f"[识别] 整段相似度 {sim:.4f} 落入灰区 {gray}，分段精判...")
            segs = segment_sims(hub, rec_path, emb_wake)
            if segs:
                sim = 0.7 * max(segs) + 0.3 * sim
                result["refined"] = True
        result["similarity"] = sim
        log(f"[识别] 余弦相似度: {sim:.4f} (阈值 {threshold:.2f})")

        if sim >= threshold:
            result["accepted"] = True
            stage("asr")
            log("[识别] 判为目标说话人，SC-Paraformer 转写...")
            pcm, sr = sf.read(rec_path, dtype="float32")
            assert sr == 16000, f"采样率 {sr} != 16000"
            batch = {
                "speech": torch.from_numpy(pcm).unsqueeze(0),
                "speech_lengths": torch.tensor([len(pcm)]),
                "spk_emb": torch.from_numpy(
                    np.asarray(emb_wake, dtype="float32").ravel()).unsqueeze(0),
            }
            device = next(sc_model.parameters()).device
            result["text"] = sc_greedy_decode(
                sc_model, sc_kwargs["frontend"], sc_kwargs["tokenizer"],
                batch, device)[0]
        else:
            log("[识别] 判为非目标说话人，拒识（输出空）")
    except Exception as e:
        result["error"] = str(e)
        log(f"[识别] 失败: {e}")

    result["elapsed"] = time.time() - t0
    log(f"[识别] 完成，总耗时 {result['elapsed']:.2f}s")
    return result


def load_sc_engine(device=None, sc_checkpoint=None, log=print):
    """加载 SC 混合系统的 ASR 侧（门控侧 wespeaker 由 hub 提供）"""
    from lib.sc_model import build_sc_model
    ckpt = sc_checkpoint or DEFAULT_SC_CKPT
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"SC checkpoint 不存在: {ckpt}\n请用 --sc-checkpoint 指定")
    device = device or ("cuda:0"
                        if __import__("torch").cuda.is_available() else "cpu")
    log(f"[识别] 加载 SC-Paraformer（{device}）: {os.path.basename(ckpt)}")
    return build_sc_model(device=device, sc_checkpoint=ckpt, log=log)


# ---------------------------------------------------------------
# sx 模式（第四周最终系统）：SX 提取 + 声纹门控 + SC-scx 识别
# ---------------------------------------------------------------
def load_final_engine(device=None, sx_checkpoint=None, sc_checkpoint=None,
                      log=print):
    """加载最终系统的引擎侧：SX 提取器 + SC-scx ASR。
    返回 (sx_model, sc_model, sc_kwargs)，门控侧 wespeaker 由 hub 提供。
    缺任一 checkpoint 抛 FileNotFoundError（供 UI 显示加载失败）。"""
    from lib.sx_model import sx_load
    from lib.sc_model import build_sc_model
    sx_ckpt = sx_checkpoint or DEFAULT_SX_CKPT
    sc_ckpt = sc_checkpoint or DEFAULT_SCX_CKPT
    if not os.path.isfile(sx_ckpt) or not os.path.isfile(sc_ckpt):
        raise FileNotFoundError(
            f"最终系统权重缺失：\n  SX   = {sx_ckpt}\n  SC-scx = {sc_ckpt}\n"
            f"请用 --sx-checkpoint / --sc-checkpoint 指定")
    device = device or ("cuda:0"
                        if __import__("torch").cuda.is_available() else "cpu")
    log(f"[识别] 加载 SX 提取器（{device}）: {os.path.basename(sx_ckpt)}")
    sx = sx_load(sx_ckpt, device=device, log=log)
    log(f"[识别] 加载 SC-scx ASR（{device}）: {os.path.basename(sc_ckpt)}")
    sc_model, sc_kwargs = build_sc_model(device=device,
                                         sc_checkpoint=sc_ckpt, log=log)
    return sx, sc_model, sc_kwargs


def recognize_sx(hub, sx_model, sc_model, sc_kwargs, wake_path, rec_path,
                 threshold, gray=SC_GRAY_ZONE, on_stage=None, log=print):
    """
    最终系统识别流程（与 evaluate.py --final 的 sx-gate/sx-asr 同口径）：
      1. 提取唤醒音频声纹嵌入（on_stage("wake")）
      2. SX 提取器把目标人声从 rec 中抽出（on_stage("sx")）
      3. 在提取波形上算 sim_sx；落灰区 → 滑窗分段精判，sim = 0.7*max(窗)+0.3*整段
      4. sim >= 阈值 → SC-scx 在提取波形上贪心解码（on_stage("asr")），否则拒识
    返回 dict 字段与 recognize() 一致，另加 refined（是否触发分段精判）。
    """
    import numpy as np
    import soundfile as sf
    import torch
    from app.evaluate import segment_sims_pcm
    from lib.sc_model import sc_greedy_decode

    t0 = time.time()
    result = {"similarity": None, "accepted": False, "text": None,
              "error": None, "elapsed": 0.0, "refined": False}

    def stage(name):
        if on_stage:
            on_stage(name)

    try:
        log(f"[识别] 唤醒音频: {wake_path}")
        log(f"[识别] 识别音频: {rec_path}")

        stage("wake")
        log("[识别] 提取唤醒音频声纹...")
        emb_wake = extract_embedding(hub, wake_path)

        stage("sx")
        log("[识别] SX 提取目标人声...")
        pcm, sr = sf.read(rec_path, dtype="float32")
        assert sr == 16000, f"采样率 {sr} != 16000"
        device = next(sc_model.parameters()).device
        with torch.no_grad():
            out = sx_model.separate(
                torch.from_numpy(pcm).unsqueeze(0).to(device),
                torch.from_numpy(np.asarray(emb_wake, dtype="float32").ravel())
                .unsqueeze(0).to(device))
        sx_pcm = out.squeeze(0).cpu().numpy().astype("float32")

        sim = cosine_similarity(emb_wake, extract_embedding_pcm(hub, sx_pcm))
        if gray and gray[0] < sim < gray[1]:
            log(f"[识别] 整段相似度 {sim:.4f} 落入灰区 {gray}，分段精判...")
            segs = segment_sims_pcm(hub, sx_pcm, emb_wake)
            if segs:
                sim = 0.7 * max(segs) + 0.3 * sim
                result["refined"] = True
        result["similarity"] = sim
        log(f"[识别] 余弦相似度: {sim:.4f} (阈值 {threshold:.2f})")

        if sim >= threshold:
            result["accepted"] = True
            stage("asr")
            log("[识别] 判为目标说话人，SC-scx 在提取音频上转写...")
            batch = {
                "speech": torch.from_numpy(sx_pcm).unsqueeze(0),
                "speech_lengths": torch.tensor([len(sx_pcm)]),
                "spk_emb": torch.from_numpy(
                    np.asarray(emb_wake, dtype="float32").ravel()).unsqueeze(0),
            }
            result["text"] = sc_greedy_decode(
                sc_model, sc_kwargs["frontend"], sc_kwargs["tokenizer"],
                batch, device)[0]
        else:
            log("[识别] 判为非目标说话人，拒识（输出空）")
    except Exception as e:
        result["error"] = str(e)
        log(f"[识别] 失败: {e}")

    result["elapsed"] = time.time() - t0
    log(f"[识别] 完成，总耗时 {result['elapsed']:.2f}s")
    return result


# ---------------------------------------------------------------
# 结果打印（控制台入口与交互模式共用）
# ---------------------------------------------------------------
def print_result(result, threshold, mode="baseline"):
    if result["error"]:
        print(f"[结果] 识别失败: {result['error']}")
        return
    sim = result["similarity"]
    tag = {"sc": "SC 混合", "sx": "最终系统"}.get(mode, "基线")
    extra = "（分段精判后）" if result.get("refined") else ""
    print(f"[结果] 模式: {tag}  声纹相似度: {sim:.4f}{extra}  阈值: {threshold:.2f}")
    if result["accepted"]:
        print(f"[结果] 判决: 目标说话人")
        print(f"[结果] 转写: {result['text'] or '（无内容）'}")
    else:
        print("[结果] 判决: 非目标说话人 — 已拒识（输出空）")
    print(f"[结果] 耗时: {result['elapsed']:.2f}s")


# ---------------------------------------------------------------
# 入口
# ---------------------------------------------------------------
def _ask(prompt, default=""):
    s = input(f"  {prompt} [{default}] > ").strip().strip('"').strip("'")
    return s or default


def interactive():
    """交互模式：循环询问音频对与模式，供 JigBas.py 菜单使用。
    wespeaker 两种模式都要用，启动即加载；baseline 的 funasr 与
    SC 模型各自在首次用到时才加载（避免双份 Paraformer 常驻）。"""
    hub = ModelHub()
    hub._load_wespeaker(print)
    sc_engine = None   # (model, kwargs)，sc 模式首次使用时加载
    sx_engine = None   # (sx, sc_model, sc_kwargs)，sx 模式首次使用时加载
    while True:
        print()
        try:
            mode = _ask("模式 baseline/sc/sx", "sc").lower()
            if mode not in ("baseline", "sc", "sx"):
                print("[演示] 模式仅支持 baseline / sc / sx")
                continue
            wake = _ask("唤醒音频路径（回车退出）")
            if not wake:
                break
            rec = _ask("识别音频路径")
            default_th = {"sc": SC_DEFAULT_THRESHOLD,
                          "sx": SX_DEFAULT_THRESHOLD}.get(mode,
                                                          DEFAULT_THRESHOLD)
            th = float(_ask("拒识阈值", str(default_th)))
        except (EOFError, KeyboardInterrupt):
            print("\n已退出演示。")
            break
        if not os.path.isfile(wake) or not os.path.isfile(rec):
            print("[演示] 文件不存在，请重试")
            continue
        if mode == "baseline":
            if hub.funasr is None:
                hub._load_funasr(print)
            print_result(recognize(hub, wake, rec, th), th, mode)
        elif mode == "sc":
            if sc_engine is None:
                try:
                    sc_engine = load_sc_engine()
                except Exception as e:
                    print(f"[演示] SC 模型加载失败: {e}")
                    continue
            print_result(
                recognize_sc(hub, sc_engine[0], sc_engine[1], wake, rec, th),
                th, mode)
        else:  # sx：最终系统
            if sx_engine is None:
                try:
                    sx_engine = load_final_engine()
                except Exception as e:
                    print(f"[演示] 最终系统加载失败: {e}")
                    continue
            print_result(
                recognize_sx(hub, sx_engine[0], sx_engine[1], sx_engine[2],
                             wake, rec, th),
                th, mode)
    return 0


def main():
    ap = argparse.ArgumentParser(description="JigBas 基础演示：单条目标说话人识别")
    ap.add_argument("--wake", help="唤醒音频路径")
    ap.add_argument("--rec", help="识别音频路径")
    ap.add_argument("--mode", choices=["baseline", "sc", "sx"],
                    default="baseline",
                    help="baseline=旧基线（声纹+Paraformer）；"
                         "sc=混合系统（两级门控+SC-Paraformer）；"
                         "sx=最终系统（SX提取+声纹门控+SC-scx）")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"拒识阈值（默认 baseline {DEFAULT_THRESHOLD} / "
                         f"sc {SC_DEFAULT_THRESHOLD} / sx {SX_DEFAULT_THRESHOLD}）")
    ap.add_argument("--sc-checkpoint", default=None,
                    help="SC/sx 模式 ASR 权重（sc 默认 train8k v2，"
                         "sx 默认 SC-scx 最终权重）")
    ap.add_argument("--sx-checkpoint", default=None,
                    help="sx 模式 SX 提取器权重（默认最终系统 SX）")
    ap.add_argument("--gate-refine", default="0.05:0.60",
                    help="SC/sx 模式两级门控灰区 LO:HI；off 关闭分段精判")
    ap.add_argument("--device", default=None, help="cpu / cuda:0（默认自动）")
    args = ap.parse_args()

    if not args.wake or not args.rec:
        return interactive()

    hub = ModelHub(device=args.device)
    if args.mode == "baseline":
        hub.load()
        if not hub.ready():
            print("[演示] 模型加载失败，退出")
            return 1
        th = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
        print_result(recognize(hub, args.wake, args.rec, th), th, args.mode)
    elif args.mode == "sc":
        hub._load_wespeaker(print)
        try:
            model, kwargs = load_sc_engine(device=args.device,
                                           sc_checkpoint=args.sc_checkpoint)
        except Exception as e:
            print(f"[演示] SC 模型加载失败: {e}")
            return 1
        th = (args.threshold if args.threshold is not None
              else SC_DEFAULT_THRESHOLD)
        gray = (None if args.gate_refine == "off"
                else tuple(float(x) for x in args.gate_refine.split(":")))
        print_result(
            recognize_sc(hub, model, kwargs, args.wake, args.rec, th,
                         gray=gray),
            th, args.mode)
    else:  # sx：最终系统
        hub._load_wespeaker(print)
        try:
            sx, model, kwargs = load_final_engine(
                device=args.device, sx_checkpoint=args.sx_checkpoint,
                sc_checkpoint=args.sc_checkpoint)
        except Exception as e:
            print(f"[演示] 最终系统加载失败: {e}")
            return 1
        th = (args.threshold if args.threshold is not None
              else SX_DEFAULT_THRESHOLD)
        gray = (None if args.gate_refine == "off"
                else tuple(float(x) for x in args.gate_refine.split(":")))
        print_result(
            recognize_sx(hub, sx, model, kwargs, args.wake, args.rec, th,
                         gray=gray),
            th, args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
