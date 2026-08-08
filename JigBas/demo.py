# ==============================================================
# JigBas 基础演示 — 单条语音的目标说话人识别
#
# 流水线：唤醒音频声纹比对 → 余弦相似度过阈值才 ASR 转写
# 模型加载见 models.py；UI「基础演示」模块与此脚本等价：
#   python demo.py --wake wake.wav --rec rec.wav --threshold 0.30
# 不带参数时进入交互式问答（供 JigBas.py 菜单调用）。
# ==============================================================

import argparse
import os
import sys
import time

from models import ModelHub, extract_embedding, cosine_similarity

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_THRESHOLD = 0.30  # 第 1 周基线最优阈值（见 Records 交接文档）


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
# 结果打印（控制台入口与交互模式共用）
# ---------------------------------------------------------------
def print_result(result, threshold):
    if result["error"]:
        print(f"[结果] 识别失败: {result['error']}")
        return
    sim = result["similarity"]
    print(f"[结果] 声纹相似度: {sim:.4f}  阈值: {threshold:.2f}")
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
    """交互模式：循环询问音频对，供 JigBas.py 菜单使用"""
    hub = ModelHub()
    hub.load()
    if not hub.ready():
        print("[演示] 模型加载失败，退出")
        return 1
    while True:
        print()
        try:
            wake = _ask("唤醒音频路径（回车退出）")
            if not wake:
                break
            rec = _ask("识别音频路径")
            th = float(_ask("拒识阈值", str(DEFAULT_THRESHOLD)))
        except (EOFError, KeyboardInterrupt):
            print("\n已退出演示。")
            break
        if not os.path.isfile(wake) or not os.path.isfile(rec):
            print("[演示] 文件不存在，请重试")
            continue
        print_result(recognize(hub, wake, rec, th), th)
    return 0


def main():
    ap = argparse.ArgumentParser(description="JigBas 基础演示：单条目标说话人识别")
    ap.add_argument("--wake", help="唤醒音频路径")
    ap.add_argument("--rec", help="识别音频路径")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"拒识阈值（默认 {DEFAULT_THRESHOLD}）")
    ap.add_argument("--device", default=None, help="cpu / cuda:0（默认自动）")
    args = ap.parse_args()

    if not args.wake or not args.rec:
        return interactive()

    hub = ModelHub(device=args.device)
    hub.load()
    if not hub.ready():
        print("[演示] 模型加载失败，退出")
        return 1
    print_result(recognize(hub, args.wake, args.rec, args.threshold),
                 args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
