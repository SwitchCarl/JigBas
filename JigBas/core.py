# ==============================================================
# JigBas Core — 模型加载与识别流水线
# 所有使用模型的功能集中在此模块，ui.py 仅负责界面显示
# ==============================================================

import os
import time
from contextlib import redirect_stdout

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
FUNASR_MODEL_DIR = os.path.join(MODELS_DIR, "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
FUNASR_MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

# 模型加载状态
STATUS_WAITING = "等待"
STATUS_LOADING = "加载中"
STATUS_READY = "就绪"
STATUS_FAILED = "失败"


class ModelHub:
    """集中管理声纹模型与 ASR 模型的加载与访问"""

    def __init__(self):
        self.wespeaker = None
        self.funasr = None
        self.status = {
            "wespeaker": STATUS_WAITING,
            "funasr": STATUS_WAITING,
        }

    def ready(self):
        return all(v == STATUS_READY for v in self.status.values())

    def load(self, log=print):
        """顺序加载全部模型；失败时未完成项标记为失败"""
        try:
            self._load_wespeaker(log)
            self._load_funasr(log)
        except Exception as e:
            for k, v in self.status.items():
                if v != STATUS_READY:
                    self.status[k] = STATUS_FAILED
            log(f"[模型] 加载失败: {e}")

    def _load_wespeaker(self, log):
        self.status["wespeaker"] = STATUS_LOADING
        log("[模型] 正在加载 Wespeaker 声纹模型...")
        import wespeaker
        with open(os.devnull, "w") as f, redirect_stdout(f):
            self.wespeaker = wespeaker.load_model("chinese")
        self.status["wespeaker"] = STATUS_READY
        log("[模型] Wespeaker 声纹模型加载完成")

    def _load_funasr(self, log):
        self.status["funasr"] = STATUS_LOADING
        log("[模型] 正在加载 FunASR ASR 模型...")
        from funasr import AutoModel
        if os.path.isdir(FUNASR_MODEL_DIR) and os.listdir(FUNASR_MODEL_DIR):
            self.funasr = AutoModel(model=FUNASR_MODEL_DIR)
        else:
            self.funasr = AutoModel(model=FUNASR_MODEL_ID)
        self.status["funasr"] = STATUS_READY
        log("[模型] FunASR ASR 模型加载完成")


# ---------------------------------------------------------------
# 声纹工具
# ---------------------------------------------------------------
def extract_embedding(hub, path):
    """提取声纹嵌入（屏蔽库自身的刷屏输出），返回 numpy 向量"""
    with open(os.devnull, "w") as f, redirect_stdout(f):
        emb = hub.wespeaker.extract_embedding(path)
    return emb.cpu().numpy() if hasattr(emb, "cpu") else emb


def cosine_similarity(a, b):
    """余弦相似度"""
    import numpy as np
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


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
