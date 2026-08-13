# ==============================================================
# JigBas Models — 模型加载与声纹工具
# （原 core.py；单条识别流水线已独立为 demo.py）
# 所有模型加载集中在此模块，ui.py / demo.py / evaluate.py 共用
# ==============================================================

import os
from contextlib import redirect_stdout

from lib.paths import FUNASR_MODEL_DIR

FUNASR_MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

# 模型加载状态
STATUS_WAITING = "等待"
STATUS_LOADING = "加载中"
STATUS_READY = "就绪"
STATUS_FAILED = "失败"


def _default_device():
    """有 CUDA 用 GPU，否则回退 CPU"""
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class ModelHub:
    """集中管理声纹模型与 ASR 模型的加载与访问"""

    def __init__(self, device=None, wespeaker_device="cpu"):
        # wespeaker 固定 CPU：其 fbank 前端在库内不随 set_device 迁移，
        # 放 GPU 会设备不匹配崩溃；且 CPU 提取嵌入仅 ~0.4s/条，收益微小
        self.device = device or _default_device()
        self.wespeaker_device = wespeaker_device
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
        log(f"[模型] 正在加载 Wespeaker 声纹模型（{self.wespeaker_device}）...")
        import wespeaker
        with open(os.devnull, "w") as f, redirect_stdout(f):
            self.wespeaker = wespeaker.load_model("chinese")
        self.wespeaker.set_device(self.wespeaker_device)
        self.status["wespeaker"] = STATUS_READY
        log("[模型] Wespeaker 声纹模型加载完成")

    def _load_funasr(self, log):
        self.status["funasr"] = STATUS_LOADING
        log(f"[模型] 正在加载 FunASR ASR 模型（{self.device}）...")
        from funasr import AutoModel
        kwargs = {"device": self.device, "disable_update": True}
        if os.path.isdir(FUNASR_MODEL_DIR) and os.listdir(FUNASR_MODEL_DIR):
            self.funasr = AutoModel(model=FUNASR_MODEL_DIR, **kwargs)
        else:
            self.funasr = AutoModel(model=FUNASR_MODEL_ID, **kwargs)
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


def extract_embedding_pcm(hub, pcm, sample_rate=16000):
    """从 float32 单声道波形（numpy 数组）提取声纹嵌入（分段精判用）"""
    import torch
    with open(os.devnull, "w") as f, redirect_stdout(f):
        emb = hub.wespeaker.extract_embedding_from_pcm(
            torch.from_numpy(pcm).unsqueeze(0), sample_rate)
    return emb.cpu().numpy() if hasattr(emb, "cpu") else emb


def cosine_similarity(a, b):
    """余弦相似度"""
    import numpy as np
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
