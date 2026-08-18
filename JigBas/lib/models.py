# ==============================================================
# JigBas Models — 模型加载与声纹工具
# （原 core.py；单条识别流水线已独立为 demo.py）
# 所有模型加载集中在此模块，ui.py / demo.py / evaluate.py 共用
# ==============================================================

import os
from contextlib import redirect_stdout

from lib.paths import FUNASR_MODEL_DIR

FUNASR_MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"


def _patch_torch_jit_load():
    """兼容 torch.jit.load 的中文路径问题（归档迁移到 E:\归档 等含中文路径时必需）。

    背景：torch.jit.load 底层用 C 的 fopen 打开文件，Windows 上 ANSI 编码的
    fopen 无法打开含中文的路径；torch.load 则无此问题。项目代码全部用
    torch.load / 不受影响，只有第三方 silero_vad（经 wespeaker）用
    torch.jit.load 加载 silero_vad.jit，归档到中文路径后即失败。

    补丁逻辑：探测到路径含非 ASCII 字符时，先复制到系统临时目录
    （纯 ASCII）再加载，加载完即删除副本。幂等，可安全重复调用。
    """
    try:
        import torch
    except ImportError:
        return  # torch 未安装，无需打补丁

    if getattr(torch.jit, "_jigbas_cn_path_patched", False):
        return  # 已打过补丁

    _orig = torch.jit.load

    def _load(path, *args, **kwargs):
        p = str(path)
        try:
            p.encode("ascii")
            return _orig(path, *args, **kwargs)
        except UnicodeEncodeError:
            import shutil
            import tempfile
            tmp = os.path.join(tempfile.gettempdir(), os.path.basename(p))
            shutil.copy2(p, tmp)
            try:
                return _orig(tmp, *args, **kwargs)
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    torch.jit.load = _load
    torch.jit._jigbas_cn_path_patched = True


# 模块加载即打补丁（本模块是所有模型加载的集中入口）
_patch_torch_jit_load()

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
