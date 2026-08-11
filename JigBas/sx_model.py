# ==============================================================
# SX-Extractor 重叠目标提取前端（第三周 阶段 C）
#
# VoiceFilter 式掩码网络：
#   混合波形 STFT(n_fft=512, win=400 25ms, hop=160 10ms) → 幅度 (B,257,T)
#   wake 声纹嵌入（256 维 wespeaker ECAPA，复用 SC 的 spk_emb 缓存）
#     → Linear(256→257) 投影，逐帧与 log1p(幅度) 拼接（514 维）
#   → 2×BLSTM(400) → Linear → sigmoid 时频掩码
#   → 掩码 × 混合幅度；混合相位 ISTFT 回波形（推理侧 separate()）
#
# 训练监督（build_dataset --save-clean 生成的 clean_audio）：
#   正样本有重叠 → 与混合波形逐点对齐的干净目标轨
#                  （重叠前的目标音轨，保留噪声/混响——模型只学分人）
#   正样本无重叠 → rec_audio 本身（生成时不重复写盘，manifest 复用路径）
#   拒识样本     → 全零（监督掩码输出 0；推理时输出近零能量，
#                  能量门控白送拒识信号——阶段 D 接入）
# 损失：线性幅度 L1（帧掩码按样本长度剔除补零帧）；SI-SDR 仅作监控指标。
#
# 实现备注：网络输入的幅度经 log1p 压缩（数值稳定，原始幅度动态范围
# 两个数量级以上）；掩码与损失都在线性幅度域（与"幅度 L1"设计一致）。
# ==============================================================

import torch
from torch import nn

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400          # 25ms
HOP_LENGTH = 160          # 10ms
N_FREQ = N_FFT // 2 + 1   # 257
SPK_EMB_DIM = 256         # wespeaker ECAPA-TDNN 嵌入维度（与 sc_model 一致）

# 默认模型配置（checkpoint 内保存，sx_load 重建用）
DEFAULT_CONFIG = {
    "emb_dim": SPK_EMB_DIM,
    "n_freq": N_FREQ,
    "lstm_hidden": 400,
    "lstm_layers": 2,
}


class SXExtractor(nn.Module):
    """目标说话人提取网络（掩码式 VoiceFilter）"""

    def __init__(self, emb_dim=SPK_EMB_DIM, n_freq=N_FREQ,
                 lstm_hidden=400, lstm_layers=2):
        super().__init__()
        self.n_freq = n_freq
        # persistent=False：state_dict 不存窗函数，与旧 checkpoint 兼容
        self.register_buffer("window", torch.hann_window(WIN_LENGTH),
                             persistent=False)
        self.emb_proj = nn.Linear(emb_dim, n_freq)
        self.blstm = nn.LSTM(n_freq * 2, lstm_hidden, lstm_layers,
                             batch_first=True, bidirectional=True)
        self.mask_head = nn.Linear(lstm_hidden * 2, n_freq)

    # ---- STFT / ISTFT ----
    def stft(self, wav):
        """(B,L) float32 → 复数谱 (B,F,T)，T = L//hop + 1"""
        return torch.stft(wav, N_FFT, HOP_LENGTH, WIN_LENGTH,
                          window=self.window.to(wav.device),
                          return_complex=True)

    def istft(self, mag, phase, length):
        """幅度 + 相位 → (B,length) 波形"""
        spec = torch.polar(mag, phase)
        return torch.istft(spec, N_FFT, HOP_LENGTH, WIN_LENGTH,
                           window=self.window.to(mag.device),
                           length=length)

    # ---- 前向：混合波形 → 目标幅度估计 ----
    def forward(self, mix_wav, spk_emb):
        """
        mix_wav (B,L) float32，spk_emb (B,emb_dim)
        返回 (est_mag, mix_mag, mix_phase)，均为 (B,F,T)
        （est_mag = sigmoid 掩码 × 混合幅度；训练损失只需 est_mag 与参考幅度）
        """
        spec = self.stft(mix_wav)
        mag = spec.abs()
        x = torch.log1p(mag).transpose(1, 2)                # (B,T,F)
        emb = self.emb_proj(spk_emb).unsqueeze(1)           # (B,1,F)
        emb = emb.expand(-1, x.size(1), -1)                 # 逐帧广播拼接
        h, _ = self.blstm(torch.cat([x, emb], dim=-1))      # (B,T,2H)
        mask = torch.sigmoid(self.mask_head(h)).transpose(1, 2)  # (B,F,T)
        est_mag = mask * mag
        return est_mag, mag, spec.angle()

    # ---- 推理：混合波形 → 提取目标波形（混合相位 ISTFT）----
    @torch.no_grad()
    def separate(self, mix_wav, spk_emb):
        self.eval()
        est_mag, _, phase = self.forward(mix_wav, spk_emb)
        return self.istft(est_mag, phase, mix_wav.shape[-1])


# ---------------------------------------------------------------
# 损失与监控指标
# ---------------------------------------------------------------
def frame_mask(lengths, T, device):
    """按样本长度生成有效帧掩码（center=True 时有效帧数 = L//hop + 1）"""
    valid = torch.div(lengths, HOP_LENGTH, rounding_mode="floor") + 1
    ar = torch.arange(T, device=device)
    return (ar.unsqueeze(0) < valid.unsqueeze(1)).float()   # (B,T)


def mag_l1_loss(est_mag, ref_mag, lengths):
    """
    线性幅度 L1：逐帧对频率取均值 → 剔除补零帧 → 逐样本归一化 → batch 均值。
    逐样本归一化避免长音频主导损失；拒识样本 ref=0，直接把估计压向 0。
    """
    T = est_mag.shape[-1]
    fm = frame_mask(lengths, T, est_mag.device)             # (B,T)
    l1 = (est_mag - ref_mag).abs().mean(dim=1)              # (B,T) 频率均值
    per_sample = (l1 * fm).sum(dim=1) / fm.sum(dim=1).clamp(min=1.0)
    return per_sample.mean()


def si_sdr(est, ref, lengths=None, eps=1e-8):
    """
    SI-SDR（dB），仅作训练监控（拒识样本 ref 全零无意义，调用方过滤）。
    est/ref (B,L) 波形。
    """
    if lengths is not None:
        L = est.shape[-1]
        sm = (torch.arange(L, device=est.device).unsqueeze(0)
              < lengths.unsqueeze(1)).float()
        est, ref = est * sm, ref * sm
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    alpha = ((est * ref).sum(-1, keepdim=True)
             / ((ref ** 2).sum(-1, keepdim=True) + eps))
    proj = alpha * ref
    noise = est - proj
    ratio = (proj ** 2).sum(-1) / ((noise ** 2).sum(-1) + eps)
    return 10 * torch.log10(ratio + eps)


# ---------------------------------------------------------------
# checkpoint 加载（评估/演示共用）
# ---------------------------------------------------------------
def sx_load(checkpoint, device="cuda:0", log=print):
    """从 sx_train 保存的 checkpoint 重建模型（含 config 的新格式优先）"""
    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        config = ckpt.get("config", DEFAULT_CONFIG)
        state = ckpt["state_dict"]
    else:  # 兼容纯 state_dict
        config, state = DEFAULT_CONFIG, ckpt
    model = SXExtractor(**config)
    model.load_state_dict(state)
    log(f"[模型] 已加载 SX 权重: {checkpoint}")
    return model.to(device).eval()
