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
# 损失：波形域 waveform_loss（正样本负 SNR + 拒识 log 能量抑制）——
#   首轮训练诊断发现单纯幅度 L1 会收敛到"全抑制"退化解（零输出代价
#   ≈ mean(ref_mag)，与恒等掩码同量级，分离梯度被淹没），波形域损失
#   让"压掉一切"在正样本上代价最大化，迫使模型真正保留目标能量。
#   SI-SDR / 幅度 L1 仅作监控指标。
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
    "use_film": True,      # FiLM 乘性门（第三轮修复）
    "input_mode": "log",   # 对数谱输入（第三轮修复）；初版为 "log1p"
}


class SXExtractor(nn.Module):
    """目标说话人提取网络（掩码式 VoiceFilter）

    条件注入（阶段 C 第三轮修正）：单次线性投影拼接太弱——诊断发现
    模型对重叠样本只做"恒等/全抑制"二选一（SI-SDR≈SIR），忽略嵌入。
    修复 = 对数谱输入 + FiLM 乘性调制（与第二周 SC-Paraformer 同款
    归纳偏置）：x = log(mag) · (1 + tanh(emb_gate(emb)))，初始≈恒等，
    再与 emb_proj(emb) 拼接进 BLSTM（乘性 + 加性双通路）。
    use_film=False / input_mode="log1p" 仅用于加载初版 checkpoint。
    """

    def __init__(self, emb_dim=SPK_EMB_DIM, n_freq=N_FREQ,
                 lstm_hidden=400, lstm_layers=2,
                 use_film=True, input_mode="log"):
        super().__init__()
        self.n_freq = n_freq
        self.use_film = use_film
        self.input_mode = input_mode
        # persistent=False：state_dict 不存窗函数，与旧 checkpoint 兼容
        self.register_buffer("window", torch.hann_window(WIN_LENGTH),
                             persistent=False)
        if use_film:
            self.emb_gate = nn.Linear(emb_dim, n_freq)   # FiLM 乘性门（tanh）
        self.emb_proj = nn.Linear(emb_dim, n_freq)       # 加性投影（拼接）
        self.blstm = nn.LSTM(n_freq * 2, lstm_hidden, lstm_layers,
                             batch_first=True, bidirectional=True)
        self.mask_head = nn.Linear(lstm_hidden * 2, n_freq)
        # 掩码偏置初始化为 +2（初始掩码≈sigmoid(2)=0.88，近恒等）。
        # 全量训练第四轮诊断：默认零偏置下拒识样本的抑制梯度在最初几步
        # 就把掩码压进 sigmoid 饱和区（输出≈0 处梯度≈0），正样本的分离
        # 梯度再也拉不回来——"全抑制"死锁。近恒等起点让正负样本的梯度
        # 都在 sigmoid 活跃区内竞争，网络有条件嵌入可用，学得会区分。
        nn.init.constant_(self.mask_head.bias, 2.0)

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
        # 对数谱输入：mag 均值仅 ~0.1，log1p 后特征 std ~0.1，BLSTM 门控
        # 近线性区驱动不足（过拟合诊断：选择类映射学不会，恒等/全抑制这种
        # 偏置级解正常）。log(clamp) 把特征拉到 std~1.5 的正常量级
        if self.input_mode == "log1p":
            x = torch.log1p(mag).transpose(1, 2)        # 初版（弱特征，弃用）
        else:
            x = torch.log(mag.clamp(min=1e-5)).transpose(1, 2)  # (B,T,F)
        if self.use_film:
            g = torch.tanh(self.emb_gate(spk_emb)).unsqueeze(1)  # (B,1,F)
            x = x * (1.0 + g)        # FiLM 乘性调制（初始≈恒等）
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

    ⚠️ 阶段 C 首轮训练教训（2026-08-12 诊断）：单独使用该损失会收敛到
    "全抑制"退化解——零输出代价 ≈ mean(ref_mag) ≈ 0.095，与恒等掩码
    （0.057）同量级，分离梯度被淹没。现仅作诊断量；主损失用 waveform_loss。
    """
    T = est_mag.shape[-1]
    fm = frame_mask(lengths, T, est_mag.device)             # (B,T)
    l1 = (est_mag - ref_mag).abs().mean(dim=1)              # (B,T) 频率均值
    per_sample = (l1 * fm).sum(dim=1) / fm.sum(dim=1).clamp(min=1.0)
    return per_sample.mean()


def waveform_loss(est, ref, lengths, types, eps=1e-8, rej_weight=0.3):
    """
    波形域提取损失（阶段 C 修正版主损失，破除"全抑制"退化解）：
      正样本: -SNR = 10·log10(‖est−ref‖² / ‖ref‖²)（按样本长度掩码）
              —— 零输出代价有界（0 dB）但梯度持续指向保留目标能量，
                 且直接惩罚错误电平（下游声纹门控需要正确的输出能量）
      拒识样本: log 能量 10·log10(mean(est²)+eps)，压向 0；
                clamp(min=-60dB) 兜底——压到 rms≈0.001 后梯度归零，
                避免无限增大的抑制梯度淹没正样本的分离梯度
                （过拟合诊断发现：无兜底时拒识项 -52dB 仍持续下压，
                 部分正样本被连带压成全零）
      rej_weight: 拒识项权重（默认 0.3）。全量训练第四轮诊断：拒识占
                30% 且抑制任务远易于分离任务，等权时抑制梯度在训练初期
                占主导，把掩码压进 sigmoid 饱和区形成死锁；降权后正负
                梯度量级匹配（配合 mask_head 偏置 +2 的近恒等初始化）。
    est/ref: (B,L) 波形；types: 每样本 "positive"/"rejection"。
    """
    L = est.shape[-1]
    sm = (torch.arange(L, device=est.device).unsqueeze(0)
          < lengths.unsqueeze(1)).float()
    est, ref = est * sm, ref * sm
    is_pos = torch.tensor([t == "positive" for t in types],
                          device=est.device, dtype=torch.bool)
    losses = est.new_zeros(est.shape[0])
    if is_pos.any():
        e, r = est[is_pos], ref[is_pos]
        err = ((e - r) ** 2).sum(-1)
        sig = (r ** 2).sum(-1).clamp(min=eps)
        losses[is_pos] = 10 * torch.log10(err.clamp(min=eps) / sig)
    if (~is_pos).any():
        e = est[~is_pos]
        n = lengths[~is_pos].float().clamp(min=1)
        losses[~is_pos] = rej_weight * (
            10 * torch.log10((e ** 2).sum(-1) / n + eps)
        ).clamp(min=-60.0)   # 抑制到 rms≈0.001 后释放梯度
    return losses.mean()


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
        config = {**DEFAULT_CONFIG, **ckpt.get("config", {})}
        state = ckpt["state_dict"]
    else:  # 兼容纯 state_dict
        config, state = DEFAULT_CONFIG, ckpt
    # 初版 checkpoint（FiLM 加入前）无 emb_gate 权重且训练时输入是 log1p，
    # 按 state_dict 键自动退回旧结构/旧输入模式，保证旧模型可评估
    if "emb_gate.weight" not in state:
        config = {**config, "use_film": False, "input_mode": "log1p"}
    model = SXExtractor(**config)
    model.load_state_dict(state)
    log(f"[模型] 已加载 SX 权重: {checkpoint}")
    return model.to(device).eval()
