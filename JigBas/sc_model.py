# ==============================================================
# SC-Paraformer 模型侧工具（第二周）
#
#   apply_cif_fix() — 修复 funasr cif_v1 的越界索引 bug（阶段 C 发现）：
#     当 batch 中末尾样本 0 次 fire（空目标/短音频时 alphas 总和略低于
#     阈值 1.0）时，shift_batch_idxs 会等于总 fire 数，导致
#     shift_frames[shift_batch_idxs] 在 GPU 上 device-side assert。
#     修复方式：索引前过滤越界位置（0-fire 样本本来就没有帧可清零）。
#     训练/实验脚本在使用模型前先调用 apply_cif_fix()。
#
#   SCParaformer — FiLM 说话人条件改造（阶段 D）：
#     缓存的 ECAPA 嵌入（256 维）→ Linear(256→1024) → γ/β（各 512 维），
#     在 encode() 输出端调制：out = (1+γ)·x + β。
#     spk_proj 零初始化 → 初始为恒等映射，不破坏预训练模型表现。
#     说话人嵌入经 forward 的 spk_emb 关键字参数传入（仅当前调用生效）。
#
#   build_sc_model() — 一步构建：原始模型 + cif 补丁 + FiLM 改造
#     （+ 可选加载 SC 微调权重），训练与评估共用。
# ==============================================================

import torch
from torch import nn


def _cif_v1_fixed(hidden, alphas, threshold):
    """cif_v1 的越界修复版（逻辑与 funasr 原版一致，仅过滤越界批索引）"""
    from funasr.models.paraformer.cif_predictor import cif_wo_hidden_v1

    fires, fire_idxs = cif_wo_hidden_v1(alphas, threshold, return_fire_idxs=True)

    device = hidden.device
    dtype = hidden.dtype
    batch_size, len_time, hidden_size = hidden.size()
    if fire_idxs.sum() == 0:
        max_label_len = torch.round(alphas.sum(-1)).int().max()
        return (
            torch.zeros(batch_size, max_label_len, hidden_size,
                        dtype=dtype, device=device),
            fires,
        )

    prefix_sum_hidden = torch.cumsum(
        alphas.unsqueeze(-1).repeat((1, 1, hidden_size)) * hidden, dim=1)

    frames = prefix_sum_hidden[fire_idxs]
    shift_frames = torch.roll(frames, 1, dims=0)

    batch_len = fire_idxs.sum(1)
    batch_idxs = torch.cumsum(batch_len, dim=0)
    shift_batch_idxs = torch.roll(batch_idxs, 1, dims=0)
    shift_batch_idxs[0] = 0
    # 修复点：0-fire 样本的批起始索引 == 总 fire 数，越界，过滤掉
    valid = shift_batch_idxs < frames.size(0)
    shift_frames[shift_batch_idxs[valid]] = 0

    remains = fires - torch.floor(fires)
    remain_frames = (remains[fire_idxs].unsqueeze(-1)
                     .repeat((1, hidden_size)) * hidden[fire_idxs])

    shift_remain_frames = torch.roll(remain_frames, 1, dims=0)
    shift_remain_frames[shift_batch_idxs[valid]] = 0  # 修复点：同上

    frames = frames - shift_frames + shift_remain_frames - remain_frames

    max_label_len = torch.round(alphas.sum(-1)).int().max()
    frame_fires = torch.zeros(batch_size, max_label_len, hidden_size,
                              dtype=dtype, device=device)
    indices = torch.arange(max_label_len, device=device).expand(batch_size, -1)
    frame_fires_idxs = indices < batch_len.unsqueeze(1)
    frame_fires[frame_fires_idxs] = frames
    return frame_fires, fires


_CIF_FIXED = False


def apply_cif_fix(log=print):
    """猴子补丁：用修复版替换 funasr 的 cif_v1（幂等）"""
    global _CIF_FIXED
    if _CIF_FIXED:
        return
    import funasr.models.paraformer.cif_predictor as cif_mod
    cif_mod.cif_v1 = _cif_v1_fixed
    _CIF_FIXED = True
    log("[补丁] cif_v1 越界修复已应用")


def _cif_predictor_forward_fixed(self, hidden, target_label=None, mask=None,
                                 ignore_id=-1, mask_chunk_predictor=None,
                                 target_label_length=None):
    """
    CifPredictorV2.forward 的稳定性修复版（阶段 E 发现）：
    唯一改动——alphas 缩放的分母 token_num 加下限 clamp(min=0.1)。
    空目标训练时 predictor 被 loss_pre 压向很低的 token 数，token_num
    偶尔趋近 0 → target_length/token_num 爆炸甚至除零 → alphas=inf/NaN
    → 污染权重 → 崩溃。其余逻辑与 funasr 原版逐行一致。
    """
    from torch.cuda.amp import autocast
    from funasr.models.paraformer.cif_predictor import cif_v1

    with autocast(False):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))
        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(
            alphas * self.smooth_factor - self.noise_threshold)
        if mask is not None:
            mask = mask.transpose(-1, -2).float()
            alphas = alphas * mask
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        mask = mask.squeeze(-1)
        if target_label_length is not None:
            target_length = target_label_length.squeeze(-1)
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)
        if target_length is not None:
            # 修复点：原版为 target_length / token_num，分母可趋近 0
            alphas *= (target_length / token_num.clamp(min=0.1))[:, None] \
                .repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            if self.tail_mask:
                hidden, alphas, token_num = self.tail_process_fn(
                    hidden, alphas, token_num, mask=mask
                )
            else:
                hidden, alphas, token_num = self.tail_process_fn(
                    hidden, alphas, token_num, mask=None
                )

        acoustic_embeds, cif_peak = cif_v1(hidden, alphas, self.threshold)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]

    return acoustic_embeds, token_num, alphas, cif_peak


_PREDICTOR_FIXED = False


def apply_predictor_fix(log=print):
    """猴子补丁：CifPredictorV2.forward 的 token_num 防除零修复（幂等）"""
    global _PREDICTOR_FIXED
    if _PREDICTOR_FIXED:
        return
    from funasr.models.paraformer.cif_predictor import CifPredictorV2
    CifPredictorV2.forward = _cif_predictor_forward_fixed
    _PREDICTOR_FIXED = True
    log("[补丁] CifPredictorV2 token_num 防除零修复已应用")


# ---------------------------------------------------------------
# FiLM 说话人条件改造
# ---------------------------------------------------------------
from funasr.models.paraformer.model import Paraformer

SPK_EMB_DIM = 256  # wespeaker ECAPA-TDNN 嵌入维度


class SCParaformer(Paraformer):
    """
    在 Paraformer 基础上加 FiLM 说话人条件：
      spk_proj: spk_emb (B,256) → γ/β (各 B,encoder_dim)
      encoder_out = (1+γ)·encoder_out + β
    零初始化保证初始等价于原模型。

    注意：实例不通过构造函数创建，而是由 to_sc_paraformer() 从
    build_model 建好的 Paraformer 原地转换（__class__ 替换），
    以完整复用 config.yaml 的构造参数与已加载的预训练权重。
    """

    def _init_spk(self, spk_emb_dim=SPK_EMB_DIM):
        d = self.encoder.output_size()
        self.spk_proj = nn.Linear(spk_emb_dim, 2 * d)
        nn.init.zeros_(self.spk_proj.weight)
        nn.init.zeros_(self.spk_proj.bias)
        self._spk_emb = None

    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        spk_emb = kwargs.pop("spk_emb", None)
        if self.training and spk_emb is None:
            raise RuntimeError("SCParaformer 训练时必须传入 spk_emb")
        self._spk_emb = spk_emb
        try:
            return super().forward(speech, speech_lengths, text,
                                   text_lengths, **kwargs)
        finally:
            self._spk_emb = None

    def encode(self, speech, speech_lengths, **kwargs):
        encoder_out, encoder_out_lens = super().encode(
            speech, speech_lengths, **kwargs)
        if self._spk_emb is not None:
            emb = self._spk_emb.to(encoder_out.device, encoder_out.dtype)
            gamma, beta = self.spk_proj(emb).chunk(2, dim=-1)
            encoder_out = ((1.0 + gamma).unsqueeze(1) * encoder_out
                           + beta.unsqueeze(1))
        return encoder_out, encoder_out_lens


def to_sc_paraformer(model, spk_emb_dim=SPK_EMB_DIM):
    """把 build_model 建好的 Paraformer 原地转换为 SCParaformer"""
    if isinstance(model, SCParaformer):
        return model
    assert isinstance(model, Paraformer), f"意外类型: {type(model)}"
    model.__class__ = SCParaformer  # 纯 Python 子类，无 __slots__，布局兼容
    SCParaformer._init_spk(model, spk_emb_dim)
    return model


def build_sc_model(device="cuda:0", sc_checkpoint=None, log=print):
    """
    构建 SC 模型：原始 model.pt + cif 补丁 + FiLM 改造。
    sc_checkpoint 给出时加载微调后的 state_dict（含 spk_proj）。
    返回 (model, kwargs)（kwargs 内含 tokenizer / frontend）。
    """
    apply_cif_fix(log)
    apply_predictor_fix(log)
    from funasr.auto.auto_model import AutoModel
    from models import FUNASR_MODEL_DIR

    model, kwargs = AutoModel.build_model(model=FUNASR_MODEL_DIR, device=device)
    model = to_sc_paraformer(model)
    if sc_checkpoint:
        state = torch.load(sc_checkpoint, map_location="cpu")
        model.load_state_dict(state)
        log(f"[模型] 已加载 SC 权重: {sc_checkpoint}")
    return model.to(device), kwargs
