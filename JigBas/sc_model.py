# ==============================================================
# SC-Paraformer 模型侧工具（第二周）
#
# 目前包含：
#   apply_cif_fix() — 修复 funasr cif_v1 的越界索引 bug（阶段 C 发现）：
#     当 batch 中末尾样本 0 次 fire（空目标/短音频时 alphas 总和略低于
#     阈值 1.0）时，shift_batch_idxs 会等于总 fire 数，导致
#     shift_frames[shift_batch_idxs] 在 GPU 上 device-side assert。
#     修复方式：索引前过滤越界位置（0-fire 样本本来就没有帧可清零）。
#     训练/实验脚本在使用模型前先调用 apply_cif_fix()。
#
# 后续（阶段 D）将在此文件加入：FiLM 说话人条件包装类。
# ==============================================================

import torch


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
