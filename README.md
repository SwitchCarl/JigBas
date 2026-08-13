# JigBas — 抗干扰语音指令识别系统 

目标说话人识别 + 拒识 + 指令转写（抗干扰语音指令识别比赛项目）。

**最终系统综合分 0.7813**（CER 37.48% / RR 93.75%）。管线：rec 音频 →
SX 提取目标人声 → wespeaker 声纹门控（灰区滑窗精判，sim_sx ≥ 0.30 接受）→
SC-scx 在提取音频上转写。详见 `Temp/Record/` 交接文档。

## 目录结构（分层）

```
JigBas/
├── JigBas.py                  # 总入口：子命令 + 无参交互菜单
├── lib/                       # 纯库（被 app/tools 引用，不直接运行）
│   ├── paths.py               #   路径单点（REPO_ROOT / MODELS_DIR / FUNASR_MODEL_DIR）
│   ├── models.py              #   模型加载与声纹工具（wespeaker / funasr）
│   ├── datasets.py            #   数据集注册与路径解析
│   ├── sc_model.py            #   SC-Paraformer（FiLM 说话人条件）与解码
│   ├── sx_model.py            #   SX-Extractor（VoiceFilter 式掩码提取）
│   └── sc_data.py             #   SC 数据管线（嵌入缓存 / Dataset / collate）
├── app/                       # 运行时（交付物核心）
│   ├── demo.py                #   单条识别（baseline / sc / sx 三模式）
│   ├── ui.py                  #   控制台界面
│   ├── evaluate.py            #   数据集评估（CER / 拒识率 / 耗时，--final 一键）
│   └── submit.py              #   官方提交格式推理（赛方测试集验证）
├── tools/                     # 开发工具（训练 / 造数据 / 自检，不交付）
│   ├── build_dataset.py       #   三元组数据集构建（Lhotse 混音）
│   ├── build_scx_dataset.py   #   用 SX 提取生成 scx8k 数据集
│   ├── sc_train.py            #   训练 SC-Paraformer
│   ├── sx_train.py            #   训练 SX 提取器
│   ├── verify_env.py          #   环境自检（依赖 / 模型 / 转写）
│   └── package_release.py     #   生成可交付的 submit/ 目录
├── Models/                    # 本地预训练权重（gitignore）
└── requirements.txt
```

## 快速开始

```bash
# 环境自检
python JigBas.py verify

# 单条演示（sx = 最终系统）
python JigBas.py demo --mode sx --wake wake.wav --rec rec.wav

# 自有数据集评估（--final 一键最终系统，输出 5 配置消融表，sx-gate/sx-asr 行即最终系统）
python JigBas.py eval --dataset latest --final

# 官方提交格式推理（在赛方指定测试集上验证模型）
python JigBas.py submit --test <测试集目录或json> --out result.json

# 无参数进入交互菜单
python JigBas.py
```

每个子命令参数透传给对应脚本，等价于直接运行该脚本；`python JigBas.py <命令> --help` 可看参数。

## 脚本索引

| 层 | 脚本 | 职责 | 常用命令 |
|---|---|---|---|
| **运行** | `app/demo.py` | 单条目标说话人识别（三模式） | `demo --mode sx --wake w --rec r` |
| **运行** | `app/ui.py` | 控制台界面（模块栏：演示 / 数据集） | `ui` |
| **运行** | `app/evaluate.py` | 数据集评估：CER / RR / FAR / FRR / RTF | `eval --dataset latest --final` |
| **运行** | `app/submit.py` | 官方提交格式推理（中文键 / final_cer / duration） | `submit --test <测试集> --out result.json` |
| **训练** | `tools/sc_train.py` | 训练 SC-Paraformer（说话人条件 ASR） | `train-sc --dataset <名>` |
| **训练** | `tools/sx_train.py` | 训练 SX 提取器（重叠目标提取） | `train-sx --dataset <名>` |
| **训练** | `tools/build_scx_dataset.py` | SX 提取生成 scx8k 训练集 | `build-scx` |
| **工具** | `tools/build_dataset.py` | 三元组数据集构建（Lhotse 混音） | `build --alias <名>` |
| **工具** | `tools/verify_env.py` | 环境自检（依赖 / 模型 / 转写） | `verify` |
| **工具** | `tools/package_release.py` | 生成可交付 `submit/` 目录 | `package_release` |
| **库** | `lib/models.py` 等 | 模型 / 数据 / 路径共享模块 | （被调用，不直接运行） |

> 「训练」与「工具」类脚本为开发期资产，保留路径供复现；实际比赛交付走
> `app/` 的 `eval` / `submit`，交付包由 `tools/package_release.py` 生成。

## 官方提交格式（Context.txt）

```json
{"result": {"results": [{"id":"识别音频名字","content":"推理文本",
                          "label":"识别标签","cer":"xx"}, ...],
             "final_cer":"xx", "duration":"t"}}
```

- `id` = 测试音频名字；拒识样本 `content`/`label` 为空。
- `final_cer` = 全部正样本聚合编辑距离 / 总字数（与 `evaluate` 同口径）。
- `duration` = batch=1 推理全部音频的总秒数（对应效率指标 10%）。

## 最终系统权重

| 模型 | 路径 |
|---|---|
| SX 提取器（阶段C 纯分离，未微调） | `Temp/Datasets/20260811_1914_sxtrain/checkpoints/sx_20260812_235228/step_6000.pt` |
| SC-scx ASR（提取音频上微调，abs 8000 步） | `Temp/Datasets/20260813_0105_scx8k/checkpoints/sc_20260813_cont4000/step_4000.pt` |

权重在仓库外（`E:/Desktop/Jigbas/Temp/Datasets/`），demo / eval --final / submit
默认读取；可用 `--sx-checkpoint` / `--sc-checkpoint` 覆盖。

