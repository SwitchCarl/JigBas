# ==============================================================
# 项目路径单点（分层后统一入口）
#
# 分层前各脚本自行定义 PROJECT_ROOT = dirname(__file__)，分层后目录变深
# （lib/、app/、tools/ 各降一级）会失效。这里统一为 REPO_ROOT：
#   项目根 = JigBas/JigBas/（含 Models/、requirements.txt、JigBas.py）
# lib/ 在项目根下一级，故上两级即项目根；app/、tools/ 同理（都在根下一级）。
# 所有脚本统一 `from lib.paths import REPO_ROOT`，不要再各自定义路径常量。
# ==============================================================

import os

# 项目根（仓库内代码根）
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型权重目录（仓库内，gitignore）
MODELS_DIR = os.path.join(REPO_ROOT, "Models")

# 项目外层根（Temp/、文档所在层；= 仓库内代码根上两级）
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", ".."))

# 训练数据集 / 模型检查点目录（仓库外，gitignore）。相对推导，
# 项目整体迁移路径时自动跟随，无需改硬编码。
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "Temp", "Datasets")

# funasr Paraformer-large 本地权重（无网环境离线加载用）
FUNASR_MODEL_DIR = os.path.join(
    MODELS_DIR, "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
