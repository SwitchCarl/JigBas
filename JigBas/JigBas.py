# ==============================================================
# JigBas — 抗干扰语音指令识别系统（总入口）
#
# 子命令模式（参数原样透传给目标模块，等价于直接运行对应脚本）：
#   python JigBas.py ui                          # 控制台界面
#   python JigBas.py demo --mode sx --wake w.wav --rec r.wav   # 单条演示
#   python JigBas.py eval --dataset latest --final              # 评估（一键最终系统）
#   python JigBas.py submit --test <测试集> --out result.json   # 官方提交格式
#   python JigBas.py build --alias baseline      # 搭建训练数据集
#   python JigBas.py train-sc --dataset ...      # 训练 SC-Paraformer
#   python JigBas.py train-sx --dataset ...      # 训练 SX 提取器
#   python JigBas.py build-scx --src ... --dst ...              # 生成 scx8k 数据集
#   python JigBas.py verify                      # 环境自检
#   python JigBas.py list                        # 列出已有数据集
#   python JigBas.py --help                      # 全部子命令
#   python JigBas.py                             # 无参数 → 交互菜单
#
# 目录分层：
#   lib/   纯库（models/datasets/sc_model/sx_model/sc_data/paths）
#   app/   运行时（demo/ui/evaluate/submit，交付物核心）
#   tools/ 开发工具（build_dataset/build_scx_dataset/sc_train/sx_train/
#          verify_env，保留路径但实际使用中不交付）
# ==============================================================

import importlib
import logging
import os
import sys
import warnings

# 屏蔽所有无用的第三方库 WARNING / NOTICE
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
os.environ["TORCHAUDIO_BACKEND_WARNING"] = "0"

from lib.paths import REPO_ROOT

# 子命令表：名字 → (目标模块, 一句话说明)。module 为 None 表示纯内置命令。
COMMANDS = {
    "ui":        ("app.ui",                "启动控制台界面"),
    "demo":      ("app.demo",              "单条目标说话人识别（baseline/sc/sx）"),
    "eval":      ("app.evaluate",          "数据集评估（CER/拒识率/耗时，--final 一键）"),
    "submit":    ("app.submit",            "官方提交格式推理（最终系统）"),
    "build":     ("tools.build_dataset",   "搭建训练数据集（Lhotse 混音）"),
    "train-sc":  ("tools.sc_train",        "训练 SC-Paraformer（说话人条件 ASR）"),
    "train-sx":  ("tools.sx_train",        "训练 SX 提取器（重叠目标提取）"),
    "build-scx": ("tools.build_scx_dataset", "用 SX 提取生成 scx8k 数据集"),
    "verify":    ("tools.verify_env",      "环境自检（依赖/模型/转写）"),
    "list":      (None,                    "列出已有数据集"),
}


def _show_banner():
    print("=" * 50)
    print("  JigBas — 抗干扰语音指令识别系统")
    print("=" * 50)


def _show_deps_status():
    """快速显示核心依赖状态（不导入包，< 0.1s），返回是否全部就绪"""
    from importlib.metadata import version, PackageNotFoundError

    deps = {}
    for pkg in ["torch", "torchaudio", "funasr", "wespeaker"]:
        try:
            deps[pkg] = version(pkg)
        except PackageNotFoundError:
            deps[pkg] = "MISSING"

    for name, ver in deps.items():
        status = "OK" if ver != "MISSING" else "MISSING"
        print(f"  [{status}] {name}: {ver}")

    return all(v != "MISSING" for v in deps.values())


def _deps_ok():
    """静默依赖检查（CLI 路径用，不打印状态表）"""
    from importlib.metadata import version, PackageNotFoundError

    for pkg in ["torch", "torchaudio", "funasr", "wespeaker"]:
        try:
            version(pkg)
        except PackageNotFoundError:
            return False
    return True


def _ask(prompt, default):
    s = input(f"  {prompt} [{default}] > ").strip().strip('"').strip("'")
    return s or default


def _list_datasets():
    """列出已有数据集（按元数据摘要），供搭建/评估前查看。返回 0。"""
    from lib import datasets as ds
    entries = ds.list_datasets()
    if not entries:
        print("  （暂无数据集）")
    else:
        print("  已有数据集：")
        for e in entries[:12]:
            print(f"    {ds.one_line_summary(e)}")
    return 0


def _run_module(module, argv):
    """重组 sys.argv = [<脚本名>] + 剩余参数，再调目标模块的 main()，
    参数天然透传（目标模块用 argparse 自己解析）。"""
    # 先打 torch.jit.load 中文路径补丁（见 lib/models.py 说明），
    # 确保被调模块内 wespeaker/silero_vad 加载不受归档中文路径影响。
    from lib.models import _patch_torch_jit_load
    _patch_torch_jit_load()

    mod = importlib.import_module(module)
    sys.argv = [mod.__file__] + list(argv)
    return mod.main()


def _print_help():
    print("用法: python JigBas.py <命令> [参数...]")
    print("      python JigBas.py --help        # 本帮助")
    print("      python JigBas.py               # 无参数 → 交互菜单")
    print()
    print("子命令：")
    for name, (_, help_txt) in COMMANDS.items():
        print(f"  {name:<11} {help_txt}")
    print()
    print("每个子命令的参数透传给对应脚本，等价于直接运行该脚本；")
    print("可用 `python JigBas.py <命令> --help` 查看该命令的参数。")


# ---------------------------------------------------------------
# 交互菜单（无参数回退）
# ---------------------------------------------------------------
def menu_build_dataset():
    """搭建数据集 — 等价 python tools/build_dataset.py（回车使用默认值）"""
    import argparse
    from tools import build_dataset as bd

    _list_datasets()
    print()
    print("数据集构建参数（直接回车使用默认值，路径为相对项目根）：")
    try:
        args = argparse.Namespace(
            alias=_ask("数据集别名", "run"),
            clean_dir=_ask("干净语料目录", bd.DEFAULT_CLEAN_DIR),
            transcript=_ask("转写文件（留空自动探测）", bd.DEFAULT_TRANSCRIPT) or None,
            noise_dir=_ask("噪声库目录", bd.DEFAULT_NOISE_DIR),
            rir_dir=_ask("RIR 混响目录", bd.DEFAULT_RIR_DIR),
            output=None,  # 自动生成 <时间>_<别名>
            num_train=int(_ask("train 样本数", "2000")),
            num_dev=int(_ask("dev 样本数", "200")),
            reject_ratio=float(_ask("拒识样本占比", "0.3")),
            overlap_prob=float(_ask("双人重叠概率", "0.4")),
            dev_speaker_ratio=0.1,
            trim=True,
            seed=int(_ask("随机种子", "42")),
            progress=False,
        )
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return

    bd.build(args)


def menu_evaluate():
    """评估数据集 — 等价 python app/evaluate.py（回车使用默认值）"""
    import argparse
    from app import evaluate as ev

    _list_datasets()
    print()
    print("评估参数（直接回车使用默认值；--final 一键最终系统请用 CLI）：")
    try:
        args = argparse.Namespace(
            dataset=_ask("数据集（latest / 别名 / 时间前缀）", "latest"),
            split=_ask("评估划分", "dev"),
            manifest=None,
            data_root=None,
            threshold=None,
            sweep=_ask("阈值扫描 lo:hi:step", "0.30:0.75:0.05"),
            device=None,
            limit=int(_ask("仅评估前 N 条（0=全部）", "0")),
            output=None,
            detail=True,
            progress=False,
        )
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return

    ev.run(args)


def menu_train_sc():
    """训练 SC-Paraformer"""
    _list_datasets()
    print()
    print("SC-Paraformer 训练参数（直接回车使用默认值）：")
    try:
        dataset = _ask("数据集（latest / 别名）", "latest")
        epochs = _ask("训练轮数", "10")
        steps = _ask("固定步数（0=按轮数）", "0")
        batch = _ask("batch size", "8")
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return
    _run_module("tools.sc_train",
                ["--dataset", dataset, "--epochs", epochs,
                 "--steps", steps, "--batch-size", batch, "--progress"])


def menu_train_sx():
    """训练 SX 提取器"""
    _list_datasets()
    print()
    print("SX 提取器训练参数（直接回车使用默认值）：")
    try:
        dataset = _ask("数据集（latest / 别名）", "latest")
        epochs = _ask("训练轮数", "30")
        steps = _ask("固定步数（0=按轮数）", "0")
        batch = _ask("batch size", "16")
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return
    _run_module("tools.sx_train",
                ["--dataset", dataset, "--epochs", epochs,
                 "--steps", steps, "--batch-size", batch, "--progress"])


def menu_build_scx():
    """用 SX 提取器生成 scx8k 数据集"""
    from tools import build_scx_dataset as bscx

    print()
    print("生成 scx8k 数据集（SX 提取波形；直接回车使用默认路径）：")
    try:
        src = _ask("源数据集目录", bscx.DEFAULT_SRC)
        dst = _ask("目标数据集目录", bscx.DEFAULT_DST)
        ckpt = _ask("SX 提取器权重", bscx.DEFAULT_CKPT)
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return
    _run_module("tools.build_scx_dataset",
                ["--src", src, "--dst", dst, "--ckpt", ckpt])


def menu_submit():
    """官方提交格式推理（最终系统）"""
    _list_datasets()
    print()
    print("官方提交推理（最终系统）——在赛方指定数据集上验证模型：")
    print("  测试集支持目录（自动探测 test.json / test_manifest.jsonl）或单 JSON。")
    try:
        test = _ask("测试集目录 / JSON 路径", "")
        if not test:
            print("已取消。")
            return
        out = _ask("输出 JSON 路径", os.path.join(REPO_ROOT, "result.json"))
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return
    _run_module("app.submit",
                ["--test", test, "--out", out, "--progress"])


def _interactive_menu():
    _show_banner()
    deps_ok = _show_deps_status()
    if not deps_ok:
        print("\n[WARN] 部分依赖缺失，请先运行: pip install -r requirements.txt")

    menu = [
        ("1", "启动界面",            lambda: _run_module("app.ui", [])),
        ("2", "基础演示",            lambda: _run_module("app.demo", [])),
        ("3", "搭建数据集",          menu_build_dataset),
        ("4", "评估数据集",          menu_evaluate),
        ("5", "训练 SC-Paraformer",  menu_train_sc),
        ("6", "训练 SX 提取器",      menu_train_sx),
        ("7", "生成 scx8k 数据集",   menu_build_scx),
        ("8", "官方提交（submit）",  menu_submit),
        ("9", "环境自检",            lambda: _run_module("tools.verify_env", [])),
        ("0", "退出",                None),
    ]

    while True:
        print()
        print("-" * 50)
        for key, label, _ in menu:
            print(f"  {key}. {label}")
        print("-" * 50)

        try:
            choice = input("请选择 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if choice == "0":
            print("退出。")
            break

        hit = next((fn for k, _, fn in menu if k == choice), None)
        if hit is None:
            print(f"  无效选项: {choice}")
            continue
        if not deps_ok:
            print("  依赖缺失，无法执行。请先 pip install -r requirements.txt")
            continue
        print()
        try:
            hit()
        except KeyboardInterrupt:
            print("\n已取消。")

    return 0


def main():
    # 无参数 → 交互菜单（菜单内自行打印横幅与依赖状态）
    if len(sys.argv) < 2:
        return _interactive_menu()

    _show_banner()
    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd in ("-h", "--help", "help"):
        _print_help()
        return 0

    entry = COMMANDS.get(cmd)
    if entry is None:
        print(f"  未知命令: {cmd}（输入 --help 查看全部子命令）")
        return 2

    module = entry[0]
    if module is None:  # 纯内置命令：list
        return _list_datasets()

    # CLI 路径：静默检查依赖，缺失时给出友好提示
    if not _deps_ok():
        print("\n[WARN] 部分依赖缺失，请先运行: pip install -r requirements.txt")
        return 1
    return _run_module(module, rest)


if __name__ == "__main__":
    sys.exit(main())
