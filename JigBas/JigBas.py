# ==============================================================
# JigBas — 抗干扰语音指令识别系统
# 主入口：交互式菜单导航（与 UI 模块栏平行）
#   1. 启动界面   — 控制台 UI（模块栏：基础演示 / 数据集搭建）
#   2. 基础演示   — 单条语音目标说话人识别（等价 python demo.py）
#   3. 搭建数据集 — 等价 python build_dataset.py（构建后自动写元数据）
#   4. 评估数据集 — 等价 python evaluate.py（结果写入数据集 evals/）
# ==============================================================

import sys
import os
import logging
import warnings

# 屏蔽所有无用的第三方库 WARNING / NOTICE
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
os.environ["TORCHAUDIO_BACKEND_WARNING"] = "0"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _show_banner():
    print("=" * 50)
    print("  JigBas — 抗干扰语音指令识别系统")
    print("=" * 50)


def _show_deps_status():
    """快速显示核心依赖状态（不导入包，< 0.1s）"""
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


def _ask(prompt, default):
    s = input(f"  {prompt} [{default}] > ").strip().strip('"').strip("'")
    return s or default


def _list_datasets():
    """列出已有数据集（按元数据摘要），供搭建/评估前查看"""
    import datasets as ds
    entries = ds.list_datasets()
    if not entries:
        print("  （暂无数据集）")
    else:
        print("  已有数据集：")
        for e in entries[:8]:
            print(f"    {ds.one_line_summary(e)}")
    return entries


def menu_ui():
    """1. 启动界面 — 打开图形化操作界面"""
    print("正在启动 UI 界面...")
    from ui import main as ui_main
    ui_main()


def menu_demo():
    """2. 基础演示 — 单条语音目标说话人识别（等价 python demo.py）"""
    import demo
    demo.interactive()


def menu_build_dataset():
    """3. 搭建数据集 — 等价 python build_dataset.py（回车使用默认值）"""
    import argparse
    import build_dataset as bd

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
    """4. 评估数据集 — 等价 python evaluate.py（回车使用默认值）"""
    import argparse
    import evaluate as ev

    _list_datasets()
    print()
    print("评估参数（直接回车使用默认值）：")
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


def main():
    _show_banner()
    deps_ok = _show_deps_status()

    if not deps_ok:
        print("\n[WARN] 部分依赖缺失，请先运行: pip install -r requirements.txt")
        sys.exit(1)

    menu = {
        "1": ("启动界面",   menu_ui),
        "2": ("基础演示",   menu_demo),
        "3": ("搭建数据集", menu_build_dataset),
        "4": ("评估数据集", menu_evaluate),
        "0": ("退出",       None),
    }

    while True:
        print()
        print("-" * 50)
        for key, (label, _) in menu.items():
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

        if choice in menu:
            _, func = menu[choice]
            if func:
                print()
                func()
        else:
            print(f"  无效选项: {choice}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
