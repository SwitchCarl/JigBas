# ==============================================================
# JigBas — 抗干扰语音指令识别系统
# 主入口：交互式菜单导航
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


def menu_verify():
    """1. 验证环境 — 运行完整验证流程"""
    from verify_env import run_verification
    run_verification()


def menu_ui():
    """2. 启动界面 — 打开图形化操作界面"""
    print("正在启动 UI 界面...")
    from ui import main as ui_main
    ui_main()


def main():
    _show_banner()
    deps_ok = _show_deps_status()

    if not deps_ok:
        print("\n[WARN] 部分依赖缺失，请先运行: pip install -r requirements.txt")
        sys.exit(1)

    menu = {
        "1": ("验证环境", menu_verify),
        "2": ("启动界面", menu_ui),
        "0": ("退出",     None),
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