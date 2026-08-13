# ==============================================================
# JigBas UI — 抗干扰语音指令识别系统界面
# 控制台风格界面：方向键 + Enter 控制
# 仅负责界面显示与交互；模型见 models.py，识别见 demo.py
#
# - 启动后先进入模型加载页：仅进度条 + 动画，就绪后才进入正式 UI
# - 顶部模块栏：基础演示 / 数据集搭建（启用模块用双制表符+亮色）
# - 基础演示：单条目标说话人识别（进程内跑，等价 python demo.py）
# - 数据集搭建（单页）：左上入口（搭建/评估），右侧数据集竖列表，
#   下方信息框显详情/调参数；入口上按 Enter = 派生子进程执行等价
#   命令行（--progress 结构化进度），日志回传原窗口
# - 窗口大小固定，选中项亮青色高亮
# ==============================================================

import os
import re
import sys
import json
import time
import threading
import logging
import warnings
import unicodedata

# 直接运行本脚本时把项目根加入 sys.path（python app/ui.py --ui-child）
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import datasets as ds
from lib.models import ModelHub
from lib.paths import REPO_ROOT
from tools import build_dataset as bd
from app.demo import (recognize, recognize_sc, load_sc_engine,
                      recognize_sx, load_final_engine,
                      DEFAULT_THRESHOLD, SC_DEFAULT_THRESHOLD,
                      SX_DEFAULT_THRESHOLD, SC_GRAY_ZONE)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

UI_CHILD_FLAG = "--ui-child"
WINDOW_TITLE = "JigBas — 抗干扰语音指令识别系统"

# 窗口尺寸（字符）
WIN_COLS = 94
WIN_ROWS = 36
CONTENT_W = WIN_COLS - 2

# ANSI 颜色
C_RESET = "\x1b[0m"
C_DIM = "\x1b[90m"
C_SEL = "\x1b[96m"
C_TITLE = "\x1b[97m"
C_OK = "\x1b[92m"
C_WARN = "\x1b[93m"
C_ERR = "\x1b[91m"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

PROGRESS_PREFIX = "[PROGRESS] "


# ---------------------------------------------------------------
# 显示宽度工具（忽略 ANSI 转义；中文按 2 列计算）
# ---------------------------------------------------------------
def strip_ansi(s):
    return ANSI_RE.sub("", s)


def _char_width(c):
    return 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1


def disp_width(s):
    return sum(_char_width(c) for c in strip_ansi(s))


def pad_right(s, w):
    return s + " " * max(0, w - disp_width(s))


def truncate(s, w):
    """按可见宽度截断（尾部省略），保留 ANSI 转义序列完整性"""
    if disp_width(s) <= w:
        return s
    out = []
    width = 0
    pos = 0
    stopped = False
    for m in ANSI_RE.finditer(s):
        for c in s[pos:m.start()]:
            cw = _char_width(c)
            if width + cw > w:
                stopped = True
                break
            out.append(c)
            width += cw
        if stopped:
            break
        out.append(m.group(0))
        pos = m.end()
    if not stopped:
        for c in s[pos:]:
            cw = _char_width(c)
            if width + cw > w:
                break
            out.append(c)
            width += cw
    return "".join(out) + C_RESET


def _take_prefix(s, w):
    out, width = [], 0
    for c in s:
        cw = _char_width(c)
        if width + cw > w:
            break
        out.append(c)
        width += cw
    return "".join(out)


def _take_suffix(s, w):
    out, width = [], 0
    for c in reversed(s):
        cw = _char_width(c)
        if width + cw > w:
            break
        out.append(c)
        width += cw
    return "".join(reversed(out))


def truncate_middle(s, w):
    """超长时中间用 ... 省略，保留首尾（适用于纯文本路径）"""
    if disp_width(s) <= w:
        return s
    keep = w - 3
    head = _take_prefix(s, (keep + 1) // 2)
    tail = _take_suffix(s, keep - disp_width(head))
    return head + "..." + tail


def fit(s, w):
    return pad_right(truncate(s, w), w)


def prog_bar(pct, width=30):
    """进度条：pct ∈ [0,1]"""
    pct = max(0.0, min(1.0, pct))
    n = int(pct * width)
    return f"{C_SEL}{'█' * n}{C_DIM}{'░' * (width - n)}{C_RESET}"


# ---------------------------------------------------------------
# 制表符盒子
# ---------------------------------------------------------------
SINGLE = ("┌", "┐", "└", "┘", "─", "│")
DOUBLE = ("╔", "╗", "╚", "╝", "═", "║")


def make_box(title, lines, width, selected=False, min_height=None, style=None):
    """生成一个盒子，返回字符串行列表（每行可见宽度 == width）
    style=(制表符六元组, 颜色) 可覆盖 selected 推导出的默认样式"""
    if style:
        (tl, tr, bl, br, h, v), color = style
    else:
        tl, tr, bl, br, h, v = DOUBLE if selected else SINGLE
        color = C_SEL if selected else C_DIM
    inner = width - 2

    lines = list(lines)
    if min_height is not None:
        lines = (lines + [""] * min_height)[:min_height]

    title_seg = f" {title} " if title else ""
    top = tl + h + title_seg + h * max(0, width - 2 - disp_width(title_seg) - 1) + tr
    rows = [f"{color}{top}{C_RESET}"]
    for line in lines:
        rows.append(f"{color}{v}{C_RESET}{fit(' ' + line, inner)}{color}{v}{C_RESET}")
    bottom = bl + h * (width - 2) + br
    rows.append(f"{color}{bottom}{C_RESET}")
    return rows


def hcat(box_a, box_b, gap=2):
    """并排放置两个等高的盒子"""
    n = max(len(box_a), len(box_b))
    box_a = box_a + [box_a[-1]] * (n - len(box_a))
    box_b = box_b + [box_b[-1]] * (n - len(box_b))
    return [a + " " * gap + b for a, b in zip(box_a, box_b)]


def vcat(*groups):
    rows = []
    for g in groups:
        rows += g
    return rows


# ---------------------------------------------------------------
# UI 输出通道
# 子进程中: 写入自身控制台 CONOUT$（日志走 stdout 管道回父进程）
# ---------------------------------------------------------------
_con_out = None
_con_handle = None


def _setup_console_io():
    """子进程模式：打开自身控制台输出，固定窗口大小，启用 ANSI"""
    global _con_out, _con_handle
    import ctypes
    import msvcrt

    # 控制台代码页切换为 UTF-8，保证制表符正确显示
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleCP(65001)
    kernel32.SetConsoleOutputCP(65001)

    _con_out = open("CONOUT$", "w", encoding="utf-8", buffering=1)
    _con_handle = msvcrt.get_osfhandle(_con_out.fileno())

    mode = ctypes.c_ulong()
    if kernel32.GetConsoleMode(_con_handle, ctypes.byref(mode)):
        kernel32.SetConsoleMode(_con_handle, mode.value | 0x0004)  # VT 转义

    _fix_window_size()

    _emit(f"\x1b]0;{WINDOW_TITLE}\x07")  # 窗口标题
    _emit("\x1b[?25l")                   # 隐藏光标


def _fix_window_size():
    """固定控制台窗口为 WIN_COLS x WIN_ROWS，并尽量禁止拉伸"""
    import ctypes
    from ctypes import wintypes

    if _con_handle is None:
        return

    class COORD(ctypes.Structure):
        _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]

    class SMALL_RECT(ctypes.Structure):
        _fields_ = [("Left", wintypes.SHORT), ("Top", wintypes.SHORT),
                    ("Right", wintypes.SHORT), ("Bottom", wintypes.SHORT)]

    kernel32 = ctypes.windll.kernel32

    # 先缩窗口，再设缓冲，最后恢复到目标大小（避免缓冲小于窗口导致失败）
    kernel32.SetConsoleWindowInfo(_con_handle, True, ctypes.byref(SMALL_RECT(0, 0, 1, 1)))
    kernel32.SetConsoleScreenBufferSize(_con_handle, COORD(WIN_COLS, WIN_ROWS))
    kernel32.SetConsoleWindowInfo(
        _con_handle, True,
        ctypes.byref(SMALL_RECT(0, 0, WIN_COLS - 1, WIN_ROWS - 1)))

    # 去掉拉伸边框与最大化按钮（经典 conhost 下生效）
    hwnd = kernel32.GetConsoleWindow()
    if hwnd:
        GWL_STYLE = -16
        WS_SIZEBOX = 0x00040000
        WS_MAXIMIZEBOX = 0x00010000
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(hwnd, GWL_STYLE)
        user32.SetWindowLongW(hwnd, GWL_STYLE, style & ~WS_SIZEBOX & ~WS_MAXIMIZEBOX)


def _emit(text):
    if _con_out is not None:
        _con_out.write(text + "\n")
        _con_out.flush()
    else:
        print(text, flush=True)


def _emit_raw(text):
    if _con_out is not None:
        _con_out.write(text)
        _con_out.flush()
    else:
        sys.stdout.write(text)
        sys.stdout.flush()


def _clear_screen():
    _emit_raw("\x1b[2J\x1b[3J\x1b[H")


def _show_cursor(show):
    _emit_raw("\x1b[?25h" if show else "\x1b[?25l")


def render(lines):
    """整帧渲染（超出窗口高度的行截断）"""
    _clear_screen()
    _emit("\n".join(" " + line for line in lines[:WIN_ROWS - 1]))


# ---------------------------------------------------------------
# 模型中心（models.py 统一管理，此处仅镜像状态用于显示）
# ---------------------------------------------------------------
hub = ModelHub()
model_status = hub.status  # 与 models 共享状态字典

# SC 混合系统（第二周）：选中 sc 模式后后台预载，避免首次识别干等
sc_engine = None        # (model, kwargs)，就绪后由 demo.recognize_sc 使用
sc_status = "未加载"    # 未加载 / 加载中 / 就绪 / 失败


def ensure_sc_engine():
    """后台预载 SC-Paraformer（幂等）；已就绪或失败不再重复触发"""
    global sc_engine, sc_status
    if sc_engine is not None or sc_status in ("加载中", "失败"):
        return

    def run():
        global sc_engine, sc_status
        try:
            sc_engine = load_sc_engine(log=lambda m: print(m, flush=True))
            sc_status = "就绪"
        except Exception as e:
            print(f"[UI] SC 模型加载失败: {e}", flush=True)
            sc_status = "失败"

    sc_status = "加载中"
    threading.Thread(target=run, daemon=True).start()


# 最终系统（第四周定版）：SX 提取器 + SC-scx，选中 sx 模式后后台预载
final_engine = None        # (sx_model, sc_model, sc_kwargs)，就绪后由 demo.recognize_sx 使用
final_status = "未加载"    # 未加载 / 加载中 / 就绪 / 失败


def ensure_final_engine():
    """后台预载最终系统（幂等）；已就绪或失败不再重复触发"""
    global final_engine, final_status
    if final_engine is not None or final_status in ("加载中", "失败"):
        return

    def run():
        global final_engine, final_status
        try:
            final_engine = load_final_engine(log=lambda m: print(m, flush=True))
            final_status = "就绪"
        except Exception as e:
            print(f"[UI] 最终系统加载失败: {e}", flush=True)
            final_status = "失败"

    final_status = "加载中"
    threading.Thread(target=run, daemon=True).start()


def load_progress():
    """模型加载总进度 [0,1]：Wespeaker 占前 50%，FunASR 占后 50%"""
    seg = {"等待": 0.0, "加载中": 0.5, "就绪": 1.0, "失败": 1.0}
    return (seg.get(model_status["wespeaker"], 0.0)
            + seg.get(model_status["funasr"], 0.0)) / 2.0


def load_failed():
    return any(v == "失败" for v in model_status.values())


# ---------------------------------------------------------------
# 界面状态
# ---------------------------------------------------------------
MODULES = ["基础演示", "数据集搭建"]

# 构建参数表：(key, 标签, 默认值, 类型)  类型: text/int/float/dir/file
BUILD_PARAMS = [
    ("alias",        "数据集别名",   "run",                 "text"),
    ("clean_dir",    "干净语料目录", bd.DEFAULT_CLEAN_DIR,  "dir"),
    ("transcript",   "转写文件",     bd.DEFAULT_TRANSCRIPT, "file"),
    ("noise_dir",    "噪声库目录",   bd.DEFAULT_NOISE_DIR,  "dir"),
    ("rir_dir",      "RIR 混响目录", bd.DEFAULT_RIR_DIR,    "dir"),
    ("num_train",    "train 样本数", "2000",                "int"),
    ("num_dev",      "dev 样本数",   "200",                 "int"),
    ("reject_ratio", "拒识样本占比", "0.3",                 "float"),
    ("overlap_prob", "双人重叠概率", "0.4",                 "float"),
    ("seed",         "随机种子",     "42",                  "int"),
]

# 数值型参数的 ←→ 步长与进度条量程: key -> (lo, hi, step)
# 有明确含义区间的比值参数才显示进度条；样本数/种子只步进不显示进度条
BUILD_ADJUST = {
    "num_train":    (0, None, 100),
    "num_dev":      (0, None, 50),
    "reject_ratio": (0.0, 1.0, 0.05),
    "overlap_prob": (0.0, 1.0, 0.05),
    "seed":         (0, None, 1),
}
# 显示进度条的参数（有固定量程的比值）
BUILD_BAR = {"reject_ratio", "overlap_prob"}

# 评估参数（数据集在右侧列表选择）：(key, 标签, 类型)
EVAL_PARAMS = [
    ("split",  "划分",     "toggle:train:dev"),
    ("sweep",  "阈值扫描", "text"),
    ("limit",  "条数限制", "num"),
    ("final",  "最终系统", "toggle"),
    ("detail", "逐样本明细", "toggle"),
]


class AppState:
    def __init__(self):
        self.load_gate = True     # 模型加载页（就绪或失败后确认才关闭）
        self.layer = "bar"        # bar=焦点在模块栏 / content=焦点在模块内容
        self.module = 0           # 当前启用的模块
        self.message = ""         # 底部提示

        # 基础演示
        self.wake_path = ""
        self.audio_path = ""
        self.demo_mode = "baseline"  # baseline / sc / sx（三模式）
        self.threshold = DEFAULT_THRESHOLD
        self.demo_focus = 0
        self.is_running = False
        self.run_t0 = 0.0
        self.run_stage = ""
        self.run_stage_t0 = 0.0
        self.result_lines = []

        # 数据集搭建（单页：zone=entry/param/ds）
        self.zone = "entry"       # entry=入口行 / param=信息框调参 / ds=右侧数据集列表
        self.entry = 0            # 0=搭建数据集 1=评估数据集
        self.param_idx = 0
        self.ds_entries = []      # 已有数据集摘要
        self.ds_sel = 0           # 右侧列表当前选中（评估目标）
        self.build_vals = {k: v for k, _, v, _ in BUILD_PARAMS}
        self.eval_split = "dev"
        self.eval_sweep = "0.30:0.75:0.05"
        self.eval_limit = "0"
        self.eval_final = False
        self.eval_detail = True

        # 长任务（构建/评估）运行状态
        self.task = None          # 见 start_task()
        self.task_acked = True    # 完成后是否已被用户按键确认（切回选择视图）


state = AppState()

# 基础演示可选模块（焦点顺序）
DEMO_MODULES = ["wake", "rec", "mode", "threshold", "start"]
# 各模块在焦点列表中的下标（避免魔法数字）
DEMO_IDX = {name: i for i, name in enumerate(DEMO_MODULES)}

# 基础演示三模式（左右键循环）与各模式的推荐阈值
DEMO_MODES = ("baseline", "sc", "sx")
DEMO_MODE_THRESHOLD = {"baseline": DEFAULT_THRESHOLD,
                       "sc": SC_DEFAULT_THRESHOLD,
                       "sx": SX_DEFAULT_THRESHOLD}

# 识别各阶段: (起始进度, 结束进度, 预估秒数, 描述)
RUN_STAGES = {
    "wake": (0.05, 0.30, 1.5, "提取唤醒音频声纹"),
    "rec":  (0.30, 0.55, 1.5, "提取识别音频声纹"),
    "sx":   (0.30, 0.55, 2.0, "SX 提取目标人声"),
    "asr":  (0.55, 0.95, 5.0, "ASR 语音转写"),
}

# 长任务各阶段: (起始进度, 结束进度)
TASK_PHASES = {
    "scan":  (0.00, 0.05),
    "train": (0.05, 0.65),
    "dev":   (0.65, 0.95),
    "infer": (0.05, 0.95),
    "done":  (1.00, 1.00),
}

SPINNER = "-\\|/"


# ---------------------------------------------------------------
# 键盘输入（子进程拥有独立控制台，msvcrt 直接读取其输入）
# ---------------------------------------------------------------
def read_key():
    import msvcrt
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        code = msvcrt.getwch()
        return {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}.get(code, "")
    if ch == "\r":
        return "ENTER"
    if ch == "\n":
        return "CTRL_ENTER"
    if ch == "\x1b":
        return "ESC"
    return ch


def read_key_timeout(sec):
    """带超时的按键读取，超时返回空串（用于周期性刷新界面）"""
    import msvcrt
    deadline = time.time() + sec
    while time.time() < deadline:
        if msvcrt.kbhit():
            return read_key()
        time.sleep(0.02)
    return ""


# ---------------------------------------------------------------
# 模型加载（models 执行，日志走 stdout → 管道 → 原窗口）
# ---------------------------------------------------------------
def load_models_in_background():
    def run():
        hub.load(log=lambda m: print(m, flush=True))

    threading.Thread(target=run, daemon=True).start()


def models_ready():
    return hub.ready()


# ---------------------------------------------------------------
# 模型加载页（就绪前唯一界面）
# ---------------------------------------------------------------
def draw_loading(frame):
    pct = load_progress()
    spin = SPINNER[frame % len(SPINNER)]
    rows = ["", "", ""]
    title = WINDOW_TITLE
    rows.append(" " * max(0, (WIN_COLS - disp_width(title)) // 2)
                + f"{C_TITLE}{title}{C_RESET}")
    rows.append("")
    bar = f"{prog_bar(pct, 60)} {int(pct * 100):3d}%"
    rows.append(" " * max(0, (WIN_COLS - 66) // 2) + bar)
    rows.append("")
    for key, label in (("wespeaker", "声纹模型"), ("funasr", "ASR 模型")):
        st = model_status[key]
        mark = {"就绪": f"{C_OK}[OK]{C_RESET}", "失败": f"{C_ERR}[X]{C_RESET}"}.get(
            st, f"{C_SEL}{spin}{C_RESET}" if st == "加载中" else "  ")
        line = f"{mark} {label}: {st}"
        rows.append(" " * max(0, (WIN_COLS - 24) // 2) + line)
    if load_failed():
        rows.append("")
        rows.append(" " * max(0, (WIN_COLS - 40) // 2)
                    + f"{C_ERR}模型加载失败，详见原窗口日志{C_RESET}")
        rows.append(" " * max(0, (WIN_COLS - 40) // 2)
                    + f"{C_DIM}Enter 仍要进入  ESC 退出{C_RESET}")
    render(rows)


# ---------------------------------------------------------------
# 模块栏
# ---------------------------------------------------------------
def module_bar_rows():
    """顶部模块栏（仅左右竖线，无上下框）：启用模块双竖线+亮色"""
    tabs = []
    for i, name in enumerate(MODULES):
        active = (i == state.module)
        focused = active and state.layer == "bar"
        v = "║" if active else "│"
        color = C_SEL if focused else (C_TITLE if active else C_DIM)
        inner = fit(f" {i + 1} {name} ", 20)
        tabs.append([f"{color}{v}{C_RESET}{inner}{color}{v}{C_RESET}"])
    rows = tabs[0]
    for t in tabs[1:]:
        rows = hcat(rows, t, gap=2)
    return rows


# ---------------------------------------------------------------
# 基础演示页
# ---------------------------------------------------------------
def audio_box(title, path, selected):
    """音频选择盒：单行显示路径，超长中间省略"""
    if path:
        file_line = truncate_middle(path, 40)
    else:
        file_line = f"{C_DIM}(未选择){C_RESET}"
    return make_box(title, [file_line], 45, selected)


def result_box_lines():
    """识别结果盒内容：运行中显示进度，否则显示最终结果"""
    if state.is_running:
        pct = run_progress()
        stage = RUN_STAGES.get(state.run_stage)
        stage_desc = stage[3] if stage else "准备中"
        elapsed = time.time() - state.run_t0
        return [
            f"{prog_bar(pct, 56)} {int(pct * 100):3d}%",
            "",
            f"{C_SEL}{stage_desc}{C_RESET} ... 已用时 {elapsed:.1f}s",
        ]
    if state.result_lines:
        return state.result_lines
    return [f"{C_DIM}尚未开始识别{C_RESET}"]


def demo_rows():
    rows = []
    sel = state.layer == "content"

    # 唤醒音频 + 识别音频（并排）
    wake = audio_box("唤醒音频", state.wake_path,
                     sel and state.demo_focus == DEMO_IDX["wake"])
    rec = audio_box("识别音频", state.audio_path,
                    sel and state.demo_focus == DEMO_IDX["rec"])
    rows += hcat(wake, rec)
    rows.append("")

    # 识别模式（基线 / SC 混合 / 最终系统，左右键循环）
    if state.demo_mode == "sc":
        mode_txt = "SC 混合（两级门控 + SC-Paraformer）"
        mode_note = f"  SC 模型: {sc_status}"
    elif state.demo_mode == "sx":
        mode_txt = "最终系统（SX 提取 + 声纹门控 + SC-scx）"
        mode_note = f"  最终引擎: {final_status}"
    else:
        mode_txt = "基线（声纹比对 + Paraformer）"
        mode_note = ""
    mode_lines = [f"◀ {C_TITLE}{mode_txt}{C_RESET} ▶{C_DIM}{mode_note}{C_RESET}"]
    mode_box = make_box("识别模式", mode_lines, CONTENT_W,
                        sel and state.demo_focus == DEMO_IDX["mode"])
    rows += mode_box
    rows.append("")

    # 拒识阈值 + 开始识别（并排）
    th = [f"余弦相似度阈值: {C_TITLE}{state.threshold:.2f}{C_RESET}  {prog_bar(state.threshold, 20)}",
          f"{C_DIM}相似度 ≥ 阈值判为目标说话人，否则拒识{C_RESET}"]
    th_box = make_box("拒识阈值", th, 56,
                      sel and state.demo_focus == DEMO_IDX["threshold"])
    label = (">>> 开 始 识 别 <<<" if state.demo_focus == DEMO_IDX["start"]
             else "开 始 识 别")
    pad = max(0, (34 - 4 - disp_width(label)) // 2)
    st_box = make_box("", [" " * pad + label], 34,
                      sel and state.demo_focus == DEMO_IDX["start"],
                      min_height=2)
    rows += hcat(th_box, st_box)
    rows.append("")

    # 识别结果（内联，运行中实时刷新）
    rows += make_box("识别结果", result_box_lines(), CONTENT_W,
                     state.is_running, min_height=9)
    return rows


# ---------------------------------------------------------------
# 文本输入 / 文件浏览
# ---------------------------------------------------------------
def edit_text(title, hint_lines, current=""):
    """全屏输入界面：返回输入文本（空 = 取消）"""
    lines = make_box(title, hint_lines + [""], CONTENT_W, True)
    render(lines)
    _emit_raw(f"  当前值: {current}\n  新值 > " if current else "  > ")
    _show_cursor(True)
    try:
        val = input().strip().strip('"').strip("'")
    except (EOFError, KeyboardInterrupt):
        val = ""
    _show_cursor(False)
    return val


def pick_explorer(kind):
    """Ctrl+Enter：系统对话框选文件/目录"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if kind == "file":
            path = filedialog.askopenfilename(
                title="选择文件",
                filetypes=[("音频/文本", "*.wav *.mp3 *.flac *.txt"),
                           ("所有文件", "*.*")])
        else:
            path = filedialog.askdirectory(title="选择目录")
        root.destroy()
    except ImportError:
        state.message = "当前环境无 tkinter，请改用手动键入"
        return ""
    except Exception as e:
        state.message = f"打开对话框失败: {e}"
        return ""
    return os.path.normpath(path) if path else ""


# ---------------------------------------------------------------
# 基础演示：识别流程（demo.recognize 执行，结果内联刷新）
# ---------------------------------------------------------------
def _set_stage(name):
    state.run_stage = name
    state.run_stage_t0 = time.time()


def run_progress():
    """识别总进度 [0,1]：按阶段插值，平滑推进"""
    stage = RUN_STAGES.get(state.run_stage)
    if stage is None:
        return 0.02
    lo, hi, est, _ = stage
    frac = min(1.0, (time.time() - state.run_stage_t0) / est)
    return lo + (hi - lo) * frac


def start_recognition():
    """启动识别（非阻塞，主界面结果盒实时刷新）"""
    if state.is_running:
        state.message = "识别进行中，请稍候"
        return
    if not state.wake_path or not os.path.isfile(state.wake_path):
        state.message = "请先选择唤醒音频"
        return
    if not state.audio_path or not os.path.isfile(state.audio_path):
        state.message = "请先选择识别音频"
        return

    state.is_running = True
    state.run_t0 = time.time()
    state.result_lines = []
    _set_stage("wake")

    def work():
        if state.demo_mode == "sx":
            # 最终系统：SX 提取器 + SC-scx，首次加载较久（~60s+）
            ensure_final_engine()
            while final_engine is None and final_status != "失败":
                time.sleep(0.2)
            if final_engine is None:
                state.result_lines = [
                    f"{C_ERR}最终系统加载失败，详见原窗口日志{C_RESET}"]
                state.is_running = False
                return
            result = recognize_sx(
                hub, final_engine[0], final_engine[1], final_engine[2],
                state.wake_path, state.audio_path, state.threshold,
                gray=SC_GRAY_ZONE,
                on_stage=_set_stage, log=lambda m: print(m, flush=True))
        elif state.demo_mode == "sc":
            # SC 混合系统：引擎未就绪则等待后台预载（首次约 40s）
            ensure_sc_engine()
            while sc_engine is None and sc_status != "失败":
                time.sleep(0.2)
            if sc_engine is None:
                state.result_lines = [
                    f"{C_ERR}SC 模型加载失败，详见原窗口日志{C_RESET}"]
                state.is_running = False
                return
            result = recognize_sc(
                hub, sc_engine[0], sc_engine[1],
                state.wake_path, state.audio_path, state.threshold,
                gray=SC_GRAY_ZONE,
                on_stage=_set_stage, log=lambda m: print(m, flush=True))
        else:
            result = recognize(
                hub, state.wake_path, state.audio_path, state.threshold,
                on_stage=_set_stage, log=lambda m: print(m, flush=True))

        if result["error"]:
            state.result_lines = [f"{C_ERR}识别失败: {result['error']}{C_RESET}"]
        else:
            sim = result["similarity"]
            tag = {"sc": "SC 混合", "sx": "最终系统"}.get(state.demo_mode,
                                                        "基线")
            refine_note = ("（分段精判后）" if result.get("refined") else "")
            lines = [
                f"唤醒: {truncate_middle(state.wake_path, 76)}",
                f"识别: {truncate_middle(state.audio_path, 76)}",
                f"{C_DIM}等价命令: python demo.py --mode {state.demo_mode} "
                f"--wake \"{state.wake_path}\" "
                f"--rec \"{state.audio_path}\" --threshold {state.threshold:.2f}{C_RESET}",
                "",
                f"声纹相似度: {C_TITLE}{sim:.4f}{C_RESET}{refine_note}   阈值: {state.threshold:.2f}",
            ]
            if result["accepted"]:
                text = result["text"] or ""
                lines += [
                    f"判决结果: {C_OK}目标说话人{C_RESET}",
                    f"转写内容: {C_TITLE}{text if text else '（无内容）'}{C_RESET}",
                ]
            else:
                lines += [
                    f"判决结果: {C_WARN}非目标说话人 — 已拒识{C_RESET}",
                    f"转写内容: {C_DIM}（空）{C_RESET}",
                ]
            lines += ["", f"推理耗时: {result['elapsed']:.2f}s   模式: {tag}"]
            state.result_lines = lines

        state.is_running = False

    threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------
# 长任务（构建/评估）：子进程 + [PROGRESS] 解析
# ---------------------------------------------------------------
def start_task(kind, cmd):
    """派生子进程执行等价命令行，线程读取 stdout 驱动进度"""
    if state.task and not state.task["done"]:
        state.message = "任务进行中，请稍候（ESC 中止）"
        return

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    import subprocess
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        cwd=REPO_ROOT, env=env)
    state.task = {
        "kind": kind, "proc": proc, "t0": time.time(),
        "progress": {"phase": "scan" if kind == "build" else "infer",
                     "done": 0, "total": 0},
        "logs": [], "done": False, "rc": None,
    }
    state.task_acked = False
    print(f"[UI] 执行: {' '.join(cmd)}", flush=True)

    def work():
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)  # 日志回传原窗口
            if line.startswith(PROGRESS_PREFIX):
                try:
                    state.task["progress"] = json.loads(
                        line[len(PROGRESS_PREFIX):])
                except ValueError:
                    pass
            elif line.strip():
                state.task["logs"].append(line)
                state.task["logs"] = state.task["logs"][-5:]
        proc.wait()
        state.task["rc"] = proc.returncode
        state.task["done"] = True
        if kind == "build":
            refresh_datasets()

    threading.Thread(target=work, daemon=True).start()


def cancel_task():
    t = state.task
    if t and not t["done"]:
        t["proc"].terminate()
        t["logs"].append("[UI] 已中止")
        state.message = "任务已中止"


def task_running():
    return state.task is not None and not state.task["done"]


def task_progress():
    """长任务总进度 [0,1]：按阶段映射 + 阶段内 done/total 插值"""
    p = state.task["progress"]
    lo, hi = TASK_PHASES.get(p.get("phase"), (0.0, 0.05))
    total = p.get("total") or 0
    frac = (p.get("done", 0) / total) if total else 0.0
    return lo + (hi - lo) * min(1.0, frac)


def task_box_lines():
    """长任务可视化：进度条 + 阶段 + ETA + 日志尾部 / 完成后摘要"""
    t = state.task
    if t is None:
        return [f"{C_DIM}尚未运行{C_RESET}"]
    if not t["done"]:
        p = t["progress"]
        pct = task_progress()
        elapsed = time.time() - t["t0"]
        done, total = p.get("done", 0), p.get("total", 0)
        eta = (elapsed * (total - done) / done) if done else 0
        lines = [
            f"{prog_bar(pct, 48)} {int(pct * 100):3d}%",
            f"阶段: {C_SEL}{p.get('phase', '?')}{C_RESET}   "
            f"{done}/{total or '?'}   已用 {elapsed:.0f}s"
            + (f"   预计剩余 {eta:.0f}s" if done else ""),
            "",
        ]
        lines += [f"{C_DIM}{l}{C_RESET}" for l in t["logs"][-3:]]
        return lines

    ok = t["rc"] == 0
    head = (f"{C_OK}完成{C_RESET}" if ok
            else f"{C_ERR}失败（退出码 {t['rc']}）{C_RESET}")
    lines = [f"{head}   耗时 {time.time() - t['t0']:.0f}s", ""]
    lines += task_summary(t)
    return lines


def task_summary(t):
    """任务完成后的结果摘要（从元数据/评估结果文件读取）"""
    if t["rc"] != 0:
        return [f"{C_ERR}{l}{C_RESET}" for l in t["logs"][-4:]]
    out = []
    if t["kind"] == "build":
        entries = ds.list_datasets()
        if not entries:
            return ["（未找到新数据集）"]
        meta = entries[0]["meta"] or {}
        out.append(f"数据集: {C_TITLE}{entries[0]['name']}{C_RESET}")
        for split, d in meta.get("splits", {}).items():
            out.append(f"{split}: {d['total']} 条 (正 {d['positive']} / 拒 {d['rejection']})"
                       f"  重叠 {d['overlap_pct']:.0%}")
    else:
        p = t["progress"]
        if p.get("phase") == "done" and "cer" in p:
            out.append(f"最优阈值 {p.get('best_threshold'):.2f}: "
                       f"CER {C_TITLE}{p['cer']:.2%}{C_RESET}  "
                       f"拒识率 {C_TITLE}{p['rr']:.2%}{C_RESET}  "
                       f"RTF {p.get('rtf')}")
            out.append(f"结果文件: evals/{p.get('output', '')}")
        out.append(f"{C_DIM}明细与全阈值表见数据集 evals/ 目录{C_RESET}")
    return out


# ---------------------------------------------------------------
# 数据集搭建模块（单页）
# ---------------------------------------------------------------
DATA_LEFT_W = 56          # 左列宽（入口 + 信息框）
DATA_RIGHT_W = CONTENT_W - DATA_LEFT_W - 2   # 右侧数据集列表宽
INFO_H = 14               # 信息框内容行数


def refresh_datasets():
    state.ds_entries = ds.list_datasets()
    if state.ds_sel >= len(state.ds_entries):
        state.ds_sel = max(0, len(state.ds_entries) - 1)


def _num(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def param_visual(key, val):
    """有固定量程的比值参数：进度条 + 当前值"""
    lo, hi, _ = BUILD_ADJUST[key]
    v = _num(val, lo)
    frac = (v - lo) / max(1e-9, hi - lo)
    return f"{prog_bar(frac, 14)} {val}"


def build_param_lines():
    """信息框内容：构建参数列表（比值项带进度条）"""
    lines = []
    for i, (key, label, _, kind) in enumerate(BUILD_PARAMS):
        focused = state.zone == "param" and state.entry == 0 \
            and state.param_idx == i
        mark = f"{C_SEL}>{C_RESET} " if focused else "  "
        val = state.build_vals[key]
        disp = param_visual(key, val) if key in BUILD_BAR else val
        if focused:
            disp = f"{C_SEL}{disp}{C_RESET}"
        lines.append(mark + truncate(f"{label}: {disp}", 50))
    return lines


def eval_param_lines():
    """信息框内容：评估参数列表"""
    cur = state.ds_entries[state.ds_sel] if state.ds_entries else None
    ds_name = cur["name"] if cur else "（无可用数据集）"
    rows = [("目标数据集", ds_name, None),
            ("划分", state.eval_split, 0),
            ("阈值扫描", state.eval_sweep, 1),
            ("条数限制(0=全部)", state.eval_limit, 2),
            ("最终系统", "是" if state.eval_final else "否", 3),
            ("逐样本明细", "是" if state.eval_detail else "否", 4)]
    lines = []
    for label, val, idx in rows:
        focused = (idx is not None and state.zone == "param"
                   and state.entry == 1 and state.param_idx == idx)
        mark = f"{C_SEL}>{C_RESET} " if focused else "  "
        disp = f"{C_SEL}{val}{C_RESET}" if focused else val
        lines.append(mark + truncate(f"{label}: {disp}", 50))
    return lines


def dataset_detail_lines(entry):
    """信息框内容：数据集详细信息（来自 metadata + 最近评估）"""
    meta = entry["meta"] or {}
    lines = [f"{C_TITLE}{entry['name']}{C_RESET}",
             f"创建: {meta.get('created_at', '?')}   别名: {meta.get('alias', '?')}"]
    for split, d in meta.get("splits", {}).items():
        s = (f"{split}: {d['total']} 条 (正 {d['positive']} / 拒 {d['rejection']})"
             f"  重叠 {d['overlap_pct']:.0%}")
        if "low_snr_pct" in d:
            s += f"  低SNR {d['low_snr_pct']:.0%}"
        lines.append(s)
    p = meta.get("params", {})
    if p:
        lines.append(f"{C_DIM}参数: seed={p.get('seed')} reject={p.get('reject_ratio')}"
                     f" overlap={p.get('overlap_prob')} trim={p.get('trim')}{C_RESET}")
    ev = entry.get("latest_eval")
    m = ds.best_metric(ev)
    if m:
        lines.append(f"最近评估: 阈值 {m['threshold']}  CER {C_TITLE}{m['cer']:.1%}{C_RESET}"
                     f"  RR {C_TITLE}{m['rr']:.1%}{C_RESET}"
                     f"  RTF {ev.get('rtf')}")
    else:
        lines.append(f"{C_DIM}（尚未评估）{C_RESET}")
    return lines


def info_box_lines():
    """信息框内容分派：任务视图 > 数据集详情 > 参数列表"""
    if state.task is not None and (task_running() or not state.task_acked):
        return task_box_lines()
    if state.zone == "ds" and state.ds_entries:
        return dataset_detail_lines(state.ds_entries[state.ds_sel])
    if state.entry == 0:
        return build_param_lines()
    return eval_param_lines()


def data_rows():
    """数据集搭建单页：左上入口 + 右侧数据集竖列表 + 下方信息框"""
    sel = state.layer == "content"

    # 入口行（左列顶部，并排）
    entries = []
    for i, name in enumerate(["搭建数据集", "评估数据集"]):
        focused = sel and state.zone == "entry" and state.entry == i
        label = f">>> {name} <<<" if focused else name
        pad = max(0, (27 - 4 - disp_width(label)) // 2)
        entries.append(make_box("", [" " * pad + label], 27, focused))
    entry_rows = hcat(entries[0], entries[1], gap=2)

    # 信息框（左列下方）
    info_title = "信息"
    if state.task is not None and (task_running() or not state.task_acked):
        info_title = "运行状态"
    elif state.zone == "ds":
        info_title = "数据集详情"
    elif state.entry == 0:
        info_title = "构建参数"
    else:
        info_title = "评估参数"
    info_rows = make_box(info_title, info_box_lines(), DATA_LEFT_W,
                         sel and state.zone == "param", min_height=INFO_H)

    left = vcat(entry_rows, [" " * DATA_LEFT_W], info_rows)

    # 右侧：数据集竖列表（只显示 日期_别名）
    lst = []
    for i, e in enumerate(state.ds_entries[:INFO_H]):
        focused = sel and state.zone == "ds" and state.ds_sel == i
        mark = f"{C_SEL}>{C_RESET}" if focused else " "
        name = f"{C_SEL}{e['name']}{C_RESET}" if focused else e["name"]
        lst.append(f"{mark} {name}")
    if not state.ds_entries:
        lst.append(f"{C_DIM}（暂无数据集）{C_RESET}")
    right = make_box("数据集", lst, DATA_RIGHT_W,
                     sel and state.zone == "ds", min_height=len(left) - 2)

    return hcat(left, right)


# ---------------------------------------------------------------
# 主界面渲染
# ---------------------------------------------------------------
def main_signature():
    """动态内容签名：变化时才重绘，避免闪烁"""
    parts = [
        state.layer, str(state.module), state.message,
        str(state.demo_focus), f"{state.threshold:.2f}",
        state.demo_mode, sc_status, final_status,
        state.wake_path, state.audio_path,
        str(state.is_running), state.run_stage,
        "".join(strip_ansi(l) for l in state.result_lines),
        state.zone, str(state.entry), str(state.param_idx), str(state.ds_sel),
        json.dumps(state.build_vals, sort_keys=True),
        state.eval_split, state.eval_sweep, state.eval_limit,
        str(state.eval_final), str(state.eval_detail), str(state.task_acked),
        "|".join(e["name"] for e in state.ds_entries),
    ]
    if state.is_running:
        parts.append(str(int(run_progress() * 56)))
        parts.append(f"{time.time() - state.run_t0:.1f}")
    if state.task:
        t = state.task
        parts += [json.dumps(t["progress"], sort_keys=True),
                  str(t["done"]), "|".join(t["logs"])]
        if not t["done"]:
            parts.append(f"{time.time() - t['t0']:.0f}")
    return "|".join(parts)


def hint_line():
    if state.layer == "bar":
        return f" {C_DIM}←→ 切换模块  Enter 进入  ESC 退出{C_RESET}"
    if state.module == 0:
        return (f" {C_DIM}↑↓ 切换  ←→ 调整  Enter 选择/开始  "
                f"Ctrl+Enter 浏览  ESC 返回{C_RESET}")
    if state.zone == "entry":
        return f" {C_DIM}←→ 切换入口/数据集  ↓ 调整参数  Enter 运行  ESC 返回{C_RESET}"
    if state.zone == "param":
        return (f" {C_DIM}↑↓ 移动  ←→ 调整  Enter 修改文本项  "
                f"Ctrl+Enter 浏览  ESC 返回入口{C_RESET}")
    return f" {C_DIM}↑↓ 选择数据集  ← 返回入口  ESC 返回{C_RESET}"


def draw():
    rows = module_bar_rows()
    rows.append("")
    if state.module == 0:
        rows += demo_rows()
    else:
        rows += data_rows()
    rows.append("")
    if state.message:
        rows.append(f" {C_WARN}{state.message}{C_RESET}")
    rows.append(hint_line())
    render(rows)


# ---------------------------------------------------------------
# 按键处理
# ---------------------------------------------------------------
def handle_bar_key(key):
    if key == "LEFT":
        state.module = (state.module - 1) % len(MODULES)
    elif key == "RIGHT":
        state.module = (state.module + 1) % len(MODULES)
    elif key in ("DOWN", "ENTER"):
        state.layer = "content"
        if state.module == 1:
            refresh_datasets()
    elif key in ("1", "2"):
        state.module = int(key) - 1
        state.layer = "content"
        if state.module == 1:
            refresh_datasets()
    elif key == "ESC":
        return "exit"
    state.message = ""
    return None


def handle_demo_key(key):
    if key == "UP":
        state.demo_focus = (state.demo_focus - 1) % len(DEMO_MODULES)
    elif key == "DOWN":
        state.demo_focus = (state.demo_focus + 1) % len(DEMO_MODULES)
    elif key in ("LEFT", "RIGHT"):
        mod = DEMO_MODULES[state.demo_focus]
        if mod == "threshold":
            delta = 0.05 if key == "RIGHT" else -0.05
            state.threshold = round(min(1.0, max(0.0, state.threshold + delta)), 2)
        elif mod == "mode":
            # 三模式循环：baseline → sc → sx；切到各模式用推荐阈值，
            # sc/sx 首次选中即后台预载对应引擎
            i = DEMO_MODES.index(state.demo_mode)
            state.demo_mode = DEMO_MODES[(i + 1) % len(DEMO_MODES)]
            state.threshold = DEMO_MODE_THRESHOLD[state.demo_mode]
            if state.demo_mode == "sc":
                ensure_sc_engine()
            elif state.demo_mode == "sx":
                ensure_final_engine()
        elif mod in ("wake", "rec"):
            state.demo_focus = (DEMO_IDX["rec"]
                                if state.demo_focus == DEMO_IDX["wake"]
                                else DEMO_IDX["wake"])
    elif key == "ENTER":
        mod = DEMO_MODULES[state.demo_focus]
        if mod in ("wake", "rec"):
            title = "唤醒音频" if mod == "wake" else "识别音频"
            cur = state.wake_path if mod == "wake" else state.audio_path
            val = edit_text(f"键入{title}路径",
                            ["支持 WAV/MP3/FLAC，直接 Enter 取消。"], cur)
            if val:
                if os.path.isfile(val):
                    _set_audio_path(mod, val)
                else:
                    state.message = f"文件不存在: {truncate(val, 60)}"
            return "redraw"
        elif mod == "start":
            start_recognition()
    elif key == "CTRL_ENTER":
        mod = DEMO_MODULES[state.demo_focus]
        if mod in ("wake", "rec"):
            path = pick_explorer("file")
            if path:
                _set_audio_path(mod, path)
            return "redraw"
    elif key == "ESC":
        state.layer = "bar"
    return None


def _set_audio_path(target, path):
    if target == "wake":
        state.wake_path = path
    else:
        state.audio_path = path
    state.message = f"已选择: {os.path.basename(path)}"


def _ack_task():
    """任意选择变化视为已查看任务结果"""
    if state.task and state.task["done"]:
        state.task_acked = True


def handle_data_key(key):
    """数据集搭建模块：zone=entry/param/ds"""
    if task_running():
        if key == "ESC":
            cancel_task()
        return None  # 运行中锁定其它操作

    if key == "ESC":
        _ack_task()
        if state.zone == "entry":
            state.layer = "bar"
        else:
            state.zone = "entry"
        return None

    if state.zone == "entry":
        if key == "LEFT":
            state.entry = 0
            _ack_task()
        elif key == "RIGHT":
            if state.entry == 0:
                state.entry = 1
            elif state.ds_entries:
                state.zone = "ds"  # 评估入口再右移 → 数据集列表
            _ack_task()
        elif key == "DOWN":
            state.zone = "param"
            state.param_idx = 0
        elif key == "ENTER":
            if state.entry == 0:
                launch_build()
            else:
                launch_eval()
    elif state.zone == "param":
        params = BUILD_PARAMS if state.entry == 0 else EVAL_PARAMS
        n = len(params)
        if key == "UP":
            state.param_idx -= 1
            if state.param_idx < 0:
                state.zone = "entry"
                state.param_idx = 0
        elif key == "DOWN":
            state.param_idx += 1
            if state.param_idx >= n:
                state.param_idx = 0
                if state.entry == 0:
                    state.entry = 1
                    state.zone = "entry"
                else:
                    state.zone = "ds"
        elif key in ("LEFT", "RIGHT"):
            adjust_param(-1 if key == "LEFT" else 1)
        elif key == "ENTER":
            return edit_param()
        elif key == "CTRL_ENTER":
            return browse_param()
    elif state.zone == "ds":
        if key == "UP":
            if state.ds_sel > 0:
                state.ds_sel -= 1
            else:
                state.zone = "entry"
                state.entry = 1
            _ack_task()
        elif key == "DOWN":
            if state.ds_sel < len(state.ds_entries) - 1:
                state.ds_sel += 1
            _ack_task()
        elif key == "LEFT":
            state.zone = "entry"
    state.message = ""
    return None


def adjust_param(delta):
    """←→ 可视化调节数值/开关参数"""
    if state.entry == 0:
        key = BUILD_PARAMS[state.param_idx][0]
        if key not in BUILD_ADJUST:
            return
        lo, hi, step = BUILD_ADJUST[key]
        v = _num(state.build_vals[key], lo) + delta * step
        v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        state.build_vals[key] = (f"{v:.2f}" if isinstance(step, float)
                                 else str(int(v)))
    else:
        item = EVAL_PARAMS[state.param_idx][0]
        if item == "split":
            state.eval_split = "train" if state.eval_split == "dev" else "dev"
        elif item == "detail":
            state.eval_detail = not state.eval_detail
        elif item == "final":
            state.eval_final = not state.eval_final
        elif item == "limit":
            v = max(0, int(_num(state.eval_limit)) + delta * 50)
            state.eval_limit = str(v)


def edit_param():
    """Enter 修改文本型参数"""
    if state.entry == 0:
        key, label, _, kind = BUILD_PARAMS[state.param_idx]
        val = edit_text(f"修改 {label}", ["直接 Enter 取消。"],
                        state.build_vals[key])
        if val:
            state.build_vals[key] = val
        return "redraw"
    item, label, _ = EVAL_PARAMS[state.param_idx]
    if item == "sweep":
        val = edit_text("修改 阈值扫描", ["格式 lo:hi:step，直接 Enter 取消。"],
                        state.eval_sweep)
        if val:
            state.eval_sweep = val
        return "redraw"
    if item == "limit":
        val = edit_text("修改 条数限制", ["0 = 全部，直接 Enter 取消。"],
                        state.eval_limit)
        if val:
            state.eval_limit = val
        return "redraw"
    adjust_param(1)  # toggle 项 Enter = 翻转
    return None


def browse_param():
    """Ctrl+Enter 浏览选择路径"""
    if state.entry != 0:
        return None
    key, label, _, kind = BUILD_PARAMS[state.param_idx]
    if kind not in ("dir", "file"):
        return None
    path = pick_explorer(kind)
    if path:
        state.build_vals[key] = ds.rel_path(path)
    return "redraw"


def launch_build():
    v = state.build_vals
    cmd = [sys.executable, "-u",
           os.path.join(REPO_ROOT, "tools", "build_dataset.py"),
           "--alias", v["alias"],
           "--clean-dir", v["clean_dir"],
           "--noise-dir", v["noise_dir"],
           "--rir-dir", v["rir_dir"],
           "--num-train", str(int(_num(v["num_train"], 2000))),
           "--num-dev", str(int(_num(v["num_dev"], 200))),
           "--reject-ratio", f"{_num(v['reject_ratio'], 0.3)}",
           "--overlap-prob", f"{_num(v['overlap_prob'], 0.4)}",
           "--seed", str(int(_num(v["seed"], 42))),
           "--progress"]
    if v["transcript"]:
        cmd += ["--transcript", v["transcript"]]
    start_task("build", cmd)


def launch_eval():
    if not state.ds_entries:
        state.message = "无可用数据集，请先搭建"
        return
    name = state.ds_entries[state.ds_sel]["name"]
    cmd = [sys.executable, "-u",
           os.path.join(REPO_ROOT, "app", "evaluate.py"),
           "--dataset", name,
           "--split", state.eval_split,
           "--sweep", state.eval_sweep,
           "--progress"]
    limit = int(_num(state.eval_limit))
    if limit > 0:
        cmd += ["--limit", str(limit)]
    if state.eval_final:
        cmd += ["--final"]
    if state.eval_detail:
        cmd += ["--detail"]
    start_task("eval", cmd)


def handle_key(key):
    """返回 'exit' / 'redraw' / None"""
    if state.layer == "bar":
        return handle_bar_key(key)
    if state.module == 0:
        return handle_demo_key(key)
    return handle_data_key(key)


# ---------------------------------------------------------------
# UI 主循环（运行于独立子窗口）
# ---------------------------------------------------------------
def run_ui():
    _setup_console_io()
    load_models_in_background()

    last_sig = ""
    last_fix = 0.0
    frame = 0

    try:
        while True:
            if state.load_gate:
                # 模型加载页：仅进度与动画
                if hub.ready():
                    state.load_gate = False
                    last_sig = ""
                    continue
                draw_loading(frame)
                frame += 1
                key = read_key_timeout(0.15)
                if key == "ESC":
                    break
                if key == "ENTER" and load_failed():
                    state.load_gate = False
                continue

            # 内容变化时才重绘（识别/任务进度实时推进）
            sig = main_signature()
            if sig != last_sig:
                draw()
                last_sig = sig

            key = read_key_timeout(0.2)

            # 周期性强制恢复窗口大小（防止用户拉伸 / 终端重排）
            if time.time() - last_fix > 2.0:
                _fix_window_size()
                last_fix = time.time()

            if not key:
                continue

            action = handle_key(key)
            if action == "exit":
                break
            if action == "redraw":
                last_sig = ""  # 输入界面清过屏，强制重绘主界面
    finally:
        _show_cursor(True)
        _clear_screen()
        _emit("退出。")


# ---------------------------------------------------------------
# 启动入口：派生独立窗口，日志回传本窗口
# ---------------------------------------------------------------
def main():
    if UI_CHILD_FLAG in sys.argv:
        run_ui()
        return

    import subprocess
    cmd = [sys.executable, os.path.abspath(__file__), UI_CHILD_FLAG]
    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        cwd=REPO_ROOT,
    )
    print("[UI] 界面已在独立窗口启动，日志输出如下：", flush=True)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        proc.wait()
    print("[UI] 界面窗口已关闭。", flush=True)


if __name__ == "__main__":
    main()
