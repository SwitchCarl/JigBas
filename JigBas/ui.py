# ==============================================================
# JigBas UI — 抗干扰语音指令识别系统界面
# 控制台风格界面：方向键 + 回车 控制
#
# - 启动时派生独立控制台窗口运行界面，日志回传原窗口
# - 窗口大小固定（定时强制恢复 + 移除拉伸样式）
# - 模块由制表符盒子包围，选中模块亮青色双边框高亮
# - 识别结果内联显示在主界面，识别过程进度实时刷新
# - 音频模块：回车 = 键入路径，Ctrl+回车 = Explorer 选择
# - 支持唤醒音频 + 识别音频：声纹相似度达标才转写，否则拒识
# ==============================================================

import os
import re
import sys
import time
import threading
import logging
import warnings
import unicodedata
from contextlib import redirect_stdout

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
FUNASR_MODEL_DIR = os.path.join(MODELS_DIR, "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

UI_CHILD_FLAG = "--ui-child"
WINDOW_TITLE = "JigBas — 抗干扰语音指令识别系统"

# 窗口尺寸（字符）
WIN_COLS = 94
WIN_ROWS = 30
CONTENT_W = WIN_COLS - 2
LEFT_W = 30  # 左侧“模型状态”列宽

# ANSI 颜色
C_RESET = "\x1b[0m"
C_DIM = "\x1b[90m"
C_SEL = "\x1b[96m"
C_TITLE = "\x1b[97m"
C_OK = "\x1b[92m"
C_WARN = "\x1b[93m"
C_ERR = "\x1b[91m"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


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


def make_box(title, lines, width, selected=False, min_height=None):
    """生成一个盒子，返回字符串行列表（每行可见宽度 == width）"""
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
    """整帧渲染"""
    _clear_screen()
    _emit("\n".join(" " + line for line in lines))


# ---------------------------------------------------------------
# 全局模型引用（延迟加载）
# ---------------------------------------------------------------
wespeaker_model = None
funasr_model = None

# 模型加载状态: "等待" / "加载中" / "就绪" / "失败"
model_status = {"wespeaker": "等待", "funasr": "等待"}


def load_progress():
    """模型加载总进度 [0,1]：Wespeaker 占前 50%，FunASR 占后 50%"""
    seg = {"等待": 0.0, "加载中": 0.5, "就绪": 1.0, "失败": 1.0}
    return (seg.get(model_status["wespeaker"], 0.0)
            + seg.get(model_status["funasr"], 0.0)) / 2.0


# ---------------------------------------------------------------
# 界面状态
# ---------------------------------------------------------------
class AppState:
    def __init__(self):
        self.wake_path = ""           # 唤醒音频
        self.audio_path = ""          # 识别音频
        self.threshold = 0.50         # 拒识阈值（余弦相似度）
        self.focus = 0                # 当前选中模块
        self.message = ""             # 底部提示
        # 识别状态（后台线程驱动，主界面内联刷新）
        self.is_running = False
        self.run_t0 = 0.0
        self.run_stage = ""           # 识别阶段: wake / rec / asr
        self.run_stage_t0 = 0.0
        self.result_lines = []        # 最终识别结果（空 = 尚未识别）

state = AppState()

# 主界面可选模块（焦点顺序）
MODULES = ["wake", "rec", "threshold", "start"]

# 识别各阶段: (起始进度, 结束进度, 预估秒数, 描述)
RUN_STAGES = {
    "wake": (0.05, 0.30, 1.5, "提取唤醒音频声纹"),
    "rec":  (0.30, 0.55, 1.5, "提取识别音频声纹"),
    "asr":  (0.55, 0.95, 5.0, "ASR 语音转写"),
}


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
# 模型加载（日志走 stdout → 管道 → 原窗口）
# ---------------------------------------------------------------
def load_models_in_background():
    def run():
        global wespeaker_model, funasr_model
        try:
            model_status["wespeaker"] = "加载中"
            print("[模型] 正在加载 Wespeaker 声纹模型...", flush=True)
            import wespeaker
            with open(os.devnull, "w") as f, redirect_stdout(f):
                wespeaker_model = wespeaker.load_model("chinese")
            model_status["wespeaker"] = "就绪"
            print("[模型] Wespeaker 声纹模型加载完成", flush=True)

            model_status["funasr"] = "加载中"
            print("[模型] 正在加载 FunASR ASR 模型...", flush=True)
            from funasr import AutoModel
            if os.path.isdir(FUNASR_MODEL_DIR) and os.listdir(FUNASR_MODEL_DIR):
                funasr_model = AutoModel(model=FUNASR_MODEL_DIR)
            else:
                funasr_model = AutoModel(
                    model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
                )
            model_status["funasr"] = "就绪"
            print("[模型] FunASR ASR 模型加载完成", flush=True)
        except Exception as e:
            for k, v in model_status.items():
                if v != "就绪":
                    model_status[k] = "失败"
            state.message = "模型加载失败，详见原窗口日志"
            print(f"[模型] 加载失败: {e}", flush=True)

    threading.Thread(target=run, daemon=True).start()


def models_ready():
    return model_status["wespeaker"] == "就绪" and model_status["funasr"] == "就绪"


# ---------------------------------------------------------------
# 主界面渲染
# ---------------------------------------------------------------
def status_mark(s):
    marks = {"等待": (C_DIM, "[  ]"), "加载中": (C_WARN, "[~ ]"),
             "就绪": (C_OK, "[OK]"), "失败": (C_ERR, "[X ]")}
    color, mark = marks.get(s, (C_DIM, "[??]"))
    return f"{color}{mark}{C_RESET} {s}"


def audio_box(title, path, selected):
    """音频模块：单行显示路径，超长中间省略"""
    if path:
        file_line = truncate_middle(path, 40)
    else:
        file_line = f"{C_DIM}(未选择){C_RESET}"
    lines = [file_line,
             f"{C_DIM}回车: 键入路径  Ctrl+回车: Explorer{C_RESET}"]
    return make_box(title, lines, 45, selected)


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


def main_signature():
    """主界面动态内容签名：变化时才重绘，避免闪烁"""
    parts = [
        model_status["wespeaker"], model_status["funasr"],
        str(state.focus), f"{state.threshold:.2f}",
        state.wake_path, state.audio_path, state.message,
        str(state.is_running), state.run_stage,
        "".join(strip_ansi(l) for l in state.result_lines),
    ]
    if state.is_running:
        # 运行中按进度条格数与 0.1s 粒度刷新
        parts.append(str(int(run_progress() * 56)))
        parts.append(f"{time.time() - state.run_t0:.1f}")
    return "|".join(parts)


def draw_main():
    rows = []

    # 唤醒音频 + 识别音频（并排）
    wake = audio_box("唤醒音频", state.wake_path, state.focus == 0)
    rec = audio_box("识别音频", state.audio_path, state.focus == 1)
    rows += hcat(wake, rec)
    rows.append("")

    # 右列：拒识阈值 + 开始识别
    right_w = CONTENT_W - LEFT_W - 2
    th = [f"余弦相似度阈值: {C_TITLE}{state.threshold:.2f}{C_RESET}  {prog_bar(state.threshold, 20)}",
          f"{C_DIM}相似度 ≥ 阈值判为目标说话人，否则拒识。←→ 微调 ±0.05{C_RESET}"]
    right_rows = make_box("拒识阈值", th, right_w, state.focus == 2)
    right_rows.append(" " * right_w)
    label = ">>> 开 始 识 别 <<<" if state.focus == 3 else "开 始 识 别"
    pad = max(0, (right_w - 4 - disp_width(label)) // 2)
    right_rows += make_box("", [" " * pad + label], right_w, state.focus == 3)

    # 左列：模型状态（高度与右列匹配）
    pct = load_progress()
    ms = [f"{prog_bar(pct, 16)} {int(pct * 100):3d}%",
          "",
          f"声纹: {status_mark(model_status['wespeaker'])}",
          f"ASR:  {status_mark(model_status['funasr'])}"]
    left_box = make_box("模型状态", ms, LEFT_W, False,
                        min_height=len(right_rows) - 2)
    rows += hcat(left_box, right_rows)
    rows.append("")

    # 识别结果（内联，运行中实时刷新）
    rows += make_box("识别结果", result_box_lines(), CONTENT_W,
                     state.is_running, min_height=8)

    # 底部提示
    rows.append("")
    if state.message:
        rows.append(f" {C_WARN}{state.message}{C_RESET}")
    rows.append(f" {C_DIM}↑↓ 切换模块  ←→ 调整  回车 键入路径  Ctrl+回车 Explorer  ESC 退出{C_RESET}")
    render(rows)


# ---------------------------------------------------------------
# 音频路径选择
# ---------------------------------------------------------------
def pick_path_manual(target):
    """回车：键入路径"""
    title = "唤醒音频" if target == "wake" else "识别音频"
    lines = make_box(f"手动键入{title}路径", [
        "请输入音频文件的完整路径（支持 WAV/MP3/FLAC），",
        "直接回车取消。",
        "",
    ], CONTENT_W, True)
    render(lines)
    _emit_raw("  路径 > ")
    _show_cursor(True)
    try:
        path = input().strip().strip('"').strip("'")
    except (EOFError, KeyboardInterrupt):
        path = ""
    _show_cursor(False)
    if not path:
        return
    if os.path.isfile(path):
        _set_audio_path(target, path)
    else:
        state.message = f"文件不存在: {truncate(path, 60)}"


def pick_path_explorer(target):
    """Ctrl+回车：通过系统 Explorer 对话框选择"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac"), ("所有文件", "*.*")],
        )
        root.destroy()
    except ImportError:
        state.message = "当前环境无 tkinter，请改用手动键入路径"
        return
    except Exception as e:
        state.message = f"打开文件对话框失败: {e}"
        return
    if path:
        _set_audio_path(target, os.path.normpath(path))


def _set_audio_path(target, path):
    if target == "wake":
        state.wake_path = path
    else:
        state.audio_path = path
    state.message = f"已选择: {os.path.basename(path)}"


# ---------------------------------------------------------------
# 识别流程（后台线程驱动，结果内联刷新到主界面）
# ---------------------------------------------------------------
def _cosine(a, b):
    import numpy as np
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _extract_embedding(path):
    """提取声纹嵌入（屏蔽库自身的刷屏输出）"""
    with open(os.devnull, "w") as f, redirect_stdout(f):
        emb = wespeaker_model.extract_embedding(path)
    return emb.cpu().numpy() if hasattr(emb, "cpu") else emb


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
    if not models_ready():
        state.message = "模型尚未加载完成，请稍候"
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
        holder = {}
        try:
            print(f"[识别] 唤醒音频: {state.wake_path}", flush=True)
            print(f"[识别] 识别音频: {state.audio_path}", flush=True)

            _set_stage("wake")
            print("[识别] 提取唤醒音频声纹...", flush=True)
            emb_wake = _extract_embedding(state.wake_path)

            _set_stage("rec")
            print("[识别] 提取识别音频声纹...", flush=True)
            emb_rec = _extract_embedding(state.audio_path)

            sim = _cosine(emb_wake, emb_rec)
            holder["sim"] = sim
            print(f"[识别] 余弦相似度: {sim:.4f} (阈值 {state.threshold:.2f})", flush=True)

            if sim >= state.threshold:
                holder["accepted"] = True
                _set_stage("asr")
                print("[识别] 判为目标说话人，执行 ASR 转写...", flush=True)
                asr_result = funasr_model.generate(input=state.audio_path)
                text = ""
                if asr_result and len(asr_result) > 0:
                    text = (asr_result[0].get("text", "")
                            if isinstance(asr_result[0], dict) else str(asr_result[0]))
                holder["text"] = text
            else:
                holder["accepted"] = False
                print("[识别] 判为非目标说话人，拒识（输出空）", flush=True)
        except Exception as e:
            holder["error"] = str(e)
            print(f"[识别] 失败: {e}", flush=True)

        elapsed = time.time() - state.run_t0
        print(f"[识别] 完成，总耗时 {elapsed:.2f}s", flush=True)

        if "error" in holder:
            state.result_lines = [f"{C_ERR}识别失败: {holder['error']}{C_RESET}"]
        else:
            sim = holder.get("sim", 0.0)
            lines = [
                f"唤醒: {truncate_middle(state.wake_path, 76)}",
                f"识别: {truncate_middle(state.audio_path, 76)}",
                "",
                f"声纹相似度: {C_TITLE}{sim:.4f}{C_RESET}   阈值: {state.threshold:.2f}",
            ]
            if holder.get("accepted"):
                text = holder.get("text", "")
                lines += [
                    f"判决结果: {C_OK}目标说话人{C_RESET}",
                    f"转写内容: {C_TITLE}{text if text else '（无内容）'}{C_RESET}",
                ]
            else:
                lines += [
                    f"判决结果: {C_WARN}非目标说话人 — 已拒识{C_RESET}",
                    f"转写内容: {C_DIM}（空）{C_RESET}",
                ]
            lines += ["", f"推理耗时: {elapsed:.2f}s"]
            state.result_lines = lines

        state.is_running = False

    threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------
# UI 主循环（运行于独立子窗口）
# ---------------------------------------------------------------
def run_ui():
    _setup_console_io()
    load_models_in_background()

    last_sig = ""
    last_fix = 0.0

    try:
        while True:
            # 内容变化时才重绘（加载/识别进度实时推进）
            sig = main_signature()
            if sig != last_sig:
                draw_main()
                last_sig = sig

            key = read_key_timeout(0.2)

            # 周期性强制恢复窗口大小（防止用户拉伸 / 终端重排）
            if time.time() - last_fix > 2.0:
                _fix_window_size()
                last_fix = time.time()

            if not key:
                continue

            if key == "UP":
                state.focus = (state.focus - 1) % len(MODULES)
                state.message = ""
            elif key == "DOWN":
                state.focus = (state.focus + 1) % len(MODULES)
                state.message = ""
            elif key in ("LEFT", "RIGHT"):
                mod = MODULES[state.focus]
                if mod == "threshold":
                    delta = 0.05 if key == "RIGHT" else -0.05
                    state.threshold = round(min(1.0, max(0.0, state.threshold + delta)), 2)
                elif mod in ("wake", "rec"):
                    state.focus = 1 - state.focus  # 左右互换
            elif key == "ENTER":
                mod = MODULES[state.focus]
                if mod in ("wake", "rec"):
                    pick_path_manual(mod)
                    last_sig = ""  # 键入界面清过屏，强制重绘主界面
                elif mod == "start":
                    start_recognition()
            elif key == "CTRL_ENTER":
                mod = MODULES[state.focus]
                if mod in ("wake", "rec"):
                    pick_path_explorer(mod)
                    last_sig = ""  # Explorer 返回后强制重绘主界面
            elif key == "ESC":
                break
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
        cwd=PROJECT_ROOT,
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
