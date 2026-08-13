# ==============================================================
# JigBas 数据集注册与路径解析
#
# 目录约定：
#   原始语料:  <项目>/Datasets/            （仓库内，gitignore）
#   生成数据集: <项目>/Temp/Datasets/<创建时间>_<别名>/
#               ├─ train/ dev/ *_manifest.jsonl
#               ├─ metadata.json          （构建时自动生成）
#               └─ evals/<评估时间>.json  （每次评估的结果）
#   latest.txt 记录最近构建的数据集文件夹名（Windows 不用 symlink）
#
# 路径策略：存储与显示一律用相对路径（相对项目根 JigBas/JigBas），
# 运行时经 resolve_path 解析为绝对路径。
# ==============================================================

import json
import os
import time

from lib.paths import REPO_ROOT

DATASETS_ROOT_REL = os.path.join("..", "..", "Temp", "Datasets")
LATEST_FILE = "latest.txt"
METADATA_FILE = "metadata.json"
EVALS_DIR = "evals"


# ---------------------------------------------------------------
# 路径
# ---------------------------------------------------------------
def resolve_path(p, base=REPO_ROOT):
    """相对路径按项目根解析为绝对路径；绝对路径原样返回"""
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def rel_path(p, base=REPO_ROOT):
    """绝对路径尽量转回相对项目根的相对路径（用于存储/显示）"""
    if not p or not os.path.isabs(p):
        return p
    try:
        return os.path.relpath(p, base)
    except ValueError:
        return p  # 跨盘符时无法相对化，保留绝对路径


def datasets_root():
    return resolve_path(DATASETS_ROOT_REL)


# ---------------------------------------------------------------
# 数据集枚举与定位
# ---------------------------------------------------------------
def read_metadata(folder):
    path = os.path.join(folder, METADATA_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_metadata(folder, meta):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, METADATA_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def list_datasets(root=None):
    """
    扫描数据集根目录，返回按创建时间倒序的摘要列表：
    [{name, path, alias, created, meta, latest_eval}]
    无 metadata.json 的文件夹也会列出（meta=None）。
    """
    root = root or datasets_root()
    out = []
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        meta = read_metadata(folder)
        out.append({
            "name": name,
            "path": folder,
            "alias": (meta or {}).get("alias", ""),
            "created": (meta or {}).get("created_at", ""),
            "meta": meta,
            "latest_eval": latest_eval(folder),
        })
    out.sort(key=lambda d: d["name"], reverse=True)
    return out


def latest_eval(folder):
    """数据集中最新的评估结果 JSON（无则 None）"""
    d = os.path.join(folder, EVALS_DIR)
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d)
                   if f.endswith(".json") and not f.endswith("_detail.jsonl"))
    if not files:
        return None
    try:
        with open(os.path.join(d, files[-1]), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def update_latest(name, root=None):
    root = root or datasets_root()
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, LATEST_FILE), "w", encoding="utf-8") as f:
        f.write(name)


def resolve_dataset(name, root=None):
    """
    定位数据集文件夹：
      "latest"（默认）→ latest.txt 指向的文件夹
      精确文件夹名 → 唯一别名 → 唯一创建时间前缀
    失败时抛出 ValueError（消息列出候选）。
    """
    root = root or datasets_root()
    entries = list_datasets(root)
    if not entries:
        raise ValueError(f"数据集根目录为空: {root}（请先构建数据集）")

    if not name or name == "latest":
        latest_path = os.path.join(root, LATEST_FILE)
        if os.path.isfile(latest_path):
            latest = open(latest_path, encoding="utf-8").read().strip()
            for e in entries:
                if e["name"] == latest:
                    return e
        return entries[0]  # latest.txt 缺失/失效时退化为最新

    for e in entries:
        if e["name"] == name:
            return e
    hits = [e for e in entries if e["alias"] == name]
    if not hits:
        hits = [e for e in entries if e["name"].startswith(name)]
    if len(hits) == 1:
        return hits[0]
    candidates = ", ".join(e["name"] for e in entries[:8])
    if not hits:
        raise ValueError(f"找不到数据集 '{name}'，候选: {candidates}")
    raise ValueError(f"'{name}' 匹配到多个数据集，请用完整文件夹名。候选: {candidates}")


def best_metric(ev):
    """
    评估结果中“最优阈值”对应的指标 dict（UI / 列表摘要共用）。
    兼容两种结构：metrics=list（普通评估）与 metrics=dict<配置名,[阈值指标]>
    （--final 的 5 配置消融）。无则返回 None。
    """
    if not ev:
        return None
    best = ev.get("best_threshold")
    ms = ev.get("metrics")
    if isinstance(ms, dict):
        res = [m for group in ms.values() for m in group
               if isinstance(m, dict) and m.get("threshold") == best]
    else:
        res = [m for m in (ms or [])
               if isinstance(m, dict) and m.get("threshold") == best]
    return res[0] if res else None


def one_line_summary(entry):
    """数据集列表的一行摘要（UI / 控制台共用）"""
    meta = entry["meta"] or {}
    splits = meta.get("splits", {})
    n_train = splits.get("train", {}).get("total", "?")
    n_dev = splits.get("dev", {}).get("total", "?")
    ev_txt = ""
    m = best_metric(entry.get("latest_eval"))
    if m:
        ev_txt = f" | CER {m['cer']:.1%} / RR {m['rr']:.1%}"
    return f"{entry['name']} | train {n_train} / dev {n_dev}{ev_txt}"


def new_dataset_folder(alias, root=None, now=None):
    """创建 <时间>_<别名> 文件夹，返回 (绝对路径, 文件夹名)"""
    root = root or datasets_root()
    stamp = time.strftime("%Y%m%d_%H%M", now or time.localtime())
    safe_alias = "".join(c if (c.isalnum() or c in "-_") else "_"
                         for c in (alias or "run"))
    name = f"{stamp}_{safe_alias}"
    folder = os.path.join(root, name)
    suffix = 2
    while os.path.exists(folder):  # 同一分钟重名时加序号
        name = f"{stamp}_{safe_alias}_{suffix}"
        folder = os.path.join(root, name)
        suffix += 1
    os.makedirs(folder)
    return folder, name
