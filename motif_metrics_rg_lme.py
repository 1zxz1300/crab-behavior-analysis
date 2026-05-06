"""Compute per-motif radius of gyration and relative limb motion energy from SLEAP CSV exports."""

import os
import re
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 你要改的配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
OUT_METRICS = BASE_DIR / "metrics_rg_lme.csv"      # 输出结果表路径
FPS = 25.0
MIN_SCORE = 0.2

BODY = "tou"

# 计算 LME_rel 时纳入的附肢关键点
# 说明：根据当前 SLEAP 导出格式，额外加入左右尾肢 weizhi1 / weizhi2
APPENDAGES = [
    "aozuqian1", "aozuqian2",
    "aozuhou1", "aozuhou2",
    "diyi1", "diyi2",
    "dier1", "dier2",
    "disan1", "disan2",
    "weizhi1", "weizhi2",
]

# 文件名解析：支持 motif01_rep01.csv / motif1-rep8.csv / Motif_01 (7)_rep02.csv 等
FNAME_REGEX = re.compile(r"motif[_\- ]*(\d+).*rep[_\- ]*(\d+)", re.IGNORECASE)

# ✅ 直接在脚本里写映射（把下面按你的真实划分改）
# 规范：键必须是 motif01~motif10 这种形式（两位数）
MOTIF_TO_TYPE = {
    "motif01": "mobile",
    "motif03": "mobile",
    "motif05": "tactile",
    "motif06": "tactile",
    "motif07": "mobile",
    "motif08": "mobile",
    "motif09": "mobile",
    "motif10": "mobile",
    "motif13": "mobile",
    "motif14": "mobile",
}
# =========================


def _valid_xy(df: pd.DataFrame, kp: str, min_score: float) -> Tuple[np.ndarray, np.ndarray]:
    x = df.get(f"{kp}.x", pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
    y = df.get(f"{kp}.y", pd.Series([np.nan] * len(df))).to_numpy(dtype=float)

    score_col = f"{kp}.score"
    if score_col in df.columns:
        s = df[score_col].to_numpy(dtype=float)
        mask = (s >= min_score) & np.isfinite(x) & np.isfinite(y)
    else:
        mask = np.isfinite(x) & np.isfinite(y)

    x2 = x.copy()
    y2 = y.copy()
    x2[~mask] = np.nan
    y2[~mask] = np.nan
    return x2, y2


def compute_radius_of_gyration(body_x: np.ndarray, body_y: np.ndarray) -> float:
    mask = np.isfinite(body_x) & np.isfinite(body_y)
    if mask.sum() < 5:
        return np.nan
    x = body_x[mask]
    y = body_y[mask]
    xm, ym = np.mean(x), np.mean(y)
    return float(np.sqrt(np.mean((x - xm) ** 2 + (y - ym) ** 2)))


def compute_lme_rel(df: pd.DataFrame, fps: float, min_score: float) -> float:
    """附肢强度：相对身体中心的平方速度能量（平均）"""
    dt = 1.0 / fps

    bx, by = _valid_xy(df, BODY, min_score=min_score)
    if np.isfinite(bx).sum() < 5:
        return np.nan

    per_kp_means = []
    for kp in APPENDAGES:
        kx, ky = _valid_xy(df, kp, min_score=min_score)

        # 相对身体中心坐标（剔除整体平移）
        rx = kx - bx
        ry = ky - by

        dx = np.diff(rx)
        dy = np.diff(ry)
        valid = np.isfinite(dx) & np.isfinite(dy)
        if valid.sum() < 3:
            continue

        v = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2) / dt
        per_kp_means.append(np.mean(v ** 2))

    if len(per_kp_means) == 0:
        return np.nan
    return float(np.mean(per_kp_means))


def parse_motif_rep(fname: str) -> Tuple[str, str]:
    """
    从文件名抓 motif编号与rep编号，并统一成 motif01、rep01 这种格式。
    """
    m = FNAME_REGEX.search(fname)
    if not m:
        raise ValueError(
            f"无法从文件名解析 motif/rep：{fname}\n"
            f"请改成类似 motif01_rep01.csv，或修改 FNAME_REGEX。"
        )
    motif_num = int(m.group(1))
    rep_num = int(m.group(2))
    motif = f"motif{motif_num:02d}"
    rep = f"rep{rep_num:02d}"
    return motif, rep


def main():
    rows = []

    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith(".csv"):
            continue

        motif, rep = parse_motif_rep(fname)
        etype = MOTIF_TO_TYPE.get(motif, "unknown")

        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path)

        if "frame_idx" in df.columns:
            df = df.sort_values("frame_idx").reset_index(drop=True)

        bx, by = _valid_xy(df, BODY, min_score=MIN_SCORE)
        rg = compute_radius_of_gyration(bx, by)
        lme = compute_lme_rel(df, fps=FPS, min_score=MIN_SCORE)

        rows.append({
            "file": fname,
            "motif": motif,
            "rep": rep,
            "etype": etype,
            "n_frames": len(df),
            "Rg": rg,
            "LME_rel": lme
        })

    out = pd.DataFrame(rows).sort_values(["motif", "rep"]).reset_index(drop=True)
    out.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")

    print(f"[OK] 指标表已输出：{OUT_METRICS}")
    print(out.head(10))
    print("\netype 统计：")
    print(out["etype"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
