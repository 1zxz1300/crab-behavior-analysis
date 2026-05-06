"""Plot framewise confusion matrices between expert labels and VAME labels and compute strict agreement."""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 你需要改的配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "perframe_labels_multi.xlsx"   # 你的xlsx路径
SHEET_NAME = 0                              # 0=第一个sheet；或写sheet名，如 "Sheet1"

# 按你的xlsx真实列名改这里（非常关键）
EXPERT_COLS = {
    "E1": "behavior_E1",
    "E2": "behavior_E2",
    "E3": "behavior_E3",
}
VAME_COL = "behavior_vame"

# （可选）固定行为显示顺序；不填(None)就自动从数据中收集
BEHAVIOR_ORDER = None

# 输出图片文件名前缀
OUT_PREFIX = "confusion"

# =========================
# 工具函数
# =========================
NONE_TOKENS = {"none", "nan", "null", "", " "}

def _normalize_label(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def _is_none_like(s: str) -> bool:
    return s.strip().lower() in NONE_TOKENS

def framewise_confusion_row_normalized(df: pd.DataFrame, expert_col: str, vame_col: str, order=None):
    """
    严格逐帧交叉表：
    行=expert标签，列=vame标签，值=该expert标签下分到各vame标签的帧占比（行归一）
    """
    sub = df[[expert_col, vame_col]].copy()
    sub[expert_col] = sub[expert_col].map(_normalize_label)
    sub[vame_col] = sub[vame_col].map(_normalize_label)

    # 不显示 None：直接剔除包含 None/空 的帧
    mask = (~sub[expert_col].map(_is_none_like)) & (~sub[vame_col].map(_is_none_like))
    sub = sub.loc[mask].copy()

    # 标签集合（不包含None）
    if order is None:
        labels = list(pd.unique(pd.concat([sub[expert_col], sub[vame_col]], ignore_index=True)))
        labels = [l for l in labels if not _is_none_like(l)]
    else:
        labels = [l for l in order if not _is_none_like(str(l))]

    # 逐帧计数交叉表
    ct = pd.crosstab(sub[expert_col], sub[vame_col])

    # 补齐缺失行列（保证显示一致）
    ct = ct.reindex(index=labels, columns=labels, fill_value=0)

    # 行归一化（row-normalized）
    row_sum = ct.sum(axis=1).replace(0, np.nan)
    ct_row = ct.div(row_sum, axis=0).fillna(0)

    # 严格逐帧一致率
    acc = (sub[expert_col].values == sub[vame_col].values).mean() if len(sub) else np.nan

    return ct_row, acc, len(sub)

def plot_heatmap_second_style(mat: pd.DataFrame, expert_name: str, out_png: str):
    """
    字体加大，其它绘图参数保持不变
    """
    # 仅字体大小配置（可按示例图再微调）
    FONT_TITLE = 20
    FONT_LABEL = 16
    FONT_TICK  = 14
    FONT_ANNOT = 14

    plt.figure(figsize=(8.5, 6.5))
    ax = sns.heatmap(
        mat,
        cmap="viridis",
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": FONT_ANNOT},  # 数值标注字体
        linewidths=0,        # 保持：无格子间空隙
        cbar=True
    )

    ax.set_title(f"Expert {expert_name} vs VAME", fontsize=FONT_TITLE)
    ax.set_xlabel("VAME behavior", fontsize=FONT_LABEL)
    ax.set_ylabel(f"Expert {expert_name} behavior", fontsize=FONT_LABEL)

    # 坐标刻度字体
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # 色条刻度字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def main():
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    # 校验列名
    missing = []
    for _, col in EXPERT_COLS.items():
        if col not in df.columns:
            missing.append(col)
    if VAME_COL not in df.columns:
        missing.append(VAME_COL)
    if missing:
        raise KeyError(
            "以下列名在xlsx里找不到，请检查/修改 EXPERT_COLS 和 VAME_COL：\n"
            + "\n".join(missing)
            + "\n\n当前xlsx列名有：\n"
            + ", ".join(df.columns)
        )

    for e, col in EXPERT_COLS.items():
        mat, acc, n_used = framewise_confusion_row_normalized(
            df=df, expert_col=col, vame_col=VAME_COL, order=BEHAVIOR_ORDER
        )
        out_png = BASE_DIR / f"{OUT_PREFIX}_expert{e}_vs_vame_row.png"
        plot_heatmap_second_style(mat, expert_name=e, out_png=out_png)

        if np.isnan(acc):
            print(f"[Expert {e}] No valid frames after removing None-like labels.")
        else:
            print(f"[Expert {e}] valid frames used (None removed): {n_used}")
            print(f"[Expert {e}] frame-wise strict agreement with VAME: {acc:.3f}")
        print(f"[Expert {e}] saved: {out_png}\n")

if __name__ == "__main__":
    main()






