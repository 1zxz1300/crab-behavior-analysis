"""Plot a framewise timeline comparison among three experts and VAME labels."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

pf = pd.read_csv(BASE_DIR / "perframe_labels_multi.csv")

# 只画前 N 帧，避免太长
N_SHOW = 12000
pf = pf.iloc[:N_SHOW].copy()

cols   = ["behavior_E1", "behavior_E2", "behavior_E3", "behavior_vame"]
yticks = ["Expert 1",   "Expert 2",    "Expert 3",    "VAME"]

# 收集所有标签
beh_all = set()
for c in cols:
    beh_all |= set(pf[c].fillna("None").astype(str))
beh_all = sorted(beh_all)
beh2id = {b: i for i, b in enumerate(beh_all)}
print("行为→ID 映射:", beh2id)

# 每一行转成整数 ID
rows = []
for c in cols:
    arr = pf[c].fillna("None").astype(str).map(beh2id).to_numpy()
    rows.append(arr)

data = np.vstack(rows)   # shape: (4, N)

# ===== 绘图：修改部分开始 =====
fig, ax = plt.subplots(figsize=(12, 3))

im = ax.imshow(
    data,
    aspect="auto",
    interpolation="nearest",
    vmin=-0.5,
    vmax=len(beh_all) - 0.5
)

ax.set_yticks(np.arange(len(cols)))
ax.set_yticklabels(yticks)
ax.set_xlabel("Frame")

# colorbar 显示具体行为名称
cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(beh_all)))
cbar.set_label("Behavior")
cbar.ax.set_yticklabels(beh_all)
# ===== 绘图：修改部分结束 =====

plt.tight_layout()
plt.savefig(BASE_DIR / "timeline_3experts_vs_vame.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ 已保存 timeline_3experts_vs_vame.png")
