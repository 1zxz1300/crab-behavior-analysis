"""Convert a community transition matrix into a Gephi-ready edge list with filtered and rescaled weights."""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

mat_path = BASE_DIR / "meanMG.xlsx"   # 你的 .xlsx 文件

# 用 read_excel 读取，注意 engine='openpyxl'
df = pd.read_excel(mat_path, index_col=0, engine="openpyxl")

eps_zero = 1e-5
P = df.to_numpy(dtype=float)
P[P < eps_zero] = 0.0

# 比如做平方根 + 归一化变换
P_sqrt = np.zeros_like(P)
mask_nonzero = P > 0
P_sqrt[mask_nonzero] = np.sqrt(P[mask_nonzero])

thr_keep = 1e-3
mask_keep = P > thr_keep

vals = P_sqrt[mask_keep]
w_min, w_max = vals.min(), vals.max()

W = np.zeros_like(P_sqrt)
W[mask_keep] = (P_sqrt[mask_keep] - w_min) / (w_max - w_min)

rows, cols = np.where(mask_keep)
sources = df.index[rows]
targets = df.columns[cols]
weights = W[rows, cols]

edges = pd.DataFrame({
    "Source": sources,
    "Target": targets,
    "Weight": weights
})
edges.to_excel(BASE_DIR / "community_edges_for_gephi.xlsx", index=False, engine="openpyxl")
print("完成！已导出 community_edges_for_gephi.xlsx")

