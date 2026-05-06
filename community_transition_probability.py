"""Convert motif-level transition probabilities into community-level transition matrices and summarize replicates."""

import numpy as np
import pandas as pd
from pathlib import Path

# ======================
# 1. 定义社区划分（索引都是 0-based，对应转移矩阵里的行列号）
# ======================
community_groups = {
    "a": [6, 7],
    "b": [1],
    "c": [2, 3],
    "d": [0, 4, 15, 18],
    "e": [5, 10, 11, 12, 13, 14, 16],
    "f": [8, 9, 17, 19],
}
communities = list(community_groups.keys())
C = len(communities)


# ======================
# 2. 封装一个函数：给一个平行的 P 和 n，计算社区层面矩阵 P_comm
# ======================
def compute_community_transition(P: np.ndarray,
                                 n: np.ndarray,
                                 community_groups: dict,
                                 communities: list) -> np.ndarray:
    """
    P: 20x20 motif 转移概率矩阵
    n: 长度 20 的 motif 使用次数/权重
    返回: 6x6 的社区转移概率矩阵
    """
    C = len(communities)
    P_comm = np.zeros((C, C), dtype=float)

    for A_idx, A in enumerate(communities):
        S_A = community_groups[A]
        den_A = n[S_A].sum()  # 社区 A 的总权重

        if den_A == 0:
            # 这个社区根本没出现过，整行设为 NaN 或 0，按你需求
            P_comm[A_idx, :] = np.nan
            continue

        for B_idx, B in enumerate(communities):
            S_B = community_groups[B]

            # 对 i∈S_A 求 n_i * sum_{j∈S_B} P_ij
            # 先取出 A 社区到 B 社区的子矩阵 P[S_A, S_B]
            sub_P = P[np.ix_(S_A, S_B)]           # 形状: (|S_A|, |S_B|)
            p_i_to_B = sub_P.sum(axis=1)          # 每个 i∈S_A 到 B 的总概率
            num_AB = (n[S_A] * p_i_to_B).sum()    # 加权和

            P_comm[A_idx, B_idx] = num_AB / den_A

    return P_comm


# ======================
# 3. 列出所有平行的文件名（自己按实际情况改）
BASE_DIR = Path(__file__).resolve().parent
#    假设每个平行有一对文件：
#    - transition_matrix_<rep>.xlsx
#    - motif_usage_<rep>.csv
# ======================
rep_ids = ["D011213", "D011217", "D011218", "D021215", "D031213"]  # 这里填你的 5 个平行的名字

transition_files = [BASE_DIR / f"transition_matrix_{rid}.csv" for rid in rep_ids]
usage_files       = [BASE_DIR / f"motif_usage_{rid}.csv"       for rid in rep_ids]

# 用来存每个平行的社区矩阵
P_comm_list = []

# 为了把所有结果写到同一个 Excel 里
writer = pd.ExcelWriter(BASE_DIR / "community_transition_all_reps.xlsx", engine="xlsxwriter")

for rid, trans_path, usage_path in zip(rep_ids, transition_files, usage_files):
    print(f"Processing {rid} ...")

    # 读 20x20 转移概率矩阵
    df_p = pd.read_csv(trans_path, index_col=0)
    P = df_p.values

    # 读 motif 使用次数/比例，保证顺序和 P 的行一致
    usage_df = pd.read_csv(usage_path)
    # 这里假设 usage_df 里已经按 motif_0, motif_1, ... 排好；
    # 如果不是，就按你的规则重新排序
    n = usage_df["count"].to_numpy()

    # 计算这个平行的社区转移矩阵
    P_comm = compute_community_transition(P, n, community_groups, communities)
    P_comm_list.append(P_comm)

    # 保存到 Excel 的一个 sheet
    df_comm = pd.DataFrame(P_comm, index=communities, columns=communities)
    df_comm.to_excel(writer, sheet_name=rid)

# ======================
# 4. 计算所有平行的平均社区转移矩阵（以及可选的标准差 / SEM）
# ======================
P_comm_array = np.stack(P_comm_list, axis=0)  # 形状: (n_rep, 6, 6)

P_comm_mean = np.nanmean(P_comm_array, axis=0)  # 平均
P_comm_std  = np.nanstd(P_comm_array, axis=0, ddof=1)  # 标准差
P_comm_sem  = P_comm_std / np.sqrt(len(rep_ids))       # 标准误，可选

df_mean = pd.DataFrame(P_comm_mean, index=communities, columns=communities)
df_std  = pd.DataFrame(P_comm_std,  index=communities, columns=communities)
df_sem  = pd.DataFrame(P_comm_sem,  index=communities, columns=communities)

df_mean.to_excel(writer, sheet_name="mean")
df_std.to_excel(writer,  sheet_name="std")
df_sem.to_excel(writer,  sheet_name="sem")

writer.close()

print("Done. 所有平行的社区转移矩阵以及平均值已写入 community_transition_all_reps.xlsx")

