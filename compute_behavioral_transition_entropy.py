"""Compute weighted behavioral transition entropy from community usage and community transition matrices."""

import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 1. 文件路径
# =========================
BASE_DIR = Path(__file__).resolve().parent
usage_file = BASE_DIR / "community_usage_summary.xlsx"
bg_file = BASE_DIR / "BG组社区转移概率.xlsx"
sg_file = BASE_DIR / "SG组社区转移概率.xlsx"
mg_file = BASE_DIR / "MG组社区转移概率.xlsx"

# 输出文件
output_file = BASE_DIR / "总体行为转移熵计算结果_6社区.xlsx"

# 社区顺序，必须与使用率列和转移矩阵行列一致
states = ["a", "b", "c", "d", "e", "f"]


# =========================
# 2. 工具函数
# =========================
def normalize_replicate_name(name: str) -> str:
    """
    将 usage 文件中的 replicate 名称统一成转移矩阵 sheet 名称格式。
    例如：
    D01_0902 -> D010902
    """
    return str(name).replace("_", "")


def safe_entropy(prob_array):
    """
    计算 Shannon 熵，底为 2。
    约定 0 * log2(0) = 0。
    """
    prob_array = np.array(prob_array, dtype=float)
    prob_array = prob_array[prob_array > 0]
    if len(prob_array) == 0:
        return 0.0
    return -np.sum(prob_array * np.log2(prob_array))


def read_transition_matrix(excel_file: Path, sheet_name: str, states_list):
    """
    从指定 Excel 的指定 sheet 读取转移矩阵。
    返回 DataFrame，行列都是 states_list。
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0, index_col=0)

    # 去掉可能存在的首尾空格
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()

    # 只保留目标状态顺序
    df = df.loc[states_list, states_list].copy()

    # 转成 float
    df = df.astype(float)

    return df


def calc_row_entropies(trans_mat: pd.DataFrame):
    """
    计算每个状态（每一行）的条件转移熵 H_i
    """
    row_entropy = {}
    for state in trans_mat.index:
        probs = trans_mat.loc[state].values
        row_entropy[state] = safe_entropy(probs)
    return row_entropy


def calc_total_entropy(usage_row, row_entropy_dict, states_list):
    """
    用社区使用率加权计算总体转移熵
    H_total = sum(pi_i * H_i)
    """
    usage = np.array([usage_row[s] for s in states_list], dtype=float)

    # 如果使用率没有严格归一化，这里再归一化一次
    usage_sum = usage.sum()
    if usage_sum <= 0:
        raise ValueError("使用率总和 <= 0，无法计算总体转移熵。")
    usage = usage / usage_sum

    row_H = np.array([row_entropy_dict[s] for s in states_list], dtype=float)

    total_H = np.sum(usage * row_H)
    return total_H, usage


def process_treatment(treatment_name, transition_file, usage_df, states_list):
    """
    处理单个 treatment（BG / SG / MG）
    返回：
    1）每个重复的总体熵明细
    2）每个重复的各状态条件熵
    """
    sub_usage = usage_df[usage_df["treatment"] == treatment_name].copy()

    replicate_results = []
    row_entropy_results = []

    # 读取该转移矩阵文件中的所有 sheet 名
    xls = pd.ExcelFile(transition_file)
    sheet_names = xls.sheet_names

    # 只保留真正的重复，不保留 mean/std/sem
    valid_sheets = [s for s in sheet_names if s.lower() not in ["mean", "std", "sem"]]

    for _, row in sub_usage.iterrows():
        rep_usage_name = row["replicate"]
        rep_sheet_name = normalize_replicate_name(rep_usage_name)

        if rep_sheet_name not in valid_sheets:
            print(f"[警告] {treatment_name} 处理中的重复 {rep_usage_name} 对应 sheet {rep_sheet_name} 未找到，跳过。")
            continue

        # 读取该重复的转移矩阵
        trans_mat = read_transition_matrix(transition_file, rep_sheet_name, states_list)

        # 计算每个状态的行熵
        row_entropy_dict = calc_row_entropies(trans_mat)

        # 计算总体转移熵
        total_H, norm_usage = calc_total_entropy(row, row_entropy_dict, states_list)

        # 保存总体结果（自动适配社区数）
        result_dict = {
            "treatment": treatment_name,
            "replicate": rep_usage_name,
            "sheet_name": rep_sheet_name,
            "H_total": total_H,
            "H_total_norm_by_log2K": total_H / np.log2(len(states_list))
        }
        for idx, s in enumerate(states_list):
            result_dict[f"usage_{s}"] = norm_usage[idx]
        replicate_results.append(result_dict)

        # 保存各状态行熵（自动适配社区数）
        row_result_dict = {
            "treatment": treatment_name,
            "replicate": rep_usage_name,
            "sheet_name": rep_sheet_name
        }
        for s in states_list:
            row_result_dict[f"H_{s}"] = row_entropy_dict[s]
        row_entropy_results.append(row_result_dict)

    replicate_df = pd.DataFrame(replicate_results)
    row_entropy_df = pd.DataFrame(row_entropy_results)

    return replicate_df, row_entropy_df


# =========================
# 3. 读取使用率文件
# =========================
usage_df = pd.read_excel(usage_file)

# 统一列名，防止有空格
usage_df.columns = usage_df.columns.astype(str).str.strip()

# 只保留有效列
usage_df = usage_df[["treatment", "replicate"] + states].copy()

# 删除空行
usage_df = usage_df.dropna(subset=["treatment", "replicate"])

# 确保数值列为 float
for s in states:
    usage_df[s] = pd.to_numeric(usage_df[s], errors="coerce")

# 删除社区使用率缺失的行
usage_df = usage_df.dropna(subset=states)


# =========================
# 4. 分处理计算
# =========================
bg_rep_df, bg_row_df = process_treatment("BG", bg_file, usage_df, states)
sg_rep_df, sg_row_df = process_treatment("SG", sg_file, usage_df, states)
mg_rep_df, mg_row_df = process_treatment("MG", mg_file, usage_df, states)

all_rep_df = pd.concat([bg_rep_df, sg_rep_df, mg_rep_df], ignore_index=True)
all_row_df = pd.concat([bg_row_df, sg_row_df, mg_row_df], ignore_index=True)


# =========================
# 5. 计算各处理汇总统计
# =========================
summary_df = (
    all_rep_df.groupby("treatment", as_index=False)
    .agg(
        n=("H_total", "count"),
        H_total_mean=("H_total", "mean"),
        H_total_std=("H_total", "std"),
        H_total_sem=("H_total", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
        H_total_norm_mean=("H_total_norm_by_log2K", "mean"),
        H_total_norm_std=("H_total_norm_by_log2K", "std"),
        H_total_norm_sem=("H_total_norm_by_log2K", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan)
    )
)

# 各状态行熵的处理均值和标准差（自动适配社区数）
agg_dict = {}
for s in states:
    agg_dict[f"H_{s}_mean"] = (f"H_{s}", "mean")
for s in states:
    agg_dict[f"H_{s}_std"] = (f"H_{s}", "std")

row_summary_df = all_row_df.groupby("treatment", as_index=False).agg(**agg_dict)


# =========================
# 6. 打印结果
# =========================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)

print("\n================ 每个重复的总体行为转移熵 ================\n")
print(all_rep_df)

print("\n================ 每个重复的各状态条件转移熵 ================\n")
print(all_row_df)

print("\n================ 各处理总体行为转移熵汇总 ================\n")
print(summary_df)

print("\n================ 各处理各状态条件转移熵汇总 ================\n")
print(row_summary_df)


# =========================
# 7. 保存到 Excel
# =========================
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    usage_df.to_excel(writer, sheet_name="原始使用率", index=False)
    all_rep_df.to_excel(writer, sheet_name="各重复总体转移熵", index=False)
    all_row_df.to_excel(writer, sheet_name="各重复状态行熵", index=False)
    summary_df.to_excel(writer, sheet_name="处理总体转移熵汇总", index=False)
    row_summary_df.to_excel(writer, sheet_name="处理状态行熵汇总", index=False)

print(f"\n结果已保存到：{output_file}")