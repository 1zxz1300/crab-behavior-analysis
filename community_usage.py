"""Aggregate motif usage into community-level usage for each treatment and replicate."""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ===== 1) 社区定义 =====
COMMUNITIES = {
    "a": [0, 2, 12, 16],
    "b": [4, 11, 15, 17],
    "c": [5, 6],
    "d": [1, 3, 7, 8, 9, 10, 13, 14]
}

# ===== 2) 三个处理的 Excel 路径 =====
TREATMENT_FILES = {
    "BG": BASE_DIR / "BG.xlsx",
    "SG": BASE_DIR / "SG.xlsx",
    "MG": BASE_DIR / "MG.xlsx",
}


def compute_community_usage(df: pd.DataFrame):
    """
    df 两列：
      motif: 0-19
      usage: 使用率
    返回：dict (a-f)
    """
    # 统一列名容错
    cols = {c.lower(): c for c in df.columns}
    motif_col = cols.get("motif")
    usage_col = cols.get("usage")

    if motif_col is None or usage_col is None:
        raise ValueError(f"Sheet columns must include motif & usage. Found: {list(df.columns)}")

    tmp = df[[motif_col, usage_col]].copy()
    tmp[motif_col] = tmp[motif_col].astype(int)

    usage_map = dict(zip(tmp[motif_col], tmp[usage_col]))

    out = {}
    for comm, motifs in COMMUNITIES.items():
        out[comm] = sum(float(usage_map.get(m, 0.0)) for m in motifs)

    return out


def process_treatment(treatment_name: str, file_path: str):
    """
    一个处理文件有 5 个 sheet，每个 sheet 是一个平行
    返回记录列表
    """
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names

    records = []
    for sn in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sn)
        comm_usage = compute_community_usage(df)

        records.append({
            "treatment": treatment_name,
            "replicate": sn,  # 用 sheet 名作为平行标识
            **comm_usage
        })

    return records


def main():
    all_records = []

    for treatment, path in TREATMENT_FILES.items():
        recs = process_treatment(treatment, path)
        all_records.extend(recs)

    result = pd.DataFrame(all_records)

    # 列顺序
    result = result[["treatment", "replicate", "a", "b", "c", "d"]]

    print("\n=== Community usage per replicate ===")
    print(result)

    # 保存
    out_path = BASE_DIR / "community_usage_summary.xlsx"
    result.to_excel(out_path, index=False)
    print(f"\n[OK] Saved summary to: {out_path}")


if __name__ == "__main__":
    main()
