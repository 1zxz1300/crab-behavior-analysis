"""Summarize community usage by time segment across replicated treatment Excel files."""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def load_treatment_excel(file_path: str, treatment: str) -> pd.DataFrame:
    """
    读取一个处理组的 Excel（含多个sheet=多个平行），返回长表：
    columns: treatment, replicate, segment, community, usage
    """
    xls = pd.ExcelFile(file_path)
    all_long = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)

        # 兼容列名大小写/空格
        df.columns = [str(c).strip().lower() for c in df.columns]

        # 必要列：segment + 社区列(a,b,c,d)
        if "segment" not in df.columns:
            raise ValueError(f"[{treatment}] {file_path} 的 sheet={sheet} 找不到 'segment' 列，当前列：{df.columns.tolist()}")

        # 找社区列：优先按 a/b/c/d
        community_cols = [c for c in ["a", "b", "c", "d"] if c in df.columns]
        if len(community_cols) == 0:
            # 如果不是 a/b/c/d 命名，就把除 segment/total_frames 之外的列都当社区列
            community_cols = [c for c in df.columns if c not in ("segment", "total_frames")]

        # 去掉空行
        df = df.dropna(subset=["segment"])

        # segment 转为整数（1-5）
        df["segment"] = pd.to_numeric(df["segment"], errors="coerce")
        df = df.dropna(subset=["segment"])
        df["segment"] = df["segment"].astype(int)

        # 宽转长
        long_df = df.melt(
            id_vars=["segment"],
            value_vars=community_cols,
            var_name="community",
            value_name="usage",
        )

        # usage 转数字
        long_df["usage"] = pd.to_numeric(long_df["usage"], errors="coerce")

        long_df["treatment"] = treatment
        long_df["replicate"] = sheet

        all_long.append(long_df)

    return pd.concat(all_long, ignore_index=True)


def summarize_by_community(all_data: pd.DataFrame, treatments_order=("BG", "SG", "MG")) -> dict:
    """
    按 community 分社区输出：每个社区一个透视表
    行：segment
    列：BG_mean,BG_sd,SG_mean,SG_sd,MG_mean,MG_sd
    """
    # 统计：均值/标准差/样本数
    summary = (
        all_data
        .groupby(["community", "treatment", "segment"], as_index=False)
        .agg(mean=("usage", "mean"), sd=("usage", "std"), n=("usage", "count"))
    )

    community_tables = {}
    for comm in sorted(summary["community"].unique()):
        sub = summary[summary["community"] == comm].copy()

        # pivot：index=segment, columns=treatment, values=mean/sd
        pivot = sub.pivot(index="segment", columns="treatment", values=["mean", "sd"])

        # 按指定顺序排列处理组
        # 可能某些处理缺失，做安全处理
        cols = []
        for stat in ["mean", "sd"]:
            for tr in treatments_order:
                if (stat, tr) in pivot.columns:
                    cols.append((stat, tr))
        pivot = pivot[cols]

        # 扁平化列名：BG_mean, BG_sd ...
        pivot.columns = [f"{tr}_{stat}" for (stat, tr) in pivot.columns]
        pivot = pivot.reset_index()

        community_tables[comm] = pivot

    return community_tables


def main():
    # ======= 1) 改成你自己的三个文件路径 =======
    BG_FILE = BASE_DIR / "BG_segment_usage.xlsx"
    SG_FILE = BASE_DIR / "SG_segment_usage.xlsx"
    MG_FILE = BASE_DIR / "MG_segment_usage.xlsx"

    # 输出文件
    OUT_FILE = BASE_DIR / "community_usage_summary.xlsx"

    # ======= 2) 读取三组数据（每组多个sheet=多个平行） =======
    bg = load_treatment_excel(BG_FILE, "BG")
    sg = load_treatment_excel(SG_FILE, "SG")
    mg = load_treatment_excel(MG_FILE, "MG")

    all_data = pd.concat([bg, sg, mg], ignore_index=True)

    # ======= 3) 计算每社区-每处理-每时间段的均值/标准差 =======
    community_tables = summarize_by_community(all_data, treatments_order=("BG", "SG", "MG"))

    # ======= 4) 输出：每个社区一个sheet =======
    out_path = Path(OUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # 总长表（可选，方便追溯）
        all_data.to_excel(writer, sheet_name="ALL_LONG", index=False)

        # 分社区统计表
        for comm, table in community_tables.items():
            sheet_name = f"community_{comm}"
            table.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Done! Saved to:", out_path)


if __name__ == "__main__":
    main()
