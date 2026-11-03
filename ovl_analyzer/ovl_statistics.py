"""
OVL 统计计算模块
计算每片 Wafer 各测量点的均值 (ave) 和 3σ 值
"""

import numpy as np
import pandas as pd
from .mesr_parser import MESRData


def compute_statistics(data: MESRData, wafer_id: int) -> pd.DataFrame:
    """
    计算指定 Wafer 的 OVL 统计数据

    参数:
        data: 解析后的 MESRData 对象
        wafer_id: Wafer 编号 (1-based)
    返回:
        DataFrame，包含每个 MESP 点和 Total 的统计数据
        列: Point, Xave, X3sig, Yave, Y3sig, N
    """
    if data.abs_coords_df is None:
        return pd.DataFrame()

    # 筛选该 wafer 的有效数据 (valid_flag == 0 表示有效)
    df = data.abs_coords_df[
        (data.abs_coords_df['wafer_id'] == wafer_id) &
        (data.abs_coords_df['valid_flag'] == 0)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    results = []

    # 按 MESP 点分组统计
    for mesp_id in sorted(df['mesp_id'].unique()):
        subset = df[df['mesp_id'] == mesp_id]
        n = len(subset)
        x_ave = subset['dx'].mean()
        y_ave = subset['dy'].mean()
        # 3σ = 3 * sqrt(Σ(xi - μ)² / (N-1))  即 3 倍样本标准差
        x_3sig = 3.0 * subset['dx'].std(ddof=1) if n > 1 else 0.0
        y_3sig = 3.0 * subset['dy'].std(ddof=1) if n > 1 else 0.0

        results.append({
            'Point': str(mesp_id),
            'Xave': round(x_ave, 4),
            'X3sig': round(x_3sig, 4),
            'Yave': round(y_ave, 4),
            'Y3sig': round(y_3sig, 4),
            'N': n
        })

    # Total 行：所有有效测量点汇总统计
    n_total = len(df)
    x_ave_total = df['dx'].mean()
    y_ave_total = df['dy'].mean()
    x_3sig_total = 3.0 * df['dx'].std(ddof=1) if n_total > 1 else 0.0
    y_3sig_total = 3.0 * df['dy'].std(ddof=1) if n_total > 1 else 0.0

    results.append({
        'Point': 'Total',
        'Xave': round(x_ave_total, 4),
        'X3sig': round(x_3sig_total, 4),
        'Yave': round(y_ave_total, 4),
        'Y3sig': round(y_3sig_total, 4),
        'N': n_total
    })

    return pd.DataFrame(results)


def compute_all_wafer_statistics(data: MESRData) -> dict:
    """
    计算所有 Wafer 的统计数据

    返回:
        {wafer_id: DataFrame} 字典
    """
    stats = {}
    if data.meas_df is None or data.meas_df.empty or "wafer_id" not in data.meas_df.columns:
        return stats

    for wid in sorted(data.meas_df['wafer_id'].unique()):
        stats[wid] = compute_statistics(data, wid)

    return stats


if __name__ == "__main__":
    import sys
    from .mesr_parser import parse_mesr_file

    if len(sys.argv) > 1:
        data = parse_mesr_file(sys.argv[1])
        all_stats = compute_all_wafer_statistics(data)
        for wid, stat_df in all_stats.items():
            print(f"\n=== Wafer {wid} Statistics ===")
            print(stat_df.to_string(index=False))
