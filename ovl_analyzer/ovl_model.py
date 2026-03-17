"""
OVL 10 参数模型拟合模块
实现 Interfield (Grid) + Intrafield (Shot) 线性模型分解

10 参数定义:
  ┌─ Interfield (Grid/Wafer 级别) ──────────────────────┐
  │  Tx  : Wafer 平移 X                                  │
  │  Ty  : Wafer 平移 Y                                  │
  │  Sx  : Wafer 缩放 X (Magnification X)                │
  │  Sy  : Wafer 缩放 Y (Magnification Y)                │
  │  Rw  : Wafer 旋转 (Rotation)                         │
  │  Ow  : Wafer 正交性误差 (Orthogonality/非正交)        │
  └──────────────────────────────────────────────────────┘
  ┌─ Intrafield (Shot 内部) ────────────────────────────┐
  │  tx  : Shot 平移 X                                   │
  │  ty  : Shot 平移 Y                                   │
  │  sx  : Shot 缩放 X (Magnification X)                 │
  │  sy  : Shot 缩放 Y (Magnification Y)                 │
  │  rs  : Shot 旋转 (Rotation)                          │
  │  os  : Shot 正交性误差 (Orthogonality)                │
  └──────────────────────────────────────────────────────┘

模型公式:
  dx = Tx + Sx*Xw + Ow*Yw  +  tx + sx*xs + os*ys
  dy = Ty + Rw*Xw + Sy*Yw  +  ty + rs*xs + sy*ys

  其中 (Xw, Yw) 为 Shot 中心的 Wafer 坐标 (mm)
       (xs, ys) 为测量点在 Shot 内的相对坐标 (mm)

  注：Rw 和 rs 含符号约定，Rotation 使 dx 中含 -Rw*Yw 项
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from .mesr_parser import MESRData


@dataclass
class ModelResult:
    """10 参数模型拟合结果"""
    # Interfield (Grid) 参数
    Tx: float = 0.0    # Wafer Translation X (μm)
    Ty: float = 0.0    # Wafer Translation Y (μm)
    Sx: float = 0.0    # Wafer Scale X (ppm)
    Sy: float = 0.0    # Wafer Scale Y (ppm)
    Rw: float = 0.0    # Wafer Rotation (μrad)
    Ow: float = 0.0    # Wafer Orthogonality (μrad)

    # Intrafield (Shot) 参数
    tx: float = 0.0    # Shot Translation X (μm)
    ty: float = 0.0    # Shot Translation Y (μm)
    sx: float = 0.0    # Shot Scale X (ppm)
    sy: float = 0.0    # Shot Scale Y (ppm)
    rs: float = 0.0    # Shot Rotation (μrad)
    os_: float = 0.0   # Shot Orthogonality (μrad)  ('os' 避免与内置 os 冲突)

    # 拟合残差统计
    residual_dx: Optional[np.ndarray] = None
    residual_dy: Optional[np.ndarray] = None
    raw_dx: Optional[np.ndarray] = None
    raw_dy: Optional[np.ndarray] = None
    r_squared_x: float = 0.0
    r_squared_y: float = 0.0

    def to_dict(self) -> dict:
        """转为字典，用于表格显示"""
        return {
            'Parameter': [
                'Tx (Translation X)', 'Ty (Translation Y)',
                'Sx (Scale X)', 'Sy (Scale Y)',
                'Rw (Rotation)', 'Ow (Orthogonality)',
                'tx (Shot Trans X)', 'ty (Shot Trans Y)',
                'sx (Shot Scale X)', 'sy (Shot Scale Y)',
                'rs (Shot Rotation)', 'os (Shot Ortho)',
            ],
            'Value': [
                self.Tx, self.Ty, self.Sx, self.Sy, self.Rw, self.Ow,
                self.tx, self.ty, self.sx, self.sy, self.rs, self.os_,
            ],
            'Unit': [
                'μm', 'μm', 'ppm', 'ppm', 'μrad', 'μrad',
                'μm', 'μm', 'ppm', 'ppm', 'μrad', 'μrad',
            ],
            'Type': [
                'Grid', 'Grid', 'Grid', 'Grid', 'Grid', 'Grid',
                'Shot', 'Shot', 'Shot', 'Shot', 'Shot', 'Shot',
            ]
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict())

    def residual_stats(self) -> dict:
        """残差统计"""
        if self.residual_dx is None or self.residual_dy is None:
            return {}
        res_dx = self.residual_dx
        res_dy = self.residual_dy
        raw_dx = self.raw_dx
        raw_dy = self.raw_dy
        return {
            'Raw Xave (μm)': np.mean(raw_dx),
            'Raw X3σ (μm)': 3 * np.std(raw_dx, ddof=1),
            'Raw Yave (μm)': np.mean(raw_dy),
            'Raw Y3σ (μm)': 3 * np.std(raw_dy, ddof=1),
            'Residual Xave (μm)': np.mean(res_dx),
            'Residual X3σ (μm)': 3 * np.std(res_dx, ddof=1),
            'Residual Yave (μm)': np.mean(res_dy),
            'Residual Y3σ (μm)': 3 * np.std(res_dy, ddof=1),
            'R² (X)': self.r_squared_x,
            'R² (Y)': self.r_squared_y,
        }


def fit_10param_model(data: MESRData, wafer_id: int) -> ModelResult:
    """
    对指定 Wafer 进行 10 参数（实际 12 参数）模型拟合

    模型:
      dx = Tx + Sx*Xw - Rw*Yw  +  tx + sx*xs - rs*ys
      dy = Ty + Ow*Xw + Sy*Yw  +  ty + os*xs + sy*ys

    使用最小二乘法求解

    参数:
        data: MESRData 对象
        wafer_id: Wafer 编号
    返回:
        ModelResult 拟合结果
    """
    result = ModelResult()

    if data.abs_coords_df is None:
        return result

    df = data.abs_coords_df[
        (data.abs_coords_df['wafer_id'] == wafer_id) &
        (data.abs_coords_df['valid_flag'] == 0)
    ].copy()

    if df.empty:
        return result

    # Wafer 坐标 (Shot 中心坐标, mm)
    all_cols = [v[0] for v in data.shots.values()]
    all_rows = [v[1] for v in data.shots.values()]
    col_center = (min(all_cols) + max(all_cols)) / 2.0
    row_center = (min(all_rows) + max(all_rows)) / 2.0

    Xw = ((df['col'] - col_center) * data.step_pitch_x / 1000.0).values  # mm
    Yw = ((df['row'] - row_center) * data.step_pitch_y / 1000.0).values  # mm

    # Shot 内相对坐标 (mm)
    xs = (df['x_offset'] / 1000.0).values  # mm
    ys = (df['y_offset'] / 1000.0).values  # mm

    # 测量值 (μm)
    dx_meas = df['dx'].values
    dy_meas = df['dy'].values
    n = len(dx_meas)

    # 构建设计矩阵 —— 对 dx 和 dy 分别拟合
    # dx = Tx + Sx*Xw - Rw*Yw + tx + sx*xs - rs*ys
    #     → [1, Xw, -Yw, 0, 0, 0, 1, xs, -ys, 0, 0, 0] * [Tx,Sx,Rw,Ty,Ow,Sy, tx,sx,rs,ty,os,sy]
    # 但实际上更好的做法是联立求解

    # 为 dx 构建设计矩阵: dx = Tx*1 + Sx*Xw + Rw*(-Yw) + tx*1 + sx*xs + rs*(-ys)
    Ax = np.column_stack([
        np.ones(n),     # Tx
        Xw,             # Sx * Xw  (Xw 单位 mm, Sx 拟合出来后转 ppm)
        -Yw,            # -Rw * Yw
        np.ones(n),     # tx
        xs,             # sx * xs
        -ys,            # -rs * ys
    ])

    # 为 dy 构建设计矩阵: dy = Ty*1 + Ow*Xw + Sy*Yw + ty*1 + os*xs + sy*ys
    Ay = np.column_stack([
        np.ones(n),     # Ty
        Xw,             # Ow * Xw
        Yw,             # Sy * Yw
        np.ones(n),     # ty
        xs,             # os * xs
        ys,             # sy * ys
    ])

    # 注意: Tx 和 tx 都是常数项，在单独拟合时无法区分
    # 标准做法是用约束或分步拟合。这里采用分步法：
    #   Step 1: 先按 Shot 中心 (Xw, Yw) 拟合 Grid 参数
    #   Step 2: 用残差对 (xs, ys) 拟合 Intrafield 参数

    # === Step 1: Grid (Interfield) 拟合 ===
    # 先对每个 Shot 求 dx/dy 均值，得到 Shot 级别数据
    shot_means = df.groupby('shot_id').agg({
        'dx': 'mean', 'dy': 'mean',
        'col': 'first', 'row': 'first'
    }).reset_index()

    Xw_shot = ((shot_means['col'] - col_center) * data.step_pitch_x / 1000.0).values
    Yw_shot = ((shot_means['row'] - row_center) * data.step_pitch_y / 1000.0).values
    dx_shot_mean = shot_means['dx'].values
    dy_shot_mean = shot_means['dy'].values
    n_shots = len(shot_means)

    # dx_grid = Tx + Sx*Xw - Rw*Yw
    A_grid_x = np.column_stack([
        np.ones(n_shots), Xw_shot, -Yw_shot
    ])
    # dy_grid = Ty + Ow*Xw + Sy*Yw
    A_grid_y = np.column_stack([
        np.ones(n_shots), Xw_shot, Yw_shot
    ])

    # 最小二乘拟合
    params_gx, _, _, _ = np.linalg.lstsq(A_grid_x, dx_shot_mean, rcond=None)
    params_gy, _, _, _ = np.linalg.lstsq(A_grid_y, dy_shot_mean, rcond=None)

    result.Tx = params_gx[0]                    # μm
    result.Sx = params_gx[1] * 1000.0           # mm⁻¹ -> ppm (×1e6/1e3=×1e3... 实际 μm/mm = 1e-3, ×1e6=×1e3)
    result.Rw = params_gx[2] * 1000.0           # μm/mm -> μrad

    result.Ty = params_gy[0]                    # μm
    result.Ow = params_gy[1] * 1000.0           # ppm
    result.Sy = params_gy[2] * 1000.0           # ppm

    # Grid 拟合值
    dx_grid_fit = A_grid_x @ params_gx
    dy_grid_fit = A_grid_y @ params_gy

    # 扩展回逐点: 每个点的 Grid 贡献
    shot_id_to_idx = {sid: i for i, sid in enumerate(shot_means['shot_id'].values)}
    shot_indices = df['shot_id'].map(shot_id_to_idx).values

    # 逐点 Grid 拟合
    Xw_all = ((df['col'] - col_center) * data.step_pitch_x / 1000.0).values
    Yw_all = ((df['row'] - row_center) * data.step_pitch_y / 1000.0).values
    dx_grid_all = params_gx[0] + params_gx[1] * Xw_all + params_gx[2] * (-Yw_all)
    dy_grid_all = params_gy[0] + params_gy[1] * Xw_all + params_gy[2] * Yw_all

    # === Step 2: Intrafield (Shot) 拟合 ===
    # 残差 = 测量值 - Grid 拟合值
    dx_intra = dx_meas - dx_grid_all
    dy_intra = dy_meas - dy_grid_all

    xs_all = (df['x_offset'] / 1000.0).values
    ys_all = (df['y_offset'] / 1000.0).values

    # dx_intra = tx + sx*xs - rs*ys
    A_intra_x = np.column_stack([
        np.ones(n), xs_all, -ys_all
    ])
    # dy_intra = ty + os*xs + sy*ys
    A_intra_y = np.column_stack([
        np.ones(n), xs_all, ys_all
    ])

    params_ix, _, _, _ = np.linalg.lstsq(A_intra_x, dx_intra, rcond=None)
    params_iy, _, _, _ = np.linalg.lstsq(A_intra_y, dy_intra, rcond=None)

    result.tx = params_ix[0]                     # μm
    result.sx = params_ix[1] * 1000.0            # ppm
    result.rs = params_ix[2] * 1000.0            # μrad

    result.ty = params_iy[0]                     # μm
    result.os_ = params_iy[1] * 1000.0           # ppm (正交性)
    result.sy = params_iy[2] * 1000.0            # ppm

    # === 计算总残差 ===
    dx_intra_fit = A_intra_x @ params_ix
    dy_intra_fit = A_intra_y @ params_iy

    result.residual_dx = dx_intra - dx_intra_fit
    result.residual_dy = dy_intra - dy_intra_fit
    result.raw_dx = dx_meas
    result.raw_dy = dy_meas

    # R² 计算
    ss_res_x = np.sum(result.residual_dx ** 2)
    ss_tot_x = np.sum((dx_meas - np.mean(dx_meas)) ** 2)
    result.r_squared_x = 1 - ss_res_x / ss_tot_x if ss_tot_x > 0 else 0

    ss_res_y = np.sum(result.residual_dy ** 2)
    ss_tot_y = np.sum((dy_meas - np.mean(dy_meas)) ** 2)
    result.r_squared_y = 1 - ss_res_y / ss_tot_y if ss_tot_y > 0 else 0

    return result


def compute_shot_overlay(data: MESRData, wafer_id: int) -> pd.DataFrame:
    """
    计算每个 Shot 的 OVL 统计（均值、3σ、最大值）

    返回 DataFrame:
        shot_id, col, row, dx_mean, dy_mean, ovl_mean, dx_3sig, dy_3sig, ovl_max
    """
    if data.abs_coords_df is None:
        return pd.DataFrame()

    df = data.abs_coords_df[
        (data.abs_coords_df['wafer_id'] == wafer_id) &
        (data.abs_coords_df['valid_flag'] == 0)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    results = []
    for shot_id, group in df.groupby('shot_id'):
        col, row = data.shots.get(shot_id, (0, 0))
        dx_vals = group['dx'].values
        dy_vals = group['dy'].values
        ovl = np.sqrt(dx_vals**2 + dy_vals**2)

        results.append({
            'shot_id': shot_id,
            'col': col,
            'row': row,
            'dx_mean': np.mean(dx_vals),
            'dy_mean': np.mean(dy_vals),
            'ovl_mean': np.mean(ovl),
            'dx_3sig': 3 * np.std(dx_vals, ddof=1) if len(dx_vals) > 1 else 0,
            'dy_3sig': 3 * np.std(dy_vals, ddof=1) if len(dy_vals) > 1 else 0,
            'ovl_max': np.max(ovl),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    from mesr_parser import parse_mesr_file

    if len(sys.argv) > 1:
        data = parse_mesr_file(sys.argv[1])

        for wid in sorted(data.meas_df['wafer_id'].unique()):
            print(f"\n{'='*60}")
            print(f"  Wafer {wid} - 10 Parameter Model Fitting")
            print(f"{'='*60}")

            result = fit_10param_model(data, wid)
            param_df = result.to_dataframe()
            print(param_df.to_string(index=False))

            print(f"\n--- 残差统计 ---")
            for k, v in result.residual_stats().items():
                print(f"  {k}: {v:.4f}")

            print(f"\n--- Shot OVL 统计 (Top 5 by ovl_mean) ---")
            shot_df = compute_shot_overlay(data, wid)
            shot_df_sorted = shot_df.sort_values('ovl_mean', ascending=False).head(5)
            print(shot_df_sorted.to_string(index=False))
