"""
MESR 文件解析模块
用于解析 Nikon 曝光机/量测机台生成的 .MESR 数据文件
"""

import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class MESRData:
    """MESR 文件解析后的结构化数据"""
    # 全局参数
    file_name: str = ""
    eval_mode: str = ""
    meas_mode: str = ""
    meas_sens: str = ""
    exps_machine: str = ""
    meas_machine: str = ""
    meas_date: str = ""
    comment: str = ""
    wafer_size: float = 300.0
    step_pitch_x: float = 0.0  # 单位：μm
    step_pitch_y: float = 0.0  # 单位：μm
    wafer_num: int = 0
    shot_num: int = 0

    # Shot 定义: {shot_id: (col, row)}
    shots: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    # MESP 定义: {mesp_id: (x_offset, y_offset, type, name)}
    mesp_num: int = 0
    mesps: Dict[int, Tuple[float, float, str, str]] = field(default_factory=dict)

    # 测量数据 DataFrame，列: wafer_id, shot_id, mesp_id, valid_flag, dx, dy
    meas_df: Optional[pd.DataFrame] = None

    # 计算后的绝对坐标 DataFrame
    abs_coords_df: Optional[pd.DataFrame] = None


def parse_mesr_file(filepath: str) -> MESRData:
    """
    解析 .MESR 文件，返回结构化数据

    参数:
        filepath: MESR 文件路径
    返回:
        MESRData 对象
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    data = MESRData()

    # --- 解析全局参数 ---
    data.file_name = _extract_str(content, "FILE_NAME")
    data.eval_mode = _extract_str(content, "EVAL_MODE")
    data.meas_mode = _extract_str(content, "MEAS_MODE")
    data.meas_sens = _extract_str(content, "MEAS_SENS")
    data.exps_machine = _extract_str(content, "EXPS_MACHINE")
    data.meas_machine = _extract_str(content, "MEAS_MACHINE")
    data.meas_date = _extract_str(content, "MEAS_DATE")
    data.comment = _extract_str(content, "COMMENT")

    # 晶圆尺寸
    wafer_size_match = re.search(r"WAFER_SIZE:=\s*([\d.]+)", content)
    if wafer_size_match:
        data.wafer_size = float(wafer_size_match.group(1))

    # Step Pitch (X, Y)
    pitch_match = re.search(r"STEP_PITCH:=\s*([\d.]+)\s*,\s*([\d.]+)", content)
    if pitch_match:
        data.step_pitch_x = float(pitch_match.group(1))
        data.step_pitch_y = float(pitch_match.group(2))

    # Wafer / Shot 数量
    wn_match = re.search(r"WAFER_NUM:=\s*(\d+)", content)
    if wn_match:
        data.wafer_num = int(wn_match.group(1))
    sn_match = re.search(r"SHOT_NUM:=\s*(\d+)", content)
    if sn_match:
        data.shot_num = int(sn_match.group(1))
    # --- 解析 Shot 坐标映射 ---
    shot_pattern = re.compile(r"SHOT\((\d+)\):=\s*(\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)")
    for m in shot_pattern.finditer(content):
        shot_id = int(m.group(2))
        col = int(m.group(3))
        row = int(m.group(4))
        data.shots[shot_id] = (col, row)

    # 如果 SHOT_NUM 未正确解析，从实际 SHOT 定义中推断
    if data.shot_num == 0:
        data.shot_num = len(data.shots)

    # --- 解析 MESP (测量点相对坐标) ---
    mesp_num_match = re.search(r"MESP_NUM:=\s*(\d+)", content)
    if mesp_num_match:
        data.mesp_num = int(mesp_num_match.group(1))

    mesp_pattern = re.compile(
        r"MESP\((\d+)\):=\s*(\d+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*'(\w+)'\s*,\s*'(\w+)'"
    )
    for m in mesp_pattern.finditer(content):
        mesp_id = int(m.group(2))
        x_off = float(m.group(3))
        y_off = float(m.group(4))
        mtype = m.group(5)
        mname = m.group(6)
        data.mesps[mesp_id] = (x_off, y_off, mtype, mname)

    # --- 解析测量数据 ---
    meas_records = []
    current_wafer = 0

    # 按行分割（MESR文件可能没有换行，需要按关键字分割）
    # 先处理 WAFER 标记和 MEAS_DATA
    wafer_pattern = re.compile(r"WAFER\((\d+)\):=\s*(\d+)")
    meas_pattern = re.compile(
        r"MEAS_DATA\(\d+\):=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)"
    )

    # 找到所有 WAFER 标记的位置
    wafer_positions = [(m.start(), int(m.group(1))) for m in wafer_pattern.finditer(content)]

    for i, (pos, wafer_idx) in enumerate(wafer_positions):
        # 确定该 wafer 数据的范围
        end_pos = wafer_positions[i + 1][0] if i + 1 < len(wafer_positions) else len(content)
        wafer_section = content[pos:end_pos]

        for m in meas_pattern.finditer(wafer_section):
            shot_id = int(m.group(1))
            mesp_id = int(m.group(2))
            valid_flag = int(m.group(3))
            dx = float(m.group(4))
            dy = float(m.group(5))
            meas_records.append({
                'wafer_id': wafer_idx,
                'shot_id': shot_id,
                'mesp_id': mesp_id,
                'valid_flag': valid_flag,
                'dx': dx,
                'dy': dy
            })

    data.meas_df = pd.DataFrame(meas_records)

    # --- 计算绝对坐标 ---
    _compute_absolute_coords(data)

    return data


def _extract_str(content: str, key: str) -> str:
    """提取带引号的字符串值"""
    pattern = rf"{key}:=\s*'([^']*)'"
    m = re.search(pattern, content)
    return m.group(1) if m else ""


def _compute_absolute_coords(data: MESRData):
    """
    计算每个测量点在晶圆上的绝对物理坐标
    X_abs = col * step_pitch_x + x_offset  (以晶圆中心为原点)
    Y_abs = row * step_pitch_y + y_offset

    注意：需要进行中心点偏移校准。
    Grid 的 (col, row) 需要以中心 shot 为参考计算偏移。
    """
    if data.meas_df is None or data.meas_df.empty:
        return

    df = data.meas_df.copy()

    # 计算 grid 的中心偏移（取所有 shot 的 col/row 平均值作为中心参考）
    all_cols = [v[0] for v in data.shots.values()]
    all_rows = [v[1] for v in data.shots.values()]
    col_center = (min(all_cols) + max(all_cols)) / 2.0
    row_center = (min(all_rows) + max(all_rows)) / 2.0

    # 构建 shot_id -> (col, row) 映射列
    df['col'] = df['shot_id'].map(lambda sid: data.shots.get(sid, (0, 0))[0])
    df['row'] = df['shot_id'].map(lambda sid: data.shots.get(sid, (0, 0))[1])

    # 构建 mesp_id -> (x_offset, y_offset) 映射列
    df['x_offset'] = df['mesp_id'].map(lambda mid: data.mesps.get(mid, (0, 0, '', ''))[0])
    df['y_offset'] = df['mesp_id'].map(lambda mid: data.mesps.get(mid, (0, 0, '', ''))[1])

    # 计算绝对坐标 (单位: μm -> mm)
    df['x_abs'] = ((df['col'] - col_center) * data.step_pitch_x + df['x_offset']) / 1000.0
    df['y_abs'] = ((df['row'] - row_center) * data.step_pitch_y + df['y_offset']) / 1000.0

    data.abs_coords_df = df


if __name__ == "__main__":
    # 简单测试
    import sys
    if len(sys.argv) > 1:
        result = parse_mesr_file(sys.argv[1])
        print(f"文件: {result.file_name}")
        print(f"机台: {result.meas_machine}")
        print(f"日期: {result.meas_date}")
        print(f"Wafer数: {result.wafer_num}, Shot数: {result.shot_num}")
        print(f"Step Pitch: ({result.step_pitch_x}, {result.step_pitch_y}) μm")
        print(f"MESP数: {result.mesp_num}")
        print(f"测量数据行数: {len(result.meas_df)}")
        print(f"\n前10行数据:\n{result.abs_coords_df.head(10)}")
