"""
不依赖 Qt 的 MESR 图。

桌面窗口和 Nikon Expert「OVL 分析」都调用这里。

INC 与 DIS 的按 Shot 图是同一套面板：每个 Shot 用自己的场内坐标画矢量，
不把整片晶圆的平移叠成一张大图。DIS 另外保留整片晶圆总览。
套刻（BASFIA 等）仍是原来的整片矢量图。
"""

import math
import warnings

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from .mesr_reader import grid_center


WAFER_COLORS = [
    "#0000FF", "#FF0000", "#00AA00", "#FF8800", "#AA00AA",
    "#00AAAA", "#888800", "#FF00FF", "#8888FF", "#FF8888",
]

_FRAME_COLUMNS = [
    "wafer_id", "shot_id", "col", "row", "point_id",
    "x_local_mm", "y_local_mm", "x_abs_mm", "y_abs_mm", "dx", "dy", "valid",
]

_QUIVER = dict(
    angles="xy", scale_units="xy", scale=1,
    width=0.002, headwidth=5, headlength=5, headaxislength=4, alpha=0.8,
)
_QUIVER_PANEL = dict(
    angles="xy", scale_units="xy", scale=1,
    width=0.012, headwidth=4, headlength=5, headaxislength=4, alpha=0.9,
)


def _step_pitch(data):
    pitch = getattr(data, "step_pitch", None)
    if isinstance(pitch, tuple) and len(pitch) == 2:
        return float(pitch[0]), float(pitch[1])
    return float(getattr(data, "step_pitch_x", 0.0) or 0.0), float(getattr(data, "step_pitch_y", 0.0) or 0.0)


def draw_wafer_outline(ax, wafer_radius):
    """绘制晶圆外圆和 Notch。"""
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(wafer_radius * np.cos(theta), wafer_radius * np.sin(theta), "k-", linewidth=1.5)
    notch = 3.0
    ax.plot([-notch, 0, notch], [-wafer_radius, -wafer_radius + notch, -wafer_radius], "k-", lw=1.5)


def draw_shot_grid(ax, data, wafer_radius=None):
    """绘制 Shot 网格矩形。wafer_radius 保留给旧调用。"""
    del wafer_radius
    pitch_x, pitch_y = _step_pitch(data)
    shots = getattr(data, "shots", None) or {}
    if not shots or not pitch_x or not pitch_y:
        return
    half_sx = pitch_x / 2.0 / 1000.0
    half_sy = pitch_y / 2.0 / 1000.0
    col_center, row_center = grid_center(shots)
    for _, (col, row) in shots.items():
        cx = (col - col_center) * pitch_x / 1000.0
        cy = (row - row_center) * pitch_y / 1000.0
        ax.add_patch(mpatches.Rectangle(
            (cx - half_sx, cy - half_sy), 2 * half_sx, 2 * half_sy,
            lw=0.5, ec="#AAAAAA", fc="#F5F5F5", alpha=0.4,
        ))


def setup_wafer_axes(ax, wafer_radius, title=""):
    """设置晶圆坐标轴。"""
    margin = 10
    ax.set_xlim(-wafer_radius - margin, wafer_radius + margin)
    ax.set_ylim(-wafer_radius - margin, wafer_radius + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)", fontsize=10)
    ax.set_ylabel("Y (mm)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")


def displayed_component(mf, component=None) -> str:
    """INC 用增量矢量，DIS / 同时含两对矢量时用 DIS。"""
    if component and component != "auto" and mf.vector_cols.get(component):
        return component
    if mf.kind == "inc" and mf.vector_cols.get("inc"):
        return "inc"
    if mf.kind in ("dis", "both") and mf.vector_cols.get("dis"):
        return "dis"
    return "primary"


def _per_shot_kind(mf, component) -> bool:
    comp = displayed_component(mf, component)
    if component in ("inc", "dis"):
        return True
    if mf.kind in ("inc", "dis", "both"):
        return True
    return comp in ("inc", "dis") and mf.kind != "ovl"


def _overview_enabled(mf, component, show_wafer_overview) -> bool:
    """整片总览只属于 DIS。INC 保持原来的按 Shot 图。"""
    if not show_wafer_overview:
        return False
    if component == "inc" or (component in (None, "auto") and mf.kind == "inc"):
        return False
    return mf.kind in ("dis", "both") or component == "dis"


def _shot_key(mf, shot_id):
    col, row = mf.shots.get(int(shot_id), (0, 0))
    return (-int(row), int(col), int(shot_id))


def ordered_shot_ids(mf, frame=None):
    """按晶圆上的位置排 Shot：行从上到下，列从左到右。"""
    if frame is None:
        frame = shot_vectors(mf, mf.data_array())
        if len(frame):
            frame = frame[frame["valid"].astype(int) == 0]
    if frame is None or len(frame) == 0:
        return []
    return sorted({int(s) for s in frame["shot_id"]}, key=lambda sid: _shot_key(mf, sid))


def shot_vectors(mf, arr, component="auto") -> pd.DataFrame:
    """每个测量点的场内坐标和晶圆坐标。

    x_local_mm / y_local_mm 是 Shot 中心为原点的位置，按 Shot 面板用这个。
    x_abs_mm / y_abs_mm 与 mesr_parser 的绝对坐标同一公式，整片图用这个。
    """
    empty = pd.DataFrame(columns=_FRAME_COLUMNS)
    if arr is None or len(arr) == 0 or not mf.is_vector_type:
        return empty
    comp = displayed_component(mf, component)
    pair = mf.vector_cols.get(comp) or mf.vector_cols.get("primary")
    if not pair:
        return empty

    named = np.asarray(arr[:, :mf.n_fields], dtype=float)
    wafer = np.rint(arr[:, -1]).astype(int)
    n = named.shape[0]
    layout = mf.layout or {}

    def column(idx):
        if idx is None:
            return None
        return named[:, int(idx)]

    shot_col = column(layout.get("shot"))
    shot = np.rint(shot_col).astype(int) if shot_col is not None else np.zeros(n, dtype=int)
    point_col = column(layout.get("point"))
    point = np.rint(point_col).astype(int) if point_col is not None else np.full(n, -1, dtype=int)
    valid_col = column(layout.get("valid"))
    valid = np.rint(valid_col).astype(int) if valid_col is not None else np.zeros(n, dtype=int)

    col_center, row_center = grid_center(mf.shots)
    pitch_x, pitch_y = mf.step_pitch
    grid_col = np.array([mf.shots.get(int(sid), (0, 0))[0] for sid in shot], dtype=float)
    grid_row = np.array([mf.shots.get(int(sid), (0, 0))[1] for sid in shot], dtype=float)
    cx = (grid_col - col_center) * pitch_x
    cy = (grid_row - row_center) * pitch_y

    x_idx, y_idx = layout.get("x"), layout.get("y")
    if x_idx is not None and y_idx is not None:
        x_um = named[:, int(x_idx)]
        y_um = named[:, int(y_idx)]
        if layout.get("absolute"):
            x_local_um = x_um - cx
            y_local_um = y_um - cy
            x_abs_um = x_um
            y_abs_um = y_um
        else:
            x_local_um = x_um
            y_local_um = y_um
            x_abs_um = cx + x_um
            y_abs_um = cy + y_um
    elif point_col is not None and mf.mesps:
        offsets = [mf.mesps.get(int(pid), (0.0, 0.0, "", "")) for pid in point]
        x_local_um = np.array([item[0] for item in offsets], dtype=float)
        y_local_um = np.array([item[1] for item in offsets], dtype=float)
        x_abs_um = cx + x_local_um
        y_abs_um = cy + y_local_um
    else:
        x_local_um = np.zeros(n)
        y_local_um = np.zeros(n)
        x_abs_um = cx
        y_abs_um = cy

    return pd.DataFrame({
        "wafer_id": wafer,
        "shot_id": shot,
        "col": grid_col.astype(int),
        "row": grid_row.astype(int),
        "point_id": point,
        "x_local_mm": x_local_um / 1000.0,
        "y_local_mm": y_local_um / 1000.0,
        "x_abs_mm": x_abs_um / 1000.0,
        "y_abs_mm": y_abs_um / 1000.0,
        "dx": named[:, int(pair[0])],
        "dy": named[:, int(pair[1])],
        "valid": valid,
    })


def _filter_wafers(frame, visible_wafers):
    if frame is None or len(frame) == 0 or not visible_wafers:
        return frame
    keep = {int(wid) for wid, shown in visible_wafers.items() if shown}
    if not keep:
        return frame.iloc[0:0]
    return frame[frame["wafer_id"].astype(int).isin(keep)]


def _valid(frame):
    if frame is None or len(frame) == 0:
        return frame
    return frame[frame["valid"].astype(int) == 0]


def _three_sigma(values) -> float:
    if len(values) < 2:
        return 0.0
    return float(3.0 * np.std(values, ddof=1))


def summarize(mf, arr) -> str:
    """给网页信息栏的纯描述，不含判超标或处理建议。"""
    n_rows = 0 if arr is None else len(arr)
    lines = [f"记录 {n_rows}    晶圆 {len(mf.wafer_ids())}    Shot {len(mf.shots)}"]
    if mf.kind == "inc":
        lines.append("视图：按 Shot")
    elif mf.kind in ("dis", "both"):
        lines.append("视图：按 Shot（与 INC 相同），并保留整片晶圆")
    elif mf.kind == "ovl":
        lines.append("视图：整片晶圆矢量")
    else:
        lines.append("视图：曲线")
    if mf.is_vector_type and n_rows:
        frame = _valid(shot_vectors(mf, arr))
        if len(frame):
            dx = frame["dx"].to_numpy()
            dy = frame["dy"].to_numpy()
            lines.append(f"dX 均值 {np.mean(dx):+.4f} μm    3σ {_three_sigma(dx):.4f} μm")
            lines.append(f"dY 均值 {np.mean(dy):+.4f} μm    3σ {_three_sigma(dy):.4f} μm")
    return "\n".join(lines)


def _auto_scale(mf, magnitudes) -> float:
    """典型箭头大约是 Shot 尺寸的 30%。网页上的倍率滑块再乘这个数。"""
    mag = np.asarray(magnitudes, dtype=float)
    mag = mag[np.isfinite(mag) & (mag > 0)]
    pitch_x, pitch_y = mf.step_pitch
    if pitch_x and pitch_y:
        pitch_mm = min(pitch_x, pitch_y) / 1000.0
    elif mf.wafer_size:
        pitch_mm = float(mf.wafer_size) / 10.0
    else:
        pitch_mm = 20.0
    if mag.size == 0:
        return 1000.0
    return max(pitch_mm * 0.3 / float(np.percentile(mag, 75)), 1e-3)


def _paint_wafer(ax, data, frame, scale, show_grid, title):
    radius = float(getattr(data, "wafer_size", 300.0) or 300.0) / 2.0
    draw_wafer_outline(ax, radius)
    if show_grid:
        draw_shot_grid(ax, data, radius)
    handles = []
    use = _valid(frame)
    if use is not None and len(use):
        for wid, group in use.groupby("wafer_id", sort=True):
            color = WAFER_COLORS[(int(wid) - 1) % len(WAFER_COLORS)]
            ax.quiver(
                group["x_abs_mm"].to_numpy(), group["y_abs_mm"].to_numpy(),
                group["dx"].to_numpy() * scale, group["dy"].to_numpy() * scale,
                color=color, **_QUIVER,
            )
            handles.append(mpatches.Patch(color=color, label=f"Wafer {int(wid)}"))
    setup_wafer_axes(ax, radius, title)
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=9)
    return handles


def _paint_shot_panel(ax, mf, frame, scale, shot_id):
    pitch_x, pitch_y = mf.step_pitch
    half_x = pitch_x / 2.0 / 1000.0 if pitch_x else 1.0
    half_y = pitch_y / 2.0 / 1000.0 if pitch_y else 1.0
    ax.add_patch(mpatches.Rectangle(
        (-half_x, -half_y), 2 * half_x, 2 * half_y,
        lw=0.6, ec="#666666", fc="#F7F7F7",
    ))
    group = frame[frame["shot_id"].astype(int) == int(shot_id)] if frame is not None else frame
    xs = np.array([0.0])
    ys = np.array([0.0])
    if group is not None and len(group):
        xs = group["x_local_mm"].to_numpy()
        ys = group["y_local_mm"].to_numpy()
        xe = xs + group["dx"].to_numpy() * scale
        ye = ys + group["dy"].to_numpy() * scale
        for wid, part in group.groupby("wafer_id", sort=True):
            color = WAFER_COLORS[(int(wid) - 1) % len(WAFER_COLORS)]
            ax.quiver(
                part["x_local_mm"].to_numpy(), part["y_local_mm"].to_numpy(),
                part["dx"].to_numpy() * scale, part["dy"].to_numpy() * scale,
                color=color, **_QUIVER_PANEL,
            )
        span_x = max(half_x, float(np.max(np.abs(np.concatenate([xs, xe])) )))
        span_y = max(half_y, float(np.max(np.abs(np.concatenate([ys, ye])) )))
    else:
        span_x, span_y = half_x, half_y
    pad = 1.12
    ax.set_xlim(-span_x * pad, span_x * pad)
    ax.set_ylim(-span_y * pad, span_y * pad)
    ax.set_aspect("equal", adjustable="box")
    col, row = mf.shots.get(int(shot_id), (0, 0))
    ax.set_title(f"S{int(shot_id)} ({int(col)},{int(row)})", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, linestyle="--")


def _finish(fig, with_title):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            if with_title:
                fig.tight_layout(rect=(0, 0, 1, 0.96))
            else:
                fig.tight_layout()
        except Exception:
            pass


def _draw_per_shot(fig, mf, panel_frame, scale, overview_frame, label):
    ids = ordered_shot_ids(mf, panel_frame)
    show_overview = overview_frame is not None
    if not ids and not show_overview:
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center")
        return

    n_panels = max(len(ids), 1)
    cols = 1 if n_panels == 1 else min(6, n_panels)
    rows = int(math.ceil(n_panels / cols))
    extra = 1 if show_overview else 0
    fig.set_size_inches(max(2.7 * (cols + extra), 7.4), max(2.6 * rows, 4.0))
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        pass
    gs = GridSpec(rows, cols + extra, figure=fig)
    for i, shot_id in enumerate(ids):
        row, col = divmod(i, cols)
        ax = fig.add_subplot(gs[row, col])
        _paint_shot_panel(ax, mf, panel_frame, scale, shot_id)
        if n_panels > 8:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    if not ids:
        ax = fig.add_subplot(gs[:, :cols])
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center")
    if show_overview:
        ax_ov = fig.add_subplot(gs[:, -1])
        _paint_wafer(ax_ov, mf, overview_frame, scale, True, "DIS wafer")
    if len(ids) == 1 and not show_overview:
        fig.suptitle(f"{label} shot {ids[0]}", fontsize=12, fontweight="bold")
    else:
        fig.suptitle(f"{label} per shot", fontsize=12, fontweight="bold")
    # constrained layout 已经排过；tight_layout 和等比例坐标轴会叠在一起。
    if fig.get_layout_engine() is None:
        _finish(fig, True)


def _reset_fig(fig):
    fig.clear()
    try:
        fig.set_layout_engine("none")
    except Exception:
        pass


def draw_curves(fig, mf, arr, column_names, visible_wafers=None):
    """非矢量 MESR：按记录序号把各数值列画成曲线。"""
    _reset_fig(fig)
    ax = fig.add_subplot(111)
    if arr is None or len(arr) == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    data = np.asarray(arr, dtype=float)
    data = _filter_curve_wafers(data, visible_wafers)
    names = list(column_names)
    skip = {"Shot", "Point", "Valid"}
    plotted = False
    for index, name in enumerate(names):
        if name in skip or index >= data.shape[1]:
            continue
        ax.plot(np.arange(len(data)), data[:, index], lw=1, label=str(name))
        plotted = True
    if plotted:
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("record")
    ax.set_title(mf.eval_mode or "MESR")
    ax.grid(True, alpha=0.3, linestyle="--")
    _finish(fig, False)


def _filter_curve_wafers(data, visible_wafers):
    if data.size == 0 or not visible_wafers or data.shape[1] < 1:
        return data
    keep = {int(wid) for wid, shown in visible_wafers.items() if shown}
    if not keep:
        return data[:0]
    mask = np.isin(np.rint(data[:, -1]).astype(int), list(keep))
    return data[mask]


def draw_vector(
    fig,
    mf,
    arr,
    scale,
    visible_wafers=None,
    show_grid=True,
    shot_id=None,
    show_wafer_overview=True,
    component=None,
):
    """矢量图。

    INC：每个 Shot 一块面板。
    DIS：同样的 Shot 面板，并在右侧保留整片晶圆。
    套刻：一整张晶圆矢量图。
    shot_id 有值时，INC 和 DIS 都只放大那一个 Shot，交互相同。
    """
    _reset_fig(fig)
    if mf is None or not mf.is_vector_type:
        draw_curves(fig, mf, arr, mf.column_names() if mf is not None else [], visible_wafers)
        return

    comp = displayed_component(mf, component)
    frame = _filter_wafers(_valid(shot_vectors(mf, arr, comp)), visible_wafers)
    if not _per_shot_kind(mf, component):
        ax = fig.add_subplot(111)
        _paint_wafer(ax, mf, frame, scale, show_grid, "OVL Vector Map")
        _finish(fig, False)
        return

    panel_frame = frame
    if shot_id is not None and frame is not None and len(frame):
        panel_frame = frame[frame["shot_id"].astype(int) == int(shot_id)]
    label = "INC" if comp == "inc" or (comp == "primary" and mf.kind == "inc") else "DIS"
    overview = frame if _overview_enabled(mf, component, show_wafer_overview) else None
    _draw_per_shot(fig, mf, panel_frame, scale, overview, label)


def draw_legacy_vector_map(fig, data, visible_wafers, scale, show_grid=True):
    """桌面版原来的整片 OVL 图，坐标直接用 parse_mesr_file 的结果。"""
    _reset_fig(fig)
    ax = fig.add_subplot(111)
    frame = _frame_from_parsed(data)
    frame = _filter_wafers(frame, visible_wafers)
    _paint_wafer(ax, data, frame, scale, show_grid, "OVL Vector Map")
    _finish(fig, False)


def _frame_from_parsed(data) -> pd.DataFrame:
    if data is None or data.abs_coords_df is None or data.abs_coords_df.empty:
        return pd.DataFrame(columns=_FRAME_COLUMNS)
    df = data.abs_coords_df
    return pd.DataFrame({
        "wafer_id": df["wafer_id"].astype(int),
        "shot_id": df["shot_id"].astype(int),
        "col": df["col"].astype(int),
        "row": df["row"].astype(int),
        "point_id": df["mesp_id"].astype(int),
        "x_local_mm": df["x_offset"] / 1000.0,
        "y_local_mm": df["y_offset"] / 1000.0,
        "x_abs_mm": df["x_abs"],
        "y_abs_mm": df["y_abs"],
        "dx": df["dx"],
        "dy": df["dy"],
        "valid": df["valid_flag"].astype(int),
    })


def draw_residual_map_fig(fig, data, wafer_id, model_result, vector_scale):
    """10 参数：原始矢量与残差并排。给网页和桌面共用。"""
    _reset_fig(fig)
    if data is None or data.abs_coords_df is None or model_result is None or model_result.residual_dx is None:
        return

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    radius = data.wafer_size / 2.0
    df = data.abs_coords_df[
        (data.abs_coords_df["wafer_id"] == wafer_id) &
        (data.abs_coords_df["valid_flag"] == 0)
    ]
    x_pos = df["x_abs"].to_numpy()
    y_pos = df["y_abs"].to_numpy()
    panels = (
        (ax1, df["dx"].to_numpy(), df["dy"].to_numpy(), "Raw Data", "#0000FF"),
        (ax2, model_result.residual_dx, model_result.residual_dy, "Residual (after 10-param)", "#FF0000"),
    )
    for ax, dx_data, dy_data, title, color in panels:
        draw_wafer_outline(ax, radius)
        draw_shot_grid(ax, data, radius)
        ax.quiver(
            x_pos, y_pos,
            np.asarray(dx_data) * vector_scale, np.asarray(dy_data) * vector_scale,
            color=color, **_QUIVER,
        )
        setup_wafer_axes(ax, radius, f"{title} - Wafer {wafer_id}")
        dx_3s = 3 * np.std(dx_data, ddof=1) if len(dx_data) > 1 else 0.0
        dy_3s = 3 * np.std(dy_data, ddof=1) if len(dy_data) > 1 else 0.0
        info = f"X3σ={dx_3s:.4f}μm\nY3σ={dy_3s:.4f}μm"
        ax.text(
            0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
    _finish(fig, False)
