"""
OVL 矢量图分析工具 - 主 GUI 应用 (增强版)
使用 PySide6 + Matplotlib 构建桌面端 GUI
包含: Vector Map / Shot Heatmap / 10-Parameter Model / Residual Map
"""

import sys
import os
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QCheckBox, QSlider, QComboBox,
    QSplitter, QHeaderView, QMessageBox, QSizePolicy, QSpinBox,
    QTabWidget, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from .mesr_parser import parse_mesr_file, MESRData
from .ovl_statistics import compute_all_wafer_statistics
from .ovl_model import fit_10param_model, compute_shot_overlay, ModelResult


# Wafer 颜色映射
WAFER_COLORS = [
    '#0000FF', '#FF0000', '#00AA00', '#FF8800', '#AA00AA',
    '#00AAAA', '#888800', '#FF00FF', '#8888FF', '#FF8888',
]


class MultiCanvas(FigureCanvas):
    """通用 Matplotlib 画布"""
    def __init__(self, parent=None, width=7, height=7, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


def _draw_wafer_outline(ax, wafer_radius):
    """绘制晶圆外圆和 Notch"""
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(wafer_radius * np.cos(theta),
            wafer_radius * np.sin(theta), 'k-', linewidth=1.5)
    notch = 3.0
    ax.plot([-notch, 0, notch],
            [-wafer_radius, -wafer_radius + notch, -wafer_radius], 'k-', lw=1.5)


def _draw_shot_grid(ax, data, wafer_radius):
    """绘制 Shot 网格矩形"""
    half_sx = data.step_pitch_x / 2.0 / 1000.0
    half_sy = data.step_pitch_y / 2.0 / 1000.0
    all_cols = [v[0] for v in data.shots.values()]
    all_rows = [v[1] for v in data.shots.values()]
    col_center = (min(all_cols) + max(all_cols)) / 2.0
    row_center = (min(all_rows) + max(all_rows)) / 2.0

    for _, (col, row) in data.shots.items():
        cx = (col - col_center) * data.step_pitch_x / 1000.0
        cy = (row - row_center) * data.step_pitch_y / 1000.0
        rect = mpatches.Rectangle(
            (cx - half_sx, cy - half_sy), 2 * half_sx, 2 * half_sy,
            lw=0.5, ec='#AAAAAA', fc='#F5F5F5', alpha=0.4
        )
        ax.add_patch(rect)


def _setup_wafer_axes(ax, wafer_radius, title=""):
    """设置晶圆坐标轴通用参数"""
    margin = 10
    ax.set_xlim(-wafer_radius - margin, wafer_radius + margin)
    ax.set_ylim(-wafer_radius - margin, wafer_radius + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')


# =====================================================================
#  Tab 1: Vector Map 绘图
# =====================================================================
def draw_vector_map(canvas, data, visible_wafers, vector_scale, show_grid=True):
    """绘制 OVL 矢量地图"""
    canvas.fig.clear()
    ax = canvas.fig.add_subplot(111)

    if data.abs_coords_df is None:
        canvas.draw()
        return

    wafer_radius = data.wafer_size / 2.0
    _draw_wafer_outline(ax, wafer_radius)
    if show_grid:
        _draw_shot_grid(ax, data, wafer_radius)

    df = data.abs_coords_df
    legend_handles = []

    for wafer_id, visible in visible_wafers.items():
        if not visible:
            continue
        wdf = df[(df['wafer_id'] == wafer_id) & (df['valid_flag'] == 0)]
        if wdf.empty:
            continue
        color = WAFER_COLORS[(wafer_id - 1) % len(WAFER_COLORS)]
        ax.quiver(
            wdf['x_abs'].values, wdf['y_abs'].values,
            wdf['dx'].values * vector_scale, wdf['dy'].values * vector_scale,
            angles='xy', scale_units='xy', scale=1,
            color=color, width=0.002, headwidth=5, headlength=5,
            headaxislength=4, alpha=0.8
        )
        legend_handles.append(mpatches.Patch(color=color, label=f'Wafer {wafer_id}'))

    _setup_wafer_axes(ax, wafer_radius, 'OVL Vector Map')
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
    canvas.fig.tight_layout()
    canvas.draw()


# =====================================================================
#  Tab 2: Shot Heatmap 绘图
# =====================================================================
def draw_shot_heatmap(canvas, data, wafer_id, metric='ovl_mean'):
    """
    绘制 Shot 级别热力图

    metric 可选: 'ovl_mean', 'dx_mean', 'dy_mean', 'dx_3sig', 'dy_3sig', 'ovl_max'
    """
    canvas.fig.clear()
    ax = canvas.fig.add_subplot(111)

    if data.abs_coords_df is None:
        canvas.draw()
        return

    wafer_radius = data.wafer_size / 2.0
    _draw_wafer_outline(ax, wafer_radius)

    shot_df = compute_shot_overlay(data, wafer_id)
    if shot_df.empty:
        canvas.draw()
        return

    all_cols = [v[0] for v in data.shots.values()]
    all_rows = [v[1] for v in data.shots.values()]
    col_center = (min(all_cols) + max(all_cols)) / 2.0
    row_center = (min(all_rows) + max(all_rows)) / 2.0

    half_sx = data.step_pitch_x / 2.0 / 1000.0
    half_sy = data.step_pitch_y / 2.0 / 1000.0

    # 颜色映射
    values = shot_df[metric].values
    # 对于 mean 类型指标使用发散色谱，对于 sigma/max 使用顺序色谱
    if 'mean' in metric and metric != 'ovl_mean':
        vmin, vmax = np.min(values), np.max(values)
        abs_max = max(abs(vmin), abs(vmax))
        norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
        cmap = matplotlib.colormaps['RdBu_r']
    else:
        norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
        cmap = matplotlib.colormaps['YlOrRd']

    for _, row in shot_df.iterrows():
        col_val, row_val = row['col'], row['row']
        cx = (col_val - col_center) * data.step_pitch_x / 1000.0
        cy = (row_val - row_center) * data.step_pitch_y / 1000.0
        val = row[metric]

        facecolor = cmap(norm(val))
        rect = mpatches.Rectangle(
            (cx - half_sx, cy - half_sy), 2 * half_sx, 2 * half_sy,
            lw=0.8, ec='#666666', fc=facecolor, alpha=0.85
        )
        ax.add_patch(rect)
        # 在 Shot 中心标注数值
        ax.text(cx, cy, f'{val:.3f}', ha='center', va='center',
                fontsize=6, fontweight='bold', color='black')

    # Colorbar
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = canvas.fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(f'{metric} (μm)', fontsize=10)

    metric_labels = {
        'ovl_mean': 'OVL Mean', 'dx_mean': 'dX Mean', 'dy_mean': 'dY Mean',
        'dx_3sig': 'dX 3σ', 'dy_3sig': 'dY 3σ', 'ovl_max': 'OVL Max'
    }
    title = f'Shot Heatmap - Wafer {wafer_id} - {metric_labels.get(metric, metric)}'
    _setup_wafer_axes(ax, wafer_radius, title)
    canvas.fig.tight_layout()
    canvas.draw()


# =====================================================================
#  Tab 3: Residual Vector Map (模型残差矢量图)
# =====================================================================
def draw_residual_map(canvas, data, wafer_id, model_result, vector_scale):
    """绘制 10 参数模型残差矢量图"""
    canvas.fig.clear()

    if data.abs_coords_df is None or model_result.residual_dx is None:
        canvas.draw()
        return

    # 2 子图: 左=Raw, 右=Residual
    ax1 = canvas.fig.add_subplot(121)
    ax2 = canvas.fig.add_subplot(122)

    wafer_radius = data.wafer_size / 2.0
    df = data.abs_coords_df[
        (data.abs_coords_df['wafer_id'] == wafer_id) &
        (data.abs_coords_df['valid_flag'] == 0)
    ]

    x_pos = df['x_abs'].values
    y_pos = df['y_abs'].values

    for ax, dx_data, dy_data, title, color in [
        (ax1, df['dx'].values, df['dy'].values, 'Raw Data', '#0000FF'),
        (ax2, model_result.residual_dx, model_result.residual_dy, 'Residual (after 10-param)', '#FF0000'),
    ]:
        _draw_wafer_outline(ax, wafer_radius)
        _draw_shot_grid(ax, data, wafer_radius)
        ax.quiver(
            x_pos, y_pos,
            dx_data * vector_scale, dy_data * vector_scale,
            angles='xy', scale_units='xy', scale=1,
            color=color, width=0.002, headwidth=5, headlength=5,
            headaxislength=4, alpha=0.8
        )
        _setup_wafer_axes(ax, wafer_radius, f'{title} - Wafer {wafer_id}')

        # 标注统计信息
        dx_3s = 3 * np.std(dx_data, ddof=1)
        dy_3s = 3 * np.std(dy_data, ddof=1)
        info = f'X3σ={dx_3s:.4f}μm\nY3σ={dy_3s:.4f}μm'
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    canvas.fig.tight_layout()
    canvas.draw()


# =====================================================================
#  主窗口
# =====================================================================
class OVLAnalyzerApp(QMainWindow):
    """OVL 分析工具主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OVL Vector Map Analyzer - Nikon MESR 数据分析")
        self.setMinimumSize(1400, 850)

        self.mesr_data: MESRData = None
        self.all_stats: dict = {}
        self.model_results: dict = {}  # {wafer_id: ModelResult}
        self.wafer_checkboxes: dict = {}
        self.vector_scale = 5.0

        self._init_ui()

    def _init_ui(self):
        """初始化 UI 布局"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ===== 左侧：Tab 页签切换不同视图 =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        self.view_tabs = QTabWidget()

        # --- Tab 1: Vector Map ---
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.canvas_vector = MultiCanvas(tab1, width=7, height=7)
        self.toolbar_vector = NavigationToolbar(self.canvas_vector, tab1)
        tab1_layout.addWidget(self.toolbar_vector)
        tab1_layout.addWidget(self.canvas_vector)
        self.view_tabs.addTab(tab1, "矢量地图 (Vector Map)")

        # --- Tab 2: Shot Heatmap ---
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        self.canvas_heatmap = MultiCanvas(tab2, width=7, height=7)
        self.toolbar_heatmap = NavigationToolbar(self.canvas_heatmap, tab2)
        tab2_layout.addWidget(self.toolbar_heatmap)
        tab2_layout.addWidget(self.canvas_heatmap)
        self.view_tabs.addTab(tab2, "Shot 热力图 (Heatmap)")

        # --- Tab 3: Residual Map ---
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        self.canvas_residual = MultiCanvas(tab3, width=12, height=6)
        self.toolbar_residual = NavigationToolbar(self.canvas_residual, tab3)
        tab3_layout.addWidget(self.toolbar_residual)
        tab3_layout.addWidget(self.canvas_residual)
        self.view_tabs.addTab(tab3, "残差矢量图 (Residual)")

        self.view_tabs.currentChanged.connect(self._on_tab_changed)
        left_layout.addWidget(self.view_tabs)
        splitter.addWidget(left_widget)

        # ===== 右侧：控制面板（可滚动） =====
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMaximumWidth(520)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # --- 文件信息区 ---
        file_group = QGroupBox("文件信息 / File Info")
        file_layout = QVBoxLayout(file_group)
        self.btn_open = QPushButton("打开 MESR 文件...")
        self.btn_open.clicked.connect(self._on_open_file)
        file_layout.addWidget(self.btn_open)

        self.lbl_filename = QLabel("文件: -")
        self.lbl_machine = QLabel("机台: -")
        self.lbl_date = QLabel("日期: -")
        self.lbl_comment = QLabel("备注: -")
        self.lbl_wafer_info = QLabel("Wafer: - / Shot: - / MESP: -")
        self.lbl_pitch = QLabel("Step Pitch: - , -")
        for lbl in [self.lbl_filename, self.lbl_machine, self.lbl_date,
                     self.lbl_comment, self.lbl_wafer_info, self.lbl_pitch]:
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 11px;")
            file_layout.addWidget(lbl)
        right_layout.addWidget(file_group)

        # --- Vector Scale ---
        scale_group = QGroupBox("矢量缩放 / Vector Scale")
        scale_layout = QHBoxLayout(scale_group)
        scale_layout.addWidget(QLabel("Scale:"))
        self.scale_spinbox = QSpinBox()
        self.scale_spinbox.setRange(1, 500)
        self.scale_spinbox.setValue(5)
        self.scale_spinbox.setSuffix(" x")
        self.scale_spinbox.valueChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(self.scale_spinbox)
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(1, 500)
        self.scale_slider.setValue(5)
        self.scale_slider.valueChanged.connect(self._on_slider_changed)
        scale_layout.addWidget(self.scale_slider)
        right_layout.addWidget(scale_group)

        # --- Wafer 勾选 ---
        self.wafer_group = QGroupBox("Wafer 显示 / Wafer Selection")
        self.wafer_checkbox_layout = QVBoxLayout(self.wafer_group)
        self.chk_show_grid = QCheckBox("显示 Shot 网格")
        self.chk_show_grid.setChecked(True)
        self.chk_show_grid.stateChanged.connect(self._on_redraw)
        self.wafer_checkbox_layout.addWidget(self.chk_show_grid)
        right_layout.addWidget(self.wafer_group)

        # --- Wafer 选择 + Heatmap 指标 ---
        select_group = QGroupBox("数据选择 / Data Selection")
        select_layout = QVBoxLayout(select_group)

        wf_layout = QHBoxLayout()
        wf_layout.addWidget(QLabel("Wafer:"))
        self.cmb_stat_wafer = QComboBox()
        self.cmb_stat_wafer.currentIndexChanged.connect(self._on_stat_wafer_changed)
        wf_layout.addWidget(self.cmb_stat_wafer)
        select_layout.addLayout(wf_layout)

        hm_layout = QHBoxLayout()
        hm_layout.addWidget(QLabel("热力图指标:"))
        self.cmb_heatmap_metric = QComboBox()
        self.cmb_heatmap_metric.addItems([
            'ovl_mean', 'dx_mean', 'dy_mean', 'dx_3sig', 'dy_3sig', 'ovl_max'
        ])
        self.cmb_heatmap_metric.currentIndexChanged.connect(self._on_heatmap_metric_changed)
        hm_layout.addWidget(self.cmb_heatmap_metric)
        select_layout.addLayout(hm_layout)

        right_layout.addWidget(select_group)

        # --- 统计数据表 ---
        stat_group = QGroupBox("统计数据 / Statistics (per MESP)")
        stat_layout = QVBoxLayout(stat_group)
        self.stat_table = QTableWidget()
        self.stat_table.setColumnCount(6)
        self.stat_table.setHorizontalHeaderLabels(['Point', 'Xave', 'X3sig', 'Yave', 'Y3sig', 'N'])
        self.stat_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stat_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stat_table.setAlternatingRowColors(True)
        self.stat_table.setStyleSheet("""
            QTableWidget { font-size: 11px; gridline-color: #CCCCCC; }
            QTableWidget::item { padding: 2px 4px; }
            QHeaderView::section { background-color: #E0E0E0; font-weight: bold;
                font-size: 11px; border: 1px solid #CCCCCC; padding: 3px; }
        """)
        stat_layout.addWidget(self.stat_table)
        right_layout.addWidget(stat_group)

        # --- 10 参数模型表 ---
        model_group = QGroupBox("10 参数模型 / 10-Parameter Model")
        model_layout = QVBoxLayout(model_group)
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(4)
        self.model_table.setHorizontalHeaderLabels(['Parameter', 'Value', 'Unit', 'Type'])
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.model_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.setStyleSheet("""
            QTableWidget { font-size: 11px; gridline-color: #CCCCCC; }
            QTableWidget::item { padding: 2px 4px; }
            QHeaderView::section { background-color: #D8E8F8; font-weight: bold;
                font-size: 11px; border: 1px solid #CCCCCC; padding: 3px; }
        """)
        model_layout.addWidget(self.model_table)

        # 残差统计
        self.lbl_residual_stats = QLabel("残差统计: -")
        self.lbl_residual_stats.setWordWrap(True)
        self.lbl_residual_stats.setStyleSheet("font-size: 11px; color: #333;")
        model_layout.addWidget(self.lbl_residual_stats)

        right_layout.addWidget(model_group)

        right_layout.addStretch()
        right_scroll.setWidget(right_widget)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    # =================================================================
    #  文件操作
    # =================================================================
    def _on_open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择 MESR 文件", "",
            "MESR Files (*.MESR *.mesr);;All Files (*)"
        )
        if filepath:
            self.load_file(filepath)

    def load_file(self, filepath: str):
        try:
            self.mesr_data = parse_mesr_file(filepath)
            self.all_stats = compute_all_wafer_statistics(self.mesr_data)

            # 计算所有 Wafer 的 10 参数模型
            self.model_results = {}
            for wid in sorted(self.mesr_data.meas_df['wafer_id'].unique()):
                self.model_results[wid] = fit_10param_model(self.mesr_data, wid)

            self._update_file_info()
            self._update_wafer_checkboxes()
            self._update_statistics_table()
            self._update_model_table()
            self._on_redraw()
        except Exception as e:
            QMessageBox.critical(self, "解析错误", f"无法解析文件:\n{str(e)}")

    # =================================================================
    #  UI 更新方法
    # =================================================================
    def _update_file_info(self):
        d = self.mesr_data
        self.lbl_filename.setText(f"文件: {d.file_name}")
        self.lbl_machine.setText(f"机台: {d.exps_machine} / {d.meas_machine}")
        self.lbl_date.setText(f"日期: {d.meas_date}")
        self.lbl_comment.setText(f"备注: {d.comment}")
        self.lbl_wafer_info.setText(f"Wafer: {d.wafer_num} / Shot: {d.shot_num} / MESP: {d.mesp_num}")
        self.lbl_pitch.setText(f"Step Pitch: {d.step_pitch_x:.0f} x {d.step_pitch_y:.0f} μm")

    def _update_wafer_checkboxes(self):
        for cb in self.wafer_checkboxes.values():
            self.wafer_checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self.wafer_checkboxes.clear()

        if self.mesr_data and self.mesr_data.meas_df is not None:
            for wid in sorted(self.mesr_data.meas_df['wafer_id'].unique()):
                color = WAFER_COLORS[(wid - 1) % len(WAFER_COLORS)]
                cb = QCheckBox(f"Wafer {wid}")
                cb.setChecked(True)
                cb.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
                cb.stateChanged.connect(self._on_redraw)
                self.wafer_checkbox_layout.addWidget(cb)
                self.wafer_checkboxes[wid] = cb

        self.cmb_stat_wafer.blockSignals(True)
        self.cmb_stat_wafer.clear()
        if self.all_stats:
            for wid in sorted(self.all_stats.keys()):
                self.cmb_stat_wafer.addItem(f"Wafer {wid}", wid)
        self.cmb_stat_wafer.blockSignals(False)

    def _update_statistics_table(self):
        wid = self.cmb_stat_wafer.currentData()
        if wid is None or wid not in self.all_stats:
            self.stat_table.setRowCount(0)
            return

        stat_df = self.all_stats[wid]
        self.stat_table.setRowCount(len(stat_df))
        for row_idx, (_, row) in enumerate(stat_df.iterrows()):
            for col_idx, col_name in enumerate(['Point', 'Xave', 'X3sig', 'Yave', 'Y3sig', 'N']):
                val = row[col_name]
                item = QTableWidgetItem(f"{val:.4f}" if isinstance(val, float) else str(val))
                item.setTextAlignment(Qt.AlignCenter)
                if row['Point'] == 'Total':
                    item.setBackground(QColor('#FFFACD'))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.stat_table.setItem(row_idx, col_idx, item)

    def _update_model_table(self):
        """更新 10 参数模型表"""
        wid = self.cmb_stat_wafer.currentData()
        if wid is None or wid not in self.model_results:
            self.model_table.setRowCount(0)
            self.lbl_residual_stats.setText("残差统计: -")
            return

        result = self.model_results[wid]
        param_df = result.to_dataframe()
        self.model_table.setRowCount(len(param_df))

        for row_idx, (_, row) in enumerate(param_df.iterrows()):
            for col_idx, col_name in enumerate(['Parameter', 'Value', 'Unit', 'Type']):
                val = row[col_name]
                if isinstance(val, float):
                    item = QTableWidgetItem(f"{val:.6f}")
                else:
                    item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)

                # Grid 行浅蓝背景, Shot 行浅绿背景
                if row['Type'] == 'Grid':
                    item.setBackground(QColor('#E8F0FE'))
                else:
                    item.setBackground(QColor('#E8FEE8'))

                self.model_table.setItem(row_idx, col_idx, item)

        # 更新残差统计文本
        stats = result.residual_stats()
        if stats:
            lines = []
            lines.append(f"Raw:  X3σ={stats['Raw X3σ (μm)']:.4f}μm  Y3σ={stats['Raw Y3σ (μm)']:.4f}μm")
            lines.append(f"Res:  X3σ={stats['Residual X3σ (μm)']:.4f}μm  Y3σ={stats['Residual Y3σ (μm)']:.4f}μm")
            lines.append(f"R²(X)={stats['R² (X)']:.4f}  R²(Y)={stats['R² (Y)']:.4f}")
            self.lbl_residual_stats.setText('\n'.join(lines))

    # =================================================================
    #  事件处理
    # =================================================================
    def _on_scale_changed(self, value):
        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(value)
        self.scale_slider.blockSignals(False)
        self.vector_scale = value
        self._on_redraw()

    def _on_slider_changed(self, value):
        self.scale_spinbox.blockSignals(True)
        self.scale_spinbox.setValue(value)
        self.scale_spinbox.blockSignals(False)
        self.vector_scale = value
        self._on_redraw()

    def _on_stat_wafer_changed(self, index):
        self._update_statistics_table()
        self._update_model_table()
        self._on_redraw()

    def _on_heatmap_metric_changed(self, index):
        if self.view_tabs.currentIndex() == 1:  # Heatmap tab
            self._on_redraw()

    def _on_tab_changed(self, index):
        self._on_redraw()

    def _on_redraw(self):
        if self.mesr_data is None:
            return

        current_tab = self.view_tabs.currentIndex()
        wid = self.cmb_stat_wafer.currentData()

        if current_tab == 0:  # Vector Map
            visible = {w: cb.isChecked() for w, cb in self.wafer_checkboxes.items()}
            show_grid = self.chk_show_grid.isChecked()
            draw_vector_map(self.canvas_vector, self.mesr_data,
                            visible, self.vector_scale, show_grid)

        elif current_tab == 1:  # Shot Heatmap
            if wid is not None:
                metric = self.cmb_heatmap_metric.currentText()
                draw_shot_heatmap(self.canvas_heatmap, self.mesr_data, wid, metric)

        elif current_tab == 2:  # Residual Map
            if wid is not None and wid in self.model_results:
                draw_residual_map(self.canvas_residual, self.mesr_data,
                                  wid, self.model_results[wid], self.vector_scale)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = OVLAnalyzerApp()
    window.show()

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window.load_file(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
