# OVL Analyzer

Nikon MESR 数据解析与 OVL (Overlay) 矢量图分析桌面工具。

解析 Nikon 曝光机/量测机台生成的 `.MESR` 文件，提供矢量地图可视化、Shot 级热力图、10 参数模型拟合等功能。

## 功能

- **MESR 解析**: 支持 Nikon MCSV 格式的 `.MESR` 文件，自动解析 Shot/MESP 坐标与测量数据
- **矢量地图 (Vector Map)**: 在晶圆图上绘制 OVL 矢量箭头，支持多 Wafer 叠加显示、缩放控制
- **INC / DIS**: INC 按 Shot 画场内形状。DIS 使用同一套 Shot 面板，并保留整片晶圆总览。网页「OVL 分析」调用的是 `ovl_analyzer.plots`（不依赖 Qt）
- **Shot 热力图 (Heatmap)**: 按 Shot 颜色编码 OVL 大小，支持 6 种指标 (OVL Mean/Max, dX/dY Mean, dX/dY 3σ)
- **10 参数模型**: Grid (Interfield) + Shot (Intrafield) 线性模型拟合，输出 Tx/Ty/Sx/Sy/Rw/Ow 等参数
- **残差分析**: Raw vs Residual 矢量图对比，直观评估模型校正效果
- **统计表**: 按 MESP 点分组的 Xave/Yave/X3σ/Y3σ 统计

## 截图

| Vector Map | Shot Heatmap | Residual Map |
|:---:|:---:|:---:|
| 矢量箭头显示 OVL 方向与大小 | Shot 级颜色编码 | Raw vs 10-param 残差对比 |

## 快速开始

### 方法 1: 源码运行 (推荐开发者)

```bash
# 克隆仓库
git clone https://github.com/your-username/ovl-analyzer.git
cd ovl-analyzer

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 运行
python run.py
# 或直接加载文件:
python run.py sample_data/260313_REG09.MESR
```

### 方法 2: 打包为可执行文件 (分发给其他人)

```bash
# 安装 PyInstaller
pip install pyinstaller

# 使用打包脚本
python scripts/build_exe.py

# 或打包为单个 exe
python scripts/build_exe.py --onefile

# 输出在 dist/ 目录
```

### 方法 3: pip 安装

```bash
pip install -e .
ovl-analyzer
```

## 项目结构

```
ovl-analyzer/
├── ovl_analyzer/           # 主包
│   ├── __init__.py
│   ├── __main__.py         # python -m ovl_analyzer 入口
│   ├── mesr_parser.py      # MESR 文件解析器
│   ├── ovl_statistics.py   # 统计计算模块
│   ├── ovl_model.py        # 10 参数模型拟合
│   └── main_app.py         # PySide6 GUI 主窗口
├── sample_data/            # 示例数据
│   └── 260313_REG09.MESR
├── scripts/
│   └── build_exe.py        # PyInstaller 打包脚本
├── tests/                  # 测试 (待添加)
├── run.py                  # 快捷启动脚本
├── ovl_analyzer.spec       # PyInstaller spec 文件
├── pyproject.toml          # 项目元数据 & 构建配置
├── requirements.txt        # 依赖列表
├── LICENSE                 # MIT License
└── README.md
```

## 10 参数模型说明

| 参数 | 含义 | 单位 | 级别 |
|------|------|------|------|
| Tx | Wafer 平移 X | μm | Grid |
| Ty | Wafer 平移 Y | μm | Grid |
| Sx | Wafer 缩放 X | ppm | Grid |
| Sy | Wafer 缩放 Y | ppm | Grid |
| Rw | Wafer 旋转 | μrad | Grid |
| Ow | Wafer 正交性 | μrad | Grid |
| tx | Shot 平移 X | μm | Shot |
| ty | Shot 平移 Y | μm | Shot |
| sx | Shot 缩放 X | ppm | Shot |
| sy | Shot 缩放 Y | ppm | Shot |
| rs | Shot 旋转 | μrad | Shot |
| os | Shot 正交性 | μrad | Shot |

## 技术栈

- **GUI**: PySide6 (Qt6)
- **绑图**: Matplotlib
- **数据处理**: Pandas, NumPy
- **打包**: PyInstaller

## License

MIT
