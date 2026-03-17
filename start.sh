#!/bin/bash
# OVL Analyzer 启动脚本 (macOS / Linux)

echo "============================================"
echo "  OVL Analyzer 启动脚本"
echo "============================================"
echo

# 检查 Python
echo "[1/3] 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 Python3！请先安装。"
    exit 1
fi
python3 --version

# 检查并安装依赖
echo
echo "[2/3] 检查并安装依赖..."
if ! python3 -c "import PySide6" 2>/dev/null; then
    echo "  正在安装依赖..."
    pip3 install -r requirements.txt
else
    echo "  依赖已安装，跳过。"
fi

# 启动
echo
echo "[3/3] 启动 OVL Analyzer..."
echo
python3 run.py "$@"
