@echo off
chcp 65001 >nul
title OVL Analyzer - 启动中...

echo ============================================
echo   OVL Analyzer 启动脚本
echo ============================================
echo.

:: 检查 Python
echo [1/3] 检查 Python 环境...
python --version 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [错误] 未找到 Python！请先安装 Python 3.9~3.12
    echo 下载地址: https://www.python.org/downloads/
    echo 安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)

:: 检查并安装依赖
echo.
echo [2/3] 检查并安装依赖...
pip show PySide6 >nul 2>nul
if %errorlevel% neq 0 (
    echo   正在安装依赖，首次运行可能需要几分钟...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo [错误] 依赖安装失败，尝试逐个安装...
        pip install PySide6
        pip install matplotlib
        pip install numpy
        pip install pandas
    )
) else (
    echo   依赖已安装，跳过。
)

:: 启动程序
echo.
echo [3/3] 启动 OVL Analyzer...
echo.
python run.py %*

if %errorlevel% neq 0 (
    echo.
    echo [错误] 程序异常退出，错误码: %errorlevel%
    pause
)
