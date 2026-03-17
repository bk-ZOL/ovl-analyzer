#!/usr/bin/env python3
"""
自动化打包脚本
在 Windows/macOS/Linux 上构建可执行文件

用法:
    python scripts/build_exe.py           # 使用 .spec 文件打包（推荐）
    python scripts/build_exe.py --onefile # 打包为单个 exe 文件
"""

import subprocess
import sys
import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_pyinstaller():
    """检查 PyInstaller 是否已安装"""
    try:
        import PyInstaller
        print(f"  PyInstaller {PyInstaller.__version__} 已安装")
        return True
    except ImportError:
        print("  PyInstaller 未安装，正在安装...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
        return True


def clean_build():
    """清理旧的构建文件"""
    for folder in ['build', 'dist']:
        path = os.path.join(PROJECT_ROOT, folder)
        if os.path.exists(path):
            print(f"  清理 {folder}/...")
            shutil.rmtree(path)


def build_with_spec():
    """使用 .spec 文件构建（目录模式，推荐）"""
    spec_file = os.path.join(PROJECT_ROOT, 'ovl_analyzer.spec')
    if not os.path.exists(spec_file):
        print("  错误: 找不到 ovl_analyzer.spec 文件")
        return False

    cmd = [sys.executable, '-m', 'PyInstaller', spec_file, '--noconfirm']
    print(f"  执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def build_onefile():
    """打包为单个 exe 文件"""
    run_py = os.path.join(PROJECT_ROOT, 'run.py')
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',
        '--windowed',
        '--name', 'OVL_Analyzer',
        '--noconfirm',
        '--add-data', f'sample_data{os.pathsep}sample_data',
        run_py,
    ]
    print(f"  执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    print("=" * 50)
    print("  OVL Analyzer 打包工具")
    print("=" * 50)

    os.chdir(PROJECT_ROOT)

    print("\n[1/3] 检查 PyInstaller...")
    check_pyinstaller()

    print("\n[2/3] 清理旧构建...")
    clean_build()

    onefile = '--onefile' in sys.argv

    print(f"\n[3/3] 开始打包 ({'单文件' if onefile else '目录'} 模式)...")
    if onefile:
        success = build_onefile()
    else:
        success = build_with_spec()

    if success:
        print("\n" + "=" * 50)
        print("  打包成功!")
        dist_path = os.path.join(PROJECT_ROOT, 'dist')
        print(f"  输出目录: {dist_path}")
        if not onefile:
            print(f"  可执行文件: dist/OVL_Analyzer/OVL_Analyzer{'.exe' if sys.platform == 'win32' else ''}")
        else:
            print(f"  可执行文件: dist/OVL_Analyzer{'.exe' if sys.platform == 'win32' else ''}")
        print("=" * 50)
    else:
        print("\n  打包失败，请检查错误信息。")
        sys.exit(1)


if __name__ == "__main__":
    main()
