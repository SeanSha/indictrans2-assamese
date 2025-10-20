@echo off
echo ==========================================
echo IndicTrans2 Windows 虚拟机设置助手
echo ==========================================
echo.

echo 1. 检查当前分支...
git branch
echo.

echo 2. 当前项目状态...
git status
echo.

echo 3. 项目文件列表...
dir /b
echo.

echo ==========================================
echo 虚拟机安装指南
echo ==========================================
echo.
echo 请按照以下步骤安装虚拟机:
echo.
echo 步骤 1: 下载 VirtualBox
echo   - 访问: https://www.virtualbox.org/wiki/Downloads
echo   - 下载 "Windows hosts" 版本
echo   - 文件大小约 100MB
echo.
echo 步骤 2: 下载 Ubuntu ISO
echo   - 访问: https://ubuntu.com/download/desktop
echo   - 下载 Ubuntu 22.04 LTS
echo   - 文件大小约 4.7GB
echo.
echo 步骤 3: 安装 VirtualBox
echo   - 运行下载的 .exe 文件
echo   - 按照安装向导完成
echo   - 重启电脑
echo.
echo 步骤 4: 创建虚拟机
echo   - 打开 VirtualBox
echo   - 点击 "新建"
echo   - 名称: Ubuntu-IndicTrans2
echo   - 类型: Linux, Ubuntu (64-bit)
echo   - 内存: 16384 MB (16GB)
echo   - 硬盘: 100 GB (动态分配)
echo.
echo 步骤 5: 安装 Ubuntu
echo   - 启动虚拟机
echo   - 选择 Ubuntu ISO 文件
echo   - 按照安装向导完成
echo.
echo 步骤 6: 配置环境
echo   - 在 Ubuntu 中运行:
echo     wget https://raw.githubusercontent.com/SeanSha/indictrans2-assamese/windows-vm/vm_quick_setup.sh
echo     chmod +x vm_quick_setup.sh
echo     ./vm_quick_setup.sh
echo.
echo ==========================================
echo 详细指南请查看: VM_INSTALLATION_GUIDE.md
echo ==========================================
echo.
pause
