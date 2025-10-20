# 虚拟机安装详细指南

## 📋 系统要求

### 硬件要求
- **内存**: 16GB+ (推荐 32GB)
- **存储**: 100GB+ 可用空间
- **CPU**: 4+ 核心
- **虚拟化**: 需要在 BIOS 中启用

### 软件要求
- **Windows 10/11** (64位)
- **管理员权限**

## 🛠️ 安装步骤

### 步骤 1: 启用虚拟化

1. **重启电脑进入 BIOS**
   - 开机时按 `F2`, `F12`, `Delete` 或 `Esc` (根据电脑品牌)
   - 找到 "Virtualization" 或 "Intel VT-x" 或 "AMD-V"
   - 设置为 "Enabled"
   - 保存并退出

2. **检查虚拟化是否启用**
   ```cmd
   # 在 Windows 命令提示符中运行
   systeminfo | findstr /i "hyper-v"
   ```

### 步骤 2: 下载软件

#### VirtualBox (推荐)
1. 访问: https://www.virtualbox.org/wiki/Downloads
2. 下载 "Windows hosts" 版本
3. 文件大小: ~100MB

#### Ubuntu ISO
1. 访问: https://ubuntu.com/download/desktop
2. 下载 **Ubuntu 22.04 LTS**
3. 文件大小: ~4.7GB

### 步骤 3: 安装 VirtualBox

1. **运行安装程序**
   - 双击下载的 `.exe` 文件
   - 点击 "Next" 继续

2. **选择组件**
   - 保持默认选择
   - 点击 "Next"

3. **网络警告**
   - 点击 "Yes" 继续
   - 这会暂时断开网络连接

4. **完成安装**
   - 点击 "Install"
   - 等待安装完成
   - 重启电脑

### 步骤 4: 创建虚拟机

1. **打开 VirtualBox**
   - 点击 "新建" 按钮

2. **基本设置**
   ```
   名称: Ubuntu-IndicTrans2
   类型: Linux
   版本: Ubuntu (64-bit)
   ```

3. **内存设置**
   ```
   内存大小: 16384 MB (16GB)
   ```

4. **硬盘设置**
   ```
   创建虚拟硬盘: 是
   硬盘文件类型: VDI
   存储在物理硬盘上: 动态分配
   文件位置和大小: 100 GB
   ```

### 步骤 5: 安装 Ubuntu

1. **启动虚拟机**
   - 选择刚创建的虚拟机
   - 点击 "启动"

2. **选择启动盘**
   - 选择下载的 Ubuntu ISO 文件
   - 点击 "启动"

3. **Ubuntu 安装**
   - 选择 "Install Ubuntu"
   - 选择语言: English
   - 选择键盘布局
   - 选择 "Normal installation"
   - 选择 "Erase disk and install Ubuntu"
   - 设置用户名和密码
   - 等待安装完成

### 步骤 6: 配置虚拟机

1. **安装增强功能**
   - 在虚拟机菜单中: 设备 → 安装增强功能
   - 在 Ubuntu 中运行安装脚本

2. **共享文件夹** (可选)
   - 设备 → 共享文件夹
   - 添加 Windows 文件夹到虚拟机

3. **网络设置**
   - 确保网络连接正常
   - 测试网络: `ping google.com`

## 🚀 快速配置

虚拟机安装完成后，运行快速配置脚本：

```bash
# 在 Ubuntu 虚拟机中
wget https://raw.githubusercontent.com/SeanSha/indictrans2-assamese/windows-vm/vm_quick_setup.sh
chmod +x vm_quick_setup.sh
./vm_quick_setup.sh
```

## 🔧 故障排除

### 问题 1: 虚拟机启动失败
**解决方案**:
- 检查 BIOS 虚拟化设置
- 确保 Windows 功能中关闭了 Hyper-V
- 重启电脑

### 问题 2: 内存不足
**解决方案**:
- 减少虚拟机内存分配
- 关闭其他程序释放内存
- 考虑升级物理内存

### 问题 3: 网络连接问题
**解决方案**:
- 检查网络适配器设置
- 尝试不同的网络模式 (NAT/Bridge)
- 重启虚拟机

### 问题 4: 性能慢
**解决方案**:
- 增加虚拟机内存
- 启用硬件加速
- 关闭不必要的服务

## 📊 性能优化

### VirtualBox 设置
```
内存: 16GB+
CPU: 4+ 核心
显存: 128MB+
硬件加速: 启用
```

### Ubuntu 优化
```bash
# 禁用不必要的服务
sudo systemctl disable snapd
sudo systemctl disable bluetooth

# 优化内存使用
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

## 🎯 验证安装

安装完成后，运行以下命令验证：

```bash
# 检查系统信息
uname -a
free -h
df -h

# 检查 Python 环境
python3 --version
pip3 --version

# 检查项目
cd ~/projects/indictrans2-assamese
ls -la
```

## 📞 获取帮助

如果遇到问题：
1. 查看 VirtualBox 官方文档
2. 检查 Ubuntu 安装指南
3. 参考项目文档: `docs/VM_SETUP_GUIDE.md`

---

**安装时间**: 约 2-3 小时  
**难度**: 中等  
**推荐**: 有经验的用户
