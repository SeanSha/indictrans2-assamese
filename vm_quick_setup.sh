#!/bin/bash
# 虚拟机快速配置脚本
# 在 Ubuntu 虚拟机中运行此脚本

echo "=========================================="
echo "开始配置 IndicTrans2 虚拟机环境"
echo "=========================================="

# 1. 更新系统
echo "1. 更新系统包..."
sudo apt update && sudo apt upgrade -y

# 2. 安装基础工具
echo "2. 安装基础工具..."
sudo apt install -y build-essential python3-dev git curl wget vim

# 3. 安装 Python 3.10
echo "3. 安装 Python 3.10..."
sudo apt install -y python3.10 python3.10-venv python3.10-dev
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 4. 创建项目目录
echo "4. 创建项目目录..."
mkdir -p ~/projects/indictrans2-assamese
cd ~/projects/indictrans2-assamese

# 5. 克隆项目
echo "5. 克隆项目..."
git clone https://github.com/SeanSha/indictrans2-assamese .
git checkout windows-vm

# 6. 创建虚拟环境
echo "6. 创建 Python 虚拟环境..."
python3.10 -m venv indictrans2_env
source indictrans2_env/bin/activate

# 7. 升级 pip
echo "7. 升级 pip..."
pip install --upgrade pip

# 8. 安装 PyTorch
echo "8. 安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 9. 安装其他依赖
echo "9. 安装其他依赖..."
pip install transformers==4.28.1 peft accelerate
pip install huggingface_hub sentencepiece
pip install pandas scikit-learn sacrebleu

# 10. 安装 fairseq
echo "10. 安装 fairseq..."
pip install fairseq

# 11. 安装 IndicTransToolkit
echo "11. 安装 IndicTransToolkit..."
pip install IndicTransToolkit

# 12. 创建必要的目录
echo "12. 创建项目目录结构..."
mkdir -p logs scripts data outputs

# 13. 验证安装
echo "13. 验证安装..."
echo "Python 版本: $(python --version)"
echo "PyTorch 版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA 可用: $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c "import fairseq" 2>/dev/null; then
    echo "Fairseq: 安装成功"
else
    echo "Fairseq: 安装失败"
fi

if python -c "import IndicTransToolkit" 2>/dev/null; then
    echo "IndicTransToolkit: 安装成功"
else
    echo "IndicTransToolkit: 安装失败"
fi

echo "=========================================="
echo "虚拟机环境配置完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 设置 Hugging Face 认证:"
echo "   export HF_TOKEN='your_token_here'"
echo ""
echo "2. 运行数据预处理:"
echo "   python organized_scripts/preprocess_indictrans2_fixed.py"
echo ""
echo "3. 运行模型微调:"
echo "   python organized_scripts/finetune_lora_cuda_fixed.py"
echo ""
echo "项目目录: ~/projects/indictrans2-assamese"
echo "虚拟环境: ~/projects/indictrans2-assamese/indictrans2_env"
echo "=========================================="
