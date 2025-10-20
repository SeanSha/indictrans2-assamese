# 配置设置说明

## 🔑 Hugging Face 认证

在使用项目之前，您需要设置 Hugging Face 认证。

### 方法 1: 环境变量 (推荐)
```bash
# 设置环境变量
export HF_TOKEN="your_actual_token_here"

# 或者在 Windows 上
set HF_TOKEN=your_actual_token_here
```

### 方法 2: 修改脚本文件
在以下文件中将 `YOUR_HF_TOKEN_HERE` 替换为您的真实 token：

- `debug_model_simple.py`
- `organized_scripts/preprocess_indictrans2_fixed.py`
- `organized_scripts/simple_inference.py`
- `organized_scripts/test_model_access.py`

### 方法 3: 使用 setup_hf_auth.py
```bash
python organized_scripts/setup_hf_auth.py
```

## 📋 项目配置

### 数据配置
- **训练集大小**: 50,000 句
- **验证集大小**: 2,000 句
- **测试集大小**: 2,000 句
- **迷你训练集**: 500 句

### 训练配置
- **批次大小**: 4
- **学习率**: 5e-4
- **训练轮数**: 3
- **模型**: ai4bharat/indictrans2-indic-en-dist-200M

## 🚀 快速开始

1. **设置认证**
   ```bash
   export HF_TOKEN="your_token_here"
   ```

2. **运行数据预处理**
   ```bash
   python organized_scripts/preprocess_indictrans2_fixed.py
   ```

3. **运行模型微调**
   ```bash
   python organized_scripts/finetune_lora_cuda_fixed.py
   ```

## ⚠️ 安全提醒

- 不要将真实的 token 提交到 Git 仓库
- 使用环境变量或配置文件管理敏感信息
- 定期轮换您的 Hugging Face token