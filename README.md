# IndicTrans2 阿萨姆语→英语 LoRA 微调项目

## 项目概述
本项目使用 LoRA (Low-Rank Adaptation) 技术对 IndicTrans2 模型进行微调，实现阿萨姆语到英语的机器翻译。

## 项目结构
```
indictrans2-assamese/
├── data_processing/          # 数据预处理模块
│   ├── prepare_asm_eng_data.py
│   ├── assamese_english_asm_eng_format/     # 完整数据集 (50K训练+2K验证+2K测试)
│   └── assamese_english_asm_eng_mini_format/ # 迷你数据集 (500训练+20验证+30测试)
├── training/                # 模型训练模块
│   ├── train_lora_asm_eng.py
│   ├── train_asm_eng.sbatch
│   └── outputs/             # 训练输出模型
├── evaluation/              # 模型评估模块
│   ├── evaluate_asm_eng_model.py
│   └── evaluate_asm_eng.sbatch
├── results/                 # 评估结果
│   └── evaluation_results_asm_eng/
├── logs/                   # 日志管理
│   ├── slurm_outputs/      # SLURM作业输出日志
│   └── view_logs.sh        # 日志查看脚本
├── IndicTrans2/            # 官方 IndicTrans2 代码库
└── downloads/              # 原始数据下载目录
```

## 使用流程

### 1. 数据预处理
```bash
# 运行数据预处理脚本
python data_processing/prepare_asm_eng_data.py
```

### 2. 模型训练
```bash
# 提交训练任务
sbatch training/train_asm_eng.sbatch
```

### 3. 模型评估
```bash
# 提交评估任务
sbatch evaluation/evaluate_asm_eng.sbatch
```

## 环境要求
- Python 3.8+
- PyTorch
- Transformers 4.53.2
- PEFT (LoRA)
- SacreBLEU
- 其他依赖见 requirements.txt

## 模型性能
- **BLEU分数**: 5.51
- **测试样本**: 30个阿萨姆语句子
- **翻译方向**: 阿萨姆语 → 英语

## 文件说明
- `prepare_asm_eng_data.py`: 数据预处理脚本，生成训练/验证/测试集
- `train_lora_asm_eng.py`: LoRA微调训练脚本
- `evaluate_asm_eng_model.py`: 模型评估脚本，计算BLEU分数
- 对应的 `.sbatch` 文件用于SLURM作业提交

## 日志管理
所有SLURM作业的输出日志都自动保存到 `logs/slurm_outputs/` 目录中：
```bash
# 查看日志管理
./logs/view_logs.sh

# 查看最新日志
tail -f logs/slurm_outputs/slurm-*.out

# 查看特定作业日志
cat logs/slurm_outputs/slurm-[作业ID].out
```

## 注意事项
- 确保有足够的GPU内存进行训练
- 需要Hugging Face访问权限
- 建议先使用迷你数据集测试完整流程
- 所有日志文件统一管理在 `logs/slurm_outputs/` 目录
