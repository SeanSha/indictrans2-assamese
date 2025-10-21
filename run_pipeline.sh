#!/bin/bash
# IndicTrans2 阿萨姆语→英语 LoRA 微调完整流程

echo "=== IndicTrans2 阿萨姆语→英语 LoRA 微调流程 ==="
echo "1. 数据预处理"
echo "2. 模型训练 (迷你数据集)"
echo "3. 模型评估"
echo ""

# 检查环境
if [ ! -d "data_processing" ] || [ ! -d "training" ] || [ ! -d "evaluation" ]; then
    echo "错误: 项目结构不完整，请确保在项目根目录运行"
    exit 1
fi

echo "✓ 项目结构检查通过"
echo ""

# 数据预处理
echo "步骤 1: 运行数据预处理..."
python data_processing/prepare_asm_eng_data.py
if [ $? -eq 0 ]; then
    echo "✓ 数据预处理完成"
else
    echo "✗ 数据预处理失败"
    exit 1
fi

echo ""
echo "步骤 2: 提交训练任务 (迷你数据集)..."
echo "运行: sbatch training/train_asm_eng.sbatch"
echo "请手动提交训练任务，然后等待完成"

echo ""
echo "步骤 3: 提交评估任务..."
echo "运行: sbatch evaluation/evaluate_asm_eng.sbatch"
echo "请手动提交评估任务"

echo ""
echo "=== 完整流程说明 ==="
echo "1. 数据预处理: python data_processing/prepare_asm_eng_data.py"
echo "2. 模型训练: sbatch training/train_asm_eng.sbatch"
echo "3. 模型评估: sbatch evaluation/evaluate_asm_eng.sbatch"
echo "4. 查看结果: results/evaluation_results_asm_eng/"
