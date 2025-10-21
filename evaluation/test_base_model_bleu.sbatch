#!/bin/bash
#SBATCH --job-name=test_base_model_bleu
#SBATCH --account=uppmax2025-3-5
#SBATCH --partition=node
#SBATCH -M snowy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_outputs/slurm-%j.out
#SBATCH --error=logs/slurm_outputs/slurm-%j.err

# 设置环境变量
export HUGGINGFACE_HUB_TOKEN="YOUR_HF_TOKEN_HERE"

# 激活conda环境
source /proj/uppmax2025-3-5/private/maoxuan/conda_envs/miniconda3/bin/activate
conda activate mt25_indic

# 设置工作目录
cd /home/maoxuan/project/indictrans2-assamese

# 运行基础模型BLEU测试
echo "=== 开始基础预训练模型BLEU分数测试 ==="
python evaluation/test_base_model_bleu.py

echo "基础模型BLEU测试完成"