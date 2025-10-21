#!/bin/bash
#SBATCH --job-name=evaluate_asm_eng
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

# 运行模型评估
echo "=== 开始模型评估 ==="
python evaluation/evaluate_asm_eng_model.py \
    --lora_model "training/outputs/assamese_english_lora_asm_eng_20251021_145310" \
    --data_dir "data_processing/assamese_english_asm_eng_mini_format/test" \
    --output_dir "results/evaluation_results_asm_eng_test" \
    --src_lang "asm_Beng" \
    --tgt_lang "eng_Latn" \
    --batch_size 4 \
    --max_length 512

echo "评估完成"