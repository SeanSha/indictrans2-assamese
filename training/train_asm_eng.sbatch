#!/bin/bash
#SBATCH --job-name=train_asm_eng
#SBATCH --account=uppmax2025-3-5
#SBATCH --partition=node
#SBATCH -M snowy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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

# 运行数据预处理
echo "=== 开始数据预处理 ==="
python data_processing/prepare_asm_eng_data.py

# 运行LoRA微调训练
echo "=== 开始LoRA微调训练 ==="
python training/train_lora_asm_eng.py \
    --data_dir "data_processing/assamese_english_asm_eng_mini_format" \
    --output_dir "training/outputs" \
    --src_lang "asm_Beng" \
    --tgt_lang "eng_Latn" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --warmup_steps 100 \
    --save_steps 50 \
    --eval_steps 50 \
    --logging_steps 10 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --load_best_model_at_end False \
    --metric_for_best_model "bleu" \
    --greater_is_better True \
    --report_to "none"

echo "训练完成"