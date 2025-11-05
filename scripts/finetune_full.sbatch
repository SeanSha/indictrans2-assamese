#!/bin/bash
#SBATCH --job-name=indictrans2_full_ft
#SBATCH --output=logs/full_ft_%j.out
#SBATCH --error=logs/full_ft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

source ~/projects/indictrans2-assamese/indictrans2_env/bin/activate
mkdir -p logs
cd ~/projects/indictrans2-assamese

export DIRECTION=INDIC_EN   # INDIC_EN
python organized_scripts/finetune_full_cuda.py