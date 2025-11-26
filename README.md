IndicTrans2 Assamese–English Translation Fine-Tuning Project

Project Overview
This project fine-tunes the ai4bharat/indictrans2-indic-en-dist-200M model for Assamese to English machine translation.
It uses the WMT 2025 Indic MT Task dataset and supports both LoRA fine-tuning and full-parameter training.

Objectives

Dataset: WMT 2025 Indic MT Task – English–Assamese Training Data 2025.csv

Base Model: ai4bharat/indictrans2-indic-en-dist-200M

Task: Assamese (asm_Beng) → English (eng_Latn) translation

Methods: LoRA fine-tuning and full fine-tuning

Data Split:
50,000 training
2,000 validation
2,000 test
500 mini-training subset

Branching Strategy

main: main branch containing the full project

windows-vm: optimized for Windows + virtual machine environments

school-server: optimized for university HPC servers with SLURM

Quick Start

Environment Requirements

Python 3.10+

PyTorch 2.5.1+ with CUDA

Transformers 4.28.1

PEFT (LoRA)

GPU with 8GB+ VRAM recommended

Setup Authentication
Read CONFIG_SETUP.md to configure your Hugging Face token.

Local Development (Windows + VM)

Clone the project
git clone https://github.com/SeanSha/indictrans2-assamese

cd indictrans2-assamese
git checkout windows-vm

Set up the VM environment
chmod +x setup_vm_env.sh
./setup_vm_env.sh

Activate the environment
source ~/projects/indictrans2-assamese/indictrans2_env/bin/activate

Add authentication
export HF_TOKEN="your_token_here"

Run preprocessing
python organized_scripts/preprocess_indictrans2_fixed.py

Run LoRA and full fine-tuning
python organized_scripts/finetune_lora_cuda_fixed.py
python organized_scripts/finetune_full_cuda.py

School Server Deployment

Clone the project
git clone https://github.com/SeanSha/indictrans2-assamese

cd indictrans2-assamese
git checkout school-server

Setup server environment
chmod +x setup_server_env.sh
./setup_server_env.sh

Submit training jobs
chmod +x scripts/submit_jobs.sh
./scripts/submit_jobs.sh

Monitor jobs
chmod +x scripts/monitor_jobs.sh
./scripts/monitor_jobs.sh

Project Status

Completed

Dataset preprocessing

LoRA fine-tuning

CUDA accelerated training

Project documentation

Server deployment scripts

VM environment setup

Partially Completed

Training completed successfully

Inference still has errors

Evaluation blocked due to inference issue

Pending

Fixing model generation error

Preparing production deployment

Tech Stack

Python 3.10

PyTorch 2.5.1 + CUDA 12.1

Transformers 4.28.1

PEFT (LoRA)

Hugging Face Hub

CUDA GPU acceleration

SLURM scheduling on servers

Training Results

Training finished for 3 epochs

Loss dropped from 4.5+ to 3.6

Checkpoints saved successfully

LoRA adapters generated

Known Issues

Model Generation Error
Error: AttributeError: 'NoneType' object has no attribute 'shape'
Cause: IndicTrans2 inference requires components from IndicTransToolkit or fairseq.
Solution: Install IndicTransToolkit or use Linux with proper C++ compiler.

Environment Dependency Issues
Problem: Missing C++ build tools
Impact: Cannot install IndicTransToolkit or fairseq on Windows
Solution: Use Linux or install Visual Studio Build Tools

Documentation
See:

docs/PROJECT_SUMMARY.md

docs/TECHNICAL_ISSUES_AND_SOLUTIONS.md

docs/QUICK_START_GUIDE.md

docs/SCHOOL_SERVER_MIGRATION_PLAN.md

CONFIG_SETUP.md

Resources

IndicTrans2 Repo: https://github.com/AI4Bharat/IndicTrans2

Hugging Face Model: https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M

WMT 2025 Indic MT Task dataset

LoRA paper

License
MIT License

Project Status: Fine-tuning completed, inference pending
Last Updated: 2025-10-21
