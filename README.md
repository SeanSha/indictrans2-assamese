# IndicTrans2 Assamese–English Translation Fine-Tuning Project

## 📋 Project Overview
This project fine-tunes the `ai4bharat/indictrans2-indic-en-dist-200M` model for Assamese → English machine translation.  
It uses the **WMT 2025 Indic MT Task** dataset and supports both **LoRA (Low-Rank Adaptation)** and **full-parameter fine-tuning**.

---

## 🎯 Objectives
- **Dataset:** WMT 2025 Indic MT Task – English–Assamese Training Data (2025.csv)  
- **Base Model:** [`ai4bharat/indictrans2-indic-en-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M)  
- **Task:** Assamese (`asm_Beng`) → English (`eng_Latn`) translation  
- **Methods:** LoRA fine-tuning + full fine-tuning  
- **Data Splits:**
  - 50,000 training  
  - 2,000 validation  
  - 2,000 test  
  - 500 mini-training subset  

---

## 🌿 Branching Strategy
The project uses separate branches for different environments:

### **`main`**
- Complete project  
- General documentation  

### **`windows-vm`**
- Windows + VM-specific setup  
- Scripts for Windows environment  
- VM configuration guide  
- Debugging scripts  
- Solutions for Windows-only issues  

### **`school-server`**
- SLURM job scripts  
- Training + evaluation on university HPC servers  
- GPU cluster configuration  
- Job monitoring tools  

---

## 🚀 Quick Start

### Environment Requirements
- Python 3.10+  
- PyTorch 2.5.1+ with CUDA  
- Transformers 4.28.1  
- PEFT (LoRA)  
- GPU: 8GB+ VRAM recommended  

### Setup Hugging Face Authentication
See **`CONFIG_SETUP.md`**.

---

## 🖥️ Local Development (Windows + Virtual Machine)

### 1. Clone the project
bash
git clone https://github.com/SeanSha/indictrans2-assamese
cd indictrans2-assamese
git checkout windows-vm


### Step 2 — Setup VM environment
chmod +x setup_vm_env.sh
./setup_vm_env.sh


### Step 3 — Activate environment
source ~/projects/indictrans2-assamese/indictrans2_env/bin/activate


### Step 4 — Configure authentication
export HF_TOKEN="your_token_here"


### Step 5 — Run preprocessing
python organized_scripts/preprocess_indictrans2_fixed.py

### Step 6 — Fine-tune (LoRA + Full)
python organized_scripts/finetune_lora_cuda_fixed.py
python organized_scripts/finetune_full_cuda.py


---

## 3. School Server Deployment (SLURM)

### Step 1 — Clone
git clone https://github.com/SeanSha/indictrans2-assamese
cd indictrans2-assamese
git checkout school-server


### Step 2 — Environment Setup
chmod +x setup_server_env.sh
./setup_server_env.sh


### Step 3 — Submit Jobs
chmod +x scripts/submit_jobs.sh
./scripts/submit_jobs.sh


### Step 4 — Monitor Jobs
chmod +x scripts/monitor_jobs.sh
./scripts/monitor_jobs.sh


---

## 📊 Project Status

### Completed
- Dataset preprocessing  
- LoRA fine-tuning  
- CUDA training  
- Docs ready  
- Server deployment  
- VM configuration  

### Partially Completed
- Training successful  
- Inference failing  
- Evaluation blocked  

### Pending
- Fix generation error  
- Production deployment  

---

## 🔧 Tech Stack
- Python 3.10  
- PyTorch 2.5.1 + CUDA 12.1  
- Transformers 4.28.1  
- PEFT (LoRA)  
- Hugging Face Hub  
- SLURM (server)  

---

## 📈 Training Results
- 3 epochs completed  
- Loss: 4.5 → 3.6  
- Checkpoints saved  
- LoRA adapters generated  

---

## 🚨 Known Issues

### Issue 1 — Model Generation Error


AttributeError: 'NoneType' object has no attribute 'shape'


**Cause:** IndicTrans2 generation requires IndicTransToolkit / fairseq  
**Fix:**  
- Install IndicTransToolkit (Linux recommended)  
- Or use Linux with full C++ build tools  

---

### Issue 2 — Environment Dependencies
- Missing C++ compiler prevents fairseq installation  
- Windows environment not ideal  
**Solutions:**  
- Use Linux / WSL2  
- Or install Visual Studio Build Tools  

---

## 📚 Documentation
See:  
- `docs/PROJECT_SUMMARY.md`  
- `docs/TECHNICAL_ISSUES_AND_SOLUTIONS.md`  
- `docs/QUICK_START_GUIDE.md`  
- `docs/SCHOOL_SERVER_MIGRATION_PLAN.md`  
- `CONFIG_SETUP.md`  

---

## 🔗 Resources
- IndicTrans2 Repo: https://github.com/AI4Bharat/IndicTrans2  
- Hugging Face Model: https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M  
- WMT 2025 Indic MT Dataset  
- LoRA Paper  

---

## 📝 License
MIT License  
