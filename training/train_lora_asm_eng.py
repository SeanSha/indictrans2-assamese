#!/usr/bin/env python3
"""
训练阿萨姆语→英语的LoRA模型（正确的方向）
"""

import os
import argparse
import torch
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
import huggingface_hub

def load_and_process_translation_dataset(data_dir, src_lang="asm_Beng", tgt_lang="eng_Latn"):
    """加载翻译数据集"""
    print(f"加载数据集从: {data_dir}")
    
    # 构建文件路径
    train_src_path = os.path.join(data_dir, "train", f"{src_lang}-{tgt_lang}", f"train.{src_lang}")
    train_tgt_path = os.path.join(data_dir, "train", f"{src_lang}-{tgt_lang}", f"train.{tgt_lang}")
    dev_src_path = os.path.join(data_dir, "dev", f"{src_lang}-{tgt_lang}", f"dev.{src_lang}")
    dev_tgt_path = os.path.join(data_dir, "dev", f"{src_lang}-{tgt_lang}", f"dev.{tgt_lang}")
    
    # 读取训练数据
    with open(train_src_path, "r", encoding="utf-8") as f:
        train_src = [line.strip() for line in f if line.strip()]
    with open(train_tgt_path, "r", encoding="utf-8") as f:
        train_tgt = [line.strip() for line in f if line.strip()]
    
    # 读取验证数据
    with open(dev_src_path, "r", encoding="utf-8") as f:
        dev_src = [line.strip() for line in f if line.strip()]
    with open(dev_tgt_path, "r", encoding="utf-8") as f:
        dev_tgt = [line.strip() for line in f if line.strip()]
    
    print(f"训练样本数: {len(train_src)}")
    print(f"验证样本数: {len(dev_src)}")
    
    # 创建数据集
    train_dataset = Dataset.from_dict({
        "src": train_src,
        "tgt": train_tgt
    })
    
    dev_dataset = Dataset.from_dict({
        "src": dev_src,
        "tgt": dev_tgt
    })
    
    return train_dataset, dev_dataset

def preprocess_fn(examples, tokenizer, src_lang="asm_Beng", tgt_lang="eng_Latn"):
    """预处理函数"""
    # 添加语言标签
    src_texts = [f"{src_lang} {tgt_lang} {src}" for src in examples["src"]]
    tgt_texts = examples["tgt"]
    
    # 编码输入 - 使用正确的padding和truncation
    model_inputs = tokenizer(
        src_texts,
        max_length=512,
        padding=True,  # 添加padding
        truncation=True,
        return_tensors="pt"
    )
    
    # 编码目标
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            tgt_texts,
            max_length=512,
            padding=True,  # 添加padding
            truncation=True,
            return_tensors="pt"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = argparse.ArgumentParser(description="训练阿萨姆语→英语LoRA模型")
    parser.add_argument("--data_dir", type=str, default="assamese_english_asm_eng_mini_format", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="outputs/assamese_english_lora_asm_eng_$(date +%Y%m%d_%H%M%S)", help="输出目录")
    parser.add_argument("--base_model", type=str, default="ai4bharat/indictrans2-indic-en-dist-200M", help="基础模型")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=500, help="最大训练步数")
    parser.add_argument("--save_steps", type=int, default=50, help="保存步数")
    parser.add_argument("--eval_steps", type=int, default=50, help="评估步数")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")
    parser.add_argument("--src_lang", type=str, default="asm_Beng", help="源语言")
    parser.add_argument("--tgt_lang", type=str, default="eng_Latn", help="目标语言")
    
    args = parser.parse_args()
    
    print("=== 训练阿萨姆语→英语LoRA模型 ===")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"基础模型: {args.base_model}")
    print(f"源语言: {args.src_lang}")
    print(f"目标语言: {args.tgt_lang}")
    
    # 登录Hugging Face
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        huggingface_hub.login(token=token)
        print("✓ Hugging Face登录成功")
    else:
        print("⚠ 未设置HUGGINGFACE_HUB_TOKEN环境变量")
    
    # 加载模型和tokenizer
    print(f"\n加载基础模型: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ 基础模型加载成功")
    
    # 加载数据集
    train_dataset, dev_dataset = load_and_process_translation_dataset(
        args.data_dir, args.src_lang, args.tgt_lang
    )
    
    # 预处理数据集
    print("\n预处理数据集...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_fn(x, tokenizer, args.src_lang, args.tgt_lang),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    dev_dataset = dev_dataset.map(
        lambda x: preprocess_fn(x, tokenizer, args.src_lang, args.tgt_lang),
        batched=True,
        remove_columns=dev_dataset.column_names
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        fp16=True,
        dataloader_pin_memory=False,
        do_eval=False,
        eval_strategy="no",
        load_best_model_at_end=False,
    )
    
    # 创建训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("\n开始训练...")
    trainer.train()
    
    # 保存模型
    print(f"\n保存模型到: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("✓ 训练完成！")

if __name__ == "__main__":
    main()
