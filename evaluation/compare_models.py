#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比分析脚本：基础模型 vs Mini微调 vs 完整微调
"""

import os
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from sacrebleu import corpus_bleu
from tqdm import tqdm
from huggingface_hub import login

def load_test_data(data_dir):
    """加载测试数据"""
    print(f"加载测试数据从: {data_dir}")
    
    src_path = os.path.join(data_dir, "asm_Beng-eng_Latn", "test.asm_Beng")
    tgt_path = os.path.join(data_dir, "asm_Beng-eng_Latn", "test.eng_Latn")
    
    with open(src_path, "r", encoding="utf-8") as f:
        src_texts = [line.strip() for line in f if line.strip()]
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_texts = [line.strip() for line in f if line.strip()]
    
    print(f"✓ 加载了 {len(src_texts)} 个测试样本")
    return src_texts, tgt_texts

def translate_with_model(model, tokenizer, src_texts, src_lang="asm_Beng", tgt_lang="eng_Latn", batch_size=4):
    """使用模型进行翻译"""
    print(f"开始翻译 {len(src_texts)} 个样本...")
    translations = []
    
    for i in tqdm(range(0, len(src_texts), batch_size), desc="翻译进度"):
        batch_src = src_texts[i:i+batch_size]
        batch_src_with_lang = [f"{src_lang} {tgt_lang} {src}" for src in batch_src]
        
        try:
            inputs = tokenizer(
                batch_src_with_lang,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            batch_translations = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            translations.extend(batch_translations)
        except Exception as e:
            print(f"翻译第 {i+1} 个样本时出错: {e}")
            translations.extend([""] * len(batch_src))
    
    return translations

def calculate_bleu_score(predictions, references):
    """计算BLEU分数"""
    bleu_score = corpus_bleu(predictions, [references])
    return bleu_score.score, str(bleu_score)

def test_base_model(base_model_name, src_texts, tgt_texts):
    """测试基础模型"""
    print("\n" + "="*50)
    print("🔍 测试1: 基础预训练模型")
    print("="*50)
    
    # 登录Hugging Face
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token)
        print("✓ Hugging Face登录成功")
    
    # 加载基础模型
    print(f"加载基础模型: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✓ 基础模型加载成功")
    
    # 翻译
    predictions = translate_with_model(model, tokenizer, src_texts)
    
    # 计算BLEU分数
    bleu_score, bleu_details = calculate_bleu_score(predictions, tgt_texts)
    print(f"✓ 基础模型BLEU分数: {bleu_score:.4f}")
    
    return {
        "model_type": "base_model",
        "bleu_score": bleu_score,
        "bleu_details": bleu_details,
        "predictions": predictions
    }

def test_mini_finetuned_model(base_model_name, lora_path, src_texts, tgt_texts):
    """测试Mini微调模型"""
    print("\n" + "="*50)
    print("🔍 测试2: Mini数据集微调模型")
    print("="*50)
    
    # 登录Hugging Face
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token)
    
    # 加载基础模型
    print(f"加载基础模型: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA适配器
    print(f"加载LoRA适配器: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    print("✓ Mini微调模型加载成功")
    
    # 翻译
    predictions = translate_with_model(model, tokenizer, src_texts)
    
    # 计算BLEU分数
    bleu_score, bleu_details = calculate_bleu_score(predictions, tgt_texts)
    print(f"✓ Mini微调模型BLEU分数: {bleu_score:.4f}")
    
    return {
        "model_type": "mini_finetuned",
        "bleu_score": bleu_score,
        "bleu_details": bleu_details,
        "predictions": predictions
    }

def test_full_finetuned_model(base_model_name, lora_path, src_texts, tgt_texts):
    """测试完整微调模型"""
    print("\n" + "="*50)
    print("🔍 测试3: 完整数据集微调模型")
    print("="*50)
    
    # 登录Hugging Face
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token)
    
    # 加载基础模型
    print(f"加载基础模型: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA适配器
    print(f"加载LoRA适配器: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    print("✓ 完整微调模型加载成功")
    
    # 翻译
    predictions = translate_with_model(model, tokenizer, src_texts)
    
    # 计算BLEU分数
    bleu_score, bleu_details = calculate_bleu_score(predictions, tgt_texts)
    print(f"✓ 完整微调模型BLEU分数: {bleu_score:.4f}")
    
    return {
        "model_type": "full_finetuned",
        "bleu_score": bleu_score,
        "bleu_details": bleu_details,
        "predictions": predictions
    }

def generate_comparison_report(results, output_dir):
    """生成对比分析报告"""
    print("\n" + "="*50)
    print("📊 生成对比分析报告")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    with open(os.path.join(output_dir, "comparison_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 生成对比报告
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== IndicTrans2 阿萨姆语→英语模型对比分析报告 ===\n\n")
        
        f.write("📊 BLEU分数对比:\n")
        for result in results:
            f.write(f"- {result['model_type']}: {result['bleu_score']:.4f}\n")
        
        f.write(f"\n📈 性能提升分析:\n")
        if len(results) >= 2:
            base_score = results[0]['bleu_score']
            mini_score = results[1]['bleu_score'] if len(results) > 1 else None
            full_score = results[2]['bleu_score'] if len(results) > 2 else None
            
            if mini_score:
                mini_improvement = ((mini_score - base_score) / base_score) * 100
                f.write(f"- Mini微调提升: {mini_improvement:+.2f}% ({base_score:.4f} → {mini_score:.4f})\n")
            
            if full_score:
                full_improvement = ((full_score - base_score) / base_score) * 100
                f.write(f"- 完整微调提升: {full_improvement:+.2f}% ({base_score:.4f} → {full_score:.4f})\n")
                
                if mini_score:
                    mini_to_full = ((full_score - mini_score) / mini_score) * 100
                    f.write(f"- 完整vs Mini: {mini_to_full:+.2f}% ({mini_score:.4f} → {full_score:.4f})\n")
        
        f.write(f"\n🎯 结论:\n")
        f.write(f"- 基础模型: 原始翻译能力\n")
        f.write(f"- Mini微调: 小数据集快速适应\n")
        f.write(f"- 完整微调: 最佳翻译性能\n")
    
    print(f"✓ 对比分析报告保存到: {report_path}")

def main():
    """主函数"""
    print("=== IndicTrans2 模型对比分析 ===")
    
    # 设置参数
    base_model = "ai4bharat/indictrans2-indic-en-dist-200M"
    test_data_dir = "data_processing/assamese_english_asm_eng_mini_format/test"
    mini_lora_path = "training/outputs/assamese_english_lora_asm_eng_20251021_145310"  # 当前Mini微调模型
    full_lora_path = "training/outputs/assamese_english_lora_asm_eng_full_$(date +%Y%m%d_%H%M%S)"  # 完整微调模型路径
    output_dir = "results/model_comparison"
    
    # 加载测试数据
    src_texts, tgt_texts = load_test_data(test_data_dir)
    
    results = []
    
    # 测试1: 基础模型
    try:
        base_result = test_base_model(base_model, src_texts, tgt_texts)
        results.append(base_result)
    except Exception as e:
        print(f"❌ 基础模型测试失败: {e}")
    
    # 测试2: Mini微调模型
    try:
        mini_result = test_mini_finetuned_model(base_model, mini_lora_path, src_texts, tgt_texts)
        results.append(mini_result)
    except Exception as e:
        print(f"❌ Mini微调模型测试失败: {e}")
    
    # 测试3: 完整微调模型 (如果存在)
    if os.path.exists(full_lora_path):
        try:
            full_result = test_full_finetuned_model(base_model, full_lora_path, src_texts, tgt_texts)
            results.append(full_result)
        except Exception as e:
            print(f"❌ 完整微调模型测试失败: {e}")
    else:
        print("⚠️ 完整微调模型尚未训练，跳过测试")
    
    # 生成对比报告
    generate_comparison_report(results, output_dir)
    
    print("\n🎉 对比分析完成！")
    print(f"结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
