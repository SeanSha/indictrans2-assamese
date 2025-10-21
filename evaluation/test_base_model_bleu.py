#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试基础预训练模型的BLEU分数
这是对比分析的第一步：建立基线性能
"""

import os
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu import corpus_bleu
from tqdm import tqdm
from huggingface_hub import login

def load_test_data(data_dir):
    """加载测试数据"""
    print(f"加载测试数据从: {data_dir}")
    
    src_path = os.path.join(data_dir, "asm_Beng-eng_Latn", "test.asm_Beng")
    tgt_path = os.path.join(data_dir, "asm_Beng-eng_Latn", "test.eng_Latn")
    
    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        raise FileNotFoundError(f"测试文件未找到: {src_path} 或 {tgt_path}")
    
    with open(src_path, "r", encoding="utf-8") as f:
        src_texts = [line.strip() for line in f if line.strip()]
    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_texts = [line.strip() for line in f if line.strip()]
    
    if len(src_texts) != len(tgt_texts):
        raise ValueError(f"源语言和目标语言样本数量不匹配: {len(src_texts)} vs {len(tgt_texts)}")
    
    print(f"✓ 加载了 {len(src_texts)} 个测试样本")
    return src_texts, tgt_texts

def translate_with_base_model(model, tokenizer, src_texts, src_lang="asm_Beng", tgt_lang="eng_Latn", batch_size=4):
    """使用基础模型进行翻译"""
    print(f"开始翻译 {len(src_texts)} 个阿萨姆语句子...")
    translations = []
    
    for i in tqdm(range(0, len(src_texts), batch_size), desc="翻译进度"):
        batch_src = src_texts[i:i+batch_size]
        
        # 添加语言标签 (IndicTrans2格式)
        batch_src_with_lang = [f"{src_lang} {tgt_lang} {src}" for src in batch_src]
        
        try:
            # 编码输入
            inputs = tokenizer(
                batch_src_with_lang,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            # 生成翻译
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=1,  # 使用greedy decoding
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # 解码输出
            batch_translations = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            translations.extend(batch_translations)
            
        except Exception as e:
            print(f"翻译第 {i+1} 个样本时出错: {e}")
            # 添加空字符串作为占位符
            translations.extend([""] * len(batch_src))
    
    print(f"✓ 翻译完成，生成了 {len(translations)} 个翻译结果")
    return translations

def calculate_bleu_score(predictions, references):
    """计算BLEU分数"""
    print(f"\n计算BLEU分数...")
    
    # 使用sacrebleu计算BLEU分数
    bleu_score = corpus_bleu(predictions, [references])
    
    print(f"✓ BLEU分数: {bleu_score.score:.4f}")
    print(f"✓ BLEU详细信息: {bleu_score}")
    
    return bleu_score.score, str(bleu_score)

def calculate_other_metrics(predictions, references):
    """计算其他评估指标"""
    print(f"\n计算其他评估指标...")
    metrics = {}
    
    # 平均预测长度
    pred_lengths = [len(p.split()) for p in predictions]
    avg_pred_len = sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0
    metrics["average_prediction_length"] = avg_pred_len
    
    # 平均参考长度
    ref_lengths = [len(r.split()) for r in references]
    avg_ref_len = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0
    metrics["average_reference_length"] = avg_ref_len
    
    # 长度比率
    length_ratios = [
        (len(p.split()) / len(r.split())) if len(r.split()) > 0 else 0
        for p, r in zip(predictions, references)
    ]
    avg_length_ratio = sum(length_ratios) / len(length_ratios) if length_ratios else 0
    std_length_ratio = (
        (sum((x - avg_length_ratio) ** 2 for x in length_ratios) / len(length_ratios)) ** 0.5
        if length_ratios
        else 0
    )
    metrics["average_length_ratio"] = avg_length_ratio
    metrics["length_ratio_std_dev"] = std_length_ratio
    
    # 完全匹配率
    exact_matches = sum(1 for p, r in zip(predictions, references) if p == r)
    exact_match_rate = exact_matches / len(predictions) if predictions else 0
    metrics["exact_match_rate"] = exact_match_rate
    
    print(f"✓ 平均预测长度: {avg_pred_len:.2f}")
    print(f"✓ 平均参考长度: {avg_ref_len:.2f}")
    print(f"✓ 平均长度比率: {avg_length_ratio:.4f}")
    print(f"✓ 长度比率标准差: {std_length_ratio:.4f}")
    print(f"✓ 完全匹配率: {exact_match_rate:.4f}")
    
    return metrics

def save_results(predictions, references, bleu_score, bleu_details, other_metrics, output_dir):
    """保存结果"""
    print(f"\n保存结果到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存翻译结果
    import pandas as pd
    results_df = pd.DataFrame({
        "source_asm": [""] * len(predictions),  # 这里可以添加源语言文本
        "reference_eng": references,
        "prediction_eng": predictions
    })
    results_df.to_csv(os.path.join(output_dir, "base_model_translation_results.csv"), index=False)
    print(f"✓ 翻译结果保存到: {os.path.join(output_dir, 'base_model_translation_results.csv')}")
    
    # 保存评估指标
    evaluation_metrics = {
        "model_type": "base_model",
        "bleu_score": bleu_score,
        "bleu_details": bleu_details,
        **other_metrics
    }
    with open(os.path.join(output_dir, "base_model_evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(evaluation_metrics, f, ensure_ascii=False, indent=4)
    print(f"✓ 评估指标保存到: {os.path.join(output_dir, 'base_model_evaluation_metrics.json')}")
    
    # 生成评估报告
    report_path = os.path.join(output_dir, "base_model_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== IndicTrans2 基础预训练模型评估报告 ===\n\n")
        f.write("📊 模型信息:\n")
        f.write("- 模型类型: 基础预训练模型 (未微调)\n")
        f.write("- 模型名称: ai4bharat/indictrans2-indic-en-dist-200M\n")
        f.write("- 翻译方向: 阿萨姆语 → 英语\n\n")
        
        f.write("📈 评估结果:\n")
        f.write(f"- BLEU分数: {bleu_score:.4f}\n")
        f.write(f"- BLEU详细信息: {bleu_details}\n\n")
        
        f.write("📊 其他指标:\n")
        for key, value in other_metrics.items():
            f.write(f"- {key}: {value:.4f}\n")
        
        f.write(f"\n📝 翻译样本 (前5个):\n")
        for i in range(min(5, len(predictions))):
            f.write(f"  参考: {references[i]}\n")
            f.write(f"  预测: {predictions[i]}\n\n")
    
    print(f"✓ 评估报告保存到: {report_path}")

def main():
    """主函数"""
    print("=== IndicTrans2 基础预训练模型BLEU分数测试 ===")
    
    # 设置参数
    base_model = "ai4bharat/indictrans2-indic-en-dist-200M"
    test_data_dir = "data_processing/assamese_english_asm_eng_mini_format/test"
    output_dir = "results/base_model_evaluation"
    src_lang = "asm_Beng"
    tgt_lang = "eng_Latn"
    batch_size = 4
    
    print(f"基础模型: {base_model}")
    print(f"测试数据: {test_data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"翻译方向: {src_lang} → {tgt_lang}")
    
    # 1. 登录Hugging Face
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token)
        print("✓ Hugging Face登录成功")
    else:
        print("⚠ 未设置HUGGINGFACE_HUB_TOKEN环境变量，可能无法访问私有模型。")
    
    # 2. 加载测试数据
    src_texts, tgt_texts = load_test_data(test_data_dir)
    
    # 3. 加载基础模型
    print(f"\n加载基础模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("✓ 模型已加载到GPU")
    else:
        print("⚠ 未检测到GPU，模型将在CPU上运行。")
    print("✓ 基础模型加载成功")
    
    # 4. 进行翻译
    predictions = translate_with_base_model(model, tokenizer, src_texts, src_lang, tgt_lang, batch_size)
    
    # 5. 计算BLEU分数
    bleu_score, bleu_details = calculate_bleu_score(predictions, tgt_texts)
    
    # 6. 计算其他指标
    other_metrics = calculate_other_metrics(predictions, tgt_texts)
    
    # 7. 保存结果
    save_results(predictions, tgt_texts, bleu_score, bleu_details, other_metrics, output_dir)
    
    print("\n🎉 基础模型BLEU分数测试完成！")
    print(f"BLEU分数: {bleu_score:.4f}")
    print(f"结果保存在: {output_dir}")
    
    return {
        "model_type": "base_model",
        "bleu_score": bleu_score,
        "bleu_details": bleu_details,
        **other_metrics
    }

if __name__ == "__main__":
    main()
