#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备阿萨姆语→英语的数据集（正确的方向）
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_asm_eng_data():
    """准备阿萨姆语→英语的数据集"""
    print("=== 准备阿萨姆语→英语数据集 ===")
    
    # 读取原始数据
    csv_path = "/home/maoxuan/project/downloads/WMT_INDIC_MT_Task_2025/WMT INDIC MT Task 2025/Category I/English-Assamese Training Data 2025.csv"
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # 移除标题行
    if len(df) > 0 and df.iloc[0]['en'] == 'en' and df.iloc[0]['as'] == 'as':
        df = df.iloc[1:].reset_index(drop=True)
    
    # 清理数据
    df.dropna(subset=['en', 'as'], inplace=True)
    print(f"总数据量: {len(df)}")
    
    # 创建完整数据集：阿萨姆语→英语
    if len(df) >= 54000:  # 确保有足够数据
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 分割数据：50K训练 + 2K验证 + 2K测试
        train_df = df.iloc[:50000]
        dev_df = df.iloc[50000:52000]
        test_df = df.iloc[52000:54000]
        
        print(f"训练集: {len(train_df)} 样本")
        print(f"验证集: {len(dev_df)} 样本")
        print(f"测试集: {len(test_df)} 样本")
        
        # 创建输出目录
        output_dir = "assamese_english_asm_eng_format"
        train_dir = os.path.join(output_dir, "train", "asm_Beng-eng_Latn")
        dev_dir = os.path.join(output_dir, "dev", "asm_Beng-eng_Latn")
        test_dir = os.path.join(output_dir, "test", "asm_Beng-eng_Latn")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(dev_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # 保存训练数据（阿萨姆语→英语）
        with open(os.path.join(train_dir, "train.asm_Beng"), 'w', encoding='utf-8') as f:
            for text in train_df['as']:
                f.write(f"{text.strip()}\n")
        with open(os.path.join(train_dir, "train.eng_Latn"), 'w', encoding='utf-8') as f:
            for text in train_df['en']:
                f.write(f"{text.strip()}\n")
        
        # 保存验证数据
        with open(os.path.join(dev_dir, "dev.asm_Beng"), 'w', encoding='utf-8') as f:
            for text in dev_df['as']:
                f.write(f"{text.strip()}\n")
        with open(os.path.join(dev_dir, "dev.eng_Latn"), 'w', encoding='utf-8') as f:
            for text in dev_df['en']:
                f.write(f"{text.strip()}\n")
        
        # 保存测试数据
        with open(os.path.join(test_dir, "test.asm_Beng"), 'w', encoding='utf-8') as f:
            for text in test_df['as']:
                f.write(f"{text.strip()}\n")
        with open(os.path.join(test_dir, "test.eng_Latn"), 'w', encoding='utf-8') as f:
            for text in test_df['en']:
                f.write(f"{text.strip()}\n")
        
        print(f"\n✓ 完整数据集已按照官方格式保存到: {output_dir}")
        
        # 创建迷你数据集：阿萨姆语→英语
        if len(df) >= 550:
            mini_train_df = df.iloc[:500]
            mini_dev_df = df.iloc[500:520]
            mini_test_df = df.iloc[520:550]
            
            mini_output_dir = "assamese_english_asm_eng_mini_format"
            mini_train_dir = os.path.join(mini_output_dir, "train", "asm_Beng-eng_Latn")
            mini_dev_dir = os.path.join(mini_output_dir, "dev", "asm_Beng-eng_Latn")
            mini_test_dir = os.path.join(mini_output_dir, "test", "asm_Beng-eng_Latn")
            
            os.makedirs(mini_train_dir, exist_ok=True)
            os.makedirs(mini_dev_dir, exist_ok=True)
            os.makedirs(mini_test_dir, exist_ok=True)
            
            # 保存迷你训练数据（阿萨姆语→英语）
            with open(os.path.join(mini_train_dir, "train.asm_Beng"), 'w', encoding='utf-8') as f:
                for text in mini_train_df['as']:
                    f.write(f"{text.strip()}\n")
            with open(os.path.join(mini_train_dir, "train.eng_Latn"), 'w', encoding='utf-8') as f:
                for text in mini_train_df['en']:
                    f.write(f"{text.strip()}\n")
            
            # 保存迷你验证数据
            with open(os.path.join(mini_dev_dir, "dev.asm_Beng"), 'w', encoding='utf-8') as f:
                for text in mini_dev_df['as']:
                    f.write(f"{text.strip()}\n")
            with open(os.path.join(mini_dev_dir, "dev.eng_Latn"), 'w', encoding='utf-8') as f:
                for text in mini_dev_df['en']:
                    f.write(f"{text.strip()}\n")
            
            # 保存迷你测试数据
            with open(os.path.join(mini_test_dir, "test.asm_Beng"), 'w', encoding='utf-8') as f:
                for text in mini_test_df['as']:
                    f.write(f"{text.strip()}\n")
            with open(os.path.join(mini_test_dir, "test.eng_Latn"), 'w', encoding='utf-8') as f:
                for text in mini_test_df['en']:
                    f.write(f"{text.strip()}\n")
            
            print(f"✓ 迷你数据集已按照官方格式保存到: {mini_output_dir}")
            print(f"  - 训练: {len(mini_train_df)} 样本")
            print(f"  - 验证: {len(mini_dev_df)} 样本")
            print(f"  - 测试: {len(mini_test_df)} 样本")
        
        return output_dir, mini_output_dir
    else:
        print(f"错误：数据不足，需要至少54000行，实际只有{len(df)}行")
        return None, None

if __name__ == "__main__":
    prepare_asm_eng_data()
