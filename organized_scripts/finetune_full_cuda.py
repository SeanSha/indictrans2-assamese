import os
import json
import torch
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

# 从官方格式加载数据
def load_data_from_official_format(data_dir, split="train", swap_src_tgt=False):
    split_dir = os.path.join(data_dir, split, "eng_Latn-asm_Beng")
    src_file = os.path.join(split_dir, f"{split}.eng_Latn")
    tgt_file = os.path.join(split_dir, f"{split}.asm_Beng")
    if not os.path.exists(src_file) or not os.path.exists(tgt_file):
        raise FileNotFoundError(f"数据文件不存在: {src_file} 或 {tgt_file}")
    with open(src_file, "r", encoding="utf-8") as f:
        src_lines = [x.strip() for x in f if x.strip()]
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_lines = [x.strip() for x in f if x.strip()]
    n = min(len(src_lines), len(tgt_lines))
    src_lines, tgt_lines = src_lines[:n], tgt_lines[:n]
    data = [{"sentence_SRC": None, "sentence_TGT": None} for _ in range(n)]
    for i in range(n):
        if swap_src_tgt:
            # A→E: 源=阿萨姆语，目标=英语
            data[i]["sentence_SRC"] = tgt_lines[i]
            data[i]["sentence_TGT"] = src_lines[i]
        else:
            # E→A: 源=英语，目标=阿萨姆语
            data[i]["sentence_SRC"] = src_lines[i]
            data[i]["sentence_TGT"] = tgt_lines[i]
    return data

# 预处理函数
# 预处理函数（手动提供等长的 decoder_input_ids/decoder_attention_mask，避免 batch 拼接报错）
def preprocess_fn(example, tokenizer, lang_tags):
    src_tag, tgt_tag = lang_tags
    max_src_len = 256
    max_tgt_len = 256

    # 源侧编码：不定长（交给 collator pad），但截断
    model_inputs = tokenizer(
        f"{src_tag} {tgt_tag} {example['sentence_SRC']}",
        truncation=True, padding=False, max_length=max_src_len
    )

    # 目标侧：固定长度，便于构造等长 decoder_input_ids/labels
    with tokenizer.as_target_tokenizer():
        tgt_tokens = tokenizer(
            example["sentence_TGT"],
            truncation=True, padding="max_length", max_length=max_tgt_len
        )
    tgt_ids = tgt_tokens["input_ids"]  # 长度固定为 max_tgt_len

    # labels：pad 替换为 -100
    pad_id = tokenizer.pad_token_id
    labels = [(tok if tok != pad_id else -100) for tok in tgt_ids]
    model_inputs["labels"] = labels

    # decoder_input_ids：右移一位，首位放目标语标签 ID（eng_Latn），保持等长
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_tag)
    if tgt_lang_id is None or (hasattr(tokenizer, "unk_token_id") and tgt_lang_id == tokenizer.unk_token_id):
        raise ValueError(f"未找到目标语言 token: {tgt_tag}")
    dec_inp = [tgt_lang_id] + tgt_ids[:-1]

    # decoder_attention_mask：pad=0，非 pad=1
    dec_attn = [0 if x == pad_id else 1 for x in dec_inp]

    model_inputs["decoder_input_ids"] = dec_inp
    model_inputs["decoder_attention_mask"] = dec_attn
    return model_inputs

# 计算简单指标
def compute_metrics_simple(eval_preds):
    preds, _ = eval_preds
    try:
        if isinstance(preds, tuple):
            preds = preds[0]
        return {"eval_loss": float(getattr(preds, "mean", lambda: 0.0)())}
    except Exception:
        return {"eval_loss": 0.0}

# 主函数
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 用 INDIC_EN
    model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
    lang_tags = ("asm_Beng", "eng_Latn")

    data_dir = "assamese_english_official_format"
    base_out_dir = os.environ.get("OUTPUTS_DIR_BASE", "outputs")
    os.makedirs(base_out_dir, exist_ok=True)
    out_dir = os.path.join(
        base_out_dir,
        f"assamese_english_full_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="eager",
        dropout=0.1,
    )

    # 仅设置 decoder_start_token_id = 目标语标签（eng_Latn）；禁用缓存更稳
    tgt_lang_id = tokenizer.convert_tokens_to_ids(lang_tags[1])
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tgt_lang_id
    model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
        model.generation_config.eos_token_id = model.config.eos_token_id
        model.generation_config.pad_token_id = model.config.pad_token_id
        model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    else:
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_model_config(model.config)
        model.generation_config.use_cache = False

    # 全参数微调：不使用 LoRA，解冻全部参数
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    swap = True
    train_data = load_data_from_official_format(data_dir, "train", swap_src_tgt=swap)
    dev_data = load_data_from_official_format(data_dir, "dev", swap_src_tgt=swap)
    train_ds = Dataset.from_list(train_data)
    dev_ds = Dataset.from_list(dev_data)

    train_ds = train_ds.map(lambda ex: preprocess_fn(ex, tokenizer, lang_tags))
    dev_ds = dev_ds.map(lambda ex: preprocess_fn(ex, tokenizer, lang_tags))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    # 训练参数
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        do_train=True,
        do_eval=True,
        fp16=True,  # GPU上用 fp16
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",

        # 约6小时建议（视GPU/数据略有浮动）
        max_steps=6000,              # 覆盖 num_train_epochs
        # num_train_epochs=3,        # 可保留但不会生效

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # 全局 batch ≈ 32
        eval_accumulation_steps=2,
        
        learning_rate=3e-5,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        label_smoothing_factor=0.1,

        predict_with_generate=True,     # 开启生成评估
        generation_max_length=128,
        generation_num_beams=1,         # 关键：评估禁用 beam，避开缓存分支

        logging_steps=50,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics_simple,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1e-3)],
    )

    # 强制评估阶段 generate 使用无缓存、无 beam 的最稳设置（避免 past_key_values/shape 异常）
    if getattr(trainer, "_gen_kwargs", None) is None:
        trainer._gen_kwargs = {}
    trainer._gen_kwargs.update({
        "use_cache": False,
        "num_beams": 1,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    })

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(out_dir)

    # 简单保存若干样本生成（可选，保持与评估一致的稳定路径）
    model = trainer.model
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    try:
        samples = dev_data[:5]
        outs = []
        for s in samples:
            enc = tokenizer(f"{lang_tags[0]} {lang_tags[1]} {s['sentence_SRC']}",
                            return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=128,
                    num_beams=1,
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,
                )
            pred = tokenizer.decode(gen[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            outs.append({"src": s["sentence_SRC"], "tgt": s["sentence_TGT"], "pred": pred})
        with open(os.path.join(out_dir, "simple_evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump(outs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[post-eval] generation failed: {e}")

if __name__ == "__main__":
    main()