import os
import time
import argparse
import pandas as pd
import numpy as np
from typing import Dict
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import GenerationConfig
from model.chat_model import TextToTextModel

def get_dataset(tokenizer: PreTrainedTokenizerFast, file:str, split: str, )-> Dataset:
    dataset = load_dataset(path='parquet', data_files=file, split=split)
    eos_token_id = tokenizer.eos_token_id

    def token_to_id(example: dict) -> Dict[str, list]:
        batch_prompt = example['prompt']
        batch_response = example['response']
        encoded_prompt = tokenizer(text=batch_prompt, padding=False, truncation=False, return_attention_mask=False).input_ids
        encoded_response = tokenizer(text=batch_response, padding=False, truncation=False, return_attention_mask=False).input_ids
        input_ids = [np.array(item + [eos_token_id], dtype=np.uint16) for item in encoded_prompt]
        labels = [np.array(item + [eos_token_id], dtype=np.uint16) for item in encoded_response]
        return {'input_ids': input_ids, 'labels': labels}

    dataset = dataset.map(function=token_to_id, batched=True, batch_size=2048, remove_columns=dataset.column_names)
    return dataset

def sft_train(config, is_keeptrain: bool=False) -> None:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)
    model = TextToTextModel.from_pretrained(config.finetune_from_ckp_file)

    # 冻结参数
    for layer in [model.shared, model.encoder]:
        for p in layer.parameters():
            p.requires_grad = False

    train_dataset = get_dataset(tokenizer, config.sft_train_file, split='train')
    eval_dataset = get_dataset(tokenizer, config.sft_test_file, split='train')
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, max_length=config.max_seq_len)

    generation_config = GenerationConfig()
    generation_config.remove_invalid_values = True
    generation_config.decoder_start_token_id = tokenizer.pad_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.max_new_tokens = 320
    generation_config.repetition_penalty = 1.5
    generation_config.num_beams = 1
    generation_config.do_sample = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy='epoch',
        per_device_train_batch_size=config.batch_size_per_gpu,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        log_level='info',
        logging_steps=config.logging_steps,
        save_strategy='steps',
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        bf16=config.bf16,
        bf16_full_eval=config.bf16,
        remove_unused_columns=True,
        optim='adafactor',
        report_to=['tensorboard'],
        auto_find_batch_size=True,
        predict_with_generate=True,
        generation_config=generation_config,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics='',
        callbacks=[],
    )

    if is_keeptrain:
        print('{}\n{}'.format('-' * 100, 'keep training ...'))
    trainer.train(
        resume_from_checkpoint=is_keeptrain
    )

    loss_log = pd.DataFrame(trainer.state.log_history)
    path_log = './logs'
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    loss_log.to_csv(f"{path_log}/sft_train_log_{time.strftime('%Y%m%d')}.csv", index_label='index')
    trainer.save_model(config.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sft_train_t5')
    parser.add_argument('--is_keeptrain', action='store_true', help='是否从中断处继续训练')
    parser.add_argument('--is_online', action='store_true', help='是否线上训练')
    args = parser.parse_args()

    if args.is_online:
        from config import SFTConfig
    else:
        from config_test import SFTConfig

    config = SFTConfig()
    sft_train(config, args.is_keeptrain)