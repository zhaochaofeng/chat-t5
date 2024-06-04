from typing import Dict
from config import TrainConfig, T5ModelConfig
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import GenerationConfig
from transformers import EarlyStoppingCallback
from accelerate import Accelerator
from model.chat_model import TextToTextModel
from utils.functions import get_T5_config, MyTrainerCallback
from datasets import Dataset
from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
import time
import os



def get_dataset(file: str, split: str, tokenizer: PreTrainedTokenizerFast, cache_dir: str='.cache') -> Dataset:
    dataset = load_dataset(path='parquet', data_files=file, split=split, cache_dir=cache_dir)

    def token_to_ids(example: dict) -> Dict[str, list]:
        batch_prompt = example['prompt']
        batch_response = example['response']
        eos_token_id = tokenizer.eos_token_id

        encoded_prompt = tokenizer(text=batch_prompt, padding=False, truncation=False, return_attention_mask=False)
        encoded_response = tokenizer(text=batch_response, padding=False, truncation=False, return_attention_mask=False)

        # 如果vocab size<=65535, 则可以时用无符号整数类型uint16
        input_ids = [np.array(item + [eos_token_id], dtype=np.uint16) for item in encoded_prompt['input_ids']]
        labels = [np.array(item + [eos_token_id], dtype=np.uint16) for item in encoded_response['input_ids']]

        return {
            'input_ids': input_ids,
            'labels': labels
        }

    dataset = dataset.map(function=token_to_ids, batched=True, batch_size=100, remove_columns=dataset.column_names)
    return dataset

def pre_train(config: TrainConfig) -> None:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)
    config_t5 = get_T5_config(T5ModelConfig(),
                              vocab_size=len(tokenizer),
                              decoder_start_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id)
    model = TextToTextModel(config_t5)

    # 加载 BLEU 评估指标
    bleu_metric = load_metric("sacrebleu")
    def compute_bleu_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # 替换 -100 为 pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 计算 BLEU 分数
        result = bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
        result = {"bleu": result["score"]}
        return result

    dataset = get_dataset(file=config.train_file, split='train', tokenizer=tokenizer)
    dataset_eval = get_dataset(file=config.validation_file, split='train', tokenizer=tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, max_length=config.max_seq_len)

    # accelerator = Accelerator(cpu=True)
    # model, tokenizer, data_collator, dataset = accelerator.prepare(model, tokenizer, data_collator, dataset)

    generation_config = GenerationConfig()
    generation_config.remove_invalid_values = True
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.decoder_start_token_id = tokenizer.pad_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.max_new_tokens = 320
    generation_config.num_beams = 1
    generation_config.do_sample = False

    args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy='steps',
        eval_steps=5,
        learning_rate=config.learn_rate,
        per_device_train_batch_size=config.batch_size_per_gpu,
        per_gpu_eval_batch_size=config.batch_size_per_gpu,
        weight_decay=0.1,
        save_total_limit=5,
        save_steps=config.save_steps,
        num_train_epochs=1,
        max_steps=100,
        bf16=True if config.mixed_precision == 'bf16' else False,
        fp16=True if config.mixed_precision == 'fp16' else False,
        bf16_full_eval=True if config.mixed_precision == 'bf16' else False,
        fp16_full_eval=True if config.mixed_precision == 'fp16' else False,
        auto_find_batch_size=True,
        remove_unused_columns=True,
        logging_first_step=True,
        logging_steps=config.logging_steps,
        log_level='info',
        warmup_steps=config.warmup_steps,
        report_to=['tensorboard'],
        generation_config=generation_config,
        seed=config.seed,
        optim='adafactor',
        load_best_model_at_end=True,
        metric_for_best_model='bleu',
        greater_is_better=True,
        predict_with_generate=True,

    )

    empty_cuda_cahce = MyTrainerCallback()
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.1
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=dataset_eval,
        tokenizer=tokenizer,
        # compute_metrics=compute_bleu_metrics,
        callbacks=[empty_cuda_cahce],
    )

    trainer.train(
        # resume_from_checkpoint=True
    )

    loss_log = pd.DataFrame(trainer.state.log_history)
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    loss_log.to_csv(path_or_buf=f"{log_dir}/pre_train_log_{time.strftime('%Y%m%d-%H%M%S')}.csv", index_label='index')
    trainer.save_model(output_dir=config.output_dir+'_save')
    # accelerator.save_state(output_dir=config.output_dir+'_save')

if __name__ == '__main__':
    config = TrainConfig()
    pre_train(config)


