from dataclasses import dataclass
from os.path import abspath, dirname

# /Users/chaofeng/code/ChatLM-mini-Chinese/chaofeng
PROJECT_ROOT: str = abspath(dirname(__file__))


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size_per_gpu: int = 16
    learn_rate: float = 1e-4
    div_factor: int = 50
    mixed_precision: str = 'bf16'
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 1024
    tokenizer_dir: str = PROJECT_ROOT + '/model_save/hf_tokenizer_fast'
    model_file: str = PROJECT_ROOT +'/model_save/chat_small_t5_{}.bin'
    model_config_file: str = PROJECT_ROOT +'/model_save/model_config.json'

    train_file: str = PROJECT_ROOT +'/data/my_train_dataset.parquet'
    validation_file: str = PROJECT_ROOT +'/data/my_valid_dataset.parquet'
    test_file: str = PROJECT_ROOT +'/data/my_test_dataset.parquet'

    finetune_from_ckpt_file: str = PROJECT_ROOT +'/model_save/chat_small_t5.best.bin'
    train_state_dir: str = PROJECT_ROOT + '/model_save/train_latest_state'
    output_dir: str = PROJECT_ROOT + '/model_save/pretrain'
    logging_steps: int = 50
    save_steps: int = 10000
    keep_latest_n_ckp: int = 8
    seed: int = 23333
    dataloader_buffer_size: int = 500000
    max_seq_len: int = 256

@dataclass
class T5ModelConfig:
    d_ff: int = 128
    d_model: int = 32
    num_heads: int = 1
    d_kv: int = 32
    num_decoder_layers: int = 1
    num_layers: int = 1

