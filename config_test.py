from dataclasses import dataclass
from os.path import abspath, dirname

PROJECT_ROOT: str = '/'.join(abspath(dirname(__file__)).split('\\')) if '\\' in abspath(dirname(__file__)) else abspath(dirname(__file__))

@dataclass
class TrainConfig:
    tokenizer_dir: str = PROJECT_ROOT + '/model_save/'
    train_file: str = PROJECT_ROOT + '/data/my_train_dataset.parquet'
    validation_file: str = PROJECT_ROOT + '/data/my_valid_dataset.parquet'
    test_file: str = PROJECT_ROOT + '/data/my_test_dataset.parquet'
    output_dir: str = PROJECT_ROOT + '/model_save/pretrain'

    num_train_epochs: int = 1
    batch_size_per_gpu: int = 8
    learning_rate: float = 0.001
    div_factor: int = 50
    bf16: bool = False
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 1024
    logging_steps: int = 2
    save_steps: int = 2
    save_total_limit: int = 2
    seed: int = 23333
    max_seq_len: int = 256

@dataclass
class T5ModelConfig:
    d_ff: int = 128
    d_model: int = 32
    num_heads: int = 1
    d_kv: int = 32
    num_decoder_layers: int = 2
    num_layers: int = 2

@dataclass
class SFTConfig:
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/pretrain/checkpoint-6'
    tokenizer_dir: str = PROJECT_ROOT + '/model_save/'  # tokenizer一般和model权重放在同一个文件夹
    sft_train_file: str = PROJECT_ROOT + '/data/sft_train.parquet'
    sft_test_file: str = PROJECT_ROOT + '/data/sft_test.parquet'
    output_dir: str = PROJECT_ROOT + '/model_save/sft'

    max_seq_len: int = 384 + 8  # 8 for eos token
    batch_size_per_gpu: int = 8
    num_train_epochs: int = 1
    save_steps: int = 10
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    logging_first_step: bool = True
    logging_steps: int = 2

    warmup_steps: int = 100
    bf16: bool = False
    seed: int = 23333
    save_total_limit = 3