
import os
from collections import Counter
from typing import Union
import numpy as np
import ujson
from config import T5ModelConfig
from transformers import T5Config

def get_free_space_of_disk(folder: str='./') -> float:
    statvfs = os.statvfs(folder)
    res_val = statvfs.f_frsize * statvfs.f_bavail
    return res_val / (1024 ** 3)

def get_T5_config(config: T5ModelConfig,
                  vocab_size: int,
                  decoder_start_token_id: int = 0,
                  eos_token_id: int = 1) -> T5Config:
    t5_config = T5Config()
    t5_config.d_ff = config.d_ff
    t5_config.d_kv = config.d_kv
    t5_config.d_model = config.d_model
    t5_config.num_decoder_layers = config.num_decoder_layers
    t5_config.num_heads = config.num_heads
    t5_config.num_layers = config.num_layers
    t5_config.vocab_size = vocab_size
    t5_config.decoder_start_token_id = decoder_start_token_id
    t5_config.eos_token_id = eos_token_id

    return t5_config


def save_model_config(config_dict: dict, file: str) -> None:
    with open(file, 'w', encoding='utf-8') as f:
        ujson.dump(config_dict, f, indent=4, ensure_ascii=False)

def my_average(array_list: list[float]) -> float:
    if len(array_list) == 0:
        return 0.0
    return np.average(array_list)


def get_bleu4_score(reference: Union[str, list[str]], outputs: Union[str, list[str]], n_gram: int = 4) -> float:
    '''
    获取bleu4分数
    '''

    weights = np.ones(n_gram) * (1.0 / n_gram)

    outputs_len, reference_len = len(outputs), len(reference)

    if not type(reference) is list:
        reference = list(reference)
    if not type(outputs) is list:
        outputs = list(outputs)

    outputs_counter = extract_Ngram(outputs, n_gram=n_gram)
    reference_counter = extract_Ngram(reference, n_gram=n_gram)

    ngram_counter_clip = outputs_counter & reference_counter

    clip_counter = np.zeros(n_gram)
    output_ngram_counter = np.zeros(n_gram)

    for (key, ngram), cnt in ngram_counter_clip.items():
        clip_counter[ngram - 1] += cnt

    for (key, ngram), cnt in outputs_counter.items():
        output_ngram_counter[ngram - 1] += cnt

    # print(clip_counter, output_ngram_counter)
    if np.min(clip_counter) == 0.0:
        return np.array(0.0)

    precision_scores = clip_counter / output_ngram_counter

    # bleu
    log_precision_scores = weights * np.log(precision_scores)

    # 几何平均形式求平均值然后加权
    geometric_mean = np.exp(np.sum(log_precision_scores))
    brevity_penalty = np.exp(1 - (reference_len / outputs_len))

    # brevity_penalty = 1.0,   bleu = sentence_bleu([reference], outputs)
    # brevity_penalty = 1.0

    bleu = brevity_penalty * geometric_mean

    return bleu

def extract_Ngram(words_list: list[str], n_gram: int) -> tuple:
    '''
    获取一个句子的n_grama
    return：
        ngram_counter： key = ('w1  w2 ... wn', n_gram), value: count of key
    '''
    n = len(words_list)
    ngram_counter = Counter()

    for i in range(1, n_gram + 1):
        for j in range(n - i + 1):
            key = ' '.join(words_list[j: j + i])
            ngram_counter[(key, i)] += 1

    return ngram_counter


if __name__ == '__main__':
    print(get_free_space_of_dist('/'))


