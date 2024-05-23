
import os
from config import PROJECT_ROOT

import tokenizers.normalizers
import tokenizers.decoders
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Punctuation, Digits, Metaspace

from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

def check_dir_exists(dir: str) -> None:
    if not os.path.exists(dir):
        os.makedirs(dir)

def train_tokenizer(corpus_file: str,
                    max_train_line: int=None,
                    vocab_size: int=40960,
                    token_type: str='char') -> None:
    tokenizer_slow_save_path = PROJECT_ROOT + '/model_save/hf_tokenizer_slow/tokenizer.json'
    tokenizer_fast_save_path = PROJECT_ROOT + '/model_save/hf_tokenizer_fast/'
    check_dir_exists(PROJECT_ROOT + '/model_save/hf_tokenizer_slow')
    check_dir_exists(tokenizer_fast_save_path)

    def get_training_corpus(buffer_size: int = 1000, chunk_len: int = 2048) -> list:
        line_cnt = 0
        buffer = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            curr_chunk_txt, txt_len = [], 0
            for line in f:
                curr_chunk_txt.append(line)
                txt_len += len(line)
                line_cnt += 1
                if line_cnt % 100 == 0 and line_cnt > 0:
                    print('process line: {}'.format(line_cnt))
                if txt_len >= chunk_len:
                    buffer.append(''.join(curr_chunk_txt))
                    curr_chunk_txt, txt_len = [], 0
                if len(buffer) > buffer_size:
                    yield buffer
                    buffer = []
                if isinstance(max_train_line, int) and line_cnt >= max_train_line:
                    break
            if len(buffer) > 0:
                yield buffer
            print(line_cnt)

    special_tokens = ["[PAD]", "[EOS]", "[SEP]", "[BOS]", "[CLS]", "[MASK]", "[UNK]"]
    if token_type == 'char':
        tokenizer = Tokenizer(model=BPE(unk_token='[UNK]'))
        tokenizer.normalizer = tokenizers.normalizers.Sequence([NFKC()])
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
            [Punctuation(), Digits(individual_digits=True), Metaspace()])

        tokenizer.add_special_tokens(tokens=special_tokens)
        tokenizer.decoder = tokenizers.decoders.Metaspace()
    else:
        raise Exception("token type must be char or type, but get {}".format(token_type))
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=100, show_progress=True, special_tokens=special_tokens)
    tokenizer.train_from_iterator(iterator=get_training_corpus(), trainer=trainer)

    if '\t' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['\t'])
    if '\n' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['\n'])

    tokenizer.save(path=tokenizer_slow_save_path)

    slow_tokenizer = tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=slow_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        bos_token='[BOS]',
        eos_token='[EOS]',
    )
    fast_tokenizer.save_pretrained(tokenizer_fast_save_path)
    print('save slow tokenizer in: {}'.format(tokenizer_slow_save_path))
    print('save fast tokenizer in: {}'.format(tokenizer_fast_save_path))

if __name__ == '__main__':
    corpus_file = PROJECT_ROOT + '/data/wiki.simple.txt'
    train_tokenizer(corpus_file=corpus_file,
                    token_type='char',
                    max_train_line=100000)

