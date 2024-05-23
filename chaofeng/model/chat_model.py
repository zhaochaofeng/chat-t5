

import torch
from torch import LongTensor, Tensor
from transformers import T5ForConditionalGeneration, T5Config
from transformers import GenerationConfig
from transformers import TextIteratorStreamer

class TextToTextModel(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super(TextToTextModel, self).__init__(config)

    @torch.no_grad
    def my_generate(self,
                    input_ids: LongTensor,
                    attention_mask: LongTensor,
                    max_seq_len: int = 256,
                    search_type: str = 'beam',
                    streamer: TextIteratorStreamer = None,
                    ) -> Tensor:
        gene_config = GenerationConfig()
        gene_config.remove_invalid_values = True
        gene_config.eos_token_id = 1
        gene_config.pad_token_id = 0
        gene_config.decoder_start_token_id = self.config.decoder_start_token_id
        gene_config.max_new_tokens = max_seq_len

        if search_type == 'greedy':
            gene_config.num_beams = 1
            gene_config.do_sample = False
        elif search_type == 'beam':
            gene_config.top_k = 50
            gene_config.num_beams = 5
            gene_config.do_sample = True
            gene_config.top_p = 0.95
            gene_config.no_repeat_ngram_size = 4
            gene_config.length_penalty = -2.0
            gene_config.early_stopping = True
        elif search_type == 'sampling':
            gene_config.num_beams = 1
            gene_config.do_sample = True
            gene_config.top_k = 50
            gene_config.temperature = 0.98   # 越低，贫富差距越大，越高(>1)，越趋向于均匀分布
            gene_config.top_p = 0.80
            gene_config.no_repeat_ngram_size = 4
        elif search_type == 'contrastive':
            gene_config.penalty_alpha = 0.5
            gene_config.top_k = 50

        result = self.generate(
            inputs=input_ids,
            attention_mask=attention_mask,  # 为什么要加这个参数 ？？？
            generation_config=gene_config,
            streamer=streamer,
        )
        return result


