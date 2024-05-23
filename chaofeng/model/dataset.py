
import pyarrow.parquet as pq
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from config import PROJECT_ROOT


class MyDataset(Dataset):
    def __init__(self,
                 parquet_file: str,
                 tokenizer_dir: str,
                 keep_in_memory: bool = False,
                 max_seq_len: int = 512,
                 buffer_size: int = 40960):
        super(MyDataset, self).__init__()

        if torch.cuda.device_count() >= 2 and not keep_in_memory:
            raise ValueError('when number of GPU >=2 then keep_in_memory should be True !')

        self.keep_in_memory = keep_in_memory
        self.max_seq_len = max_seq_len

        parquet_table = pq.read_table(parquet_file)
        self.length = parquet_table.num_rows

        self.buffer_size = self.length if self.length < buffer_size else buffer_size

        if keep_in_memory:
            self.data = parquet_table.to_pandas()
        else:
            self.data = parquet_table

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        self.sample_generator = self.item_generator()

    def item_generator(self):
        parquet_table = self.data
        buffer_list = []
        while True:
            for prompt, response in zip(parquet_table['prompt'], parquet_table['response']):
                if len(buffer_list) < self.buffer_size:
                    buffer_list.append((prompt.as_py(), response.as_py()))
                    continue
                np.random.shuffle(buffer_list)
                for p, r in buffer_list:
                    yield p, r

                buffer_list = []

    def __getitem__(self, index):
        if self.keep_in_memory:
            data = self.data
            prompt, response = data.iloc[index].prompt, data.iloc[index].response
        else:
            prompt, response = next(self.sample_generator)
        max_seq_len = self.max_seq_len - 5
        return "{}[EOS]".format(prompt[0: max_seq_len]), "{}[EOS]".format(response[0: max_seq_len])

    def __len__(self) -> int:
        return self.length

    def collate_fn(self, data: list[list]) -> dict:
        tokneizer = self.tokenizer
        prompt = tokneizer([item[0] for item in data], padding=True, return_token_type_ids=False)
        response = tokneizer([item[1] for item in data], padding=True, return_token_type_ids=False)

        input_ids = np.array(prompt.input_ids, dtype=np.int64)
        input_mask = np.array(prompt.attention_mask, dtype=np.int64)
        target_ids = np.array(response.input_ids, dtype=np.int64)

        ret = {
            'input_ids': torch.LongTensor(input_ids),
            'input_mask': torch.LongTensor(input_mask),
            'target_ids': torch.LongTensor(target_ids)
        }
        return ret

'''
if __name__ == '__main__':
    parquet_file = PROJECT_ROOT + '/data/my_test_dataset.parquet'
    tokenizer_dir = PROJECT_ROOT + '/model_save/hf_tokenizer_fast'

    dataset = MyDataset(parquet_file, tokenizer_dir)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=5,
        collate_fn=dataset.collate_fn,
    )
    for i, item in enumerate(dataloader):
        if i >= 2:
            break
        print(item)
        input_ids, input_mask, target_ids = item['input_ids'], item['input_mask'], item['target_ids']
        # print(input_ids, input_mask, target_ids)
'''
