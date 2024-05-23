
from accelerate import Accelerator
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_optimizer import Adafactor
from config import TrainConfig, T5ModelConfig
from dataset import MyDataset
from model.chat_model import TextToTextModel
from utils.functions import (
    get_T5_config
)

class ChatModel:
    def __init__(self, train_config: TrainConfig=None, model_config:T5ModelConfig=None):
        self.train_config = train_config
        self.model_config = model_config
        self.model = None

    def train(self):
        accelerator = Accelerator(
            mixed_precision=self.train_config.mixed_precision,
            gradient_accumulation_steps=self.train_config.gradient_accumulation_steps,
            project_dir=self.train_config.train_state_dir,
            cpu=True,
        )

        train_dataset = MyDataset(
            parquet_file=self.train_config.train_file,
            tokenizer_dir=self.train_config.tokenizer_dir,
            keep_in_memory=False,
            max_seq_len=self.train_config.max_seq_len,
        )
        valid_dataset = MyDataset(
            parquet_file=self.train_config.validation_file,
            tokenizer_dir=self.train_config.tokenizer_dir,
            keep_in_memory=False,
            max_seq_len=self.train_config.max_seq_len,
        )
        print('train len: {}, valid len: {}'.format(len(train_dataset), len(valid_dataset)))
        batch_size = self.train_config.batch_size_per_gpu

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            pin_memory=False,
            num_workers=0,
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=valid_dataset.collate_fn,
            pin_memory=False,
            num_workers=0,
        )
        tokenizer = train_dataset.tokenizer

        t5_config = get_T5_config(model_config,
                                  vocab_size=len(tokenizer),
                                  decoder_start_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id,
                                  )
        model = TextToTextModel(t5_config)
        self.model = model

        print(sum(p.numel() for p in model.parameters()))

        optimizer = Adafactor(model.parameters(), lr=self.train_config.learn_rate)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.train_config.div_factor * self.train_config.learn_rate,
            epochs=self.train_config.epochs,
            steps_per_epoch=int(np.ceil(len(train_dataset) / (batch_size * self.train_config.gradient_accumulation_steps))),
            div_factor=self.train_config.div_factor,
            cycle_momentum=False,
        )

        model, optimizer, lr_scheduler, train_dataloader, valid_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader, valid_dataloader
        )

        print(sum(p.numel() for p in model.parameters()))

        for epoch in range(self.train_config.epochs):
            model.train()
            for step, batch_data in enumerate(train_dataloader):
                input_ids, input_mask, target_ids = batch_data['input_ids'], batch_data['input_mask'], batch_data['target_ids']
                target_ids[target_ids == tokenizer.pad_token_id] = -100
                output = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    labels=target_ids
                )
                loss = output.loss.mean() / self.train_config.gradient_accumulation_steps
                print('step: {}, loss: {:.3f}'.format(step, loss))
                accelerator.backward(loss)
                if (step + 1) % self.train_config.gradient_accumulation_steps == 0:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

if __name__ == '__main__':
    train_config = TrainConfig()
    model_config = T5ModelConfig()
    chat_model = ChatModel(train_config, model_config)

    chat_model.train()
