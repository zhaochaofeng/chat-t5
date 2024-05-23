
import sys
import numpy as np
from typing import Union
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import torch
from torch.utils.data import DataLoader
from torch_optimizer import Adafactor
import signal
from psutil import virtual_memory
from config import TrainConfig, T5ModelConfig
from utils.logger import Logger
from model.dataset import MyDataset
from model.chat_model import TextToTextModel
from accelerate.utils import set_seed
from accelerate import Accelerator
from transformers import PreTrainedTokenizerFast

from utils.functions import (
    get_free_space_of_disk,
    get_T5_config,
    save_model_config,
    my_average,
    get_bleu4_score,
)

# 防止出现并行tokenizer警告日志
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class ChatTrainer:
    def __init__(self, train_config: TrainConfig, model_config: T5ModelConfig, )->None:
        self.train_config = train_config
        self.model_config = model_config

        self.logger = Logger('chat_trainer', std_out=True, save2file=True, file_name=None)
        self.model = None
        self.accelerator = None

        signal.signal(signal.SIGINT, self.process_exit_handler)

        torch.manual_seed(train_config.seed)
        torch.cuda.manual_seed_all(train_config.seed)

    def process_exit_handler(self, signal_received, frame) -> None:
        if self.accelerator and self.model:
            ask = "you are pressed `ctrl+c`,  do you want to save checkpoint? Yes (y) or No (n)"
            self.accelerator.print(ask)
            ins = input()

            if ins.lower() in ('yes', 'y'):
                self.accelerator.wait_for_everyone()
                self.accelerator.save_state(output_dir=self.train_config.train_state_dir)
                self.accelerator.print('model checkpoint has been saved in: {}'.format(self.train_config.train_state_dir))
            sys.exit(0)
        else:
            print('process not in training, exit !')
            sys.exit(0)

    def train(self, is_keep_training: bool=False, is_finetune: bool=False, )->None:
        log = self.logger
        train_config = self.train_config
        save_steps = train_config.save_steps
        logging_steps = train_config.logging_steps
        accumulation_steps = train_config.gradient_accumulation_steps

        set_seed(train_config.seed)
        accelerator = Accelerator(
            mixed_precision=train_config.mixed_precision,
            gradient_accumulation_steps=accumulation_steps,
            project_dir=train_config.train_state_dir,
            # cpu=True,
        )
        unuse_mem = virtual_memory().available / (1024 ** 3)  # GB
        unuse_dist = get_free_space_of_disk('./')

        keep_in_memory = True if unuse_mem > 48.0 or torch.cuda.device_count() >= 2 else False

        if accelerator.is_main_process:
            log.info('memory available: {:.2f} GB, dist available: {:.2f} GB, keep_in_memory: {}'.
                     format(unuse_mem, unuse_dist, keep_in_memory), save_to_file=True)
            log.info('operation: {}, keep_training: {}, loading data ...'.
                     format('fine-tuning' if is_finetune else 'train', is_keep_training))

        num_workers = 0

        train_dataset = MyDataset(
            parquet_file=train_config.train_file,
            tokenizer_dir=train_config.tokenizer_dir,
            keep_in_memory=False,
            max_seq_len=train_config.max_seq_len,
        )
        val_dataset = MyDataset(
            parquet_file=train_config.validation_file,
            tokenizer_dir=train_config.tokenizer_dir,
            keep_in_memory=False,
            max_seq_len=train_config.max_seq_len,
        )

        batch_size = train_config.batch_size_per_gpu

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            pin_memory=False,
            num_workers=num_workers,
        )
        valid_dataloder = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            pin_memory=False,
            num_workers=num_workers,
        )

        device = accelerator.device
        log.info('using device: {}'.format(device), save_to_file=True)

        tokenizer = train_dataset.tokenizer
        decoder_start_token_id = tokenizer.pad_token_id

        t5_config = get_T5_config(T5ModelConfig(),
                                  vocab_size=len(tokenizer),
                                  decoder_start_token_id=decoder_start_token_id,
                                  eos_token_id=tokenizer.eos_token_id)

        model = TextToTextModel(t5_config)

        if is_finetune:
            print('fine-tuning...')
            model.load_state_dict(torch.load(train_config.finetune_from_ckp_file))

            layer_to_freeze = [model.shared, model.encoder]

            for layer in layer_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        save_model_config(t5_config.to_diff_dict(), train_config.model_config_file)

        optimizer = Adafactor(params=model.parameters(), lr=train_config.learn_rate)

        num_gpu_used = accelerator.state.num_processes
        total_batch_size = train_config.batch_size_per_gpu
        if num_gpu_used > 1:
            total_batch_size = num_gpu_used * train_config.batch_size_per_gpu

        steps_per_epoch = int(np.ceil(len(train_dataset) / total_batch_size))
        eval_steps = int(np.ceil(len(val_dataset) / total_batch_size))

        if accelerator.is_main_process:
            log.info('train dataset size: {}, steps_per_epoch: {}, valid dataset size: {}, eval_steps: {}, num_workers: {}'.format(
                len(train_dataset), steps_per_epoch, len(val_dataset), eval_steps, num_workers
            ), save_to_file=True)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=train_config.div_factor * train_config.learn_rate,
            epochs=train_config.epochs,
            steps_per_epoch=int(np.ceil(len(train_dataloader) / (batch_size * accumulation_steps))),
            div_factor=train_config.div_factor,
            cycle_momentum=False,
        )

        # if is_keep_training:
        #     print('keep training ...')
        #     accelerator.load_state(input_dir=train_config.train_state_dir)
        #     accelerator.register_for_checkpointing(lr_scheduler)

        model, optimizer, lr_scheduler, train_dataloader, valid_dataloder = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader, valid_dataloder,
        )

        if is_keep_training:
            print('keep training ...')
            accelerator.load_state(input_dir=train_config.train_state_dir)
            accelerator.register_for_checkpointing(lr_scheduler)

        self.model = model
        self.accelerator = accelerator

        best_bleu4 = 0.0
        best_epoch = 0
        epoch_loss_list = []
        if accelerator.is_main_process:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                TextColumn("[bold blue]{task.fields[show_info]}"),
                refresh_per_second=1,
            )
            epoch_progress = progress.add_task(description='epochs: ', show_info='', total=train_config.epochs)
            steps_progress = progress.add_task(description='steps: ', show_info='', total=np.ceil(steps_per_epoch / logging_steps))
            eval_progress = progress.add_task(description='evaluate: ', show_info='', total=eval_steps, visible=False)
            self.progress = progress
            self.eval_progress = eval_progress

            progress.start()

        for epoch in range(train_config.epochs):
            if accelerator.is_main_process:
                epoch_show_txt = 'epoch: {}/{}, avg_loss: {:.6f}, best_epoch: {}, best_bleu: {}'.format(
                    epoch, train_config.epochs, my_average(epoch_loss_list), best_epoch, best_bleu4
                )
                progress.update(epoch_progress, show_info=epoch_show_txt)
                progress.reset(steps_progress)
            epoch_loss_list = []
            model.train()

            for step, batch_data in enumerate(train_dataloader):
                input_ids, input_mask, target_ids = batch_data['input_ids'], batch_data['input_mask'], batch_data['target_ids']
                target_ids[target_ids == decoder_start_token_id] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    labels=target_ids,
                )
                loss = outputs.loss.mean() / accumulation_steps
                accelerator.backward(loss)

                if (step + 1) % accumulation_steps == 0:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if (step + 1) % save_steps == 0 or step == steps_per_epoch:
                    self.save_model('epoch_{}_latest'.format(epoch))
                    accelerator.save_state(output_dir=train_config.train_state_dir)

                if step % logging_steps == 0 or step == steps_per_epoch:
                    loss_cpu = loss.detach().item() * accumulation_steps
                    epoch_loss_list.append(loss_cpu)

                    info_txt = 'training loss: epoch: {}, step: {}, loss: {}, device: {}'.\
                        format(epoch, step, loss_cpu, str(accelerator.device))
                    log.info(info_txt, std_out=False, save_to_file=True)

                    if accelerator.is_main_process:
                        step_show_txt = 'step: {}/{}, loss: {:.4f}'.format(step, steps_per_epoch, loss_cpu)
                        progress.advance(steps_progress, advance=1)
                        progress.update(steps_progress, show_info=step_show_txt)

        accelerator.wait_for_everyone()

        model.eval()

        cur_bleu4_score = self.evaluate(
            model=model,
            tokenizer=tokenizer,
            valid_dataloader=valid_dataloder,
            accelerator=accelerator,
            eval_steps=eval_steps,
        )

        if cur_bleu4_score >= best_bleu4:
            best_bleu4 = cur_bleu4_score
            best_epoch = epoch
            self.save_model('best')
            accelerator.save_state(output_dir=train_config.train_state_dir)

        if accelerator.is_main_process:
            info_txt = 'epoch log: epoch:{}, avg_loss:{}, curr_bleu4:{}, best_bleu4:{}, best_epoch:{}'.\
                format(epoch, my_average(epoch_loss_list), cur_bleu4_score, best_bleu4, best_epoch)
            self.print_and_log(info_txt, accelerator)

    def save_model(self, suffix: Union[str, int]) -> None:
        if self.model and self.accelerator:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrap_model = self.accelerator.unwrap_model(self.model)
                model_dict = self.accelerator.get_state_dict(unwrap_model)
                torch.save(model_dict, self.train_config.model_file.format(suffix))

    def print_and_log(self, info: str, accelerator: Accelerator=None) -> None:
        '''
        使用accelerator.print, 否则多进程打印会异常
        '''
        if not accelerator:
            print(info)
        else:
            accelerator.print(info)
        self.logger.info(info, std_out=False, save_to_file=True)

    def evaluate(self,
                 model: TextToTextModel,
                 tokenizer: PreTrainedTokenizerFast,
                 valid_dataloader: DataLoader,
                 accelerator: Accelerator,
                 eval_steps: int,
                 ) -> float:

        '''
        评估，返回平均的bleu分数
        '''
        max_seq_len = self.train_config.max_seq_len
        batch_decode = tokenizer.batch_decode
        bleu4_scores = []

        if accelerator.is_main_process:
            self.progress.reset(self.eval_progress)
            self.progress.update(self.eval_progress, visible=True)

        with torch.no_grad():
            for step, batch_data in enumerate(valid_dataloader):

                if accelerator.is_main_process:
                    self.progress.advance(self.eval_progress, advance=1)
                    self.progress.update(self.eval_progress, show_info='step: {}/{}'.format(step, eval_steps))

                input_ids, input_mask = batch_data['input_ids'], batch_data['input_mask']
                target_ids = batch_data['target_ids']

                outputs = accelerator.unwrap_model(model).my_generate(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    max_seq_len=max_seq_len,
                )

                # gather data from multi-gpus (used when in ddp mode)
                outputs = accelerator.gather_for_metrics(outputs).detach().cpu().numpy()
                target_ids = accelerator.gather_for_metrics(target_ids).detach().cpu().numpy()

                outputs = batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                target_ids = batch_decode(target_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                # print(outputs, target_ids)

                bleu4_scores = [get_bleu4_score(reference=target_ids[i], outputs=outputs[i]) for i in
                                range(len(target_ids))]
                bleu4_scores.extend(bleu4_scores)

                # if step >= 5: break

        avg_bleu4_score = my_average(bleu4_scores)
        if accelerator.is_main_process:
            self.progress.update(self.eval_progress, show_info='bleu4 score: {}'.format(avg_bleu4_score))
            self.progress.update(self.eval_progress, visible=False)

        return avg_bleu4_score

if __name__ == '__main__':
    train_config = TrainConfig()
    model_config = T5ModelConfig()

    chat_trainer = ChatTrainer(train_config=train_config, model_config=model_config)
    chat_trainer.train(is_finetune=False, is_keep_training=True)
