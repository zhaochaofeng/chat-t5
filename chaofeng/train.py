


import fire
from config import TrainConfig, T5ModelConfig
from model.trainer import ChatTrainer

if __name__ == '__main__':
    train_config = TrainConfig()
    t5_config = T5ModelConfig()
    chat_trainer = ChatTrainer(train_config=train_config, model_config=t5_config)
    fire.Fire(component=chat_trainer)

