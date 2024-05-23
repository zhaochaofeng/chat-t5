
'''
import torch
import torch.nn as nn
from accelerate import Accelerator

# Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize the model and accelerator
model = SimpleModel()
accelerator = Accelerator(project_dir='/Users/chaofeng/code/ChatLM-mini-Chinese/test/tmp')
optimizer = torch.optim.Adam(model.parameters())

# Register the learning rate scheduler for checkpointing
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
accelerator.register_for_checkpointing(lr_scheduler)

# Dummy training loop
for epoch in range(5):
    # Training code here
    # Simulate training interruption

    if epoch == 2:
        # Save the training state
        accelerator.save_state(output_dir=accelerator.project_dir)
        print("Training interrupted. State saved.")

# Simulate resuming training from the checkpoint
if epoch == 2:
    # Load the training state
    accelerator.load_state(input_dir=accelerator.project_dir)
    print("Training resumed from the checkpoint.")

# Continue training
for epoch in range(2, 5):
    # Training code here
    print("Epoch:", epoch)

print("Training completed.")
'''

import torch
import torch.nn as nn
from accelerate import Accelerator


# 定义一个简单的 PyTorch 模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 初始化模型和 accelerator
model = SimpleModel()
accelerator = Accelerator(project_dir='/Users/chaofeng/code/ChatLM-mini-Chinese/test/tmp',
                          cpu=True)
optimizer = torch.optim.Adam(model.parameters())
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

model, optimizer = accelerator.prepare(model, optimizer)

# 使用 Accelerator 的 register_for_checkpointing 函数记录学习率调度器的保存
accelerator.register_for_checkpointing(lr_scheduler)

# 准备一些伪数据
train_data = torch.randn(100, 10)
train_targets = torch.randn(100, 1)
train_loader = torch.utils.data.DataLoader(list(zip(train_data, train_targets)), batch_size=10)

from torch.utils.data import DataLoader

# 模拟训练循环
for epoch in range(100):
    # 模拟每个 epoch 的训练步骤
    for batch in train_loader:
        # 模拟训练代码
        data, target = batch
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print('epoch: {}, loss: {}'.format(epoch, loss.item()))

    # 模拟训练中断
    if epoch == 2:
        # 保存训练状态
        accelerator.save_state(output_dir=accelerator.project_dir)
        print("Training interrupted. State saved.")

# 模拟从检查点处恢复训练
accelerator.load_state(input_dir=accelerator.project_dir)
print("Training resumed from the checkpoint.")

# 继续训练
for epoch in range(100):
    # 模拟每个 epoch 的训练步骤
    for batch in train_loader:
        # 模拟训练代码
        data, target = batch
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print("Epoch resume:{}, loss: {}".format(epoch, loss))

print("Training completed.")

'''
epoch: 0, loss: 1.4259713888168335
Training interrupted. State saved.
epoch: 10, loss: 1.1083863973617554
epoch: 20, loss: 0.9094504117965698
epoch: 30, loss: 0.7877599000930786
epoch: 40, loss: 0.7149628400802612
epoch: 50, loss: 0.672378659248352
epoch: 60, loss: 0.6479735374450684
epoch: 70, loss: 0.634179949760437
epoch: 80, loss: 0.6263947486877441
epoch: 90, loss: 0.6219289898872375
Training resumed from the checkpoint.
Epoch resume:0, loss: 0.6192759275436401
Epoch resume:10, loss: 0.6176226735115051
Epoch resume:20, loss: 0.6165431141853333
Epoch resume:30, loss: 0.6158155202865601
Epoch resume:40, loss: 0.6153221726417542
Epoch resume:50, loss: 0.6149952411651611
Epoch resume:60, loss: 0.614791989326477
Epoch resume:70, loss: 0.6146815419197083
Epoch resume:80, loss: 0.6146403551101685
Epoch resume:90, loss: 0.6146494150161743
Training completed.
'''

