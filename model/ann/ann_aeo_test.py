#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：ann_aeo_test.py
@Author ：RockJim
@Date ：2023/8/6 18:04
@Description ：用于测试训练出来的ann_aeo模型
@Version ：1.0
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from config.config import DATASET_ROOT_PATH, BASE_DATASET_NAME
from dataset.data_load import load_data, data_processing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_samples = 1000

num_inputs = 12
num_outputs = 1


# 加载数据集
x, y = load_data(os.path.join(r'D:\Code\PythonProject\ML_project', DATASET_ROOT_PATH, BASE_DATASET_NAME))

# 数据预处理
x_train, x_test, y_train, y_test, scaler = data_processing(x, y)

# 将数据转换为PyTorch张量
X_train = torch.tensor(x_train, dtype=torch.float32)
Y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(x_test, dtype=torch.float32)
Y_test = torch.tensor(y_test, dtype=torch.float32)


# 构建神经网络模型
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 32)
        # self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_outputs)

    def forward(self, x_input):
        x_input = torch.relu(self.fc1(x_input))
        # x_input = torch.relu(self.fc2(x_input))
        x_input = self.fc3(x_input)
        return x_input


model = RegressionModel()


# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
batch_size = 1

for epoch in range(num_epochs):
    for i in range(0, num_samples, batch_size):
        inputs = X_train[i:i + batch_size]
        targets = Y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 使用模型进行预测
model.eval()
with torch.no_grad():
    test_inputs = X_test
    predictions = model(test_inputs)

# 或者使用以下代码来打印每个层的权重和偏置：
print("Model's Weights and Biases:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

print("mse损失为：", mean_squared_error(Y_test, predictions))
# evaluate.evaluate.evaluate_prediction(Y_test, predictions)
