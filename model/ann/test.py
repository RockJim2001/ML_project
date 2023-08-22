#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：test.py
@Author ：RockJim
@Date ：2023/8/9 16:16
@Description ：首先让ANN算法对数据跑起来
@Version ：1.0
"""
# import os
#
# from config.log_config import log
# from dataset.data_load import load_data, data_processing
# from dataset.data_config import DATASET_ROOT_PATH, BASE_DATASET_NAME
# from torch.utils.tensorboard import SummaryWriter
# import torch
# from torch import nn, optim
# import numpy as np
#
# from evaluate.evaluate import evaluate_prediction
#
# # 指定日志存储路径
# log_dir = r"logs/"
# logger = log().getLogger(__name__)
#
# # 生成训练数据
# # 加载数据集
# x_data, y_data = load_data(os.path.join(r'D:\Code\PythonProject\ML_project', DATASET_ROOT_PATH, '数据.csv'))
# # 数据归一化处理
# x_train, x_test, y_train, y_test, scaler = data_processing(x_data, y_data)
# N = 1000  # 训练数据样本个数。
# in_features = 12  # 单个输入样本的变量个数。
# out_features = 6  # 单个输出的变量个数。
#
# # 将这些值转为tensor格式
# X_train = torch.tensor(x_train, dtype=torch.float32)
# Y_train = torch.tensor(y_train, dtype=torch.float32)
# X_test = torch.tensor(x_test, dtype=torch.float32)
# Y_test = torch.tensor(y_test, dtype=torch.float32)
#
#
# # 创建模型
# class Net(nn.Module):
#
#     def __init__(self, input_features, hidden_dims, output_features):
#         super().__init__()
#         self.hidden_layers = nn.ModuleList()  # 使用 ModuleList 来管理多个隐藏层
#
#         prev_dim = input_features  # 记录前一层的维度
#         for dim in hidden_dims:
#             self.hidden_layers.append(nn.Linear(prev_dim, dim))
#             prev_dim = dim
#         self.output_layer = nn.Linear(prev_dim, output_features)
#
#         # 只初始化网络第一层的权重和偏置项
#         for layer in self.hidden_layers:
#             nn.init.normal_(layer.weight, mean=0, std=0.01)
#             nn.init.constant_(layer.bias, 0)
#             break
#         # nn.init.normal_(self.output_layer.weight, mean=0, std=0.01)
#         # nn.init.constant_(self.output_layer.bias, 0)
#
#     def forward(self, x):
#         for layer in self.hidden_layers:
#             x = torch.relu(layer(x))
#         return self.output_layer(x)
#
#
# def train(model, input_data, targets, epochs, learning_rate):
#     mseloss = nn.MSELoss()  # 均方误差对象作为损失函数
#     sgd = optim.SGD(model.parameters(), lr=learning_rate)  # 用随机梯度下降对象作为优化函数。
#     for epoch in range(epochs * 100):
#         y_predict = model(input_data)  # 前向传播，自动调用Module的forward方法
#         loss = mseloss(y_predict, targets)  # 计算损失函数
#         loss.backward()  # 反向传播
#         sgd.step()  # 更新参数(weight和bias)
#         sgd.zero_grad()  # 清零梯度数据
#
#         writer.add_scalar('Loss/train', loss.item(), epoch)
#         np.set_printoptions(precision=2)
#         print(
#             f"epoch:{epoch}, loss:{loss:.6f}"
#         )
#
#
# writer = SummaryWriter(log_dir=log_dir)
# hidden_dims = [8, 16, 32, 64]
# net = Net(in_features, hidden_dims, out_features)  # 创建神经网络对象
#
# # 进行预测
# learning_rate = 0.1
# epochs = 50
# train(X_train, Y_train, epochs, learning_rate)
# writer.add_graph(net, X_train[0])
# writer.close()
# net.eval()
# with torch.no_grad():
#     predictions = net(X_test)
# pred = predictions.detach().numpy()
# # 计算评估指标
# evaluate_prediction(Y_test.detach().numpy(), pred)
