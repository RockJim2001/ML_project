#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：ann_aeo.py
@Author ：RockJim
@Date ：2023/8/4 10:22
@Description ：使用AEO优化算法来查找ANN的最优参数
@Version ：1.0
"""
import os

import numpy as np
import torch
from torch import optim

from config.log_config import log
from dataset.data_config import DATASET_ROOT_PATH, BASE_DATASET_NAME
from dataset.data_load import load_data, data_processing
from evaluate.evaluate import evaluate_prediction
from model.ann.ann import ANNModel
from model.loss.loss import custom_loss
from model.optimizer.aeo import aeo_algorithm_original

logger = log().getLogger(__name__)

# 检查是否有可用的GPU，如果有则使用第一个GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 根据参数范围生成初始参数向量
def generate_initial_params(param_ranges_):
    params = {}
    for param_name, (min_val, max_val) in param_ranges_.items():
        if param_name == 'activation':
            params[param_name] = np.random.choice(max_val)
        elif isinstance(min_val, int) and isinstance(max_val, int):
            params[param_name] = np.random.randint(min_val, max_val + 1)
        elif isinstance(min_val, float) and isinstance(max_val, float):
            params[param_name] = np.random.uniform(min_val, max_val)
        elif isinstance(min_val, list):
            params[param_name] = [np.random.randint(min_val[i], max_val[i] + 1) for i in range(len(min_val))]
        elif isinstance(min_val, str):
            params[param_name] = np.random.choice(max_val)
    return params


# 将参数向量映射回参数值
def map_params_back(params, param_ranges_):
    mapped_params = {}
    for param_name, (min_val, max_val) in param_ranges_.items():
        if isinstance(min_val, int) and isinstance(max_val, int):
            mapped_params[param_name] = int(params[param_name])
        elif isinstance(min_val, float) and isinstance(max_val, float):
            mapped_params[param_name] = float(params[param_name])
        elif isinstance(min_val, list):
            mapped_params[param_name] = [int(val) for val in params[param_name]]
        elif isinstance(min_val, str):
            mapped_params[param_name] = params[param_name]
    return mapped_params


class ANN_AEO:
    def __init__(self):
        logger.info(
            "初始化ANN（Artificial Neural Network）+AEO（Artificial Ecosystem-Based Optimization）模型，并加载训练数据集和测试集数据")
        # 加载数据集
        x, y = load_data(os.path.join(os.getcwd(), DATASET_ROOT_PATH, BASE_DATASET_NAME))
        # 数据预处理
        x_train, x_test, y_train, y_test, scaler = data_processing(x, y)
        # 定义数据部分
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        # 输入输出维度部分
        self.input_dim = self.x_train.shape[1]
        self.output_dim = self.y_train.shape[1]
        # self.output_dim = 1
        logger.info("输入维度为：\t{}，输出维度为：\t{}".format(self.input_dim, self.output_dim))

        # 参数的可选范围和格式定义
        self.param_ranges = {
            'learning_rate': (0.001, 0.1),  # 学习率范围
            'batch_size': (8, 128),  # 批大小范围
            'num_hidden_layers': (1, 6),  # 隐藏层数范围
            'activation': ['relu', 'tanh'],  # 使用整数编码  激活函数选项
            'hidden_dims': (8, 64),  # 隐藏层维度范围，每层的最小和最大维度
        }
        logger.info("参数的可选范围和格式定义：\t{}".format(self.param_ranges))

    def objective_function(self, params, x, y):
        print("当前参数为：\t{}".format(params))
        logger.info("当前参数为：\t{}".format(params))
        learning_rate = params[0]  # 'learning_rate'
        batch_size = params[1]  # 'batch_size'
        num_hidden_layers = params[2]  # 转换为整数 'num_hidden_layers'
        activation = self.param_ranges['activation'][params[3]]
        # hidden_dims = [int(np.interp(params[i], [0, 1], self.param_ranges[f'hidden_dims_{i - 4}'])) for i in
        #                range(4, num_hidden_layers + 4)]
        hidden_dims = params[4]
        model = ANNModel(self.input_dim, hidden_dims, self.output_dim, activation).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        for epoch in range(1000):
            optimizer.zero_grad()
            inputs = torch.tensor(x, dtype=torch.float32, device=device)
            targets = torch.tensor(y, dtype=torch.float32, device=device)
            outputs = model(inputs)
            # loss = custom_loss(outputs.detach(), targets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # 打印日志输出
            logger.info(f'Epoch [{epoch + 1}/1000], Loss: {loss.item()}')
            print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item()}')

        return loss.item()

    def forward(self):
        # 定义AEO算法参数
        num_generations = 50
        competition_factor = 0.5
        logger.info("AEO算法迭代\t{}\t代，竞争因子为：\t{}".format(num_generations, competition_factor))
        # 使用AEO算法来优化神经网络超参数
        logger.info("使用AEO算法来优化神经网络超参数")
        best_params = aeo_algorithm_original(self.param_ranges, num_generations, competition_factor,
                                             lambda params: self.objective_function(params, self.x_train, self.y_train))

        # 解析最优超参数并训练神经网络
        logger.info("解析最优超参数:")
        best_learning_rate, best_batch_size, best_num_hidden_layers, best_activation_idx = best_params[:4]

        best_activation = self.param_ranges['activation'][best_activation_idx]
        # best_hidden_dims = best_params[4:best_num_hidden_layers + 4].astype(int)
        best_hidden_dims = best_params[4]
        logger.info("最优参数为：\t\nbest_learning_rate\tbest_batch_size\tbest_num_hidden_layers\tbest_activation\t"
                    "best_hidden_dims".format(
                                                best_learning_rate, best_batch_size, best_num_hidden_layers,
                                                best_activation, best_hidden_dims
                                              ))

        best_model = ANNModel(self.input_dim, best_hidden_dims, self.output_dim, best_activation).to(device)
        criterion = torch.nn.MSELoss()
        best_optimizer = optim.SGD(best_model.parameters(), lr=best_learning_rate)

        for epoch in range(1000):
            best_optimizer.zero_grad()
            inputs = torch.tensor(self.x_train, dtype=torch.float32, device=device)
            targets = torch.tensor(self.y_train, dtype=torch.float32, device=device)
            outputs = best_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            best_optimizer.step()
            # 打印日志输出
            logger.info(f'Epoch [{epoch + 1}/1000], Loss: {loss.item()}')
            print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item()}')

        # 使用训练好的模型进行预测
        logger.info("使用训练好的模型进行预测:")
        test_inputs = torch.tensor(self.x_test, dtype=torch.float32, device=device)
        predictions = best_model(test_inputs).cpu().numpy()
        print("最优学习率：", best_learning_rate)
        print("最优批大小：", best_batch_size)
        print("最优隐藏层数：", best_num_hidden_layers)
        print("最优激活函数：", best_activation)
        print("最优隐藏层维度：", best_hidden_dims)
        print("模型预测结果：", predictions)
        evaluate_prediction(self.x_test, predictions)
