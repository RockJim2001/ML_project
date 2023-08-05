#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：ML_project 
@Product_name ：PyCharm
@File ：aeo.py
@Author ：RockJim
@Date ：2023/8/1 16:31
@Description ：基于人工生态系统的优化算法（AEO）
@Version ：1.0 
'''
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch import optim
from config.log_config import log
from model.optimizer.Autoencoder import Autoencoder
from tools.common import generate_initial_population, print_params

logger = log().getLogger("SVR算法使用AEO优化算法来寻找最优参数")

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_categorical_param(category_list, selected_category):
    encoded_param = np.zeros(len(category_list))
    encoded_param[category_list.index(selected_category)] = 1
    return encoded_param


def aeo_algorithm_svr(param_grid):
    """
        适用于SVR的AEO 算法实现
    :param param_grid:  参数列表
    :return:
    """
    best_params = None
    best_mse = float('inf')

    # 自编码器参数
    input_dim = len(param_grid) + len(param_grid['kernel']) + len(param_grid['gamma']) - 2
    encoding_dim = 10  # 自编码器的编码维度

    print(device)
    # 构建自编码器模型
    autoencoder_model = Autoencoder(input_dim, encoding_dim).to(device)
    optimizer = optim.Adam(autoencoder_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 遍历所有参数组合
    for params in product(*param_grid.values()):
        print_params(params)
        print("当前参数为：\t\n"
              "kernel\tC\tgamma\tepsilon\tdegree\tcoef0\tshrinking\ttol\tmax_iter\tcache_size\tverbose\t\n"
              "{}\t{}\t{}\t{}\t{}\t\t{}\t{}\t{}\t{}\t{}\t{}\t\n".format(params[0], params[1], params[2], params[3],
                                                                        params[4], params[5], params[6], params[7],
                                                                        params[8], params[9], params[10]))
        param_dict = dict(zip(param_grid.keys(), params))

        # 使用独热编码处理离散特征
        kernel_encoded = encode_categorical_param(param_grid['kernel'], param_dict['kernel'])
        gamma_encoded = encode_categorical_param(param_grid['gamma'], param_dict['gamma'])
        continuous_params = [param_dict['C'], param_dict['epsilon'], param_dict['degree'], param_dict['coef0'],
                             param_dict['shrinking'], param_dict['tol'], param_dict['max_iter'],
                             param_dict['cache_size'], param_dict['verbose']]
        param_vector = np.concatenate((kernel_encoded, gamma_encoded, continuous_params), axis=None)

        # 转换为 PyTorch 张量
        param_tensor = torch.tensor(param_vector, dtype=torch.float32, device=device)

        # 使用自编码器编码和解码超参数向量
        autoencoder_model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            decoded_param = autoencoder_model(param_tensor)
            loss = criterion(decoded_param, param_tensor)
            loss.backward()
            optimizer.step()

        # 计算重构误差
        autoencoder_model.eval()
        with torch.no_grad():
            decoded_param = autoencoder_model(param_tensor)
            mse = mean_squared_error(param_tensor.numpy(), decoded_param.numpy())
        logger.info("计算得到的mse为\t{}".format(mse))
        print("计算得到的mse为\t{}".format(mse))
        # 更新最优参数和最小重构误差
        if mse < best_mse:
            best_params = param_dict
            best_mse = mse

    return best_params


# 定义AEO算法
def aeo_algorithm_original(param_ranges, num_generations, competition_factor, objective_function):
    """
        原始的AEO(Artificial Ecosystem-Based Optimization)算法实现
    :param param_ranges:   参数集
    :param num_generations: 更新的代数
    :param competition_factor:
    :param objective_function:  目标优化函数（损失函数）
    :return:  最优参数
    """
    logger.info("原始的AEO(Artificial Ecosystem-Based Optimization)算法实现……")
    param_dim = len(param_ranges)  # 学习率、批大小、隐藏层数、激活函数、隐藏层各层的维度
    population_size = 10
    # 初始化种群
    population = generate_initial_population(param_ranges, population_size)

    for generation in range(num_generations):
        """
            Tips:cpu上运行时python=3.10，GPU在conda环境下的python=3.9下运行会报错:
           " ValueError: setting an array element with a sequence. The requested array has an inhomogeneous
            shape after 2 dimensions. The detected shape was (10, 5) + inhomogeneous part."
        """
        fitness_scores = np.apply_along_axis(objective_function, 1, population)
        sort_indices = np.argsort(fitness_scores)
        sorted_population = np.take(population, sort_indices, axis=0)

        competition_strength = np.exp(-competition_factor * np.arange(population_size))

        for i in range(population_size):
            competing_indices = np.roll(np.arange(population_size), shift=i + 1)
            winners = sorted_population[competing_indices[:2]]
            # 对于学习率、批大小、隐藏层数、激活函数，进行线性平均更新

            # 更新learning_rate, 从范围内随机选择一个
            # 更新学习率，取最接近的0.001的10倍数，并限制在范围内
            min_learning_rate, max_learning_rate = param_ranges['learning_rate']
            new_learning_rate = np.clip(np.round(
                competition_strength[i] * np.mean(winners[:, 0]) + (1 - competition_strength[i]) * population[i][0],
                decimals=3),
                min_learning_rate, max_learning_rate)
            # new_learning_rate = max(min_learning_rate, min(max_learning_rate, new_learning_rate))

            # 更新batch_size，取整并限制在指定范围内
            min_batch_size, max_batch_size = param_ranges['batch_size']
            new_batch_size = np.clip(np.round(
                competition_strength[i] * np.mean(winners[:, 1]) + (1 - competition_strength[i]) * population[i][1]),
                min_batch_size, max_batch_size)

            # 更新num_hidden_layers，取整并限制在指定范围内
            min_num_hidden_layers, max_num_hidden_layers = param_ranges['num_hidden_layers']
            new_num_hidden_layers = int(np.clip(np.round(
                competition_strength[i] * np.mean(winners[:, 2]) + (1 - competition_strength[i]) * population[i][2]),
                min_num_hidden_layers, max_num_hidden_layers))

            # 更新activation，直接从范围内随机选择一个
            activation_options = param_ranges['activation']
            new_activation_idx = int(np.clip(np.round(
                competition_strength[i] * np.mean(winners[:, 3]) + (1 - competition_strength[i]) * population[i][3]),
                0, len(activation_options) - 1))

            # 对于隐藏层数和隐藏层维度，使用第一个竞争个体的参数
            if new_num_hidden_layers < winners[0][2]:
                num_hidden_layers = winners[0][2]
                hidden_dims = winners[0][4]
            else:
                num_hidden_layers = winners[1][2]
                hidden_dims = winners[1][4]

            population[i] = [new_learning_rate, new_batch_size, num_hidden_layers, new_activation_idx, hidden_dims]

    best_solution = population[np.argmin(fitness_scores)]
    return best_solution
