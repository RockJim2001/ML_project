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
from tools.common import print_params

logger = log().getLogger("SVR算法使用AEO优化算法来寻找最优参数")

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def encode_categorical_param(category_list, selected_category):
    encoded_param = np.zeros(len(category_list))
    encoded_param[category_list.index(selected_category)] = 1
    return encoded_param


def aeo_algorithm(param_grid):
    """
        AEO 算法实现
    :param param_grid:
    :return:
    """
    best_params = None
    best_mse = float('inf')

    # 自编码器参数
    input_dim = len(param_grid) + len(param_grid['kernel']) + len(param_grid['gamma']) - 2
    encoding_dim = 11  # 自编码器的编码维度

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
