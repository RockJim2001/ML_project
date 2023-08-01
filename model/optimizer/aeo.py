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
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import mean_squared_error


# 自编码器模型
def autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder_model = Model(input_layer, decoded)
    encoder_model = Model(input_layer, encoded)
    return autoencoder_model, encoder_model


def aeo_algorithm(param_grid):
    """
        AEO 算法实现
    :param param_grid:
    :return:
    """
    best_params = None
    best_mse = float('inf')

    # 自编码器参数
    input_dim = len(param_grid)
    encoding_dim = 10  # 自编码器的编码维度

    # 构建自编码器模型
    autoencoder_model, encoder_model = autoencoder(input_dim, encoding_dim)

    # 遍历所有参数组合
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        param_vector = np.array(list(param_dict.values()))

        # 使用自编码器编码和解码超参数向量
        encoded_param = encoder_model.predict(param_vector.reshape(1, -1))
        decoded_param = autoencoder_model.predict(param_vector.reshape(1, -1))

        # 计算重构误差
        mse = mean_squared_error(param_vector, decoded_param[0])

        # 更新最优参数和最小重构误差
        if mse < best_mse:
            best_params = param_dict
            best_mse = mse

    return best_params

