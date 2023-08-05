#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：ann.py
@Author ：RockJim
@Date ：2023/8/4 10:14
@Description ：自己定义ANN（人工神经网络模型）
@Version ：1.0
"""
from torch import nn
from config.log_config import log

logger = log().getLogger(__name__)


class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation):
        """
            初始化，构建模型
        :param input_dim: 输入层的维度
        :param hidden_dims:  list类型，表示隐藏层的层数以及各层的维度
        :param output_dim: 输出层的维度
        :param activation:
        """
        logger.info("构建ANN模型……")
        super(ANNModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
        logger.info("模型结构如下：\n%s", self.model)

    def forward(self, x):
        return self.model(x)
