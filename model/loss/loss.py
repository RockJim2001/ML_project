#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：loss.py
@Author ：RockJim
@Date ：2023/8/5 0:24
@Description ：定义一些基本的损失函数
@Version ：1.0
"""
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.log_config import log
from evaluate.evaluate import root_mean_squared_error

logger = log().getLogger(__name__)

# 定义权重
weights = {
    'mae': 1.0,
    'mse': 2.0,
    'rmse': 1.5,
    'r2': 0.5
}


# 定义各指标的损失计算函数（以 MAE 为例）
def mae_loss(y_pred, y_true):   # 越接近0 越好
    return mean_absolute_error(y_true, y_pred)


def mse_loss(y_pred, y_true):
    return mean_squared_error(y_true, y_pred)


def rmse_loss(y_pred, y_true):
    return root_mean_squared_error(y_true, y_pred)


def r2_loss(y_pred, y_true):
    y_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return 1 - r2


def custom_loss(y_pred, y_true):
    # mse = torch.mean(torch.abs(y_pred - y_true))
    # logger.info("mse损失为：\t{}".format(mse))
    # print("mse损失为：\t{}".format(mse))
    # 计算各指标的损失
    mae = mae_loss(y_pred, y_true)
    mse = mse_loss(y_pred, y_true)
    rmse = rmse_loss(y_pred, y_true)
    r2 = r2_loss(y_pred, y_true)
    logger.info("各项损失如下：\t\nmae\tmse\trmse\tr2\t\n{}\t{}\t{}\t{}".format(mae, mse, rmse, r2))
    # 整合为统一的损失函数
    unified_loss = weights['mae'] * mae + weights['mse'] * mse + weights['rmse'] * rmse + weights['r2'] * r2
    return unified_loss
