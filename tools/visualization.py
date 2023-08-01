#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：ML_project 
@Product_name ：PyCharm
@File ：visualization.py
@Author ：RockJim
@Date ：2023/7/26 19:35
@Description ：可视化工具
@Version ：1.0 
'''
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


def visual_prediction(y_train, y, y_prediction, scaler: StandardScaler):
    """
        可视化预测结果（svr模型）
    :param scaler:
    :param y_train: y1 组成的训练集真值
    :param y: y1 组成的全部的真值
    :param y_prediction: 测试集x_test上的预测结果
    :return:
    """
    y_test = np.array(y[-200:]).reshape(-1, 1)
    y_test = scaler.fit_transform(y_test).ravel()
    y_train_original = np.concatenate((y_train.reshape(-1, 1), y_test.reshape(-1, 1)))
    y_train_original_temp = scaler.inverse_transform(y_train_original.reshape(-1, 1))
    temp = np.concatenate((y_train.reshape(-1, 1), y_prediction.reshape(-1, 1)), axis=0)
    y_pred_original = scaler.inverse_transform(temp.reshape(-1, 1))

    # 绘制原始数据和预测结果
    # 绘制折线图
    length = list(range(len(y)))
    # 预测值

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(20, 4), dpi=600)
    plt.plot(length, y_train_original_temp, ls='-.', lw=2, color='r', label='真实值')
    plt.plot(length, y_pred_original, ls='-', lw=2, color='b', label='预测值')
    plt.axvline(x=800, color='g', linestyle='--', linewidth=1)
    # 添加一个外框（矩形）
    # rect = patches.Rectangle((280, 83), 0.4, 0.7, linewidth=1, edgecolor='red', facecolor='none')
    # plt.gca().add_patch(rect)
    # 添加左侧标签
    plt.text(350, 117, 'train', rotation=0, color=(0, 0, 0))
    # 添加右侧标签
    plt.text(850, 117, 'test', rotation=0, color=(0, 0, 0))
    # 绘制网格
    plt.grid(alpha=0.4, linestyle=':')
    plt.legend()
    plt.xlabel('样本')  # 设置x轴的标签文本
    plt.ylabel('Cooling(MWh)')  # 设置y轴的标签文本

    # 展示
    plt.show()
    # plt.savefig('result.png')
