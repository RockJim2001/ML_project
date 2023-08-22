#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：main.py
@Author ：RockJim
@Date ：2023/8/21 22:14
@Description ：测试的主函数
@Version ：1.0
"""
import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from config.log_config import log
from dataset.data_config import DATASET_NAME, ROOT_PATH
from dataset.data_load import load_data, data_processing
from evaluate.evaluate import evaluate_prediction
from model.ann.ann import Net
from model.ann.genetic_algorithm import GeneticAlogrithm
from dataset.data_config import log_dir
logger = log().getLogger(__name__)


def data_load():
    # 加载数据集
    x_data, y_data = load_data(os.path.join(ROOT_PATH, 'resource', DATASET_NAME))
    # 数据归一化处理
    x_train, x_test, y_train, y_test, scaler = data_processing(x_data, y_data)

    # 将这些值转为tensor格式
    X_train = torch.tensor(x_train, dtype=torch.float32)
    Y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(x_test, dtype=torch.float32)
    Y_test = torch.tensor(y_test, dtype=torch.float32)
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    input_features = 12  # 单个输入样本的变量个数。
    output_features = 6  # 单个输出的变量个数。
    hidden_layer_range = [2, 128]
    learning_rate = 0.1
    epochs = 20
    population_size = 20
    num_generations = 1
    mutation_rate = 0.1
    # 加载数据
    X_train, Y_train, X_test, Y_test = data_load()

    ga = GeneticAlogrithm(population_size, num_generations, mutation_rate, X_train, Y_train,
                          X_test, Y_test, input_features, hidden_layer_range, output_features)
    # 选择最佳模型结构
    best_hidden_dims = ga.forward()
    logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    logger.info("最佳隐藏层参数为：{}".format(best_hidden_dims))
    writer = SummaryWriter(log_dir=log_dir)
    # 创建模型
    best_model = Net(input_features, best_hidden_dims, output_features)
    logger.info("最佳模型结构如下：\t\n{}".format(best_model))

    mseloss = nn.MSELoss()  # 均方误差对象作为损失函数
    sgd = optim.SGD(best_model.parameters(), lr=learning_rate)  # 用随机梯度下降对象作为优化函数。
    for epoch in range(epochs * 100):
        y_predict = best_model(X_train)  # 前向传播，自动调用Module的forward方法
        loss = mseloss(y_predict, Y_train)  # 计算损失函数
        loss.backward()  # 反向传播
        sgd.step()  # 更新参数(weight和bias)
        sgd.zero_grad()  # 清零梯度数据
        logger.info(
            f"epoch:{epoch}, loss:{loss:.6f}"
        )
        writer.add_scalar('Loss/train', loss.item(), epoch)

    writer.add_graph(best_model, X_train[0])
    writer.close()
    logger.info("在测试集上进行评估………………")
    best_model.eval()
    with torch.no_grad():
        predictions = best_model(X_test)
    pred = predictions.detach().numpy()
    # 计算评估指标
    evaluate_prediction(Y_test.detach().numpy(), pred)
