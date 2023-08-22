#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：genetic_algorithm.py
@Author ：RockJim
@Date ：2023/8/21 20:56
@Description ：自定定义实现遗传算法
@Version ：1.0
"""
import math
import random
import numpy as np
import torch
from torch import nn, optim
from config.log_config import log
from model.ann.ann import Net

logger = log().getLogger(__name__)


def train(model, input_data, targets, epochs, learning_rate):
    logger.info("模型训练…………")
    mseloss = nn.MSELoss()  # 均方误差对象作为损失函数
    sgd = optim.SGD(model.parameters(), lr=learning_rate)  # 用随机梯度下降对象作为优化函数。
    for epoch in range(epochs * 100):
        y_predict = model(input_data)  # 前向传播，自动调用Module的forward方法
        loss = mseloss(y_predict, targets)  # 计算损失函数
        loss.backward()  # 反向传播
        sgd.step()  # 更新参数(weight和bias)
        sgd.zero_grad()  # 清零梯度数据
        logger.info(
            f"epoch:{epoch}, loss:{loss:.6f}"
        )
    logger.info("训练结束…………")


class GeneticAlogrithm:
    def __init__(self, population_size, num_generations, mutation_rate, x_train, y_train,
                 x_test, y_test, input_feature, hidden_layer_range, output_feature):
        """
            初始化
        :param population_size: 种群大小指的是每一代中包含的个体数量。较大的种群通常能够更好地探索搜索空间，但也会增加计算成本。
        :param num_generations:  最大迭代次数
        :param mutation_rate: 变异率表示在变异过程中每个基因发生变异的概率。变异率较低时，可能会陷入局部最优解；变异率过高时，可能会随机搜索而失去方向性。
        :param x_train: 训练集数据
        :param y_train: 训练集数据
        :return:
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.input_feature = input_feature
        self.hidden_layer_range = hidden_layer_range
        self.output_feature = output_feature

    # 定义种群初始化函数
    def initialize_population(self, pop_size):
        population = []
        min_hidden_dim, max_hidden_dim = self.hidden_layer_range

        for _ in range(pop_size):
            hidden_dims = []
            num_hidden_layers = np.random.randint(1, 6)  # 随机生成 1 到 5 个隐藏层
            for _ in range(num_hidden_layers):
                hidden_dim = 2 ** np.random.randint(*[math.log2(min_hidden_dim),
                                                      math.log2(max_hidden_dim)])  # 随机生成以2为底的维度
                hidden_dims.append(hidden_dim)
            population.append(hidden_dims)
        return population

    # 定义交叉函数
    def crossover(self, parent1, parent2):
        if len(parent1) == 1 or len(parent2) == 1:
            # If either parent has length 1, return a copy of the longer parent as child
            if len(parent1) >= len(parent2):
                return parent1[:], parent1[:]
            else:
                return parent2[:], parent2[:]

        crossover_point = np.random.randint(1, min(len(parent1), len(parent2)))

        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    # 定义变异函数
    def mutate(self, child, mutation_rate):
        for i in range(len(child)):
            if random.random() < mutation_rate:
                min_hidden_dim, max_hidden_dim = self.hidden_layer_range
                hidden_dim = 2 ** np.random.randint(*[math.log2(min_hidden_dim),
                                                      math.log2(max_hidden_dim)])  # 随机生成以2为底的维度
                child[i] = hidden_dim
        return child

    # 计算适应度函数
    @staticmethod
    def compute_fitness(model, x_test, y_test):
        logger.info("计算适应度")
        criterion = nn.MSELoss()
        total_loss = 0.0
        with torch.no_grad():
            output = model(x_test)
            loss = criterion(output, y_test)
            total_loss += loss.item()
        logger.info("total_loss:{}".format(total_loss))
        return 1.0 / (total_loss + 1e-10)  # 防止除零

    def forward(self):
        # 初始化种群
        population = self.initialize_population(self.population_size)
        # 遗传算法优化循环
        for generation in range(self.num_generations):
            # 计算适应度
            fitness_scores = []
            for hidden_dims in population:
                # 构建模型
                model = Net(self.input_feature, hidden_dims, self.output_feature)
                logger.info("当前的模型结构为：\t\n{}".format(model))
                # 进行训练
                train(model, self.x_train, self.y_train, epochs=30, learning_rate=0.1)
                fitness = self.compute_fitness(model, self.x_test, self.y_test)
                fitness_scores.append(fitness)

            # 选择
            selected_indices = np.argsort(fitness_scores)[-self.population_size // 2:]
            selected_population = [population[i] for i in selected_indices]

            # 交叉和变异
            new_population = []
            for _ in range(self.population_size // 2):
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, self.mutation_rate)
                child2 = self.mutate(child2, self.mutation_rate)
                new_population.extend([child1, child2])

            population = new_population
        return population[np.argmax(fitness_scores)]
