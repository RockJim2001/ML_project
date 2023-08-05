#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：ML_project 
@Product_name ：PyCharm
@File ：common.py
@Author ：RockJim
@Date ：2023/8/1 16:27
@Description ：公共工具函数
@Version ：1.0 
'''
import numpy as np

from config.log_config import log

logger = log().getLogger("工具类")


def print_best_params(best_params):
    logger.info("最优超参数为：\t\n"
                "kernel\tC\tepsilon\tgamma\tdegree\tcoef0\tshrinking\ttol\tmax_iter\tcache_size\tverbose\t\n"
                "{}\t{}\t{}\t{}\t{}\t\t{}\t{}\t{}\t{}\t{}\t{}\t\n".format(best_params['kernel'], best_params['C'],
                                                                          best_params['epsilon'], best_params['gamma'],
                                                                          best_params['degree'], best_params['coef0'],
                                                                          best_params['shrinking'], best_params['tol'],
                                                                          best_params['max_iter'],
                                                                          best_params['cache_size'],
                                                                          best_params['verbose']))


def print_params(param):
    logger.info("当前参数为：\t\n"
                "kernel\tC\tgamma\tepsilon\tdegree\tcoef0\tshrinking\ttol\tmax_iter\tcache_size\tverbose\t\n"
                "{}\t{}\t{}\t{}\t{}\t\t\t{}\t{}\t{}\t{}\t{}\t{}\t\n".format(param[0], param[1], param[2], param[3],
                                                                            param[4], param[5], param[6], param[7],
                                                                            param[8], param[9], param[10]))


def generate_learning_rate(value_range):
    start, stop = value_range
    # 生成以1/1000为底的对数递增的学习率
    values = []
    current_val = start
    while current_val <= stop:
        values.append(current_val)
        current_val *= 10
    return np.random.choice(values)


def generate_power_of_two(value_range):
    start, stop = value_range
    powers = np.arange(np.log2(start), np.log2(stop) + 1)
    return int(2 ** np.random.choice(powers))


def generate_initial_population(param_ranges, population_size):
    population = []
    for _ in range(population_size):
        individual = {}
        for param, value_range in param_ranges.items():
            if param == 'learning_rate':
                value = generate_learning_rate(value_range)
            elif param == 'activation':
                value = np.random.randint(0, len(value_range) - 1)  # 随机选择一个下标
            elif param == 'num_hidden_layers':
                min_value, max_value = value_range
                value = np.random.randint(min_value, max_value + 1)
            elif param == 'batch_size':
                value = generate_power_of_two(value_range)
            elif param.startswith('hidden_dims'):
                value_list = [generate_power_of_two(value_range) for _ in range(individual['num_hidden_layers'])]
                value = np.array(value_list)
            else:
                raise ValueError("Unsupported parameter: {}".format(param))
            individual[param] = value
        temp = [individual['learning_rate'], individual['batch_size'], individual['num_hidden_layers'],
                individual['activation'], individual['hidden_dims']]
        population.append(temp)
    return population
