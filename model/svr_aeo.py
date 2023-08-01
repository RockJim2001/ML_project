#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：ML_project 
@Product_name ：PyCharm
@File ：svr_aeo.py
@Author ：RockJim
@Date ：2023/8/1 16:24
@Description ：SVR算法使用AEO优化算法来寻找最优参数
@Version ：1.0 
'''
import os
import time
from sklearn.svm import SVR
from config.log_config import log
from dataset.data_config import DATASET_ROOT_PATH, BASE_DATASET_NAME
from dataset.data_load import load_data, data_processing
from evaluate.evaluate import evaluate_prediction
from model.optimizer.aeo import aeo_algorithm
from tools.common import print_best_params

logger = log().getLogger("SVR算法使用AEO优化算法来寻找最优参数")


def svr_aeo():
    # 加载数据集
    x, y = load_data(os.path.join(os.getcwd(), DATASET_ROOT_PATH, BASE_DATASET_NAME))
    # 数据预处理
    x_train, x_test, y_train, y_test, scaler = data_processing(x, y)

    # 定义参数网格，更细化参数范围
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
        'gamma': ['scale', 'auto'] + [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
        'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
        'degree': [2, 3, 4, 5],
        'coef0': [-1, 0, 1],
        'shrinking': [True, False],
        'tol': [1e-3, 1e-4, 1e-5],
        'max_iter': [100, 500, 1000, 5000],
        'cache_size': [100, 200, 500, 1000],
        'verbose': [0, 1, 2]
    }
    logger.info("定义参数网格：\t\n{}".format(param_grid))
    logger.info("使用AEO算法搜索最优参数……")
    start_time = time.time()
    # 调用AEO算法
    best_params = aeo_algorithm(param_grid)
    # 使用最优超参数构建SVR模型
    kernel, C, epsilon, gamma, degree, coef0, shrinking, tol, max_iter, cache_size, verbose = best_params
    print_best_params(best_params)
    end_time = time.time()
    run_time = start_time - end_time
    logger.info("AEO算法执行完毕，耗时{}".format(run_time))
    # 构建SVR模型
    svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma, degree=degree, coef0=coef0, shrinking=shrinking, tol=tol, max_iter=max_iter, cache_size=cache_size, verbose=verbose)

    # 训练SVR模型
    logger.info("使用最优参数进行训练……")
    start_time_train = time.time()
    svr_model.fit(x_train, y_train)
    end_time_train = time.time()
    train_time = end_time_train - start_time_train
    logger.info("SVR模型训练完毕，耗时{}".format(train_time))
    logger.info("在测试集上评估最佳模型……")
    start_time_test = time.time()
    # 在测试集上评估最佳模型
    y_pred = svr_model.predict(x_test)
    end_time_test = time.time()
    test_time = end_time_test - start_time_test
    logger.info("SVR模型训练完毕，耗时{}".format(test_time))

    mae, mse, evs, medAE, msle, rae, rse, rmse, r2 = evaluate_prediction(y_test, y_pred)
