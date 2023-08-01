#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：ML_project 
@Product_name ：PyCharm
@File ：svr_grid_search.py
@Author ：RockJim
@Date ：2023/7/25 14:33
@Description ：使用SVR支持向量机做回归预测（材料能耗）
@Version ：1.0 
'''
import logging
import os
import time

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.svm import SVR

from dataset.data_config import DATASET_ROOT_PATH, BASE_DATASET_NAME
from dataset.data_load import load_data, data_processing
from evaluate.evaluate import evaluate_prediction
from tools.visualization import visual_prediction
from config.log_config import log

logger = log().getLogger("SVR 算法做回归预测，使用网格搜索算法进行最优参数的搜索")


def svr_grid_search():
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
    param_df = pd.DataFrame(param_grid)
    # 将DataFrame以表格形式输出
    param_table = param_df.to_string(index=False)
    logger.info("定义参数网格：\t\n{}".format(param_table))
    # 创建SVR模型
    logger.info("创建SVR模型")
    svr_model = SVR()
    logger.info("定义GridSearchCV对象，指定参数网格、交叉验证的折数和评估指标:\t\n"
                "GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=5, scoring=neg_mean_squared_error,"
                "n_jobs=-1)")
    # 定义GridSearchCV对象，指定参数网格、交叉验证的折数和评估指标
    grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                               n_jobs=-1)
    logger.info("使用grid_search算法进行最优参数的检索……")
    start_time = time.time()
    # 在训练集上执行网格化搜索和交叉验证
    grid_search.fit(x_train, y_train)
    # 保存GridSearchCV对象的状态到磁盘
    joblib.dump(grid_search, '../resource/grid_search_state.pkl')
    # 打印最佳参数组合
    logging.info("Best parameters:", grid_search.best_params_)

    # 得到最佳模型
    best_svr = grid_search.best_estimator_
    end_time = time.time()
    run_time = start_time - end_time
    logger.info("grid_search算法执行完毕，耗时{}".format(run_time))

    # 在测试集上评估最佳模型
    logger.info("在测试集上评估最佳模型……")
    y_pred = best_svr.predict(x_test)

    mae, mse, evs, medAE, msle, rae, rse, rmse, r2 = evaluate_prediction(y_test, y_pred)

# 创建svR实例
# svr = SVR(C=1, kernel='rbf', epsilon=0.2)

# # 创建k-fold交叉验证对象，k=5表示5折交叉验证
# k_fold = KFold(n_splits=2)
#
#
# mse_scores = cross_val_score(svr_model, x, y, scoring='neg_mean_squared_error', cv=k_fold)
#
#
# # 将均方误差的负值转换为正值，然后计算均方根误差（RMSE）
# rmse_scores = np.sqrt(-mse_scores)
#
# # 计算均方误差（MSE）的平均值和标准差
# avg_mse = np.mean(-mse_scores)
# std_mse = np.std(-mse_scores)
#
# # 计算均方根误差（RMSE）的平均值和标准差
# avg_rmse = np.mean(rmse_scores)
# std_rmse = np.std(rmse_scores)
#
# # 计算平均绝对误差（MAE）
# mae_scores = cross_val_score(svr_model, x, y, scoring='neg_mean_absolute_error', cv=k_fold)
# avg_mae = np.mean(-mae_scores)
#
# # 计算R²（决定系数）
# r2_scores = cross_val_score(svr_model, x, y, scoring='r2', cv=k_fold)
# avg_r2 = np.mean(r2_scores)
# std_r2 = np.std(r2_scores)
#
# # 将指标以表格形式打印出来
# metrics_table = pd.DataFrame({
#     'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
#     'Average_Score': [avg_mse, avg_rmse, avg_mae, avg_r2],
#     'Standard_Deviation': [std_mse, std_rmse, 'N/A', std_r2]
# })
#
# print(metrics_table)


# svr = svr.fit(x_train, y_train)
# # 预测
# svr_predict = svr.predict(x_test)
#
# # visual_prediction(y_train, y, svr_predict, scaler)
#
# # 评价结果
# mae, mse, evs, medAE, msle, rae, rse, rmse, r2 = evaluate_prediction(y_test, svr_predict)
