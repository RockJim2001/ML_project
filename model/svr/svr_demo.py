#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：ML_project 
@Product_name ：PyCharm
@File ：svr_demo.py
@Author ：RockJim
@Date ：2023/7/24 23:45
@Description ：一个demo，使用SVR支持向量机做回归预测（房价）
@Version ：1.0 
'''

from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


def notEmpty(s):
    return s != ''


# 加载数据集
# boston = load_boston()
# x = boston.data
# y = boston.target
housing = fetch_california_housing()
x = housing["data"]
y = housing["target"]



# 拆分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
# 预处理
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)
y_train = StandardScaler().fit_transform(y_train).ravel()
y_test = StandardScaler().fit_transform(y_test).ravel()

# 创建svR实例
svr = SVR(C=1, kernel='rbf', epsilon=0.2)
svr = svr.fit(x_train, y_train)
# 预测
svr_predict = svr.predict(x_test)
# 评价结果
mae = mean_absolute_error(y_test, svr_predict)
mse = mean_squared_error(y_test, svr_predict)
evs = explained_variance_score(y_test, svr_predict)
r2 = r2_score(y_test, svr_predict)
print("MAE：", mae)
print("MSE：", mse)
print("EVS：", evs)
print("R2：", r2)
