#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：analyse.py
@Author ：RockJim
@Date ：2024/5/11 21:00
@Description ：做敏感度分析
@Version ：1.0
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
from sklearn.metrics import mean_squared_error

# 加载数据
df = pd.read_csv(r'E:\PythonProject\machine_learning\ML_project-Data_4\ML_project\resource\analyse\7层-AllCombinedResults7层楼500组-20240510.csv')

# 选择输入和输出列
X = df[[f'Z{i}' for i in range(1, 31)]]  # 输入变量 Z1 to Z30
y = df['Y1']  # 输出变量 Y1

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# 模型训练
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train_scaled.ravel())

# 模型评估
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
mse = mean_squared_error(y_test, y_pred)
print(f'Model Mean Squared Error: {mse}')

# 定义Morris问题
problem = {
    'num_vars': 30,
    'names': [f'Z{i}' for i in range(1, 31)],
    'bounds': [[-1, 1]] * 30,
    'groups': None
}

# 生成Morris样本
morris_samples = morris_sample.sample(problem, N=100, num_levels=4, optimal_trajectories=None)

# 应用标准化
morris_samples_scaled = scaler_X.transform(morris_samples)

# 使用模型进行预测
y_pred_morris_scaled = model.predict(morris_samples_scaled)
y_pred_morris = scaler_y.inverse_transform(y_pred_morris_scaled.reshape(-1, 1)).flatten()

# Morris敏感性分析
morris_results = morris_analyze.analyze(problem, morris_samples, y_pred_morris, conf_level=0.95, print_to_console=True,
                                        num_resamples=1000)

# 输出敏感性分析结果
print("Morris Analysis Results:")
print(morris_results)
