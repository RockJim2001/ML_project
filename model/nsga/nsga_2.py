#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：nsga_2.py
@Author ：RockJim
@Date ：2023/8/24 20:44
@Description ：使用nsga-2来实现多目标的优化问题
@Version ：1.0
"""
import joblib
import numpy as np
import torch
from pymoo.core.problem import Problem
from sklearn.preprocessing import StandardScaler

from config.config import log_dir
from model.ann.ann import Net

"""
    继承pymoo中的Problem类来实现自己的多目标优化问题
"""


class MultiObjectiveProblem(Problem):
    def __init__(self, model, input_feature, output_feature, constr):
        """
            初始化
        :param model: 经过筛选，最佳模型结构的网络，并且已经通过训练集训练
        :param input_feature:   12
        :param output_feature:  6
        :param constr: 一个list【dict】类型的约束，其中每个dict包含xl和xu两个key
        :return:
        """
        if len(constr) == 0:
            n_constr_ = 0
            xl_ = None
            xu_ = None
        else:
            n_constr_ = len(constr)
            xl_ = constr[0]
            xu_ = constr[1]
        # super().__init__(n_var=input_feature, n_obj=output_feature, n_constr=n_constr, xl=xl,
        #                  xu=xu)

        temp_xl = np.array([0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0])
        temp_xu = np.array([1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1])
        super().__init__(
            n_var=input_feature,  # 输入变量的数量
            n_obj=output_feature,  # 目标函数的数量
            n_constr=0,  # 约束条件的数量
            xl=xl_,  # 输入变量的下界
            xu=xu_  # 输入变量的上界
        )
        self.model = model

    def _evaluate(self, x, out, *args, **kwargs):
        """
        :param x: 种群
        :param out: 每个种群的输出
        :param args:
        :param kwargs:
        :return:
        """
        temp_x = joblib.load(log_dir + '/scaler_x.pkl')
        # 先对数据进行标准化处理
        x_test = temp_x.fit_transform(x)
        # 转换成tensor格式
        x_temp = torch.tensor(x_test, dtype=torch.float32)
        # 进行预测
        pred = self.model(x_temp).detach().numpy()

        # 反归一化
        temp_y = joblib.load(log_dir + '/scaler_y.pkl')
        predictions = temp_y.inverse_transform(pred)

        # outputs = np.array(outputs)
        temp = np.column_stack([predictions[:, 0], predictions[:, 1], predictions[:, 2],
                                predictions[:, 3], predictions[:, 4], predictions[:, 5]])  # 每个个体的目标函数值是一个包含六个输出值的向量
        out["F"] = temp
