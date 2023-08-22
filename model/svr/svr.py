#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：ML_project 
@Product_name ：PyCharm
@File ：svr.py
@Author ：RockJim
@Date ：2023/8/1 20:19
@Description ：最基本的SVR模型
@Version ：1.0 
'''
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


class custom_SVR:
    def __init__(self, params):
        self.kernel = params['kernel']
        self.C = params['C']
        self.gamma = params['gamma'],
        self.epsilon = params['epsilon']
        self.degree = params['degree']
        self.coef0 = params['coef0'],
        self.shrinking = params['shrinking']
        self.tol = params['tol']
        self.max_iter = params['max_iter'],
        self.cache_size = params['cache_size']
        self.verbose = params['verbose']
        self.svr_model = None

    def fit(self, x_train, y_train):
        self.svr_model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon,
                             gamma=self.gamma, degree=self.degree, coef0=self.coef0,
                             shrinking=self.shrinking, tol=self.tol, max_iter=self.max_iter,
                             cache_size=self.cache_size, verbose=self.verbose)
        self.svr_model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.svr_model.predict(x_test)

    def evaluate(self, x_test, y_test):
        y_pred = self.svr_model.predict(x_test)
        return mean_squared_error(y_test, y_pred)
