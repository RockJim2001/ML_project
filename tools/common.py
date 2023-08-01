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
from config.log_config import log

logger = log().getLogger("工具类")


def print_best_params(kernel, C, epsilon, gamma, degree, coef0, shrinking, tol, max_iter, cache_size, verbose):
    logger.info("最优超参数为：\t\n"
                "kernel\tC\tepsilon\tgamma\tdegree\tcoef0\tshrinking\ttol\tmax_iter\tcache_size\tverbose\t\n"
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n".format(kernel, C, epsilon, gamma, degree, coef0,
                                                                        shrinking, tol, max_iter, cache_size, verbose))

