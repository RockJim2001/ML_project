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
