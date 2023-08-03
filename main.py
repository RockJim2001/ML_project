#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：ML_project 
@Product_name ：PyCharm
@File ：main.py
@Author ：RockJim
@Date ：2023/8/1 15:36
@Description ：ML项目的主目录
@Version ：1.0 
'''
import time

from config.log_config import log
from model.svr_aeo import svr_aeo
from model.svr_grid_search import svr_grid_search

logger = log().getLogger("main函数")

if __name__ == '__main__':
    # 使用grid_search搜索的svr
    # svr_grid_search()
    # 使用aeo优化的svr
    start_time = time.time()
    svr_aeo()
    end_time = time.time()
    print(end_time-start_time)
