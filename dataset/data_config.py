#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：data_config.py
@Author ：RockJim
@Date ：2023/7/24 23:48
@Description ：数据集配置文件
@Version ：1.0
"""
import os.path

# ########数据配置#################
# 数据存储根目录
ROOT_PATH = r'D:\Code\PythonProject\ML_project'
# base-数据.csv
BASE_DATASET_NAME = 'base-数据.csv'
# 波士顿房价
BOSTON_HOUSING_DATA = 'boston_housing.data'

# 800组的数据
BASE_DATASET_NAME_800 = 'AllCombinedResults-0800组.csv'
# 900组的数据
BASE_DATASET_NAME_900 = 'AllCombinedResults-0900组.csv'

# 3000组的数据
BASE_DATASET_NAME_3000 = 'AllCombinedResults-3000组.csv'
# 5000组的数据
BASE_DATASET_NAME_5000 = 'AllCombinedResults-5000组.csv'

# ###########日志配置#############
# 指定日志存储路径
DATASET_NAME = BASE_DATASET_NAME    # 更换当前的数据集
folder = DATASET_NAME.split('.')[0]
log_dir = os.path.join(ROOT_PATH, 'logs', folder)
log_file_path = os.path.join(ROOT_PATH, 'logs', folder, 'test.log')
