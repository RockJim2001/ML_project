#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：config.py
@Author ：RockJim
@Date ：2023/7/24 23:48
@Description ：数据集配置文件
@Version ：1.0
"""
import os.path
import numpy as np

# ########数据配置#################
# 数据存储根目录
ROOT_PATH = r'E:\PythonProject\machine_learning\ML_project-Data_4\ML_project'
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
# 模型1的数据
BASE_DATASET_NAME_MODEL_1 = 'table_4_10000_cols.csv'
# 模型2的数据
BASE_DATASET_NAME_MODEL_2 = '表2的10000组-发东东版本.csv'

# 模型4的300数据
BASE_DATASET_NAME_MODEL_3 = 'table_4_50000_300_cols.csv'
# 模型4的1000数据
BASE_DATASET_NAME_MODEL_4 = 'table_4_50000_1000_cols.csv'
# 模型4的4000数据
BASE_DATASET_NAME_MODEL_5 = 'table_4_50000_4000_cols.csv'
# 模型4的6000数据
BASE_DATASET_NAME_MODEL_6 = 'table_4_50000_6000_cols.csv'
# 模型4的9000数据
BASE_DATASET_NAME_MODEL_7 = 'table_4_50000_9000_cols.csv'
# 模型4的12000数据
BASE_DATASET_NAME_MODEL_8 = 'table_4_50000_12000_cols.csv'
# 模型4的21000数据
BASE_DATASET_NAME_MODEL_9 = 'table_4_50000_21000_cols.csv'
# 模型4的30000数据
BASE_DATASET_NAME_MODEL_10 = 'table_4_50000_30000_cols.csv'
# 模型4的40000数据
BASE_DATASET_NAME_MODEL_11 = 'table_4_50000_40000_cols.csv'
# 模型4的50000数据
BASE_DATASET_NAME_MODEL_12 = 'table_4_50000_50000_cols.csv'


# 模型4的10000条数据修改版
BASE_DATASET_NAME_MODEL_13 = 'table_4_10000_change.csv'

# 最新版的table_4的10000条数据
BASE_DATASET_NAME_MODEL_14 = 'table_4_10000_cols_new.csv'


# 模型4的300数据
BASE_DATASET_NAME_MODEL_15 = '008_6_floor_table_4_50000_cols_300_cols.csv'
# 模型4的1000数据
BASE_DATASET_NAME_MODEL_16 = '008_6_floor_table_4_50000_cols_1000_cols.csv'
# 模型4的4000数据
BASE_DATASET_NAME_MODEL_17 = '008_6_floor_table_4_50000_cols_4000_cols.csv'
# 模型4的6000数据
BASE_DATASET_NAME_MODEL_18 = '008_6_floor_table_4_50000_cols_6000_cols.csv'
# 模型4的9000数据
BASE_DATASET_NAME_MODEL_19 = '008_6_floor_table_4_50000_cols_9000_cols.csv'
# 模型4的12000数据
BASE_DATASET_NAME_MODEL_20 = '008_6_floor_table_4_50000_cols_12000_cols.csv'
# 模型4的21000数据
BASE_DATASET_NAME_MODEL_21 = '008_6_floor_table_4_50000_cols_21000_cols.csv'
# 模型4的30000数据
BASE_DATASET_NAME_MODEL_22 = '008_6_floor_table_4_50000_cols_30000_cols.csv'
# 模型4的40000数据
BASE_DATASET_NAME_MODEL_23 = '008_6_floor_table_4_50000_cols_40000_cols.csv'
# 模型4的50000数据
BASE_DATASET_NAME_MODEL_24 = '008_6_floor_table_4_50000_cols_50000_cols.csv'



# 最新版的table_4的10000条数据
BASE_DATASET_NAME_MODEL_25 = '008_7_floor_table_4_10000_cols.csv'
# 模型4的300数据
BASE_DATASET_NAME_MODEL_26 = '008_7_floor_table_4_50000_cols_300_cols.csv'
# 模型4的1000数据
BASE_DATASET_NAME_MODEL_27 = '008_7_floor_table_4_50000_cols_1000_cols.csv'
# 模型4的4000数据
BASE_DATASET_NAME_MODEL_28 = '008_7_floor_table_4_50000_cols_4000_cols.csv'
# 模型4的6000数据
BASE_DATASET_NAME_MODEL_29 = '008_7_floor_table_4_50000_cols_6000_cols.csv'
# 模型4的9000数据
BASE_DATASET_NAME_MODEL_30 = '008_7_floor_table_4_50000_cols_9000_cols.csv'
# 模型4的12000数据
BASE_DATASET_NAME_MODEL_31 = '008_7_floor_table_4_50000_cols_12000_cols.csv'
# 模型4的21000数据
BASE_DATASET_NAME_MODEL_32 = '008_7_floor_table_4_50000_cols_21000_cols.csv'
# 模型4的30000数据
BASE_DATASET_NAME_MODEL_33 = '008_7_floor_table_4_50000_cols_30000_cols.csv'
# 模型4的40000数据
BASE_DATASET_NAME_MODEL_34 = '008_7_floor_table_4_50000_cols_40000_cols.csv'
# 模型4的50000数据
BASE_DATASET_NAME_MODEL_35 = '008_7_floor_table_4_50000_cols_50000_cols.csv'



BASE_DATASET_NAME_MODEL_36 = '008_9_floor_table_4_10000_cols.csv'
# 模型4的300数据
BASE_DATASET_NAME_MODEL_37 = '008_9_floor_table_4_50000_cols_300_cols.csv'
# 模型4的1000数据
BASE_DATASET_NAME_MODEL_38 = '008_9_floor_table_4_50000_cols_1000_cols.csv'
# 模型4的4000数据
BASE_DATASET_NAME_MODEL_39 = '008_9_floor_table_4_50000_cols_4000_cols.csv'
# 模型4的6000数据
BASE_DATASET_NAME_MODEL_40 = '008_9_floor_table_4_50000_cols_6000_cols.csv'
# 模型4的9000数据
BASE_DATASET_NAME_MODEL_41 = '008_9_floor_table_4_50000_cols_9000_cols.csv'
# 模型4的12000数据
BASE_DATASET_NAME_MODEL_42 = '008_9_floor_table_4_50000_cols_12000_cols.csv'
# 模型4的21000数据
BASE_DATASET_NAME_MODEL_43 = '008_9_floor_table_4_50000_cols_21000_cols.csv'
# 模型4的30000数据
BASE_DATASET_NAME_MODEL_44 = '008_9_floor_table_4_50000_cols_30000_cols.csv'
# 模型4的40000数据
BASE_DATASET_NAME_MODEL_45 = '008_9_floor_table_4_50000_cols_40000_cols.csv'
# 模型4的50000数据
BASE_DATASET_NAME_MODEL_46 = '008_9_floor_table_4_50000_cols_50000_cols.csv'


BASE_DATASET_NAME_MODEL_47 = '008_18_floor_table_4_10000_cols.csv'
# 模型4的300数据
BASE_DATASET_NAME_MODEL_48 = '008_18_floor_table_4_50000_cols_300_cols.csv'
# 模型4的1000数据
BASE_DATASET_NAME_MODEL_49 = '008_18_floor_table_4_50000_cols_1000_cols.csv'
# 模型4的4000数据
BASE_DATASET_NAME_MODEL_50 = '008_18_floor_table_4_50000_cols_4000_cols.csv'
# 模型4的6000数据
BASE_DATASET_NAME_MODEL_51 = '008_18_floor_table_4_50000_cols_6000_cols.csv'
# 模型4的9000数据
BASE_DATASET_NAME_MODEL_52 = '008_18_floor_table_4_50000_cols_9000_cols.csv'
# 模型4的12000数据
BASE_DATASET_NAME_MODEL_53 = '008_18_floor_table_4_50000_cols_12000_cols.csv'
# 模型4的21000数据
BASE_DATASET_NAME_MODEL_54 = '008_18_floor_table_4_50000_cols_21000_cols.csv'
# 模型4的30000数据
BASE_DATASET_NAME_MODEL_55 = '008_18_floor_table_4_50000_cols_30000_cols.csv'
# 模型4的40000数据
BASE_DATASET_NAME_MODEL_56 = '008_18_floor_table_4_50000_cols_40000_cols.csv'
# 模型4的50000数据
BASE_DATASET_NAME_MODEL_57 = '008_18_floor_table_4_50000_cols_50000_cols.csv'


# ###########日志配置#############
# 指定日志存储路径
DATASET_NAME = BASE_DATASET_NAME_MODEL_57  # 更换当前的数据集
folder = DATASET_NAME.split('.')[0]
log_dir = os.path.join(ROOT_PATH, 'logs', folder)
log_file_path = os.path.join(ROOT_PATH, 'logs', folder, 'test.log')
parent_dir = '008_18_floor_table_4'

# ###########随机数种子设置#############

# 设置随机数种子
seed_value = 2023
np.random.seed(seed_value)
