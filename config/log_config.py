#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：ML_project 
@Product_name ：PyCharm
@File ：log_config.py
@Author ：RockJim
@Date ：2023/8/1 15:47
@Description ：日志文件的配置
@Version ：1.0 
'''
import logging
import os.path

import colorlog
# 设置日志文件的保存路径和文件名
from dataset.data_config import log_file_path

# 确保目录存在
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
# 判断文件是否存在并进行创建
if not os.path.exists(log_file_path):
    with open(log_file_path, 'w') as file:
        file.write('This is a log file.\n')
    print(f'File "{log_file_path}" created.')
else:
    print(f'File "{log_file_path}" already exists.')

# 配置全局的logging对象
logging.basicConfig(filename=log_file_path, encoding='utf-8', level=logging.INFO, format='[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S')

# 创建一个日志记录器
logger = colorlog.getLogger()
logger.setLevel(logging.DEBUG)

# 创建一个StreamHandler，并使用colorlog库的ColoredFormatter
handler = logging.StreamHandler()

handler.setFormatter(colorlog.ColoredFormatter(
    '[%(asctime)s] %(log_color)s%(levelname)s:%(reset)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    log_colors={
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_yellow',
    }
))
logger.addHandler(handler)


def log():
    return logging
