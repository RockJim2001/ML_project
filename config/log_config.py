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
import colorlog
# 设置日志文件的保存路径和文件名
# log_file_path = 'logs/svr_aeo.log'
# log_file_path = 'logs/ann_aeo.log'
log_file_path = 'logs/test.log'
# log_file_path = r'D:\Code\PythonProject\ML_project\logs\ann_aeo_test.log'


# 配置全局的logging对象
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S')

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
