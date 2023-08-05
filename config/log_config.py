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

# 设置日志文件的保存路径和文件名
# log_file_path = 'logs/svr_aeo.log'
log_file_path = 'logs/ann_aeo.log'
# 配置全局的logging对象
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def log():
    return logging
