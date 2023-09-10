#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：ML_project
@Product_name ：PyCharm
@File ：sample.py
@Author ：RockJim
@Date ：2023/9/7 15:51
@Description ：对数据进行抽样
@Version ：1.0
"""
import os.path
import pandas as pd


def sample_and_save(data, sample_size, save_dir):
    """
        抽样并保存
    :param data: 原始数据
    :param sample_size: 抽样数量
    :param save_dir: 保存文件夹
    :return:
    """
    # 检查样本大小是否合理
    assert sample_size < len(data), print("样本大小大于数据集大小，请重新设置。")
    # 随机抽取不重复的样本
    sample_data = data.sample(n=sample_size, replace=False)

    # 按照 # 排序进行保存
    sample_data = sample_data.sort_values(by='#')

    # 重置索引并增加序号列
    # sample_data.reset_index(drop=True, inplace=True)
    # sample_data.index = sample_data.index + 1
    # sample_data.insert(0, )

    # 保存抽取的样本到新的CSV文件
    file_path = os.path.join(save_dir, 'sample_data_' + str(sample_size) + '.csv')
    sample_data.to_csv(file_path, index=False)

    print(f"成功抽取并保存了 {sample_size} 条样本到 {file_path} 文件中。")


if __name__ == '__main__':
    # 读取原始CSV文件
    original_data = pd.read_csv('../resource/30000-20230906.csv')
    save_dir = '../resource/sampled_data'
    sample_size = [300, 2000, 3500, 6700, 9900, 13100, 16300, 19500, 21700, 26000]
    # sample_size = [5, 10, 35, 67, 90]
    for item in sample_size:
        sample_and_save(original_data, item, save_dir)
    print("抽样执行结束……")
