#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：ML_project 
@Product_name ：PyCharm
@File ：transformer.py
@Author ：RockJim
@Date ：2023/7/28 20:56
@Description ：对生成的数据格式进行转换
@Version ：1.0 
'''
import csv
import os.path

import numpy as np
from numpy import genfromtxt


#
# # 文件路径
# data_path = r"C:\Users\25760\Desktop\AllCombinedResults.csv"
# save_path = r"C:\Users\25760\Desktop\AllCombinedResults_替换后.csv"
#
# # 数据读取
# assert os.path.exists(data_path), print("数据集路径{}不存在".format(data_path))
# file_data = genfromtxt(data_path, delimiter=',')
#
# # 定义替换规则
# rule = {
#     4: {
#         1.461281581: 0.684330805,
#         1.342724779: 0.744754261,
#         1.037437653: 0.963913347,
#         1.315362691: 0.76024659,
#         1.321664717: 0.756621545,
#     },
#     5: {
#         1.165435214: 0.858048554,
#         1.060853711: 0.942637038,
#         1.129366043: 0.885452512,
#         1.260847624: 0.79311725,
#         1.584347777: 0.631174553,
#     },
#     6: {
#         1.128512955: 0.886121861,
#         1.014414274: 0.985790545,
#         0.963575217: 1.037801702,
#         0.675090963: 1.481281864,
#         0.773982567: 1.292018765,
#     },
#     7: {
#         3.699580855: 3.699580855,
#         3.791348094: 3.791348094,
#         3.7780763: 3.7780763,
#         3.782396333: 3.782396333,
#         3.290854749: 3.290854749,
#     },
#     8: {
#         0.310802761: 0.310802761,
#         0.35313454: 0.35313454,
#         0.30554842: 0.30554842,
#         0.336477144: 0.336477144,
#         0.273519163: 0.273519163,
#     },
#     9: {
#         0.380312409: 0.380312409,
#         0.310237558: 0.310237558,
#         0.205393491: 0.205393491,
#         0.312830491: 0.312830491,
#         0.289584361: 0.289584361,
#     },
#     10: {
#         0.253227251: 0.253227251,
#         0.262723057: 0.262723057,
#         0.204416668: 0.204416668,
#         0.326895578: 0.326895578,
#         0.359080078: 0.359080078,
#     },
#     11: {
#         0.291657595: 0.291657595,
#         0.242627125: 0.242627125,
#         0.203904911: 0.203904911,
#         0.379576876: 0.379576876,
#         0.243728903: 0.243728903,
#     },
#     12: {
#         13.23405932: 0.271867866,
#         11.38268273: 0.230726182,
#         12.76016164: 0.261336811,
#         13.21624075: 0.271471898,
#         10.23882053: 0.205307033,
#     },
#     13: {
#         10.43796874: 0.209732547,
#         13.02927715: 0.267317153,
#         11.55148739: 0.234477395,
#         12.77407146: 0.261645918,
#         10.71259941: 0.215835448,
#     },
#     14: {
#         21.78773311: 0.219863605,
#         34.98766916: 0.366529497,
#         20.28361888: 0.203151232,
#         32.56629208: 0.339625319,
#         30.28666444: 0.314296134,
#     },
#     15: {
#         30.05076236: 0.311675001,
#         26.5997203: 0.273330106,
#         31.21038956: 0.324559742,
#         29.0558284: 0.300620184,
#         24.71246115: 0.252360569,
#     },
# }
#
#
# def transformer_rule():
#     """
#         使用上面的rule规则或者自己进行计算来进行替换
#     :return:
#     """
#     # 进行替换
#     # for item in file_data:
#     #     temp = item
#     #     for key, value in rule.items():
#     #         if np.isnan(temp[key]):
#     #             temp[key] = np.nan
#     #             continue
#     #         temp[key] = value[temp[key]]
#     # 将结果进行保存, 通过数组来定义保存哪些列
#     file_data[:, 4] = 1 / file_data[:, 4]
#     file_data[:, 5] = 1 / file_data[:, 5]
#     file_data[:, 6] = 1 / file_data[:, 6]
#
#     file_data[:, 12] = (file_data[:, 12] - 1) * 1.777777 / 80
#     file_data[:, 13] = (file_data[:, 13] - 1) * 1.777777 / 80
#
#     file_data[:, 14] = (file_data[:, 14] - 2) * 1.777777 / 160
#     file_data[:, 15] = (file_data[:, 15] - 2) * 1.777777 / 160
#
#     file_data[:, 24] = file_data[:, 23] + file_data[:, 24]
#     save_column_list = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 22, 23, 24, 26, 27]
#     save_data = file_data[:, save_column_list]
#     np.savetxt(save_path, save_data, delimiter=',', fmt='%.9f')
#

def transformer_combine():
    # 定义文件路径
    x_data_path = r"C:\Users\25760\Desktop\x_data.csv"
    y_data_path = r"C:\Users\25760\Desktop\y_data.csv"
    save_path = r"C:\Users\25760\Desktop\AllCombinedResults_替换后.csv"
    # 读取x数据
    with open(x_data_path, 'r') as file_temp:
        x_data = []
        reader = csv.DictReader(file_temp)
        for row in reader:
            x_data.append(row)
        # return x_data

    # 读取y数据
    with open(y_data_path, 'r') as file_temp:
        y_data = []
        reader = csv.DictReader(file_temp)
        for row in reader:
            y_data.append(row)
        # return y_data

    # 结果保存
    result_list = []
    for index, item in enumerate(x_data):
        result_temp = {**{'id': index}, **item}
        flag = index
        if y_data[flag]['Job_ID'] == item['Job_ID']:
            flag = index
        else:
            for y_index, y_item in enumerate(y_data):
                if y_item['Job_ID'] == item['Job_ID']:
                    flag = y_index
        y_data_item = y_data[flag]
        result_temp['c0: District Cooling'] = float(y_data_item['c0: District Cooling 1.056[kWh]']) * 1.056
        result_temp['c1: Interior Lighting'] = float(y_data_item['c1: Interior Lighting 3.167[kWh]']) * 3.167
        result_temp['c2: Total Energy [kWh]'] = y_data_item['c2: Total Energy [kWh]']
        result_temp['c4: Annual and Peak Values-Electricity'] = float(y_data_item[
                                                                          'c4: Annual and Peak Values-Electricity 3.167[kWh]']) * 3.167
        result_temp['c5: Annual and Peak Values-Cooling'] = float(y_data_item[
                                                                      'c5: Annual and Peak Values-Cooling 1.056[kWh]']) * 1.056
        result_temp['c7: Not Comfortable[Hours]'] = y_data_item['c7: Not Comfortable[Hours]']
        result_list.append(result_temp)
    print("结果保存")
    with open(save_path, 'w', newline='') as file_temp:
        fieldnames = list(result_list[0].keys())
        writer = csv.DictWriter(file_temp, fieldnames=fieldnames)

        writer.writeheader()
        for row in result_list:
            writer.writerow(row)
    print("CSV 文件已保存：", save_path)


if __name__ == '__main__':
    transformer_combine()
