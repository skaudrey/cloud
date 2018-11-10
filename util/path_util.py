#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/6 9:44
# @Author  : MiaFeng
# @Site    : 
# @File    : path_util.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import os

BASE_PATH = 'D:\Code\Python\cloud'

def getPath(category='data'):
    subPath = ''
    if category in ['data','model','result']:
        subPath = category
    elif category=='typ':
        subPath = 'data/typ'
    elif category=='all':
        subPath = 'data/land'
    elif category=='fig':
        subPath = 'result/fig'
    elif category=='showfig':
        subPath = 'result/showfig'
    elif category == 'avhrr':
        subPath = 'data/byluo/AVHRR'
    elif category == 'ml':
        subPath = 'data/byluo/ML'
    elif category=='luo':
        subPath = 'data/byluo'
    else:
        print("Please check the path category you defined")
    return "%s/%s"%(BASE_PATH,subPath)

def getCurrentPath():
    return os.getcwd()

def getParentPath(aPath):
    '''
    get the parent path of path aPath
    :param dirname:
    :return:
    '''
    return os.path.dirname(aPath)