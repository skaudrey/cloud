#!/usr/bin/env python
# encoding: utf-8
'''
@author: MiaFeng
@contact: skaudrey@163.com
@file: io_util.py
@time: 2018/10/2 11:24
@desc:
'''
from sklearn.externals import joblib
import os
from util.path_util import getPath
import pandas as pd
import pickle

def save_model( model,fileDescription, modelName=''):
    if not os.path.exists(getPath('model')):
        os.makedirs(getPath('model'))
    modelName = makeModelName(fileDescription,modelName)
    print('save model as %s' % modelName)
    modelName = '%s/%s'%(getPath('model'),modelName)
    joblib.dump(model, modelName)
    return modelName

def load_model(fileDescription,modelName=''):
    modelName = makeModelName(fileDescription, modelName)
    modelName = '%s/%s' % (getPath('model'), modelName)
    print('load --> %s'%modelName)
    model = joblib.load(modelName)
    return model

def readCSV(filename,category='data',fileType='csv',sep=',',header=None):
    basepath = getPath(category)
    filename = "%s/%s.%s" % (basepath, filename,fileType)
    print('read file --> %s' % filename)
    return pd.read_csv(filename,sep=sep, encoding='utf-8',header=header)

def saveAsCSV(df,fileName,category='typ',header = None):
    fileName = '%s/%s.csv' % (getPath(category), fileName)
    if header == 'infer':
        header = df.columns.tolist()
    df.to_csv(fileName,sep=',',encoding='utf-8',header=header,index=None)
    print("Done for saving dataframe as --> %s"%fileName)


def makeFileDescription(topo,category,chan_num,typName='',rescale=False,isTrain = None,isAddEmiss=None):
    '''
    :param topo: ['land','sea']
    :param category: ['typ','all']
    :param chan_num: [4,616]
    :param typName: ['','catfish','seahorse']
    :param rescale: [True,False]
    :param isTrain: [None,True,False]
    :return:
    '''
    # 'land_all_4_catfish_train_scale','land_typ_4_catfish_train_scale'
    fileDescrip = '%s_%s_%d'%(topo,category,chan_num)

    if isAddEmiss:
        fileDescrip += '_emiss'

    if typName!='':
        fileDescrip += '_%s'%typName
    if isTrain != None:
        if isTrain:
            fileDescrip += '_train'
        else:
            fileDescrip += '_test'
    if rescale:
        fileDescrip += '_scale'

    return fileDescrip

def makeModelName(fileDescription, modelName =''):
    model_name = fileDescription
    if modelName != '':
        model_name += '_%s' % modelName
    model_name = '%s.pkl' % (model_name)
    return model_name


def saveDataAsPKL(filename,data,category = 'result'):
    filename = '%s/%s.pkl'%(getPath(category),filename)
    output = open(filename, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(data, output)
    output.close()
    print('save data as --> %s'%filename)

def loadDataFromPKL(filename,category = 'result',encoding='utf-8'):
    filename = '%s/%s.pkl' % (getPath(category), filename)
    pkl_file = open(filename, 'rb')

    data = pickle.load(pkl_file,encoding=encoding)
    # pprint.pprint(data)

    pkl_file.close()

    return data

def saveFig(axes,figName):
    if len(figName)>0:
        axes.savefig('%s/%s.png'%(getPath('fig'),figName))