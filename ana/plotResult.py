#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 9:04
# @Author  : MiaFeng
# @Site    : 
# @File    : plotResult.py
# @Software: PyCharm
__author__ = 'MiaFeng'
from logistic.modelCloud import *
scoring = 'recall'
estimator = CEstimator()

from itertools import chain
def plotResult(topo, category, chan_num, rescale, isAddEmiss):
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)

    lc = loadDataFromPKL('%s_lc' % modelDescrip,encoding='iso-8859-1')

    plot_leaning_curve(lc['split_sizes'], lc['train_scores'], lc['test_scores'], score_name=scoring,savefigName='%s_lc' % modelDescrip)

    # feaImp= loadDataFromPKL('%s_feaImp' % modelDescrip,encoding='iso-8859-1')
    #
    # plotFeaImportance(feaImp['feaCols'], feaImp['fea_imp'], saveFigName='%s_fea_imp' % modelDescrip)

    y_pred_all = loadDataFromPKL('%s_pred' % modelDescrip,encoding='iso-8859-1')

    estimator.plot_ROC(y_pred_all['y_test'], y_pred_all['y_pred'], y_pred_all['y_name'],
                       saveFigName='%s_roc' % modelDescrip)


def plotTrain():
    # # 1. sea, typhoon, channels 4
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 4, False, False
    plotResult(topo, category, chan_num, rescale, isAddEmiss)


    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 4, True, False
    plotResult(topo, category, chan_num, rescale, isAddEmiss)


    # 2. land, typhoon, channels 4
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, False, False
    plotResult(topo, category, chan_num, rescale, isAddEmiss)


    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, True, False
    plotResult(topo, category, chan_num, rescale, isAddEmiss)

    # 3. land, typhoon, channels 4 + emissivity
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, False, True
    plotResult(topo, category, chan_num, rescale, isAddEmiss)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, True, True
    plotResult(topo, category, chan_num, rescale, isAddEmiss)

    # # 7. land, global, channels 4
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, False, False
    plotResult(topo, category, chan_num, rescale, isAddEmiss)


    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, True, False
    plotResult(topo, category, chan_num, rescale, isAddEmiss)

    # # 8. land, global, channels 4 + emissivity
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, False, True
    plotResult(topo, category, chan_num, rescale, isAddEmiss)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, True, True
    plotResult(topo, category, chan_num, rescale, isAddEmiss)

    # 4. sea, typhoon, channels 616
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 616, False, False
    plotResult(topo, category, chan_num, rescale, isAddEmiss)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 616, True, False
    plotResult(topo, category, chan_num, rescale, isAddEmiss)

    # 5. land, typhoon, channels 616
    # # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 616, False, False
    plotResult(topo, category, chan_num, rescale, isAddEmiss)

    # # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 616, True, False
    plotResult(topo, category, chan_num, rescale, isAddEmiss)

    # # 6. land, typhoon, channels 616 + emissivity
    # # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 616, False, True
    plotResult(topo, category, chan_num, rescale, isAddEmiss)

    # # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 616, True, True
    plotResult(topo, category, chan_num, rescale, isAddEmiss)

def mergeCloudAVHRR():
    avhrr_base_path = getPath(category='avhrr')
    ml_base_path = getPath(category='ml')

    df_avhrr = pd.DataFrame({})
    df_ml = pd.DataFrame({})
    # df_true_cloud = pd.DataFrame({})

    # for topo in ['land','sea']:
    #     true_lat_fileName = '%s/%s_lat.txt' % (getPath(category='luo'),  topo)
    #     true_lon_fileName = '%s/%s_lon.txt' % (getPath(category='luo'),  topo)
    #     df_true_lat = pd.read_csv(true_lat_fileName, encoding='utf-8', header=None)
    #     df_true_lon = pd.read_csv(true_lon_fileName, encoding='utf-8', header=None)
    #
    #     df_true_cloud = pd.concat(
    #         [df_true_cloud, pd.DataFrame({'lat': list(chain.from_iterable(df_true_lat.values.tolist())),
    #                                       'lon': list(chain.from_iterable(df_true_lon.values.tolist())),
    #                                       'cloud_flag': [1] * len(df_true_lon),
    #                                       'topo_flag': [-1] * len(df_true_lon)})])

    for isCloud in ['cloud','clear']:

        for topo in ['land','sea']:
            cloud_flag,topo_flag = 1,0
            if isCloud=='clear':
                cloud_flag = 0
            if topo == 'land':
                topo_flag = 1
            avhrr_lat_fileName = '%s/%s/%s_%s_lat.txt'%(avhrr_base_path,isCloud,isCloud,topo)
            avhrr_lon_fileName = '%s/%s/%s_%s_lon.txt' % (avhrr_base_path, isCloud, isCloud, topo)

            df_avhrr_lat = pd.read_csv(avhrr_lat_fileName,encoding='utf-8',header=None)
            df_avhrr_lon = pd.read_csv(avhrr_lon_fileName,encoding='utf-8',header=None)

            df_avhrr = pd.concat([df_avhrr,pd.DataFrame({'lat':list(chain.from_iterable(df_avhrr_lat.values.tolist())),
                                            'lon':list(chain.from_iterable(df_avhrr_lon.values.tolist())),
                                            'cloud_flag':[cloud_flag]*len(df_avhrr_lat),
                                            'topo_flag':[topo_flag]*len(df_avhrr_lat)
                                            })])

            ml_lat_fileName = '%s/%s/%s_%s_lat.txt' % (ml_base_path, isCloud, isCloud, topo)
            ml_lon_fileName = '%s/%s/%s_%s_lon.txt' % (ml_base_path, isCloud, isCloud, topo)

            df_ml_lat = pd.read_csv(ml_lat_fileName, encoding='utf-8', header=None)
            df_ml_lon = pd.read_csv(ml_lon_fileName, encoding='utf-8', header=None)

            df_ml = pd.concat([df_ml ,pd.DataFrame({'lat': list(chain.from_iterable(df_ml_lat.values.tolist())),
                                       'lon': list(chain.from_iterable(df_ml_lon.values.tolist())),
                                       'cloud_flag': [cloud_flag] * len(df_ml_lon),
                                       'topo_flag':[topo_flag]*len(df_ml_lat)})])


    print(len(df_avhrr))
    print(len(df_ml))

    df_avhrr.to_csv('%s/avhrr_result.csv'%getPath('data'),encoding='utf-8',index=None)
    df_ml.to_csv('%s/ml_result.csv'%getPath('data'),encoding='utf-8',index=None)

def mergeLRResult():
    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 4, True, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale,isAddEmiss=isAddEmiss)

    geo_file = makeFileDescription(topo, category, chan_num, typName='seahorse', rescale=rescale, isTrain=False)
    df_geo_sea = readCSV(geo_file,category='typ',header='infer')

    df_geo_sea = df_geo_sea[['lat','lon']]

    y_pred_sea = loadDataFromPKL('%s_pred' % modelDescrip, encoding='iso-8859-1')
    y_pred_sea = y_pred_sea['y_pred'][2].tolist()
    # y_pred_sea = list(map(lambda x:[x],y_pred_sea))

    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, True, True
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale,isAddEmiss=isAddEmiss)
    geo_file = makeFileDescription(topo, category, chan_num, typName='seahorse', rescale=rescale, isTrain=False)
    df_geo_land = readCSV(geo_file, category='typ', header='infer')

    df_geo_land = df_geo_land[['lat', 'lon']]

    y_pred_land = loadDataFromPKL('%s_pred' % modelDescrip, encoding='iso-8859-1')
    y_pred_land = y_pred_land['y_pred'][2].tolist()
    # y_pred_land = list(map(lambda x:[x],y_pred_land))
    df = pd.concat([df_geo_sea, df_geo_land],ignore_index=True)
    df_pred = pd.DataFrame({'cloud_flag':y_pred_sea+y_pred_land})
    df = pd.concat([df,df_pred],axis=1)

    df.to_csv('%s/ml_result.csv'%getPath('data'),index = None,encoding = 'utf-8')

    return df


if __name__=='__main__':
    # mergeCloudAVHRR()
    df_ml = mergeLRResult()
    df_ml[['lat', 'lon']] = df_ml[['lat', 'lon']].applymap(lambda x: round(x, 1))

    df_avhrr = pd.read_csv('%s/avhrr_result.csv'%getPath('data'))
    df_avhrr = df_avhrr.sort_values(by=['lon', 'lat'])
    df_avhrr[['lat', 'lon']] = df_avhrr[['lat', 'lon']].applymap(lambda x: round(x, 1))
    df_avhrr_merge = pd.merge(df_avhrr,df_ml,on=['lon','lat'])
    df_avhrr = df_avhrr_merge[['lon','lat','cloud_flag_x']]
    df_avhrr.rename(columns={'cloud_flag_x':'cloud_flag'},inplace=True)

    # df_avhrr_filt_lon = list(map(lambda a,b,c,d: x in y,df_avhrr['lon'].values.tolist(), [df_ml['lon'].values.tolist()]*len(df_avhrr)))
    # df_avhrr_filt_lat = list(map(lambda x,y: x in y,df_avhrr['lat'].values.tolist(), [df_ml['lat'].values.tolist()]*len(df_avhrr)))
    # df_avhrr_filt = list(map(lambda x,y: x and y, df_avhrr_filt_lon,df_avhrr_filt_lat))
    # df_avhrr = df_avhrr[df_avhrr_filt]

    # df_ml = pd.read_csv('%s/ml_result.csv'%getPath('data'))
    df_true_cloud = pd.read_csv('%s/cloud_true.csv'%getPath('data'))
    df_true_cloud[['lat', 'lon']] = df_true_cloud[['lat', 'lon']].applymap(lambda x: round(x, 1))


    # df_true_filt_lon = list(
    #     map(lambda x, y: x in y, df_avhrr['lon'].values.tolist(), [df_ml['lon'].values.tolist()] * len(df_true_cloud)))
    # df_true_filt_lat = list(
    #     map(lambda x, y: x in y, df_avhrr['lat'].values.tolist(), [df_ml['lat'].values.tolist()] * len(df_true_cloud)))
    # df_true_filt = list(map(lambda x, y: x and y, df_true_filt_lon, df_true_filt_lat))
    # df_true_cloud = df_true_cloud[df_true_filt]
    df_true_cloud = pd.merge(df_true_cloud,df_ml,on=['lon','lat'])[['lon','lat','cloud_flag_x']]
    df_true_cloud.rename(columns={'cloud_flag_x':'cloud_flag'},inplace=True)

    # print(df_avhrr.describe())
    # print(df_ml.describe())

    plotAvhrr(df_avhrr,'avhrr')
    plotAvhrr(df_ml,'logistic')
    plotAvhrr(df_true_cloud,'cloud_true')
    mergeLRResult()






