#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 17:26
# @Author  : MiaFeng
# @Site    : 
# @File    : anaResult.py
# @Software: PyCharm
__author__ = 'MiaFeng'

from util.io_util import loadDataFromPKL,saveDataAsPKL,makeFileDescription
from util.fig_util import *
scale_comp_4 = [
    [
        [0.988,0.966,0.982],
        [0.989,0.969,0.983],
    ],
    [
        [0.943,0.,0.499],
        [0.943,0,0.5],
    ],
    [
        [0.943,0.,0.499],
        [0.943,0,0.5],
    ],
    [
        [0.682, 0.502, 0.607],
        [0.682, 0.513, 0.612],
    ],
    [
        [0.732, 0.313, 0.557],
        [0.678, 0.636, 0.66],
    ]
]
scale_comp_label = ['sea_typ','land_typ','land_typ_emis','land_all','land_all_emis']
scale_comp_x_ticks = ['acc','recall','auc']



def estimateTopo():
    # # 1. sea, typhoon, channels 4
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea','typ',4,False,False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)

    # w = loadDataFromPKL()

    scores_sea_typ_4 = loadDataFromPKL('%s_scores' % modelDescrip)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 4, True, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)

    scores_sea_typ_4_scale = loadDataFromPKL('%s_scores' % modelDescrip)

    # 2. land, typhoon, channels 4
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, False, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)

    scores_land_typ_4 = loadDataFromPKL('%s_scores' % modelDescrip)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, True, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)

    scores_land_typ_4_scale = loadDataFromPKL('%s_scores' % modelDescrip)

    # 3. land, typhoon, channels 4 + emissivity
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, False, True
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)

    scores_land_typ_4_emis = loadDataFromPKL('%s_scores' % modelDescrip)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, True, True
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    scores_land_typ_4_emis_scale = loadDataFromPKL('%s_scores' % modelDescrip)

    # 7. land, global, channels 4
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, False, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    scores_land_all_4 = loadDataFromPKL('%s_scores' % modelDescrip)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, True, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    scores_land_all_4_scale = loadDataFromPKL('%s_scores' % modelDescrip)

    # 8. land, global, channels 4 + emissivity
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, False, True
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    scores_land_all_4_emis = loadDataFromPKL('%s_scores' % modelDescrip)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, True, True
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    scores_land_all_4_emis_scale = loadDataFromPKL('%s_scores' % modelDescrip)


def estimateTime():
    # # 1. sea, typhoon, channels 4
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 4, False, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_sea_typ_4 = loadDataFromPKL('%s_time' % modelDescrip)


    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 4, True, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_sea_typ_4_scale = loadDataFromPKL('%s_time' % modelDescrip)

    # 2. land, typhoon, channels 4
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, False, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)

    time_land_typ_4 = loadDataFromPKL('%s_time' % modelDescrip)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, True, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)

    time_land_typ_4_scale = loadDataFromPKL('%s_time' % modelDescrip)

    # 3. land, typhoon, channels 4 + emissivity
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, False, True
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)

    time_land_typ_4_emis = loadDataFromPKL('%s_time' % modelDescrip)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, True, True
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_land_typ_4_emis_scale = loadDataFromPKL('%s_time' % modelDescrip)

    # 7. land, global, channels 4
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, False, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_land_all_4 = loadDataFromPKL('%s_time' % modelDescrip)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, True, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_land_all_4_scale = loadDataFromPKL('%s_time' % modelDescrip)

    # 8. land, global, channels 4 + emissivity
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, False, True
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_land_all_4_emis = loadDataFromPKL('%s_time' % modelDescrip)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, True, True
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_land_all_4_emis_scale = loadDataFromPKL('%s_time' % modelDescrip)

    # 4. sea, typhoon, channels 616
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 616, False, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_sea_typ_616 = loadDataFromPKL('%s_time' % modelDescrip)
    imp_sea_typ_616 = loadDataFromPKL('%s_feaImp'%modelDescrip,encoding='iso-8859-1')


    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 616, True, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_sea_typ_616_scale = loadDataFromPKL('%s_time' % modelDescrip)
    imp_sea_typ_616_scale = loadDataFromPKL('%s_feaImp' % modelDescrip,encoding='iso-8859-1')

    # 5. land, typhoon, channels 616
    # # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 616, False, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_land_typ_616 = loadDataFromPKL('%s_time' % modelDescrip)
    imp_land_typ_616 = loadDataFromPKL('%s_feaImp' % modelDescrip,encoding='iso-8859-1')

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 616, True, False
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_land_typ_616_scale = loadDataFromPKL('%s_time' % modelDescrip)
    imp_land_typ_616_scale = loadDataFromPKL('%s_feaImp' % modelDescrip,encoding='iso-8859-1')


    # 6. land, typhoon, channels 616 + emissivity
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 616, False, True
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_land_typ_616_emis = loadDataFromPKL('%s_time' % modelDescrip)
    imp_land_typ_616_emis = loadDataFromPKL('%s_feaImp' % modelDescrip,encoding='iso-8859-1')

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 616, True, True
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale, isAddEmiss=isAddEmiss)
    time_land_typ_616_emis_scale = loadDataFromPKL('%s_time' % modelDescrip)
    imp_land_typ_616_emis_scale = loadDataFromPKL('%s_feaImp' % modelDescrip,encoding='iso-8859-1')

    fit_data_size = [[[27374]*3, [27374]*3],
                     [[7754]*3, [7754]*3],
                     [[7754] * 3, [7754] * 3],
                     [[26394]*3, [26394]*3],
                     [[7752]*3, [7752]*3],
                     [[7752] * 3, [7752] * 3]
                     ]
    fit_time = [
        [ time_sea_typ_4['fit'],time_sea_typ_4_scale['fit'] ],
        [time_land_typ_4['fit'], time_land_typ_4_scale['fit']],
        [time_land_typ_4_emis['fit'],time_land_typ_4_emis_scale['fit']],
        [time_sea_typ_616['fit'],time_sea_typ_616_scale['fit']],
        [time_land_typ_616['fit'],time_land_typ_616_scale['fit']],
        [time_land_typ_616_emis['fit'],time_land_typ_616_emis_scale['fit']]
    ]
    fit_time = np.divide(fit_time,fit_data_size)
    fit_time = np.log10(fit_time)
    labels = ['sea_typ_4','land_typ_4','land_typ_4_emis','sea_typ_616','land_typ_616','land_typ_616_emis']
    x_ticks = ['rf','xgbc','lr']



    predict_data_size = [[[9316]*3,[9316]*3],
                         [[1993]*3,[1993]*3],
                         [[1993] * 3, [1993] * 3],
                         [[8833]*3,[8833]*3],
                         [[1991]*3,[1991]*3],
                         [[1991] * 3, [1991] * 3]
                         ]

    pred_time = [
        [time_sea_typ_4['fit'], time_sea_typ_4_scale['fit']],
        [time_land_typ_4['fit'], time_land_typ_4_scale['fit']],
        [time_land_typ_4_emis['fit'], time_land_typ_4_emis_scale['fit']],
        [time_sea_typ_616['fit'], time_sea_typ_616_scale['fit']],
        [time_land_typ_616['fit'], time_land_typ_616_scale['fit']],
        [time_land_typ_616_emis['fit'], time_land_typ_616_emis_scale['fit']]
    ]
    pred_time = np.divide(pred_time,predict_data_size)
    pred_time = np.log10(pred_time)


    # plot time consumption
    # plot_multi_pair_line(pred_time, labels, x_ticks, 'model', 'predict time(log)',savefigName='typ_pred_time')
    # plot_multi_pair_line(fit_time, labels, x_ticks, 'model', 'fit time(log)', savefigName='typ_train_time')

    imp_df_sea_typ_616 = pd.DataFrame(data = [list(np.log10(imp_sea_typ_616['fea_imp']))],columns=list(imp_sea_typ_616['feaCols']))
    imp_df_sea_typ_616_scale = pd.DataFrame([list(np.log10(imp_sea_typ_616_scale['fea_imp']))],columns=imp_sea_typ_616_scale['feaCols'])
    imp_df_land_typ_616 = pd.DataFrame([list(np.log10(imp_land_typ_616['fea_imp']))],columns=imp_land_typ_616['feaCols'])
    imp_df_land_typ_616_scale = pd.DataFrame([list(np.log10(imp_land_typ_616_scale['fea_imp']))],columns=imp_land_typ_616_scale['feaCols'])
    imp_df_land_typ_616_emis = pd.DataFrame([list(np.log10(imp_land_typ_616_emis['fea_imp']))],columns=imp_land_typ_616_emis['feaCols'])
    imp_df_land_typ_616_emis_scale = pd.DataFrame([list(np.log10(imp_land_typ_616_emis_scale['fea_imp']))],columns=imp_land_typ_616_emis_scale['feaCols'])

    imp_df_list = [imp_df_sea_typ_616,imp_df_sea_typ_616_scale,imp_df_land_typ_616, imp_df_land_typ_616_scale]
    plotFeaImpCurve(imp_df_list,
                    fea_cols_common=['ch306', 'ch386', 'ch921'],
                    legend_labels=['sea_typ_616', 'sea_typ_616\nscale', 'land_typ_616','land_typ_616\nscale'],
                    savefigName='typ-616-fea-imp')

    imp_df_list = [imp_df_land_typ_616,imp_df_land_typ_616_scale,imp_df_land_typ_616_emis,imp_df_land_typ_616_emis_scale]
    plotFeaImpCurve(imp_df_list,
                    fea_cols_common=['ch306','ch386','ch921'],
                    legend_labels=['land_typ_616','land_typ_616\nscale','land_typ_616_emis','land_typ_616_emis\nscale'],
                    savefigName='typ-616-emis-fea-imp')

    print('')

if __name__=='__main__':
    import pandas as pd
    estimateTime()
    # plot_multi_pair_line(scale_comp_4,scale_comp_label,scale_comp_x_ticks,xlabel='scores\' name',ylabel='scores',savefigName='comp-scale-4')