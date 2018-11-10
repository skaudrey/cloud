#!/usr/bin/env python
# encoding: utf-8
'''
@author: MiaFeng
@contact: skaudrey@163.com
@file: estimateSea.py
@time: 2018/10/9 11:29
@desc:
'''

from logistic.modelCloud import *
from util.io_util import saveDataAsPKL
import time

# warnings.filterwarnings("ignore")

loader = CLoader()
estimator = CEstimator()
clf_xgbc = CXGBCCloud()
clf_lr = CLRCloud()
clf_rf = CRFCloud()

trainTyp = 'catfish'
testTyp = 'seahorse'

scoring = 'recall'

# def timeTicToc(func):
#     @wraps(func)
#     def runtime(*args, **kwargs):
#         start = time.clock()
#         print('start: %.6f\n', start)
#         result = func(*args, **kwargs)
#         stop = time.clock()
#         print('run_time : %.4f' % (stop - start))
#         return result
#
#     return runtime()
#

# def testSea():
#     '''
#     (1.1) chan4_topo_rescale = False --> land & sea
#         fine tuning: need to save scores in each cv, weights of lr in each cv, feature_importances
#         learning curve: need to plot learning curve
#         save model
#     (1.2) chan4_topo_rescale = True --> land & sea
#         fine tuning: need to save scores in each cv, weights of lr in each cv, feature_importances
#         learning curve: need to plot learning curve
#         save model
#     (1.3) chan616_topo_rescale = False --> land & sea
#         fine tuning: need to save scores in each cv, weights of lr in each cv, feature_importances
#         learning curve: need to plot learning curve
#         save model
#     (1.4) chan616_topo_rescale = True --> land & sea
#         fine tuning: need to save scores in each cv, weights of lr in each cv, feature_importances
#         learning curve: need to plot learning curve
#         save model
#     (3) compare scale and rescale
#         predict: plot confusion matrix, save recall, f1, acc, predict time; and the scores required by plotting roc curve.
#
#
#     :return:
#     '''
#     pass

# @timeTicToc
# def predict(clf,X_test):
#     y_pred = clf.predict(X_test)
#     return y_pred
#
# @timeTicToc
# def fit(clf,X_train,y_train):
#     model = clf.fit(X_train,y_train)
#     return model

def tuning(topo, category, chan_num,rescale,isAddEmiss):
    # test unscaled data
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale,isAddEmiss=isAddEmiss)
    X_train, y_train, X_test, y_test = loader.loadData(trainTyp, testTyp, topo, category, chan_num, rescale=rescale,
                                                       isAddEmiss=isAddEmiss)
    # fine tuning
    xgbc, xgbc_grid_scores = clf_xgbc.hypFineTuning(X_train, y_train)
    rf, rf_grid_scores = clf_rf.hypFineTuning(X_train, y_train)
    lr, lr_grid_scores = clf_lr.hypFineTuning(X_train, y_train)
    grid_scores = {
        'rf': rf_grid_scores,
        'xgbc': xgbc_grid_scores,
        'lr': lr_grid_scores
    }
    saveDataAsPKL(data=grid_scores, filename='%s_tuning_grid_scores' % modelDescrip)

    # save model
    save_model(rf, modelDescrip, 'rf')
    save_model(xgbc, modelDescrip, 'xgbc')
    save_model(lr, modelDescrip, 'lr')

def estimate(topo, category, chan_num,rescale,isAddEmiss):
    modelDescrip = makeFileDescription(topo, category, chan_num, rescale=rescale , isAddEmiss=isAddEmiss)
    rf, xgbc, lr = loadModel(modelDescrip)
    X_train, y_train, X_test, y_test = loader.loadData(trainTyp, testTyp, topo, category, chan_num, rescale=rescale,
                                                       isAddEmiss=isAddEmiss)

    # cv train -- 评价模型的性能，此时模型并未进行fit
    cv_xgbc = estimator.cvEstimate(X_train, y_train, xgbc, 10)
    cv_rf = estimator.cvEstimate(X_train, y_train, rf, 10)
    cv_lr = estimator.cvEstimate(X_train, y_train, lr, 10)
    cv = {
        'rf': cv_rf,
        'xgbc': cv_xgbc,
        'lr': cv_lr
    }
    saveDataAsPKL(data=cv, filename='%s_cv' % modelDescrip)

    # learning curve —— 数据是顺序加入的,在此之前，loadData时需要对data进行shuffle确保data在被split时，不会只有一类数据，不然learning curve那里会报错
    split_sizes, train_scores_xgbc, test_scores_xgbc = estimator.ana_data_size(xgbc, X_train, y_train, 10, 'roc_auc',
                                                                               cv=10)
    split_sizes, train_scores_rf, test_scores_rf = estimator.ana_data_size(rf, X_train, y_train, 10, 'roc_auc', cv=10)
    split_sizes, train_scores_lr, test_scores_lr = estimator.ana_data_size(lr, X_train, y_train, 10, 'roc_auc', cv=10)

    lc_train_scores = [
        train_scores_rf,
        train_scores_xgbc,
        train_scores_lr
    ]
    lc_test_scores = [
        test_scores_rf,
        test_scores_xgbc,
        test_scores_lr
    ]

    lc = {
        'split_sizes': split_sizes,
        'train_scores': lc_train_scores,
        'test_scores': lc_test_scores
    }

    saveDataAsPKL('%s_lc' % modelDescrip, lc)

    plot_leaning_curve(split_sizes, lc_train_scores, lc_test_scores, score_name=scoring,
                       savefigName='%s_lc' % modelDescrip)

    # steady estimate
    w_steady = estimator.steadyEstimate(lr, X_train, y_train)
    saveDataAsPKL('%s_w_steady' % modelDescrip, w_steady)

    # fit
    start = time.clock()
    xgbc.fit(X_train, y_train)
    xgbc_fit_time = time.clock() - start
    start = time.clock()
    rf.fit(X_train, y_train)
    rf_fit_time = time.clock() - start
    start = time.clock()
    lr.fit(X_train, y_train)
    lr_fit_time = time.clock() - start

    fit_time = [rf_fit_time, xgbc_fit_time, lr_fit_time]

    # plot feature importance
    feaCols = getChanIndex('chan_%d' % chan_num)
    if isAddEmiss:
        feaCols += getEmisIndex()

    fea_imp = rf.feature_importances_

    feaImp_save = {
        'fea_imp':fea_imp,
        'feaCols':feaCols
    }

    saveDataAsPKL('%s_feaImp' % modelDescrip, feaImp_save)

    plotFeaImportance(feaCols,fea_imp,saveFigName='%s_fea_imp' % modelDescrip)

    # predict
    start = time.clock()
    rf_y_pred = rf.predict(X_test)
    rf_pred_time = time.clock() - start
    start = time.clock()
    xgbc_y_pred = xgbc.predict(X_test)
    xgbc_pred_time = time.clock() - start
    start = time.clock()
    lr_y_pred = lr.predict(X_test)
    lr_pred_time = time.clock() - start

    predict_time = [rf_pred_time, xgbc_pred_time, lr_pred_time]

    save_time = {
        'fit': fit_time,
        'pred': predict_time
    }
    saveDataAsPKL('%s_time' % modelDescrip, save_time)

    y_pred = pd.DataFrame({
        'rf': rf_y_pred,
        'xgbc': xgbc_y_pred,
        'lr': lr_y_pred
    })

    rf_scores = estimator.eval(y_test, y_pred['rf'], ['auc', 'recall', 'accuracy', 'f1'])
    xgbc_scores = estimator.eval(y_test, y_pred['xgbc'], ['auc', 'recall', 'accuracy', 'f1'])
    lr_scores = estimator.eval(y_test, y_pred['lr'], ['auc', 'recall', 'accuracy', 'f1'])
    print('rf scores --> ')
    print(rf_scores)
    print('xgbc scores --> ')
    print(xgbc_scores)
    print('lr scores --> ')
    print(lr_scores)

    scores = {
        'rf': rf_scores,
        'xgbc': xgbc_scores,
        'lr': lr_scores
    }
    saveDataAsPKL('%s_scores' % modelDescrip, scores)

    y_pred_all = {
        'y_test':y_test,
        'y_pred':[rf_y_pred, xgbc_y_pred, lr_y_pred],
        'y_name':['rf', 'xgbc', 'lr']
    }
    saveDataAsPKL('%s_pred' % modelDescrip, y_pred_all)

    estimator.plot_ROC(y_test, [rf_y_pred, xgbc_y_pred, lr_y_pred], ['rf', 'xgbc', 'lr'],
                       saveFigName='%s_roc' % modelDescrip)

def loadModel(modelDescrip):

    rf = load_model(modelDescrip,'rf')
    xgbc = load_model(modelDescrip,'xgbc')
    lr = load_model(modelDescrip,'lr')

    return rf,xgbc,lr

def train():

    # # # 1. sea, typhoon, channels 4
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea','typ',4,False,False
    # tuning(topo,category,chan_num,rescale,isAddEmiss)
    estimate(topo,category,chan_num,rescale,isAddEmiss)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 4, True, False
    tuning(topo, category, chan_num, rescale, isAddEmiss)
    estimate(topo, category, chan_num, rescale, isAddEmiss)

    # 2. land, typhoon, channels 4
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, False, False
    tuning(topo, category, chan_num, rescale, isAddEmiss)
    estimate(topo, category, chan_num, rescale, isAddEmiss)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, True, False
    tuning(topo, category, chan_num, rescale, isAddEmiss)
    estimate(topo, category, chan_num, rescale, isAddEmiss)

    # 3. land, typhoon, channels 4 + emissivity
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, False, True
    tuning(topo, category, chan_num, rescale, isAddEmiss)
    estimate(topo, category, chan_num, rescale, isAddEmiss)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 4, True, True
    tuning(topo, category, chan_num, rescale, isAddEmiss)
    estimate(topo, category, chan_num, rescale, isAddEmiss)

    # 4. sea, typhoon, channels 616
    # unscaled
    # topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 616, False, False
    # tuning(topo, category, chan_num, rescale, isAddEmiss)
    # estimate(topo, category, chan_num, rescale, isAddEmiss)

    # scaled
    # topo, category, chan_num, rescale, isAddEmiss = 'sea', 'typ', 616, True, False
    # tuning(topo, category, chan_num, rescale, isAddEmiss)
    # estimate(topo, category, chan_num, rescale, isAddEmiss)

    # 5. land, typhoon, channels 616
    # # unscaled
    # topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 616, False, False
    # tuning(topo, category, chan_num, rescale, isAddEmiss)
    # estimate(topo, category, chan_num, rescale, isAddEmiss)
    #
    # # scaled
    # topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 616, True, False
    # tuning(topo, category, chan_num, rescale, isAddEmiss)
    # estimate(topo, category, chan_num, rescale, isAddEmiss)
    #
    # # 6. land, typhoon, channels 616 + emissivity
    # # unscaled
    # topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 616, False, True
    # tuning(topo, category, chan_num, rescale, isAddEmiss)
    # estimate(topo, category, chan_num, rescale, isAddEmiss)
    #
    # # scaled
    # topo, category, chan_num, rescale, isAddEmiss = 'land', 'typ', 616, True, True
    # tuning(topo, category, chan_num, rescale, isAddEmiss)
    # estimate(topo, category, chan_num, rescale, isAddEmiss)

    # # 7. land, global, channels 4
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, False, False
    # tuning(topo, category, chan_num, rescale, isAddEmiss)
    estimate(topo, category, chan_num, rescale, isAddEmiss)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, True, False
    tuning(topo, category, chan_num, rescale, isAddEmiss)
    estimate(topo, category, chan_num, rescale, isAddEmiss)

    # 8. land, global, channels 4 + emissivity
    # unscaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, False, True
    tuning(topo, category, chan_num, rescale, isAddEmiss)
    estimate(topo, category, chan_num, rescale, isAddEmiss)

    # scaled
    topo, category, chan_num, rescale, isAddEmiss = 'land', 'all', 4, True, True
    tuning(topo, category, chan_num, rescale, isAddEmiss)
    estimate(topo, category, chan_num, rescale, isAddEmiss)




if __name__ == '__main__':

    train()
