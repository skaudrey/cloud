# encoding: utf-8
#!/usr/bin/env python
'''
@Author: Mia
@Contact: skaudrey@163.com
@Software: PyCharm
@Site    : 
@Time    : 2018/8/6 上午11:57
@File    : modelCloud.py
@Theme   :
'''
import itertools
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,recall_score,accuracy_score,f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.learning_curve import learning_curve
from util.fig_util import *
from util.io_util import *
from preprocess.buildData import getChanIndex,getEmisIndex
from sklearn.utils import shuffle
import geopandas as gpd
from shapely.geometry import Point


class CModelCloud(object):
    def __init__(self,**params):
        self.model = None
        for key, value in params.items():
            self.params[key] = value


    def predict(self,X_test):
        return self.model.predict(X_test)

    def predict_prob(self,X_test):
        return self.model.predict_prob(X_test)


    def fit(self,X,y):
        self.model.fit(X, y)
        return self.model

    def modelName(self):
        try:
            modelStr = (self.__class__.__name__)[1:]
            modelName = ((modelStr.split('Cloud'))[0]).lower()
            return modelName
        except IndexError as err:
            print(err)
            print("Please make sure the name of your model class is named as C[Modle name in upper case]Cloud. E.g. CRFCloud,CLRCloud")
        finally:
            return modelName

    def hypFineTuning(self,X_train,y_train):
        pass

    def plotFeaImportance(self, feaImp, feaName):
        pass

    def filterFea(self, feaImp,feaNameList, ratio=0.7):
        pass

class CXGBCCloud(CModelCloud):
    def __init__(self, num_rounds=100, early_stopping_rounds=15, **params):

        self._params = {
            'booster':'gbtree',
            'objective': 'binary:logistic',
            'learning_rate': 0.1,
            'max_depth': 3,
            'seed': 0,
            'silent': 0,
            'n_estimators':num_rounds,
            'early_stopping_rounds':early_stopping_rounds
        }
        self._tuned_parameters = [{'n_estimators': [1,10, 20,40, 100, 200, 300],
                             'max_depth': [1, 3, 5, 7],  ##range(3,10,2)
                             'learning_rate': [0.5, 1.0],
                             'subsample': [0.75, 0.8, 0.85, 0.9]
                             }]

        self.model = XGBClassifier()

        for key, value in params.items():
            self._params[key] = value


    def predict(self, X_test):
        print('test with xgbc model')
        y_pred = self.model.predict(X_test)
        # score = model.score(X_test,y_test)
        return y_pred

    def predict_prob(self,X_test):
        return self.model.predict_proba(X_test)


    def fit(self, X_train, y_train):

        self.model.fit(X_train, y_train)

        return self.model.feature_importances_


    def hypFineTuning(self,X_train,y_train):
        clfTmp = GridSearchCV(
            self.model,
            param_grid=self._tuned_parameters,
            scoring='recall',
            n_jobs=8,
            iid=False,
            cv=5)

        clfTmp.fit(X_train,y_train)

        # y_true, y_pred = y_test, clfTmp.predict(X_test)

        best_param = clfTmp.best_params_

        # print "Recall Score (Train): %f" % recall_score(y_true, y_pred)
        print(clfTmp.best_params_)



        self.model.set_params(n_estimators=best_param['n_estimators'],
                         subsample=best_param['subsample'],
                         learning_rate=best_param['learning_rate'],
                         max_depth=best_param['max_depth'])
        return self.model,clfTmp.grid_scores_

    def filterFea(self, feaImp,feaNameList, ratio=0.7):
        print('xgbc filterfea')

        feaName = []
        imps = []

        for idx, imp in enumerate(feaImp):
            if imp > 0:
                imps.append(imp)
                feaName.append(feaNameList[idx])
        print(imps)

        feaImpDf = pd.DataFrame({
            'fea':feaName,
            'imp':imps
        })

        feaImpDf.sort_values(by='imp',inplace=True)

        feaFilter = feaImpDf['fea'].iloc[0:int(np.ceil(len(feaImpDf)*0.7))].values

        print('after filtering, the features for training is-->')
        print(feaFilter)
        print(len(feaFilter))
        print(80*'-')
        return feaFilter

class CRFCloud(CModelCloud):
    def __init__(self,
                 **params):
        for key, value in params.items():
            self.params[key] = value

        self.model = RandomForestClassifier()

        self.__tuned_parameters = [{"n_estimators": [4, 10, 15, 20, 25, 30],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [2, 4, 6]
                             }]


    def fit(self,X_train,y_train):
        self.model.fit(X_train, y_train)
        return self.model


    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def predict_prob(self,X_test):
        return self.model.predict_proba(X_test)

    def hypFineTuning(self,X_train,y_train):

        clfTmp = GridSearchCV(
            self.model,
            param_grid=self.__tuned_parameters,
            scoring='recall',
            n_jobs=8,
            iid=False,
            cv=5)

        clfTmp.fit(X_train,y_train)

        print(clfTmp.best_params_)

        best_param = clfTmp.best_params_

        self.model.set_params(n_estimators=best_param['n_estimators'],
                         criterion=best_param['criterion'],
                         min_samples_leaf=best_param['min_samples_leaf'])
        return self.model,clfTmp.grid_scores_

class CLRCloud(CModelCloud):
    def __init__(self,**params):
        self.model = LogisticRegression(
            random_state=1,
            penalty='l2')
        for key, value in params.items():
            self.params[key] = value
        self.__tuned_parameters = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1','l2']
        }

    def train(self, X_train, y_train):
        print("train with lr pipelined with standardScaler and PCA model")
        self.model.fit(X_train, y_train)
        return self.model

    def fit(self,X_train,y_train):
        self.model.fit(X_train,y_train)


    def predict(self, X_test):
        print('predict with pipeline logistic model')
        return self.model.predict(X_test)

    def hypFineTuning(self,X_train,y_train):
        clfTmp = GridSearchCV(
            self.model,
            param_grid=self.__tuned_parameters,
            scoring='recall',
            n_jobs=8,
            cv=5)

        clfTmp.fit(X_train, y_train)

        print(clfTmp.grid_scores_)


        print(clfTmp.best_params_)

        best_param = clfTmp.best_params_

        self.model.set_params(C=best_param['C'],penalty = best_param['penalty'])
        return self.model,clfTmp.grid_scores_

# class CDTCloud(CModelCloud):
#     def __init__(self,**params):
#         self.model = DecisionTreeClassifier(
#             random_state=1,
#             criterion="gini"
#         )
#         for key, value in params.items():
#             self.params[key] = value
#         self.__tuned_parameters = {
#
#         }
#
#
#     def train(self, X_train, y_train):
#         print(80*'-')
#         print("train with decision tree")
#         self.model.fit(X_train, y_train)
#         return self.model
#
#     def predict(self, X_test):
#         print('predict with pipeline logistic model')
#
#         return self.model.predict(X_test)
#
#     def hypFineTuning(self,X_train,y_train):
#         clfTmp = GridSearchCV(
#             self.model,
#             param_grid=self.__tuned_parameters,
#             scoring='recall',
#             cv=5)
#
#         clfTmp.fit(X_train, y_train)
#
#         print(clfTmp.grid_scores_)
#
#         print(clfTmp.best_params_)
#
#         best_param = clfTmp.best_params_
#
#         self.model.set_params(C=best_param['C'], penalty=best_param['penalty'])
#         return self.model, clfTmp.grid_scores_


class CEstimator(object):
    def __init__(self):
        pass

    def cvEstimate(self,X, y, clf, cv=10):
        from sklearn.cross_validation import cross_val_score
        scores_clf_cv = cross_val_score(clf, X, y, cv=cv)
        print(scores_clf_cv)
        return scores_clf_cv

    def steadyEstimate(self,model,X,y,epoch=50):
        w_list = []
        for i in range(0,epoch):
            model.fit(X,y)
            # w_tmp = model.coef_
            # w_tmp.append(model.intercept_)
            w_tmp = model.intercept_
            w_tmp = np.reshape(w_tmp,(1,1))
            w_tmp = np.hstack((model.coef_,w_tmp))
            w_list.append(w_tmp)
        return w_list

    def ana_data_size(self,model,X_train,y_train,split_size,scoring,cv=10):
        '''
        training with dataset in different size
        :param model: the instance of one specific model, such as your NN
        :param X_train: features with shape (#instances, #features)
        :param y_train: features with shape (#instances, 1)
        :param split_size: the dataset will be divided uniformly by this parameter
        :param scoring: scoring metrics, such as 'MAE','MSE', ect.
        :param cv: parameter of cross-validation
        :return:
        '''
        train_sizes = np.linspace(0.1, 1.0, split_size)
        train_int_sizes = train_sizes*len(X_train)
        # Make sure each chunk's labels are not unique.
        tmp_1 = 0
        for idx,i in enumerate(train_int_sizes):
            tmp_2 = int(train_int_sizes[idx])
            assert sum(y_train[tmp_1:tmp_2]==1)>0
            tmp_1 = tmp_2
        # assert sum(y_train[:int(train_int_sizes[0])] == 1) > 0
        train_sizes, train_scores, test_scores = learning_curve(estimator=model,
                                                                X=X_train,
                                                                y=y_train,
                                                                train_sizes=train_sizes,
                                                                cv=cv,
                                                                scoring=scoring)
        return train_sizes,train_scores,test_scores



    def get_roc_auc(self,y,y_pred_prob):
        fpr = []
        tpr = []
        auc = []

        for tmp in y_pred_prob:
            fpr_tmp,tpr_tmp,_ = roc_curve(y, tmp)

            fpr.append(fpr_tmp)
            tpr.append(tpr_tmp)
            auc.append(roc_auc_score(y, tmp))

        return fpr,tpr,auc


    def plot_ROC(self, y, y_predict, compareList,saveFigName = ''):
        '''

        :param y:
        :param y_predict:
        :param compareList:  models' name
        :return:
        '''
        fpr_all, tpr_all, auc_all = self.get_roc_auc(y, y_predict)
        colors = 'bgrcmykw'

        plt.figure()
        fig, ax = plt.subplots()
        colors = ["#0092c7", "#f3e59a", "#9fe0f6",
                  "#f3b59a", "#f29c9c", "#22c3aa",
                  ]
        ax.plot([0, 1], [0, 1], colors[0],linestyle='--', label='Baseline(AUC=0.5)')
        ax.plot([0, 0,1], [0, 1,1],colors[0], label='Perfect(AUC=1)')
        p_lines = []
        for idx,(fpr,tpr,auc) in enumerate(zip(fpr_all, tpr_all, auc_all)):
            legend = '%s( AUC = %.2f )' % (compareList[idx],auc)

            p_tmp = ax.plot(fpr, tpr, colors[idx+1], label=legend)

            p_lines.append(p_tmp)

        plt.xlim([-0.1, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend(loc='lower right')
        if len(saveFigName)>0:
            plt.savefig('%s/%s.png'%(getPath('fig'),saveFigName),dpi=600)

        plt.show()

    def plotConfusionMatrix(self, cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        '''
        This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
        :param cm: confusion matrix
        :param y: the real label
        :param y_predict: the predicted label
        :param classes: The class name of each class denoted by integers.
        :param normalize:
        :param title:
        :param cmap: color map
        :return:
        '''
        plt.figure()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)  # 0 for negative, 1 for positive
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def eval(self, y, y_predict, metrics=['auc','recall','accuracy','f1'], pos_label=1):

        scores = {}

        for scoring in metrics:
            if scoring == 'confusion_matrix':
                self.plotConfusionMatrix(cm=confusion_matrix(y, y_predict),
                                         classes=['negative', 'positive'])
            elif scoring == 'auc':
                auc = roc_auc_score(y, y_predict)
                scores['auc'] = auc
            elif scoring == 'recall':
                recall = recall_score(y, y_predict, pos_label=pos_label)
                # print('recall -- %s: %.2f' % (modelName, recall))
                scores['recall'] = recall

            elif scoring == 'accuracy':
                accuracy = accuracy_score(y, y_predict)
                # print('accuracy -- %s: %.2f' % (modelName, accuracy))
                scores['acc'] = accuracy
            elif scoring == 'f1_score':
                f1 = f1_score(y, y_predict)
                # print('f1 score -- %s: %.2f' % (modelName, f1))
                scores['f1'] = f1
            else:
                print("help yourself")
        return scores

class CLoader(object):
    def __init__(self):
        self.__filter = None  # define filter
        self.__filtType = '' # define filtering type: xgbc, rf_search

    def loadData(self,trainTyp,testTyp,topo,category,chan_num,rescale,isAddEmiss):
        feaCols = getChanIndex('chan_%d'%chan_num)
        if isAddEmiss:
            feaCols += getEmisIndex()
        df_train, df_test = self.__loadData(trainTyp,testTyp,chan_num,topo,category,rescale=rescale)

        X_train, y_train, X_test, y_test = self.__getDataFea(df_train,df_test,feaCols)

        return X_train, y_train, X_test, y_test

    def loadGeoData(self,filename,category='typ'):
        '''
        load data as common dataframe or geographical dataframe
        :param fileName:
        :param category: ['typ','all']
        :return:
        '''
        df = pd.read_csv("%s/%s" % (getPath(category), filename), sep=',')

        df['Coordinates'] = list(zip(df.lon, df.lat))
        df['Coordinates'] = df['Coordinates'].apply(Point)
        gdf = gpd.GeoDataFrame(df, geometry='Coordinates')
        return gdf

    def __loadData(self,trainTyp,testTyp,chan_num,topo='sea',category='typ',rescale = False):
        '''
        load data according to typhoon's name: there are two kinds of typhoon here.
        :param trainTyp:
        :param testTyp:
        :return:
        '''
        train_file_name = makeFileDescription(topo,category,chan_num,typName=trainTyp,rescale=rescale,isTrain = True)
        test_file_name = makeFileDescription(topo,category,chan_num,typName=testTyp,rescale=rescale,isTrain = False)

        df_test = readCSV(test_file_name,category=category,header='infer')
        df_train = readCSV(train_file_name,category=category,header='infer')

        return df_train, df_test

    def __check_chunk_label_unique(self,df,ten_split_size):
        '''
        check whether the chunk's labels are not unique, unless you wanna an error while analyzing the influence of data sizes.
        :param df:
        :return:
        '''
        # ten_split_size = np.linspace(0.1, 1, 10) * len(df)
        tmp_1 = 0
        for idx, i in enumerate(ten_split_size):
            tmp_2 = int(ten_split_size[idx])
            if sum(df['label'][tmp_1:tmp_2] == 1) == 0:
                return True
            else:
                tmp_1 = tmp_2
        return False

    def __getDataFea(self,df_train,df_test,feaColNameList,isShuffle = True):
        '''
        Load data for training after filtering dataset, and rescaled data.
        :param df_train:
        :param df_test:
        :param feaColNameList:
        :param rescale:
        :return:
        '''

        assert (1 in (df_train['label'].values.tolist())) == True
        assert (1 in (df_test['label'].values.tolist())) == True

        if isShuffle:
            ten_split_size = np.linspace(0.1, 1, 10) * len(df_train)
            while self.__check_chunk_label_unique(df_train,ten_split_size) : # make sure labels are not unique in case the error of splitting
                df_train = shuffle(df_train)
                df_train.reset_index(drop = True,inplace = True)
                # tmp = df_train[:int(ten_split_size[0])]
            ten_split_size = np.linspace(0.1, 1, 10) * len(df_test)
            # tmp = df_test[:int(ten_split_size[0])]
            while self.__check_chunk_label_unique(df_test,ten_split_size):  # make sure labels are not unique in case the error of splitting
                df_test = shuffle(df_test)
                df_test.reset_index(drop = True,inplace = True)
                # tmp = df_train[:int(ten_split_size[0])]

        X_train = df_train[feaColNameList].values
        y_train = df_train['label'].values

        X_test = df_test[feaColNameList].values
        y_test = df_test['label'].values
        return X_train,y_train,X_test,y_test