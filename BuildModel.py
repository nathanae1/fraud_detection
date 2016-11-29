import numpy as np
import pandas as pd
import cPickle as pickle
from preprocessing import preprocess
import xgboost

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.datasets import make_classification


class BuildModel(object):
    def __init__(self, feat_mat,labels, params, test_size, model = LogisticRegression, cv_folds=5):
        self.X = feat_mat
        self.y = labels
        self.params = params
        self.model = model(**params)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(feat_mat, labels, test_size = test_size)
        self.cv_folds = cv_folds

    def confusion_matrix(self, y_true, y_pred):
        y_true = np.array(y_true)
        tp = np.sum(y_true*y_pred*np.ones(len(y_true)))
        fp = np.sum(y_pred*(np.ones(len(y_true))-y_true))
        fn = np.sum(y_true*(np.ones(len(y_true))-y_pred))
        tn = len(y_true) - tp - fp - fn
        return np.array([[tp,fp],[fn,tn]])

    def score(self,metric):
        if metric == roc_auc_score:
            y_pred = self.model.predict_proba(self.X_test)[:,1]
        else:
            y_pred = self.model.predict(self.X_test)
        sc = metric(self.y_test, y_pred)
        print "{} = {}".format(metric.func_name,sc)

    def test(self):
        m_list = [roc_auc_score,recall_score,precision_score,accuracy_score, self.confusion_matrix]
        for m in m_list:
            self.score(m)

    def run(self, cv_scoring):
        scores = cross_val_score(self.model, self.X_train, y=self.y_train, scoring=cv_scoring,
                        cv=self.cv_folds, n_jobs=1, verbose=0,
                        fit_params=None, pre_dispatch='2*n_jobs')
        print "The {} scores are: {}".format(cv_scoring, scores)
        self.model.fit(self.X_train, self.y_train)
        self.test()
        self.model.fit(self.X, self.y)







if __name__ == '__main__':
    # # read in data from output of preprocessing script
    # dat = pd.read_csv('datafile')
    # # split data into features and target
    # X = dat.drop(['target'])
    # y = dat['target']

    # generate fake data
    X, y = preprocess('data/train_new.json', train=True)

    # instantiate model class
    params = {}
    mod = BuildModel(X, y, params, test_size = 0.1, model=xgboost.XGBClassifier)
    mod.run('accuracy')
    # run testing on the model
    # fit the model on the complete dataset


    # pickle the model for use in prediction
    with open('model.pkl', 'w') as f:
        pickle.dump(mod.model, f)
