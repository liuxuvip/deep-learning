# Author:  Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

print(__doc__)

import time
import numpy as np
import functools
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale, minmax_scale
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.model_selection import permutation_test_score
from sklearn import datasets
from load_data import load_HSI_data_ck, load_HSI_data
from sklearn.model_selection import GridSearchCV


def time_cost(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start = time.time()
        ret = func(*args, **kw)
        end = time.time()
        print('call %s():' % func.__name__, end - start)
        return ret
    return wrapper


@time_cost
def cls(X, y):
    X = X[y!=0]
    y = y[y!=0]
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf.get_n_splits(X, y)
    AA=[]
    OA=[]
    ACC=[]
    train_index_list=[]
    test_index_list=[]
    parameters = {'C': [np.exp2(i) for i in range(-1,7)], 'gamma': [np.exp2(i) for i in range(-2,3)]}
    for index, (train_index, test_index) in enumerate(skf.split(X, y)):

        X_train, X_test = X[test_index], X[train_index]
        y_train, y_test = y[test_index], y[train_index]

        svr = SVC(kernel='rbf')
        clf = GridSearchCV(svr, parameters, refit=True, n_jobs=-1, cv=5)
        clf.fit(X_train, y_train)
        print(clf.best_params_)
        y_pred = clf.predict(X_test)
        OA.append(accuracy_score(y_pred=y_pred, y_true=y_test))
        acc_for_each_class = precision_score(y_pred=y_pred, y_true=y_test, average=None)
        ACC.append(acc_for_each_class)
        AA.append(np.mean(acc_for_each_class))
        train_index_list.append(train_index)
        test_index_list.append(test_index)
    print(np.sum(OA)/10.0)
    total_AA = np.mean(AA)
    total_OA = np.mean(OA)
    total_ACC = np.mean(ACC, axis=0)
    result = {'train_index_list': train_index_list, 'test_index_list': test_index_list, 'AA': AA, 'OA': OA, 'ACC': ACC,
              'total_AA': total_AA, 'total_OA': total_OA, 'total_ACC': total_ACC}
    return result


@time_cost
def svm_benchmark():
    for name in ['indian_pines', 'salina', 'pavia', 'paviau']:
        result = cls(*load_HSI_data(name))
        sio.savemat(file_name='svm_benchmark_{}.mat'.format(name), mdict=result)


@time_cost
def svmck_benchmark():
    for name in ['indian_pines', 'salina', 'pavia', 'paviau']:
        result = cls(*load_HSI_data_ck(name))
        sio.savemat(file_name='svmck_benchmark_{}.mat'.format(name), mdict=result)


if __name__=='__main__':
    svm_benchmark()
    # result = cls(*load_indian())
    # sio.savemat(file_name='svm_benchmark_indian.mat', mdict=result)
