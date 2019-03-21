import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import GlobalDefine


# def cls(x, label):
#     x = x[label != 0]
#     label = label[label != 0]
#     skf = StratifiedKFold(n_splits=16, shuffle=True)
#     skf.get_n_splits(x, label)
#     AA = []
#     OA = []
#     ACC = []
#     train_index_list = []
#     test_index_list = []
#     parameters = {'C': [np.exp2(i) for i in range(-1, 7)], 'gamma': [np.exp2(i) for i in range(-2, 3)]}
#     for index, (train_index, test_index) in enumerate(skf.split(x, label)):
#
#         x_train, x_test = x[test_index], x[train_index]
#         y_train, y_test = label[test_index], label[train_index]
#
#         svr = SVC(kernel='rbf')
#         clf = GridSearchCV(svr, parameters, refit=True, n_jobs=-1, cv=5)
#         clf.fit(x_train, y_train)
#         print(clf.best_params_)
#         y_pred = clf.predict(x_test)
#         OA.append(accuracy_score(y_pred=y_pred, y_true=y_test))
#         acc_for_each_class = precision_score(y_pred=y_pred, y_true=y_test, average=None)
#         ACC.append(acc_for_each_class)
#         AA.append(np.mean(acc_for_each_class))
#         train_index_list.append(train_index)
#         test_index_list.append(test_index)
#     print(np.sum(OA)/10.0)
#     total_AA = np.mean(AA)
#     total_OA = np.mean(OA)
#     total_ACC = np.mean(ACC, axis=0)
#     result = {'train_index_list': train_index_list, 'test_index_list': test_index_list,
#               'AA': AA, 'OA': OA, 'ACC': ACC,
#               'total_AA': total_AA, 'total_OA': total_OA, 'total_ACC': total_ACC}
#     return result


class SVM:
    def __init__(self, model_id):
        self.OA = 0
        self.ACC = 0
        self.AA = 0
        self.id = model_id
        self.svr = SVC(kernel='rbf')
        self.parameters = {'C': [np.exp2(i) for i in range(-1, 7)],
                           'gamma': [np.exp2(i) for i in range(-2, 3)]}
        self.clf = GridSearchCV(self.svr, self.parameters, refit=True, n_jobs=-1, cv=5)

    def train(self, x, label):
        print('start training...')
        self.clf.fit(x, label)
        print('SVM train done')
        joblib.dump(self.clf, "train_svm_model_%d%s.m" % (self.id, GlobalDefine.run_version))

    def test(self, x, label):
        self.clf = joblib.load("train_svm_model_%d%s.m" % (self.id, GlobalDefine.run_version))
        print(self.clf.best_params_)
        y_pred = self.clf.predict(x)
        self.OA = accuracy_score(y_pred=y_pred, y_true=label)
        acc_for_each_class = precision_score(y_pred=y_pred, y_true=label, average=None)
        self.ACC = acc_for_each_class
        self.AA = np.mean(acc_for_each_class)

        result = {'AA': self.AA, 'OA': self.OA, 'ACC': self.ACC}
        return result
