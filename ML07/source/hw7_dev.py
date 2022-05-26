import time
import numpy as np
import pandas as pd
# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import ensemble
# Normalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
# Scoring
from sklearn.metrics import roc_auc_score

train_data = pd.read_csv("./data_processed/train_data.csv", sep=',')
train_label = pd.read_csv("./data_processed/train_label.csv", sep=',')
validate_data = pd.read_csv("./data_processed/validate_data.csv", sep=',')
validate_label = pd.read_csv("./data_processed/validate_label.csv", sep=',')

scaler = MinMaxScaler()
train_minmax = scaler.fit_transform(train_data)
validate_minmax = scaler.transform(validate_data)

scaler = StandardScaler()
train_std = scaler.fit_transform(train_data)
validate_std = scaler.transform(validate_data)

scaler = Normalizer()
train_norm = scaler.fit_transform(train_data)
validate_norm = scaler.transform(validate_data)

# Single Machine Learning Model
logger = open("single_model_result.txt", mode='at')
logger.write("====> [%s] <====\n" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
log_template = "分类器：%s. AUC = %.7f\n"


def single_test(classifier, x_train, y_train, x_test, y_test, accuracy, precondition, **kwargs):
    clf = classifier(**kwargs)
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)
    auc = accuracy(y_test, y_hat)
    logger.write(log_template % ('%r@%r@%s' % (classifier, kwargs, precondition), auc))
    return y_hat


# single_test(DecisionTreeClassifier, train_data.values, train_label.values, validate_data.values, validate_label.values,
#             roc_auc_score, 'None', max_depth=10, min_samples_split=10)
# single_test(DecisionTreeClassifier, train_minmax, train_label.values, validate_minmax, validate_label.values,
#             roc_auc_score, 'MinMax', max_depth=10, min_samples_split=10)
# single_test(DecisionTreeClassifier, train_std, train_label.values, validate_std, validate_label.values,
#             roc_auc_score, 'Standard', max_depth=10, min_samples_split=10)
# single_test(DecisionTreeClassifier, train_norm, train_label.values, validate_norm, validate_label.values,
#             roc_auc_score, 'Normalizer', max_depth=10, min_samples_split=10)
# single_test(DecisionTreeClassifier, train_data.values, train_label.values, validate_data.values, validate_label.values,
#             roc_auc_score, 'None', max_depth=5)
# for m in [2, 5, 10, 20, 30, 50, 100]:
#     single_test(DecisionTreeClassifier, train_data.values, train_label.values, validate_data.values, validate_label.values,
#                 roc_auc_score, 'None', max_depth=5, min_samples_split=m)


def mean_test(k, threshold, classifier, x_train, y_train, x_test, y_test, accuracy, precondition, **kwargs):
    results = []
    for i in range(k):
        clf = classifier(**kwargs)
        clf.fit(x_train, y_train)
        results.append(clf.predict(x_test))
    y_hat = np.mean(np.array(results), axis=0)
    auc = accuracy(y_test, [1 if y > threshold else 0 for y in y_hat])
    logger.write(log_template % ('%r@%r@%s[%dx@%.1f]' % (classifier, kwargs, precondition, k, threshold), auc))
    return y_hat


# for thres in [0.6, 0.7, 0.8, 0.9]:
#     mean_test(10, thres, DecisionTreeClassifier, train_norm, train_label.values, validate_norm, validate_label.values,
#               roc_auc_score, 'None', max_depth=10, min_samples_split=10)
#     mean_test(10, thres, DecisionTreeClassifier, train_norm, train_label.values, validate_norm, validate_label.values,
#               roc_auc_score, 'MinMax', max_depth=10, min_samples_split=10)
#     mean_test(10, thres, DecisionTreeClassifier, train_norm, train_label.values, validate_norm, validate_label.values,
#               roc_auc_score, 'Standard', max_depth=10, min_samples_split=10)
#     mean_test(10, thres, DecisionTreeClassifier, train_norm, train_label.values, validate_norm, validate_label.values,
#               roc_auc_score, 'Normalizer', max_depth=10, min_samples_split=10)
#
#
# for clf in [KNN, LinearSVC]:
# for clf in [ensemble.AdaBoostClassifier, ensemble.RandomForestClassifier, ensemble.GradientBoostingClassifier,
#             ensemble.IsolationForest, ensemble.StackingClassifier, ensemble.VotingClassifier]:
#     single_test(clf, train_data.values, train_label.values, validate_data.values, validate_label.values,
#                 roc_auc_score, 'None')
#     single_test(clf, train_minmax, train_label.values, validate_minmax, validate_label.values,
#                 roc_auc_score, 'MinMax')
#     single_test(clf, train_std, train_label.values, validate_std, validate_label.values,
#                 roc_auc_score, 'Standard')
#     single_test(clf, train_norm, train_label.values, validate_norm, validate_label.values,
#                 roc_auc_score, 'Normalizer')

for args in [{"base_estimator": DecisionTreeClassifier(max_depth=5), "n_estimators": 1000, "max_samples": 0.3},
             {"base_estimator": DecisionTreeClassifier(max_depth=1), "n_estimators": 1000, "max_samples": 0.3},
             {"base_estimator": SVC(), "n_estimators": 1000, "max_samples": 0.3},
             {"base_estimator": KNN(n_neighbors=2), "n_estimators": 1000, "max_samples": 0.3},
             {"base_estimator": KNN(n_neighbors=3), "n_estimators": 1000, "max_samples": 0.3}]:
    single_test(ensemble.BaggingClassifier, train_data.values, train_label.values,
                validate_data.values, validate_label.values,
                roc_auc_score, 'None', **args)
    single_test(ensemble.BaggingClassifier, train_minmax, train_label.values, validate_minmax, validate_label.values,
                roc_auc_score, 'MinMax', **args)
    single_test(ensemble.BaggingClassifier, train_std, train_label.values, validate_std, validate_label.values,
                roc_auc_score, 'Standard', **args)
    single_test(ensemble.BaggingClassifier, train_norm, train_label.values, validate_norm, validate_label.values,
                roc_auc_score, 'Normalizer', **args)


logger.write('\n')
logger.close()

# 另一种可能：根据单次过关记录预测，平均值作为退坑概率
