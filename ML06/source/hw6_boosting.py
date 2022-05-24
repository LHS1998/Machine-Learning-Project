import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC  # SVM 分类器
from sklearn.svm import SVR, LinearSVR  # 可以试试回归直接学概率
from sklearn.tree import DecisionTreeClassifier
import nltk
import numpy as np
import AdaBoost

train_df = pd.read_csv('./data/train.csv', sep='\t')

train_loader = np.load("train_x.npz")
train_x = csr_matrix((train_loader['data'], train_loader['indices'], train_loader['indptr']),
                      shape=train_loader['shape'])
test_loader = np.load("test_x.npz")
test_x = csr_matrix((test_loader['data'], test_loader['indices'], test_loader['indptr']),
                      shape=test_loader['shape'])
train_y = train_df["votes_up"] / train_df["votes_all"]  # label： >=0.9 为 1，可以据此判断用投票比例作为可信概率是否合适

pca = TruncatedSVD(n_components=500)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

# # 划分验证集
# RANDOM_SEED = 0xa192c122
# x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=RANDOM_SEED)
#
# for k in range(5, 30, 5):
#     print("Testing", k)
#     boosting_svm = AdaBoost.M1(SVR, k)
#     boosting_svm.fit(x_train, y_train, C=2)
#
#     y_predict = [boosting_svm.predict(test_sample.reshape(1, -1)) for test_sample in x_test]
#     auc = roc_auc_score([1 if y >= 0.9 else 0 for y in y_test], [1 if y >= 0.9 else 0 for y in y_predict])
#     err = [(pred - test) ** 2 for pred, test in zip(y_predict, y_test)]
#     print(auc, sum(err)/len(y_test))

# ==================
# 正式提交计算
# ==================

k = 20
bagging_svm = AdaBoost.M1(DecisionTreeClassifier, k)
bagging_svm.fit(train_x, np.array([1 if y >= 0.9 else 0 for y in train_y]), max_depth=10)

f = open('result_M1_DT_k=%d.csv' % k, mode='wt')
f.write('Id,Predicted\n')
for index, term in enumerate(test_x):
    result = bagging_svm.predict(term.reshape(1, -1))
    # if result < 0:
    #     result = 0
    # if result > 1:
    #     result = 1
    f.write("%d,%.2f\n" % (index, result))
f.close()
