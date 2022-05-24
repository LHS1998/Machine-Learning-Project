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
from Bagging import Bagging
import AdaBoost
from DecisionTree import DecisionTree

train_df = pd.read_csv('./data/train.csv', sep='\t')

train_loader = np.load("train_x.npz")
train_x = csr_matrix((train_loader['data'], train_loader['indices'], train_loader['indptr']),
                      shape=train_loader['shape'])
test_loader = np.load("test_x.npz")
test_x = csr_matrix((test_loader['data'], test_loader['indices'], test_loader['indptr']),
                      shape=test_loader['shape'])
train_y = train_df["votes_up"] / train_df["votes_all"]  # label： >=0.9 为 1，可以据此判断用投票比例作为可信概率是否合适

n_features = 30
pca = TruncatedSVD(n_components=n_features)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

# train_x = pd.DataFrame(data=train_x)
# test_x = pd.DataFrame(data=test_x)
#
# for c in train_x.columns:  # 遍历每一列特征，跳过标签列
#     col = train_x[c]
#     BIN_BOUND = [np.NINF]  # 区间起点 (防止分 bin 后出现 NaN)
#     for qnt in [.05, .25, .5, .75, .95]:
#         BIN_BOUND.append(col.quantile(qnt))
#     BIN_BOUND.append(np.inf)  # 区间终点
#     BIN = pd.IntervalIndex.from_tuples([(BIN_BOUND[i], BIN_BOUND[i+1]) for i in range(0, len(BIN_BOUND)-1)])
#     train_x[c] = pd.cut(col, BIN)  # 相比 qcut 这样做的好处是分位点比较整, 且不会出现分位点不唯一的情况.
#     test_x[c] = pd.cut(test_x[c], BIN)  # 使用同样分位点处理测试数据

# RANDOM_SEED = 0xa192c122
# x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=RANDOM_SEED)
#
# for i in range(10, 50, 5):
#     print("running k = %d" % i)
#     bagging_svm = Bagging(DecisionTreeClassifier, i, 1)
#     bagging_svm.fit(x_train, np.array([1 if y >= 0.9 else 0 for y in y_train]))
#     results = {}
#     y_predict = [bagging_svm.predict(test.reshape(1, -1)) for test in x_test]
#     auc = roc_auc_score([1 if y >= 0.9 else 0 for y in y_test],
#                         [1 if y >= 0.9 else 0 for y in y_predict])
#     err = [(pred - test) ** 2 for pred, test in zip(y_predict, y_test)]
#     print("k=%d, auc=%f, err=%f" % (i, auc, sum(err)/len(y_test)))

# ==================
# 正式提交计算
# ==================

k = 200
bagging_svm = Bagging(DecisionTreeClassifier, k, 1)
bagging_svm.fit(train_x, np.array([1 if y >= 0.9 else 0 for y in train_y]))

f = open('result_DT_n30_%d.csv' % k, mode='wt')
f.write('Id,Predicted\n')
for index, term in enumerate(test_x):
    result = bagging_svm.predict(term.reshape(1, -1))
    if result < 0:
        result = 0
    if result > 1:
        result = 1
    f.write("%d,%.2f\n" % (index, result))
f.close()
