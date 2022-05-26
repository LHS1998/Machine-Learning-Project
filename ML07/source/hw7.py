import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import ensemble
from Bagging import Bagging

train_data = pd.read_csv("./data_processed/all_train_data.csv", sep=',')
train_label = pd.read_csv("./data_processed/all_train_label.csv", sep=',')
test_data = pd.read_csv("./data_processed/test_data.csv", sep=',')
test_label = pd.read_csv("./data_processed/test_label.csv", sep=',')

# 特征选择
drop = [' time_stamp_%d' % i for i in range(23)]
drop += [" mean_rest_step", " most_played_hour", " excess_success_rate", " excess_tries", " excess_time"]
train_data.drop(columns=drop)
test_data.drop(columns=drop)

scaler = StandardScaler()
train_norm = scaler.fit_transform(train_data)
test_norm = scaler.transform(test_data)

# 模型训练
clf = ensemble.BaggingClassifier(DecisionTreeClassifier(max_depth=5), 160)  # 0.74229
# clf = ensemble.RandomForestClassifier(max_depth=5, n_estimators=10000)  # 0.74421
# clf = ensemble.AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5))  # 0.66192
# clf = ensemble.StackingClassifier(estimators=[('rf', ensemble.RandomForestClassifier(max_depth=5)),
#                                               ('ada', ensemble.AdaBoostClassifier()),
#                                               ('dt', DecisionTreeClassifier(max_depth=1))])  # 0.74373
# clf = ensemble.StackingClassifier(estimators=[('rf', ensemble.RandomForestClassifier(max_depth=5)),
#                                               ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42))),
#                                               ('dt', DecisionTreeClassifier(max_depth=1))])  # 0.74382
# clf = ensemble.StackingClassifier(estimators=[('rf', ensemble.RandomForestClassifier(max_depth=5)),
#                                               ('svc', ensemble.BaggingClassifier(SVC(), 30))])  # 0.74391
# clf = ensemble.BaggingClassifier(SVC(), 50)  # 全是0-1
# clf = ensemble.BaggingClassifier(DecisionTreeClassifier(max_depth=2), max_samples=0.3, bootstrap_features=True,
#                                  n_estimators=1000)  # 0.74196
# clf = ensemble.BaggingClassifier(DecisionTreeClassifier(max_depth=1), max_samples=0.3,
#                                  n_estimators=1000)  # 0.72963
# clf.fit(train_data.values, train_label.values)
# clf = Bagging(SVC, 100, 0.3)

# 使用未经预处理的数据
# clf.fit(train_data.values, train_label.values)
# y_hat = [clf.predict(test.reshape(1, -1)) for test in test_data.values]  # 0.68793

# with open("result.csv", mode='wt') as f:
#     f.write("user_id,proba\n")
#     for y, i in zip(y_hat, test_label.values):
#         f.write("%d, %.5f\n" % (i[0], y))
# y_hat = clf.predict_proba(test_data.values)

# 使用经预处理的数据
clf.fit(train_norm, train_label.values)
y_hat = clf.predict_proba(test_norm)

with open("result.csv", mode='wt') as f:
    f.write("user_id,proba\n")
    for y, i in zip(y_hat[:, 1], test_label.values):
        f.write("%d, %.5f\n" % (i[0], y))
