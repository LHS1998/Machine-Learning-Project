from collections import Counter
import pandas as pd  # 数据处理
import numpy as np  # 数学运算
from sklearn.model_selection import train_test_split, cross_validate  # 划分数据集函数
from sklearn.metrics import accuracy_score  # 准确率函数
from config import *
from DecisionTree import *

csv_data = './high_diamond_ranked_10min.csv'  # 数据路径
data_df = pd.read_csv(csv_data, sep=',')  # 读入csv文件为pandas的DataFrame
data_df = data_df.drop(columns='gameId')  # 舍去对局标号列

# 预处理
# ========
# (1) 舍去多余的信息
df = data_df.drop(columns=drop_features)  # 舍去特征列
# (2) 创建差值列
info_names = [c[3:] for c in data_df.columns if c.startswith('red')]  # blue 多一个 blueWins
for info in info_names:
    data_df['br' + info] = data_df['blue' + info] - data_df['red' + info]  # 构造由蓝色特征减去红色特征，前缀为br
df = df.drop(columns=['blueFirstBlood', 'redFirstBlood'])  # FirstBlood 为首次击杀最多有一只队伍能获得
# (3) 离散化特征
discrete_df = df.copy()  # 先复制一份数据
for c in df.columns[1:]:  # 遍历每一列特征，跳过标签列
    if c in discrete_features:
        continue
    if c in multi_discrete_feature:
        # continue
        _types = discrete_df[c].value_counts().index.tolist()  # 取值的数目
        for _type in _types:  # 创造一系列数据，每个都是 {0, 1} 取值的
            discrete_df[c + str(_type)] = [1 if _sample == _type else 0 for _sample in discrete_df[c]]
        discrete_df = discrete_df.drop(columns=c)
    if c in q_features:
        col = df[c]
        BIN_BOUND = [np.NINF]  # 区间起点 (防止分 bin 后出现 NaN)
        for qnt in QUANTILES:  # 读取 config 中选定的分位数
            BIN_BOUND.append(col.quantile(qnt))
        BIN_BOUND.append(np.inf)  # 区间终点
        BIN = pd.IntervalIndex.from_tuples([(BIN_BOUND[i], BIN_BOUND[i+1]) for i in range(0, len(BIN_BOUND)-1)])
        discrete_df[c] = pd.cut(col, BIN)  # 相比 qcut 这样做的好处是分位点比较整, 且不会出现分位点不唯一的情况.
        _types = discrete_df[c].value_counts().index.tolist()  # 取值的数目
        for _type in _types:  # 创造一系列数据，每个都是 {0, 1} 取值的
            discrete_df[c + str(_type)] = [1 if _sample == _type else 0 for _sample in discrete_df[c]]
        discrete_df = discrete_df.drop(columns=c)
    # 对某些维度可能会有更好的预处理方式，先专注于用分位数离散化

# 数据集准备
# ========
# (1) 提取标签和特征
all_y = discrete_df['blueWins'].values  # 所有标签数据
feature_names = discrete_df.columns[1:]  # 所有特征的名称
all_x = discrete_df[feature_names].values  # 所有原始特征值，pandas 的 DataFrame.values 取出为 numpy 的 array 矩阵
# (2) 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)

# 模型训练
# ========
DT = DecisionTree(classes=[0, 1], features=feature_names, max_depth=12, min_samples_split=29, impurity_t='gini')

DT.fit(x_train, y_train)  # 在训练集上训练
p_test = DT.predict(x_test)  # 在测试集上预测，获得预测值
print(p_test)  # 输出预测值

p = [0, 0]
for result in p_test:
    if result == 0:
        p[0] += 1
    else:
        p[1] += 1
print(p)

test_acc = accuracy_score(p_test, y_test)  # 将测试预测值与测试集标签对比获得准确率
print('accuracy: {:.4f}'.format(test_acc))  # 输出准确率

print(DT)
