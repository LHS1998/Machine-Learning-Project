from collections import namedtuple
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn_extra.cluster import KMedoids

csv_data = './data/[UCI] AAAI-14 Accepted Papers - Papers.csv'  # 数据路径

with open(csv_data, mode='rt') as f:
    csv_file = csv.reader(f)
    Record = namedtuple('Record', next(csv_file))
    records = []
    for row in csv_file:
        # row = [title, authors, groups, keywords, topics, abstract]
        # author应当作为一个整体加入特征, keywords 也可以考虑整体加入
        # groups 和 topics 特征有重复 (原文和缩写), 考虑选取特异性更高的缩写
        # topics 和 keywords 特征应当把 \n 处理掉
        # title, keywords 和 abstract 应当具有不同的权重
        row_ = [row[0]]
        # author 拆成人名的元组
        authors = row[1]
        if ' and ' in authors:  # 多于一个作者
            authors = authors.split(' and ')
            if ', ' in authors[0]:  # 多于两个作者
                cache = authors[1]
                authors = authors[0].split(', ')
                authors.append(cache)
        row_.append(tuple(authors))
        # group 去掉重复元素
        groups = row[2]
        groups_ = []
        if groups:  # groups 有可能为空
            if '\n' in groups:
                groups = groups.split('\n')
            else:
                groups = [groups]  # 化成字符串的列表, 以统一处理
            for term in groups:
                groups_.append(term.split("(")[1].split(")")[0])  # 取缩写
        row_.append(tuple(groups_))
        # keywords 提取
        keywords = row[3]
        if '\n' in keywords:
            keywords = keywords.split('\n')
        else:
            keywords = (keywords, )
        row_.append(tuple(keywords))
        # topics 提取
        topics = row[4]
        if '\n' in topics:
            topics = topics.split('\n')
        else:
            topics = (topics, )
        row_.append(tuple(topics))
        # abstract 可能有换行
        row_.append(row[5].replace('\n', ' '))
        # 处理完成
        records.append(Record(*row_))

del topics, term, row, row_, keywords, groups, groups_, f, csv_file, cache, authors  # 清除临时变量，便于调试

features = []
for title, authors, groups, keywords, topics, abstract in records:
    feature = (title + ' ') * 15  # 标题三倍权重
    for author in authors:
        feature += (author.replace(' ', '').replace('.', '') + ' ')  # 作者名字作为一个整体加入
    for group in groups:
        feature += (group + ' ')
    for keyword in keywords:
        feature += (keyword + ' ') * 20  # 关键字三倍权重
    for topic in topics:
        feature += (topic + ' ')  # 有待改进
    feature += abstract
    features.append(feature)

del title, authors, groups, keywords, topics, abstract, author, feature, group, keyword, topic

vectorizer = TfidfVectorizer()
vectorizer.fit(features)
vec = vectorizer.transform(features)

#########################
##  非层次聚类方法
#########################

## K-Medoids
# num = 25
# way = 'K-Medoiods'
# cluster = KMedoids(n_clusters=num)
# cluster.fit(vec)

# K-Means
num = 25
way = 'K-Means'
cluster = KMeans(n_clusters=num)
cluster.fit(vec)

## MiniBatch K-Means
# num = 25
# way = 'MiniBatch-K-Means'
# cluster = MiniBatchKMeans(n_clusters=num)
# cluster.fit(vec)

# 保存结果
result = [[] for i in range(num)]
for rec, lbl in zip(records, cluster.labels_):
    result[lbl].append(rec)

with open("{}-{}-weighted.txt".format(way, num), mode='wt') as f:
    for i, rec in enumerate(result):
        f.write('==== Cluster {} ===>\n'.format(i+1))
        for j, term in enumerate(rec):
            f.write('\t({}) {}'.format(j+1, term[0]))
            f.write('\n')
        f.write('\n\n')
