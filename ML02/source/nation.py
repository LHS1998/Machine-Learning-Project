import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate  # 划分数据集函数
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_df = pd.read_csv('./cwurData.csv')  # 读入 csv 文件为 pandas 的 DataFrame
data_df = data_df.dropna()  # 舍去包含 NaN 的 row

feature_cols = ['quality_of_faculty', 'publications', 'citations', 'alumni_employment',
                'influence', 'quality_of_education', 'patents', 'broad_impact', 'national_rank']

for column in feature_cols:
    data_df['log_' + column] = np.log2(data_df[column])

choose = ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_patents',
          'influence', 'log_influence', 'log_national_rank']

total_RMSE = []
REPEAT_TIMES = 10000

for i in range(0, REPEAT_TIMES):

    train_df = data_df.sample(frac=0.8)
    test_df = data_df.drop(train_df.index)

    flag = train_df.region == 'USA'
    data_USA = train_df[flag]
    data_NUSA = train_df[~flag]

    x1_train = data_USA[choose].values
    y1_train = data_USA['score'].values
    x2_train = data_NUSA[choose].values
    y2_train = data_NUSA['score'].values

    x_test = test_df[choose].values
    y_test = test_df['score'].values
    x_flag = (test_df['region'] == 'USA').values

    lr1 = LinearRegression()
    lr2 = LinearRegression()
    lr1.fit(x1_train, y1_train)
    lr2.fit(x2_train, y2_train)

    RMSE = 0
    for sample, label, flag in zip(x_test, y_test, x_flag):
        y_hat = lr1.predict(sample.reshape(1, -1)) if flag else lr2.predict(sample.reshape(1, -1))
        RMSE += pow(label - y_hat, 2)

    RMSE = np.sqrt(RMSE / len(y_test))
    total_RMSE.append(RMSE[0])

mean = sum(total_RMSE) / REPEAT_TIMES
print(mean)

plt.xlim([min(total_RMSE) - 0.5, max(total_RMSE) + 0.5])
plt.hist(total_RMSE, bins=30)
plt.title('Test Set RMSE Frequency')
plt.xlabel('RMSE')
plt.ylabel('freq.')
plt.savefig('RMSE_no_br_hist.pdf')