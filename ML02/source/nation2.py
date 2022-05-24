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

    flag1 = train_df.region == 'USA'
    data_USA = train_df[flag1]
    data_NUSA = train_df[~flag1]

    flag2 = data_NUSA.region == 'China'
    data_CN = data_NUSA[flag2]
    data_NCN = data_NUSA[~flag2]

    flag3 = data_NCN.region == 'Japan'
    data_JP = data_NCN[flag3]
    data_NJP = data_NCN[~flag3]

    flag4 = data_NCN.region == 'United Kingdom'
    data_UK = data_NCN[flag4]
    data_NUK = data_NCN[~flag4]

    flag5 = data_NCN.region == 'Germany'
    data_GM = data_NCN[flag5]
    data_OT = data_NCN[~flag5]

    x1_train = data_USA[choose].values
    y1_train = data_USA['score'].values
    x2_train = data_CN[choose].values
    y2_train = data_CN['score'].values
    x3_train = data_JP[choose].values
    y3_train = data_JP['score'].values
    x4_train = data_UK[choose].values
    y4_train = data_UK['score'].values
    x5_train = data_GM[choose].values
    y5_train = data_GM['score'].values
    x6_train = data_OT[choose].values
    y6_train = data_OT['score'].values

    x_test = test_df[choose].values
    y_test = test_df['score'].values
    x_flag = (test_df['region'] == 'USA').values

    lr1 = LinearRegression()
    lr2 = LinearRegression()
    lr3 = LinearRegression()
    lr4 = LinearRegression()
    lr5 = LinearRegression()
    lr6 = LinearRegression()

    lr1.fit(x1_train, y1_train)
    lr2.fit(x2_train, y2_train)
    lr3.fit(x3_train, y3_train)
    lr4.fit(x4_train, y4_train)
    lr5.fit(x5_train, y5_train)
    lr6.fit(x6_train, y6_train)

    RMSE = 0
    for sample, label, flag in zip(x_test, y_test, x_flag):
        if flag == 'USA':
            y_hat = lr1.predict(sample.reshape(1, -1))
        elif flag == 'China':
            y_hat = lr2.predict(sample.reshape(1, -1))
        elif flag == 'Japan':
            y_hat = lr3.predict(sample.reshape(1, -1))
        elif flag == 'United Kingdom':
            y_hat = lr4.predict(sample.reshape(1, -1))
        elif flag == 'Germany':
            y_hat = lr5.predict(sample.reshape(1, -1))
        else:
            y_hat = lr6.predict(sample.reshape(1, -1))
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