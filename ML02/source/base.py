import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate  # 划分数据集函数
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_df = pd.read_csv('./cwurData.csv')  # 读入 csv 文件为 pandas 的 DataFrame
data_df = data_df.dropna()  # 舍去包含 NaN 的 row

feature_cols = ['quality_of_faculty', 'publications', 'citations', 'alumni_employment',
                'influence', 'quality_of_education', 'patents']
X = data_df[feature_cols].values
Y = data_df['score'].values

log = np.vectorize(lambda x: np.log(x))
X_log = log(X)

total_RMSE = []
REPEAT_TIMES = 10000

for i in range(0, REPEAT_TIMES):
    x_train, x_test, y_train, y_test = train_test_split(X_log, Y, test_size=0.2)

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    RMSE = 0
    for sample, label in zip(x_test, y_test):
        y_hat = lr.predict(sample.reshape(1, -1))
        RMSE += pow(label - y_hat, 2)

    RMSE = np.sqrt(RMSE / len(y_test))
    total_RMSE.append(RMSE[0])

mean = sum(total_RMSE) / REPEAT_TIMES
print(mean)

# with open('RMSE_log.csv', mode='wt') as f:
#     f.write(str(total_RMSE))

plt.xlim([min(total_RMSE) - 0.5, max(total_RMSE) + 0.5])
plt.hist(total_RMSE, bins=30)
plt.title('Test Set RMSE Frequency')
plt.xlabel('RMSE')
plt.ylabel('freq.')
plt.savefig('RMSE_no_br_log_hist.pdf')
