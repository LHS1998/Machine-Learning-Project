import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate  # 划分数据集函数
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

data_df = pd.read_csv('./cwurData.csv')  # 读入 csv 文件为 pandas 的 DataFrame
data_df = data_df.dropna()  # 舍去包含 NaN 的 row

feature_cols = ['quality_of_faculty', 'publications', 'citations', 'alumni_employment',
                'influence', 'quality_of_education', 'broad_impact', 'patents']
X = data_df[feature_cols].values
Y = data_df['score'].values

log = np.vectorize(lambda x: np.log(x))
X_log = log(X)

x_train, x_test, y_train, y_test = train_test_split(X_log, Y, test_size=0.2)

inv = sm.add_constant(x_train)  # independent variable
mu = sm.OLS(y_train, inv)
est = mu.fit()

print(est.summary())
