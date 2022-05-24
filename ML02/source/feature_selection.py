import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate  # 划分数据集函数
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm

data_df = pd.read_csv('./cwurData.csv')  # 读入 csv 文件为 pandas 的 DataFrame
data_df = data_df.dropna()  # 舍去包含 NaN 的 row

# 划分训练集与测试集
train_df = data_df.sample(frac=0.8)
test_df = data_df.drop(train_df.index)

# 在训练集中划出验证集
_train_df = train_df.sample(frac=0.8)
_validation_df = train_df.drop(_train_df.index)

# 获取标签: 标签是全程不需要特殊处理的
Y_train = _train_df["score"]
Y_val = _validation_df["score"].values
Y_test = test_df["score"].values

# 生成对数项
feature_cols = ['quality_of_faculty', 'publications', 'citations', 'alumni_employment',
                'influence', 'quality_of_education', 'broad_impact', 'patents']
for column in feature_cols:
    _train_df['log_' + column] = np.log2(_train_df[column])
    _validation_df['log_' + column] = np.log2(_validation_df[column])
    test_df['log_' + column] = np.log2(test_df[column])


def cal_RMSE(coef, X, Y):  # 透过系数计算 RMSE, 因为特征选取不一定一致, 需要传入 X 和 Y
    RMSE = 0
    for row, label in zip(X, Y):
        x_use = np.insert(row, 0, 1)  # 加常数项
        hat_y = np.dot(x_use, coef)
        RMSE += pow(label-hat_y, 2)

    RMSE = np.sqrt(RMSE / len(Y))
    # print(RMSE)
    return RMSE


def _evaluate(col):  # Y = w·X[col]

    X = _train_df[col]  # 获取训练数据
    X = sm.add_constant(X)  # 加入常数项
    M = sm.OLS(Y_train, X)
    E = M.fit()
    X_val = _validation_df[col].values

    # print(E.summary())
    return E, cal_RMSE(E.params, X_val, Y_val)


def extract(col, est):
    coefficients = {}
    for i in range(0, len(col)):
        feature = col[i]
        coef = est.params[i]
        ste = est.bse[i]
        t = est.tvalues[i]
        p = est.pvalues[i]
        conf = est.conf_int(0.05)[i]
        coefficients[feature] = (coef, ste, t, p, conf)


model_base = ['const',
              'log_quality_of_faculty', 'log_alumni_employment',
              'log_quality_of_education', 'log_broad_impact', 'log_patents',
              'publications', 'citations', 'influence',
              'log_publications', 'log_citations', 'log_influence']  # 当前全体特征

good_for_now = ['log_quality_of_faculty', 'log_alumni_employment',
                'log_quality_of_education', 'log_broad_impact', 'log_patents']

fea_to_test = ['publications', 'citations', 'influence']

controls = ['Adj. R^2', 'F-stat', 'RMSE']


def print_sample(order, col, est, RMSE):
    lines = []
    # 标题：______(1)______
    if order < 10:
        title = (" " * 6) + "(" + str(order) + ")" + (" " * 6)
    else:
        title = (" " * 5) + "(" + str(order) + ")" + (" " * 6)
    lines.append(title)

    # const
    lines.append('-' * 15)
    num_part = '%10.3f' % est.params[0]
    if est.pvalues[0] < 0.01:
        star_part = '***  '
    elif est.pvalues[0] < 0.05:
        star_part = '**   '
    elif est.pvalues[0] < 0.1:
        star_part = '*    '
    else:
        star_part = '     '
    lines.append(num_part + star_part)
    se = '%.3f' % est.bse[0]
    se_part = (' ' * (8 - len(se))) + '(' + se + ')' + (' ' * 5)
    lines.append(se_part)

    # 其他特征
    for feature in model_base[1:]:
        lines.append('-' * 15)
        if feature in col:  # 被模型选择
            index = col.index(feature) + 1  # 出现顺序
            num_part = '%10.4f' % est.params[index]
            if est.pvalues[index] < 0.01:
                star_part = '***  '
            elif est.pvalues[index] < 0.05:
                star_part = '**   '
            elif est.pvalues[index] < 0.1:
                star_part = '*    '
            else:
                star_part = '     '
            lines.append(num_part + star_part)
            se = '%.3f' % est.bse[index]
            se_part = (' ' * (8 - len(se))) + '(' + se + ')' + (' ' * 5)
            lines.append(se_part)
        else:  # 被跳过的特征
            lines.append((' ' * 7) + '-' + (' ' * 7))
            lines.append(' ' * 15)

    lines.append('-' * 15)
    lines.append('%10.4f' % est.rsquared_adj + '     ')
    lines.append('%10.4f' % est.fvalue + '     ')
    lines.append('%10.4f' % RMSE + '     ')

    return lines


def print_head():
    field_width = max([len(name) for name in model_base]) + 3
    lines = [' ' * field_width]
    for name in model_base:
        lines.append('-' * field_width)
        lines.append(name + (' ' * (field_width - len(name))))
        lines.append(' ' * field_width)
    lines.append('-' * field_width)
    for name in controls:
        lines.append(name + (' ' * (field_width - len(name))))

    return lines


targets = [
    good_for_now,
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'publications'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'citations'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'publications', 'citations'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'citations', 'influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'publications', 'influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'publications', 'citations', 'influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'log_publications'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'log_citations'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'log_influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'log_publications', 'log_citations'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'log_publications', 'log_influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'log_citations', 'log_influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'log_publications', 'log_citations', 'log_influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'publications', 'log_citations'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'publications', 'log_influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'citations', 'log_influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'log_citations', 'influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'publications', 'log_citations', 'log_influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'log_publications', 'citations', 'log_influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'log_publications', 'log_citations', 'influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'publications', 'log_publications'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'citations', 'log_citations'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'influence', 'log_influence'],
    ['log_quality_of_faculty', 'log_alumni_employment', 'log_quality_of_education', 'log_broad_impact', 'log_patents',
     'publications', 'citations', 'influence', 'log_publications', 'log_citations', 'log_influence'],
]

order = 1
results = [print_head()]
for col in targets:
    est, rmse = _evaluate(col)
    results.append(print_sample(order, col, est, rmse))
    order += 1

line_num = len(results[0])
text = [''] * line_num
for line_no in range(0, line_num):
    for col in results:
        text[line_no] += col[line_no]

with open('feature_selection.txt', mode='wt') as f:
    for line in text:
        f.write(line + '\n')
    f.write("\nNote: Standard error in the parentheses.")
