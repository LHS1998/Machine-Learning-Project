RANDOM_SEED = 2020  # 固定随机种子

# 将舍去的特征列: 可以用数据计算出来的内容
drop_features = ['blueGoldDiff', 'redGoldDiff',
                 'blueExperienceDiff', 'redExperienceDiff',
                 'blueCSPerMin', 'redCSPerMin',
                 'blueGoldPerMin', 'redGoldPerMin']

# 选择各方法所处理的特征
discrete_features = ['blueWins',  # {0, 1}
                     'brFirstBlood',  # {-1, 1}
                     # 'blueEliteMonsters', 'redEliteMonsters', 'brEliteMonsters',  # {0, 1, 2}
                     'blueDragons', 'redDragons', 'brDragons',  # {0, 1}
                     'blueHeralds', 'redHeralds', 'brHeralds',  # {0, 1}
                     # 'blueTowersDestroyed', 'redTowersDestroyed', 'brTowersDestroyed',  # {0, 1, 2, 3, 4}
                     ]  # 具有少数几个可能取值的特征, 可以直接使用
multi_discrete_feature = ['blueEliteMonsters', 'redEliteMonsters', 'brEliteMonsters',  # {0, 1, 2}
                          'blueTowersDestroyed', 'redTowersDestroyed', 'brTowersDestroyed',  # {0, 1, 2, 3, 4}
                          ]
q_features = ['blueWardsPlaced', 'redWardsPlaced', 'brWardsPlaced',
              'blueWardsDestroyed', 'redWardsDestroyed', 'brWardsDestroyed',
              'blueTotalMinionsKilled', 'redTotalMinionsKilled', 'brTotalMinionsKilled',
              'blueTotalJungleMinionsKilled', 'redTotalJungleMinionsKilled', 'brTotalJungleMinionsKilled',
              'blueKills', 'redKills', 'brKills',
              'blueDeaths', 'redDeaths', 'brDeaths',
              'blueAssists', 'redAssists', 'brAssists',
              'blueTotalGold', 'redTotalGold', 'brTotalGold',
              'blueAvgLevel', 'redAvgLevel', 'brAvgLevel',
              'blueTotalExperience', 'redTotalExperience', 'brTotalExperience',
              ]  # 连续分布的特征, 考虑采用分位数离散化
difference_features = ['brEliteMonsters', 'brDragons', 'brHeralds', 'brTowersDestroyed',
                       'brWardsPlaced', 'brWardsDestroyed', 'brTotalMinionsKilled',
                       'brTotalJungleMinionsKilled', 'brKills', 'brDeaths', 'brAssists',
                       'brTotalGold', 'brAvgLevel', 'brTotalExperience']

QUANTILES = [.05, .25, .5, .75, .95]  # 分位点
