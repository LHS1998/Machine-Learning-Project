import pandas as pd
from tools import *  # 文件读取工具

train_label = read_label('./data/train.csv')
validate_label = read_label('./data/dev.csv')
test_label = read_name('./data/test.csv')
user_seq = read_seq('./data/level_seq.csv', tokenizer=seq_parser)
level_meta = read_seq('./data/level_meta.csv', tokenizer=meta_parser)

# 数据整合：
# (1) 聚合同一用户的所有序列信息，划分训练集、验证集、测试集，匹配标签
train_seq = {}
validate_seq = {}
test_seq = {}
not_found_seq = set()
for seq in user_seq:
    if seq.user_id in train_label:  # 判断所在集合
        handler = train_seq
    elif seq.user_id in validate_label:
        handler = validate_seq
    elif seq.user_id in test_label:
        handler = test_seq
    else:
        if seq.user_id in not_found_seq:
            continue
        else:
            not_found_seq.add(seq.user_id)
            print("User %d not found in label file." % seq.user_id)
            continue
    handler.setdefault(seq.user_id, []).append(seq)  # 用列表保存同一用户的若干游玩记录

# (2) 提取用户的序列信息, 匹配关卡信息（难度，通过率，耗时）做差整合信息
train_features = []
for user in train_seq:
    train_features.append(User(train_seq[user], level_meta, train_label[user]))
to_csv("data_processed/train.csv", train_features)

validate_features = []
for user in validate_seq:
    validate_features.append(User(validate_seq[user], level_meta, validate_label[user]))
to_csv("data_processed/validate.csv", validate_features)

test_features = []
for user in test_seq:
    test_features.append(User(test_seq[user], level_meta, user))  # 测试集以用户 id 作为标签，便于输出结果
to_csv("data_processed/test.csv", test_features)

all_for_test = train_features + validate_features
to_csv("data_processed/all_train.csv", all_for_test)
