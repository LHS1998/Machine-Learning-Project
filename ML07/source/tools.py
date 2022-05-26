from collections import namedtuple
from dateutil import parser
import numpy as np


def read_label(path, sep='\t'):
    labels = {}
    with open(path, mode='rt') as f:
        f.readline()  # 表头
        while True:
            line = f.readline()
            if not line:
                break
            info = line.split(sep)
            labels[int(info[0])] = int(info[1])
    return labels


def read_name(path):
    with open(path, mode='rt') as f:
        f.readline()  # 表头
        line = f.read()
        line = line.rstrip('\n')  # 去除末尾一行 \n
    return set([int(id_) for id_ in line.split('\n')])


def read_seq(path, sep='\t', tokenizer=(lambda x: x)):
    """
    提取 csv 信息
    :param path: csv 路径
    :param sep: csv 分隔符
    :param tokenizer: callable，处理行数据，接收以列表传入的一行信息，返回同维度的处理后的信息
    :return: 记录（作为具名元组）的列表
    """
    seqs = []
    with open(path, mode='rt') as f:
        header = f.readline()  # 表头
        Record = namedtuple('Record', header)
        while True:
            line = f.readline()
            if not line:
                break
            line = line.rstrip('\n')
            row = line.split(sep)
            row = tokenizer(row)
            seqs.append(Record(*row))
    return seqs


def seq_parser(row):
    """
    提取 './data/level_seq.csv' 中各列信息
    """
    return int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4]), int(row[5]), parser.isoparse(row[6])


def meta_parser(row):
    """
    提取 './data/level_meta.csv' 中各列信息
    """
    return float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])


class User:

    def __init__(self, user_seq, levels, label):
        max_level = max([level.level_id for level in levels])
        # 个人的游玩历史
        self.tries = len(user_seq)  # 游玩次数
        level_process = [seq.level_id for seq in user_seq]  # 游玩的关卡序列
        self.process = max(level_process) / max_level  # 游玩进度
        self.history = min(level_process) / max_level  # 游玩了几关
        self.success_rate = sum([seq.f_success for seq in user_seq]) / self.tries  # 通关比例
        self.time_spent = sum([seq.f_duration for seq in user_seq])  # 总游戏时间
        self.help_used = sum([seq.f_help for seq in user_seq])  # 道具使用总数
        self.time_stamp = [0] * 24  # 统计几点玩游戏
        for seq in user_seq:
            self.time_stamp[seq.time.hour] += 1
        self.most_played_hour = np.argmax(self.time_stamp)  # 常在什么时候玩游戏（具体到小时）
        rest_step = []  # 统计成功时剩余步数
        for seq in user_seq:
            if seq.f_success:
                rest_step.append(seq.f_reststep)
        if rest_step:  # 有一个样例没有赢过，np.mean([]) 返回 NaN，这个后期还不好检查出来，必须做判断
            self.mean_rest_step = np.mean(rest_step)  # 平均通关时剩余步数
        else:
            self.mean_rest_step = 0
        self.last_win = sorted(user_seq, key=(lambda seq: seq.time), reverse=True)[0].f_success
        # assert self.history * self.success_rate == self.tries  # 不一定是每关只通一次

        # 相对的游玩水平
        metadata = [levels[level-1] for level in level_process]  # 所游玩关卡的参数, 保留重复，自动加权
        self.excess_success_rate = self.success_rate - np.mean([meta.f_avg_passrate for meta in metadata])  # 超额胜率
        self.excess_tries = self.tries - sum([meta.f_avg_retrytimes for meta in metadata])  # 超额游玩次数
        time_baseline = 0
        for seq, meta in zip(user_seq, metadata):  # 计算基准的总游玩时间
            if seq.f_success:
                time_baseline += meta.f_avg_win_duration
            else:
                time_baseline += meta.f_avg_duration
        self.excess_time = self.time_spent - time_baseline  # 超额游玩时间
        self.label = label

    def __repr__(self):
        features = [self.tries, "%.5f" % self.process, "%.5f" % self.history, "%.5f" % self.success_rate,
                    self.time_spent, self.help_used, *self.time_stamp, self.most_played_hour,
                    "%.5f" % self.mean_rest_step, self.last_win, "%.5f" % self.excess_success_rate,
                    "%.5f" % self.excess_tries, "%.5f" % self.excess_time]
        features = [str(item) for item in features]
        return ", ".join(features) + '\n'


CSV_HEADER = 'tries, process, history, success_rate, time_spent, help_used, '
for i in range(24):
    CSV_HEADER += 'time_stamp_%d, ' % i
CSV_HEADER += 'most_played_hour, mean_rest_step, last_win, excess_success_rate, excess_tries, excess_time\n'


def to_csv(path, objects):
    path = path.split('.')[0]
    with open(path + '_data.csv', mode='wt') as f:
        f.write(CSV_HEADER)
        for obj in objects:
            f.write(repr(obj))
    with open(path + '_label.csv', mode='wt') as f:
        f.write("label\n")
        for obj in objects:
            f.write(str(obj.label) + "\n")


if __name__ == '__main__':
    # train_label = read_label('./data/train.csv')
    # print(10932 in train_label)

    # seq_meta = read_seq('./data/level_seq.csv', tokenizer=seq_parser)

    # level_meta = read_seq('./data/level_meta.csv', tokenizer=meta_parser)
    # meta_tokens = {record.level_id: record for record in level_meta}

    # unknown_level = set()
    print("Test Finished")
