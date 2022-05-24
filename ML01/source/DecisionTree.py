from config import *
import numpy as np
from pandas import Interval
from random import random
from math import log2


class DTNode:

    def __init__(self, mode, branch, label=0, var=0, varname='', level=0):
        """
        A tree node for class DecisionTree.

        :param mode: 'node' or 'leaf'. 'root' is a 'node'.
        :param branch: feature_type.
        :param label: 'leaf' only, class of this leaf.
        :param var: feature.
        :param level:
        """
        if mode == 'leaf':
            self.end = True  # Leaf Node
            self.label = label
            self.branch = branch
            self._level = level
        else:
            self.variable = var  # 用于判断本层节点的变量序号
            self.varname = varname  # 用于判断本层节点的变量名称
            self.branch = branch  # 来自上层节点的类别信息
            self.children = []  # 孩子节点的列表 [DTNode]
            self._level = level
            self.end = False

    def __str__(self):
        indent = '  ->' * self._level
        if self.end:
            return " ".join([indent, 'WHEN', self.branch[0], "EQUALS", str(self.branch[1]),
                             'REACHES LEAF, CLASS=', str(self.label), '\n'])
        this_layer = " ".join([indent, 'WHEN', self.branch[0], "EQUALS", str(self.branch[1]),
                               "SELECT", self.varname, '\n'])
        for child in self.children:
            this_layer += str(child)
        return this_layer


class DecisionTree(object):

    def __init__(self, classes: list, features: list,
                 max_depth=10, min_samples_split=10, min_info_gain=0.05,
                 impurity_t='entropy'):
        """
        Set up the details of the problem.

        :param classes: a list containing names for all classes.
        :param features: a list containing names of all features.
        :param max_depth: the maximum depth of decision tree.
        :param min_samples_split: the minimum number of instances in a node.
        :param impurity_t: the way to compute impurity.
        """
        self.classes = classes  # 模型的分类，如 [0, 1]
        self.feature_name = features  # 每个特征的名字
        self.max_depth = min(len(features), max_depth)  # 预剪枝：决策树时的最大深度
        self.min_samples_split = min_samples_split  # 预剪枝：到达该节点的样本数小于该值则不再分裂
        self.min_info_gain = min_info_gain
        self.impurity_t = impurity_t  # 计算混杂度（不纯度）的计算方式，例如 entropy 或 gini
        self.root = None  # 定义根节点，未训练时为空
        self.data = None  # 缓存数据句柄
        self.label = None  # 缓存数据句柄
        self.feature_unique = []  # 每个属性的取值空间，与 features 相对应
        self.info_split = []  # 每个属性的 SplitInformation，与 features 相对应
        self.mask = [True for _ in range(0, len(features))]  # 记录哪些特征已经被选择, 这样 count 只需要算一遍

    def impurity(self, mix, mode=None):
        """
        Calculate impurity by means which self.impurity_t specified.

        :param mix: [number of class_A instances, ...]
        :param mode: if to follow the self impurity
        :return: impurity value
        """
        total = sum(mix)
        if total == 0:
            return np.inf
        pars = [ele / total for ele in mix]  # 频率，Python 2 可能会出问题

        if mode:
            im = mode
        else:
            im = self.impurity_t

        if im == 'entropy':
            ent = 0
            for par in pars:
                if par == 0:
                    continue  # 约定 log2(0) = 0
                ent += par * log2(par)
            return -ent
        if im == 'gini':
            ent = 0
            for par in pars:
                ent += par * par
            return 1-ent

    def expand_node(self, mask, sample, depth=0, impur=None, branch=('root', ''), p_mix=1):
        """
        Recursively build the decision tree.

        :param mask: a list of bool with same length of self.feature_name, True if the feature is not selected.
        :param sample: a set of index, in the set if the sample arrives this node.
        :param depth:
        :param impur: impurity for current node, only need to be calculated when root.
        :param branch: the feature selected in last level, together with feature_type specified to build this branch
        :param p_mix: helps predict when no sample survive to leaf
        :return: Node handler
        """

        # 初始节点要计算当前混杂度: 只需要在初始节点计算！
        if not impur:
            mix_root = [0] * len(self.classes)
            for sam in self.label:
                mix_root[self.classes.index(sam)] += 1
            impur = self.impurity(mix_root)

        # 1. 递归基
        if impur - 0 < .00000000001:  # 纯洁，无需分裂
            return DTNode('leaf', branch, level=depth, label=self.label[sample.pop()])  # 以随意一个样本为叶的类别

        if (len(sample) <= self.min_samples_split) | (depth >= self.max_depth):  # 层数达到分裂阈值 或 到达该节点的数据太少
            mix = [0] * len(self.classes)
            for item in range(0, len(self.label)):
                if item in sample:
                    mix[self.classes.index(self.label[item])] += 1
            return DTNode('leaf', branch, level=depth,
                          label=self.classes[np.argmax(mix)] if sum(mix) != 0 else p_mix)  # 少数服从多数

        # 取 self.max_depth 为输入值和特征数的最小值隐含地包括了没有多余特征可选这一递归基

        # 2. 找到最佳分裂特征，递归调用 expand_node
        # (1) 计算各特征的信息增益
        fea_sieve = {}  # 用于存储 可选特征(index)：[gain, [每 _type 的起始 impurity(减少可能的计算)], [sample_i]]
        for _fea in range(0, len(self.feature_name)):  # 每个特征
            if mask[_fea]:  # 若可供本节点筛选
                # a. 初始化容器
                mix_i = []  # 保存混杂度计算数据
                for i in range(0, len(self.feature_unique[_fea])):  # 循环赋值避免浅拷贝问题
                    mix_i.append([0] * len(self.classes))  # mix_i[_type][_class] 为该特征分类下的某类实例数目
                sample_i = []  # 保存每个节点留下的样本编号
                for i in range(0, len(self.feature_unique[_fea])):  # 循环赋值避免浅拷贝问题
                    sample_i.append(set())  # sample_i[_type] 为分流到该特征分类下子节点的所有样本
                # b. 查找，分流
                for index in range(0, self.data.shape[0]):
                    if index in sample:  # 该样本之前没被筛掉
                        _class = self.classes.index(self.label[index])  # 该样本类别在 mix_i 中的 offset
                        _type = self.feature_unique[_fea].index(self.data[index][_fea])  # 该样本特征类型在 mix_i 中的 offset
                        mix_i[_type][_class] += 1  # 统计相应类别数目
                        sample_i[_type].add(index)  # 将该样本加入节点留下的样本中
                # c. 计算混杂度及 IG
                ig = impur
                s_all = len(sample)  # 该节点下样本总数
                child_imp = []
                for _set, _type in zip(sample_i, mix_i):  # 取出每个类型的混杂度数据 _type[_class]
                    imp = self.impurity(_type)  # _type 的混杂度
                    child_imp.append(imp)
                    ig -= imp * len(_set) / s_all
                ig /= self.info_split[_fea]  # GainRatio
                # d. 将信息加入字典
                fea_sieve[_fea] = [ig, child_imp, sample_i]
        # (2) 筛选最佳特征
        for index in range(0, len(mask)):
            if mask[index]:  # 取第一个没被筛掉的做 k_min
                k_min = index
                break
        for k in fea_sieve:
            if fea_sieve[k][0] > fea_sieve[k_min][0]:
                k_min = k
        # (3) 递归
        this = DTNode('node', branch, var=k_min, varname=self.feature_name[k_min], level=depth)
        new_mask = mask.copy()
        new_mask[k_min] = False
        mix_parent = [0] * len(self.classes)  # 为可能的空孩子节点统计本节点的类型
        for index in sample:
            mix_parent[self.classes.index(self.label[index])] += 1
        for _type in range(0, len(self.feature_unique[k_min])):  # 建立子树
            this.children.append(self.expand_node(new_mask.copy(),
                                                  fea_sieve[k_min][2][_type].copy(),
                                                  branch=(self.feature_name[k_min], self.feature_unique[k_min][_type]),
                                                  depth=depth+1,
                                                  p_mix=self.classes[np.argmax(mix_parent)],
                                                  impur=fea_sieve[k_min][1][_type]))  # 读取所保存的计算结果
        return this

        # 3. 找不到有用的分裂特征
        # 如果所有特征的信息增益都一样，那么默认是选第一个特征
        pass

    def traverse_node(self, current: DTNode, feature):
        # 递归基
        if current.end:  # 到达叶子节点
            return current.label
        # 深入子树
        label = feature[current.variable]  # 取出本节点用于分类的属性
        for index in range(0, len(self.feature_unique[current.variable])):  # 以次找子区间
            child = self.feature_unique[current.variable][index]
            if label == child:
                return self.traverse_node(current.children[index], feature)
        raise ValueError("Not Found!")

    def fit(self, feature: np.ndarray, label: np.ndarray):
        """
        Train the model.

        :param feature: train set features, i.e., np.ndarray in shape (n, m), which each row represents a sample
        :param label: train set labels, i.e., np.ndarray in shape (n, ), represent each sample's class
        :return: nothing, use the object later.
        """
        # 判断输入合法性
        assert len(self.feature_name) == len(feature[0])  # 输入数据的特征数目应该和模型定义时的特征数目相同

        # 缓存数据
        self.data = feature.copy()
        self.label = label.copy()

        # 计算特征的取值数目
        for counter in range(0, self.data.shape[1]):
            self.feature_unique.append(list(np.unique(self.data[:, counter])))

        # 计算 SplitInformation (为了计算 GainRatio)
        count = []
        for _f_num in range(0, len(self.feature_name)):  # 初始化
            count.append([])  # count[^] 这个 index 表示特征维度
            for _fe_type in self.feature_unique[_f_num]:  # 该特征的每种取值
                count[_f_num].append(0)  # count[fea][^] 值为表示特征取该值的样本数目, 初始化为 0
        for spl_i in self.data:  # 数每个特征取值的个数
            for _f_num in range(0, len(self.feature_name)):  # 每个特征
                fea_type_i = self.feature_unique[_f_num].index(spl_i[_f_num])  # 找到对应特征类型的 offset
                count[_f_num][fea_type_i] += 1  # 类别为 type_i 的第 _f_num 项特征取 fea_type_i 的样本 +1
        for _fea in count:  # 计算 SI
            self.info_split.append(self.impurity(_fea, mode='entropy'))

        # 建树
        self.root = self.expand_node([True] * len(self.feature_name),  # 初始节点可选择所有的特征
                                     set(range(0, self.data.shape[0])))  # 同时保有所有的样本

    def predict(self, feature: np.ndarray):
        """
        Make predictions by the model.

        :param feature: features for unknown input, as np.ndarray in shape (n, m), which each row represents a sample
        :return:
        """
        assert len(feature.shape) <= 2  # 只能是1维或2维

        if len(feature.shape) == 1:  # 如果是一个样本
            return self.traverse_node(self.root, feature)  # 从根节点开始路由
        return np.array([self.traverse_node(self.root, f) for f in feature])  # 如果是很多个样本

    def __str__(self):
        if self.root is None:
            return '<class DecisionTree: Untrained>'
        else:
            return str(self.root)
