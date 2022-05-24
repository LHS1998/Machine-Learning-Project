import numpy as np
import os
from PIL import Image
from random import shuffle
import math
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

current_path = os.path.dirname(__file__)
train_path = os.path.join(current_path, "data", "train")
tests_path = os.path.join(current_path, "data", "test")

train_dirs = sorted([dir for dir in os.listdir(train_path) if dir.isdigit()], key=int)  # 过滤掉 .DS_Store
tests_dirs = sorted([dir for dir in os.listdir(tests_path) if dir.isdigit()], key=int)

train = []  # 考虑到训练集得打乱, 先整体读入
for name in train_dirs:
    label = name
    working_dir = os.path.join(train_path, name)
    files = os.listdir(working_dir)
    for file in files:
        file_path = os.path.join(working_dir, file)
        img = Image.open(file_path)
        pixels = np.array(img)
        train.append((int(label), pixels.flatten()))
shuffle(train)
train_data = [data for _, data in train]
train_label = [label for label, _ in train]  # 这个集合就可以直接用于交叉验证

test_data = []
test_label = []
for name in tests_dirs:
    working_dir = os.path.join(tests_path, name)
    files = os.listdir(working_dir)
    for file in files:
        file_path = os.path.join(working_dir, file)
        img = Image.open(file_path)
        pixels = np.array(img).flatten()
        test_label.append(name)
        test_data.append(pixels)

all_x = train_data + test_data
pca = PCA(n_components=398)  # mle
reduced_x = pca.fit_transform(all_x)

train_data_ = reduced_x[:len(train_data)]
test_data_ = reduced_x[len(train_data):]


def gaussian_kernel(dis):
    sigma = 320
    func = np.vectorize(lambda d: math.exp(-pow(d/sigma, 2)))
    return func(dis)


ks = [1, 5, 10, 20, 50, 100, 120]
for k in ks:
    model = KNN(n_neighbors=k, metric="euclidean", weights=gaussian_kernel)
    model.fit(train_data_, train_label)
    prediction = model.predict(test_data_)
    acc = 0
    for pred, real in zip(prediction, test_label):
        if str(pred) == real:
            acc += 1
    acc /= len(test_label)
    print(k, acc)
