# Bagging： Bootstrap aggregating
from joblib import Parallel, delayed
from random import random
import numpy as np
import multiprocessing


class Bagging:

    def __init__(self, base_algorithm, iterations, bootstrap_fraction):
        self.classifier = base_algorithm  # Class Name
        self.T = iterations
        self.F = bootstrap_fraction
        self.ensemble = []

    def fit(self, sample, label, **kwargs):
        for i in range(self.T):
            # bootstrap
            sample_i = []
            label_i = []
            for j in range(int(len(label) * self.F + 0.5)):  # 不可以用 sample 的 len
                id = int(random() * len(label)) - 1
                sample_i.append(sample[id])
                label_i.append(label[id])
                # sample_i.append(np.random.choice(sample, size=sample.shape[1], replace=True))
            # train classifier
            new_classifier = self.classifier(**kwargs)
            new_classifier.fit(sample_i, label_i)
            # add classifier to ensemble
            self.ensemble.append(new_classifier)

    def predict(self, instance):
        # evaluate ensemble on instance
        res = []
        for classifier in self.ensemble:
            res.append(classifier.predict(instance))
        # obtain total vote of each class
        return np.mean(res)  # as probability of being helpful
        # output 1 == the model "votes up"
        # output 0 == the model "votes down"
        # sklearn 有个 predict_prob 函数也许可以用？


# class BaggingMul(Bagging):
#
#     def fit(self, sample, label, **kwargs):
#
#         num_cores = multiprocessing.cpu_count()
#         self.ensemble = Parallel(n_jobs=num_cores)\
#             (delayed(bootstrap)(sample, label, self.classifier, self.F, **kwargs) for i in range(self.T))
#
#
def bootstrap(sample, label, classifier, F, **kwargs):
    # bootstrap
    sample_i = []
    label_i = []
    for j in range(int(len(label) * F + 0.5)):  # 不可以用 sample 的 len
        id = int(random() * len(label)) - 1
        sample_i.append(sample[id])
        label_i.append(label[id])
        # sample_i.append(np.random.choice(sample, size=sample.shape[1], replace=True))
    # train classifier
    new_classifier = classifier(**kwargs)
    new_classifier.fit(sample_i, label_i)
    # add classifier to ensemble
    return new_classifier

class BaggingMul(Bagging):

    def fit(self, sample, label, **kwargs):
        def bootstrap(_):
            # bootstrap
            sample_i = []
            label_i = []
            for j in range(int(len(label) * self.F + 0.5)):  # 不可以用 sample 的 len
                id = int(random() * len(label)) - 1
                sample_i.append(sample[id])
                label_i.append(label[id])
                # sample_i.append(np.random.choice(sample, size=sample.shape[1], replace=True))
            # train classifier
            new_classifier = self.classifier(**kwargs)
            new_classifier.fit(sample_i, label_i)
            # add classifier to ensemble
            return new_classifier

        pool = multiprocessing.Pool()
        self.ensemble = pool.map(bootstrap, range(self.T))
        pool.close()
        pool.join()
