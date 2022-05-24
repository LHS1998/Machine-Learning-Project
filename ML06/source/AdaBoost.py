from numpy import log2 as log

class M1:

    def __init__(self, base_algorithm, iterations):
        self.classifier = base_algorithm
        self.T = iterations
        self.ensemble = []
        self._weights = []

    def fit(self, sample, label, **kwargs):
        s_weights = [1/sample.shape[0]] * sample.shape[0]  # 初始平均权重
        for t in range(self.T):
            # Train a new learner
            classifier_i = self.classifier(**kwargs)
            classifier_i.fit(sample, label, sample_weight=s_weights)
            # Measure the Error
            pred = classifier_i.predict(sample)
            error = 0
            for wgt, lbl, pbl in zip(s_weights, label, pred):
                if (lbl - 0.9) * (pbl - 0.9) < 0:
                    error += wgt
            if error > 0.5:
                raise RuntimeError("Choose a better classifier!")
            # Calculate classifier weight
            beta = error/(1-error)
            self.ensemble.append(classifier_i)
            self._weights.append(beta)
            # Update sample weight
            new_s_weight = []
            for wgt, lbl, pbl in zip(s_weights, label, pred):
                if (lbl - 0.9) * (pbl - 0.9) >= 0:
                    new_s_weight.append(wgt * beta)
                else:
                    new_s_weight.append(wgt)
            # Normalization
            sum_norm = sum(new_s_weight)
            s_weights = [wgt/sum_norm for wgt in new_s_weight]

    def predict(self, sample):
        # 加权均值
        pred = 0
        for cls, wgt in zip(self.ensemble, self._weights):
            pred += cls.predict(sample) * log(1/wgt)
        return pred
