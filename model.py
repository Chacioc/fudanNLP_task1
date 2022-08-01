import numpy as np


class softmax_regression:
    def __init__(self, epochs=10, learning_rate=0.01, batch_size=1):
        self.batch_size = batch_size
        self.w = None
        self.num = None
        self.feature_num = None
        self.class_num = 5
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, sim, res, learning_rate, batch_size, epochs):
        self.__init__(epochs, learning_rate, batch_size)
        self.num = sim.shape[0]
        self.feature_num = sim.shape[1]
        self.w = np.zeros((self.feature_num, 5))
        res1 = np.zeros((res.shape[0], 5))
        for i in range(res.shape[0]):
            res1[i][res[i]] = 1
        for i in range(epochs):
            idx = np.random.permutation(np.arange(sim.shape[0]))
            p = 0
            while p < self.num:
                s = 0
                loss = np.zeros_like(self.w)
                while (s < self.batch_size) & (p + s < self.num):
                    idd = idx[p + s]
                    res2 = self.softmax(sim[idd])
                    loss += np.outer(sim[idd], res1[idd] - res2)
                    s += 1
                p += s
                self.w += (loss * self.learning_rate / self.batch_size)

    def softmax(self, x):
        y = np.exp(np.dot(x, self.w))
        return y / np.sum(y)

    def predict(self, x):
        y = self.softmax(x)
        return np.argmax(y)

