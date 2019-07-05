import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class MultiLayerNet:

    def __init__(self, input_size, hidden_size: list, output_size, loss, weight_init_std=0.01):
        self.net = [input_size] + hidden_size + [output_size]
        self.weight = np.array([weight_init_std * np.random.randn(self.net[i], self.net[i + 1])
                                for i in range(len(self.net) - 1)])
        self.bias = np.array([np.zeros(self.net[i + 1]) for i in range(len(self.net) - 1)])
        self.loss = loss()

    def predict(self, x):
        x = np.array(x)
        for i, weight in enumerate(self.weight):
            if i == 0:
                self.naive = [np.dot(x, weight) + self.bias[i]]
                self.activation = list(sigmoid(np.array(self.naive)))
            elif i > 0:
                tmp = np.matmul(self.activation[i - 1], self.weight[i]) + self.bias[i]
                self.naive.append(tmp)
                self.activation.append(sigmoid(np.array(self.naive[i])))

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)


if __name__=="-__main__":
    mul = MultiLayerNet(5, [4, 4], 2, 'cee')
