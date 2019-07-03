import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    y = x > 0 # numpy-boolean
    return y.astype(np.int) # boolean > 0, 1
    # return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def plot(x, y):
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


if __name__=="__main__":
    x = np.array([-1, 1, 2])
    print(step_function(x))
    x_domain = np.arange(-5, 5., 0.1)
    plot(x_domain, step_function(x_domain))
    plot(x_domain, sigmoid(x_domain))