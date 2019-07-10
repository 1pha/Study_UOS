import numpy as np


# Mean Squared Error(MSE)
def mean_sqaured_error(y, t):
    if type(y) == list: y = np.array(y)
    if type(t) == list: t = np.array(t)
    return 0.5 * np.sum((y-t) ** 2)

# Cross Entropy Error(CEE)
def cross_entropy_error(y, t, delta=1e-7):
    if type(y) == list: y = np.array(y)
    if type(t) == list: t = np.array(t)
    return -np.sum(t * np.log(y + delta))


if __name__=="__main__":
    t = [0, 0, 1, 0, 0]
    y = [0, 0.05, 0.9, 0.05, 0]
    y2 = [0.25, 0.25, 0.0, 0.25, 0.25]

    print(mean_sqaured_error(y, t))
    print(mean_sqaured_error(y2, t))
    print(cross_entropy_error(y, t))
    print(cross_entropy_error(y2, t))