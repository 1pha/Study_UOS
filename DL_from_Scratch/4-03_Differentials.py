import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x, h=1e-4):
    return (f(x+h) - f(x-h)) / (2*h)


def square(x):
    return 0.01*x ** 2 + 0.1*x

x = np.arange(0, 20, .1)
y = square(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()