import numpy as np
from matplotlib import pyplot as plt

def get_slp_intr(x, y):
    n = len(x)

    x_mean = sum(x)/n
    y_mean = sum(y)/n

    num = 0
    denom = 0
    for i in range(n):
        num += (x[i] - x_mean)*(y[i] - y_mean)
        denom += (x[i] - x_mean)**2

    slope = num/denom
    intercept = y_mean - slope*x_mean
    return slope, intercept

x = [1, 2, 3]
y = [1.2, 1.9, 3.2]
plt.plot(x, y, 'bo')
plt.title("Distribution")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("distribution")

m, c = get_slp_intr(x, y)

x = np.array(x)
y = np.array(y)
plt.plot(x, m*x + c)
plt.plot(x, y, 'bo')
plt.title("Best fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("best_fit")