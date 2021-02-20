import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, multivariate_normal
from numpy.random import normal
from math import sqrt

m = 0.5
c = -0.3
sd = 0.2

np.random.seed(47)

x = np.linspace(-1, 1, 10)
y = []
for i in x:
    y.append(m*i + c + normal(0, sd))

plt.scatter(x, y)
plt.plot(x, m*x + c)
plt.savefig("dist.png")
plt.show()

def gaussian(x, y, w0, w1):
    k = sqrt(np.pi*2) * sd
    return k * np.exp(-0.5 * ((y - (w0 + w1*x))/sd)**2)

def posterior_sample(s0, m0, phi, beta, t):
    s0Inv = np.linalg.inv(s0)
    Sn = np.linalg.inv(s0Inv + beta * phi.T@phi)
    inner_term = s0Inv@(m0.reshape((2, 1))) + beta * phi.T * t
    Mn = Sn@inner_term
    return Mn.reshape((1, 2))[0], Sn

w0, w1 = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))
prior = multivariate_normal([0, 0], [[m, 0], [0, m]])
prior = prior.pdf(np.dstack((w0, w1)))
plt.contourf(w0, w1, prior, cmap = "jet")
plt.savefig("fig1.png")
plt.show()

axis = 0
beta = 25
m0, s0 = np.array([0, 0]), np.array([[m, 0], [0, m]])
fig, axs = plt.subplots(10, 3, figsize = (10, 40))
X = []
Y = []

for i, j in zip(x, y):
    mle = gaussian(i, j, w0, w1)
    posterior = prior * mle
    prior = posterior
    phi = np.array([1, i]).reshape((1, 2))
    m0, s0 = posterior_sample(s0, m0, phi, beta, j)
    dist = multivariate_normal.rvs(m0, s0, 10)
    X.append(i)
    Y.append(j)
    axs[axis, 0].contourf(w0, w1, mle, cmap="jet")
    axs[axis, 1].contourf(w0, w1, posterior, cmap="jet")
    for line in dist:
        axs[axis, 2].plot(x, line[0] + line[1] * x, c="blue")
        axs[axis, 2].plot(x, m0[0] + m0[1] * x, c="red")
    axs[axis, 2].scatter(X, Y)
    axis += 1
    fig.savefig("fig2.png")
    plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 3))
axes[0].contourf(w0, w1, mle, cmap="jet")
axes[0].scatter(c, m, c="black", marker="+")
axes[1].contourf(w0, w1, posterior, cmap="jet")
axes[1].scatter(c, m, c="black", marker="+")
for line in dist:
    axes[2].plot(x, line[0] + line[1] * x, c="blue")
    axes[2].plot(x, m0[0] + m0[1] * x, c="red")
axes[2].scatter(X, Y)
fig.savefig("fig3.png")
plt.show()
