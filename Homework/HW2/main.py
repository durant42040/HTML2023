import numpy as np


def Ein(s, x, y, theta):
    if theta == float('-inf'):
        return np.count_nonzero(y == -s) / len(y)
    h = np.sign(s * (x - theta))
    return np.sum(h != y) / len(y)


def DecisionStump(x, y):
    thetas = [float('-inf')] + [(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)]

    minE = float('inf')
    minS = None
    minTheta = None

    for t in thetas:
        for s in [-1, 1]:
            E = Ein(s, x, y, t)
            if E < minE:
                minE = E
                minS = s
                minTheta = t
            if E == minE and s * t < minS * minTheta:
                minE = E
                minS = s
                minTheta = t

    return [minE, minS, minTheta]


data = np.loadtxt("hw2_train.dat")
test = np.loadtxt("hw2_test.dat")

x = np.transpose(data)[:-1]
y = np.transpose(data)[-1]

g = np.array([DecisionStump(xi, y) for xi in x])

b = np.argmax(g[:, 0])
k = np.argmin(g[:, 0])

Eoutb = Ein(g[b][1], np.transpose(test)[b], np.transpose(test)[-1], g[b][2])
Eout = Ein(g[k][1], np.transpose(test)[k], np.transpose(test)[-1], g[k][2])

print(Eoutb - Eout)  # 0.34375
