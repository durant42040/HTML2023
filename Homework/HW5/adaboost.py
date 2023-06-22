import numpy
import numpy as np
import pandas as pd


def get_data(D):
    D = D[(D.iloc[:, 0] == 11) | (D.iloc[:, 0] == 26)]
    y = np.array([1 if i == 11 else -1 for i in np.array(D[0].values.tolist())])
    x = np.array([row.to_list() for _, row in D.drop(0, axis=1).iterrows()])
    return y, x


class DecisionStump:
    def __init__(self):
        self.E = float('inf')
        self.s = None
        self.theta = None
        self.i = None

    def error(self, s, x, y, theta, w):
        if theta == float('-inf'):
            return np.count_nonzero(y == -s) / len(y)
        h = np.sign(s * (x - theta))
        return np.dot(w, (h != y))

    def fit(self, x, y):
        for j in range(len(x[0])):
            Xi = x[:, j]
            thetas = [float('-inf')] + list(set([(Xi[i] + Xi[i + 1]) / 2 for i in range(len(Xi) - 1)]))
            for t in thetas:
                for s in [-1, 1]:
                    E = self.error(s, Xi, y, t, u)
                    if E < self.E:
                        self.E = E
                        self.s = s
                        self.theta = t
                        self.i = j
                    if E == self.E and s * t < self.s * self.theta:
                        self.E = E
                        self.s = s
                        self.theta = t
                        self.i = j

        return self.E, self.s, self.i, self.theta

    def predict(self, x):
        Xi = x[:, self.i]
        return np.sign(self.s * (Xi - self.theta))


data = pd.read_csv('hw5_train.tr', sep=' ', header=None)
y, X = get_data(data)

u = np.ones(len(y)) / len(y)


def adaboost():
    model = DecisionStump()
    model.fit(X, y)
    h = model.predict(X)

    e = np.array([1 if y[n] != h[n] else 0 for n in range(len(y))])
    epsilon = np.dot(e, u) / np.sum(u)
    v = np.sqrt((1 - epsilon)/epsilon)
    alpha = np.log(v)

    for i in range(len(u)):
        if e[i] == 0:
            u[i] /= v
        else:
            u[i] *= v

    return alpha, h


G = 0
for _ in range(1000):
    a, g = adaboost()
    G += a*g

G = np.sign(G)

Ein = np.mean(y != G)
print(Ein)  # 0.0
