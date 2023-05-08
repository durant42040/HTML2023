from liblinear.liblinearutil import *
import numpy as np


def polynomialTransform(x):
    d = x.shape[1]
    cross = []
    for i in range(d):
        for j in range(i + 1, d):
            cross.append(x[:, i] * x[:, j])
            cross.append((x[:, i] ** 2) * (x[:, j] ** 2))
            cross.append(x[:, i] * (x[:, j] ** 2))
            cross.append((x[:, i] ** 2) * x[:, j])
            cross.append((x[:, i] ** 3) * x[:, j])
            cross.append(x[:, i] * (x[:, j] ** 3))

            for k in range(j + 1, d):
                cross.append(x[:, i] * x[:, j] * x[:, k])
                cross.append((x[:, i] ** 2) * x[:, j] * x[:, k])
                cross.append(x[:, i] * (x[:, j] ** 2) * x[:, k])
                cross.append(x[:, i] * x[:, j] * (x[:, k] ** 2))

                for l in range(k + 1, d):
                    cross.append(x[:, i] * x[:, j] * x[:, k] * x[:, l])
    return np.c_[np.ones((len(x), 1)), x, x ** 2, x ** 3, x ** 4, np.column_stack(cross)]


data = np.loadtxt("hw4_train.dat")

x = data[:, :-1]
y = data[:, -1]

phi = polynomialTransform(x)

# C = 1/(2lambda)
C = 1 / 2 * np.array([10 ** 6, 10 ** 3, 1, 10 ** -3, 10 ** -6])

prob = problem(y, phi)
option = '-s 0 -e 0.000001 -c ' + str(C[3])

m = train(prob, parameter(option))
w = np.array([m.get_decfun_coef(i) for i in range(phi.shape[1])])

print((np.abs(w) <= 10**-6).sum())  # 2
