import numpy as np

data = np.loadtxt('hw3_train.dat')
y = data[:, -1]
X = np.c_[np.ones((len(data[:, :-1]), 1)), data[:, :-1], *[data[:, :-1] ** i for i in range(2, 9)]]

test = np.loadtxt('hw3_test.dat')
ytest = test[:, -1]
Xtest = np.c_[np.ones((len(test[:, :-1]), 1)), test[:, :-1], *[test[:, :-1] ** i for i in range(2, 9)]]


def Ein(w):
    return np.mean((np.sign(X.dot(w)) * y) < 0)


def Eout(w):
    return np.mean((np.sign(Xtest.dot(w)) * ytest) < 0)


w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(abs(Ein(w) - Eout(w)))
