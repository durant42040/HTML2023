import numpy as np
from libsvm.svm import *
from libsvm.svmutil import *
import pandas as pd
import time

from libsvm.svmutil import svm_train, svm_predict

start = time.time()
data = pd.read_csv('hw5_train.tr', sep=' ', header=None)


def get_data(D):
    y = [1 if i == 7 else -1 for i in D[0].values.tolist()]
    x = [row.to_dict() for _, row in D.drop(0, axis=1).iterrows()]
    return y, x


def data_split(D):
    test_indices = np.random.choice(D.index, size=200, replace=False)

    Dtest = D.loc[test_indices]
    Dtrain = D.drop(test_indices)

    ytrain, xtrain = get_data(Dtrain)
    ytest, xtest = get_data(Dtest)
    return xtrain, ytrain, xtest, ytest


def SVM(y, x, g):
    prob = svm_problem(y, x)
    param = svm_parameter(f'-s 0 -t 2 -c 0.1 -g {g} -q -m 2000')
    return svm_train(prob, param)


gamma_values = [0.1, 1, 10, 100, 1000]
N = {0.1: 0, 1: 0, 10: 0, 100: 0, 1000: 0}

for _ in range(500):
    Xtrain, ytrain, Xval, yval = data_split(data)
    accuracy = [svm_predict(yval, Xval, SVM(ytrain, Xtrain, g))[1][0] for g in gamma_values]
    print(accuracy)
    n = np.argmax(accuracy)
    print(n)
    N[gamma_values[n]] += 1
    print(N)



print(N)

end = time.time()
print(f'time: {end - start}s')
