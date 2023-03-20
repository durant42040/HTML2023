import numpy as np


def PLA(data):
    w = np.zeros(11)
    M = 0
    while M < 1024:
        n = np.random.randint(256)
        random_data = data[n]
        x = np.insert(random_data[0:10], 0, 0.1126)
        y = 1 if np.dot(x, w) >= 0 else -1
        if y * random_data[-1] < 0:
            M = 0
            w += x * random_data[-1]
        else:
            M += 1

    return 0.1126 * w[0]


data = np.loadtxt("hw1_train.dat")

print(np.median([PLA(data) for _ in range(1000)]))  # Output: 0.4310778400000001
