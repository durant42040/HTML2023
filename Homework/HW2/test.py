import numpy as np

tau = 0
a = np.ones(20)*2
a *= np.random.choice([1, -1], size=len(a), p=[tau, 1-tau])
print(a)