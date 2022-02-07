import numpy as np
import tensorflow as tf

a = np.array([
    [1, 2, 3, 4 ,5],
    [1, 2, 3, 4 ,5]
])

a[:, 0] *= 5
print(a)


a = np.array([])
a = np.append(a, 1)
a = np.append(a, 1)
a = np.append(a, 1)
a = np.append(a, 1)
a = np.append(a, 1)

print(a)