import numpy as np
import tensorflow as tf

a = np.array([
    [1, 2, 3, 4 ,5],
    [1, 2, 3, 4 ,5]
])

a[:, 0] *= 5
print(a)