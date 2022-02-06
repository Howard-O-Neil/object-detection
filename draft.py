import numpy as np

a = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4]
])

b = np.array([
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4],
])

# throw error
res = np.concatenate((np.expand_dims(a, 0), np.expand_dims(b, 0)), axis=0)

print(res)

