import numpy as np
import pandas as pd


def add(x, y):
    # Print pair value
    print(f"{x} --- {y}")
    return x + y

custom_add = np.frompyfunc(add, 2, 1)

# # This one is normal

a = np.array(
    [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    ]
)

b = np.array(
    [
        [4, 4, 4, 4],
        [5, 5, 5, 5],
    ]
)

print(custom_add(a, b))

# === OUTPUT ===
# 1 --- 4
# 1 --- 4
# 1 --- 4
# 1 --- 4
# 2 --- 5
# 2 --- 5
# 2 --- 5
# 2 --- 5

# [[5 5 5 5]
#  [7 7 7 7]]

# This one still normal

a = np.ascontiguousarray(np.array([[1, 2], [1, 2], [1, 2], [1, 2]]).T)
b = np.ascontiguousarray(np.array([[4, 5], [4, 5], [4, 5], [4, 5]]).T)


print(custom_add(a, b))

# === OUTPUT ===
# 1 --- 4
# 1 --- 4
# 1 --- 4
# 1 --- 4
# 2 --- 5
# 2 --- 5
# 2 --- 5
# 2 --- 5

# [[5 5 5 5]
#  [7 7 7 7]]
