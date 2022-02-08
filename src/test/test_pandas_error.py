import numpy as np
import pandas as pd


def add(x, y):
    # Print pair value
    print(f"{x} --- {y}")
    return x + y

custom_add = np.frompyfunc(add, 2, 1)

# # This one is normal

# a = np.array(
#     [
#         [1, 1, 1, 1],
#         [2, 2, 2, 2],
#     ]
# )

# b = np.array(
#     [
#         [4, 4, 4, 4],
#         [5, 5, 5, 5],
#     ]
# )

# print(custom_add(a, b))

# # === OUTPUT ===
# # 1 --- 4
# # 1 --- 4
# # 1 --- 4
# # 1 --- 4
# # 2 --- 5
# # 2 --- 5
# # 2 --- 5
# # 2 --- 5

# # [[5 5 5 5]
# #  [7 7 7 7]]


# pd.DataFrame(a).to_csv("test/a.csv")
# pd.DataFrame(b).to_csv("test/b.csv")

# This one has weird behavior

a = np.array([])
for i in range(4):
    u = np.array([1, 2])

    if a.shape[0] <= 0:
        a = np.expand_dims(u, 1)
    else: a = np.concatenate(
        (a, np.expand_dims(u, 1)), 1
    )

b = np.array([])
for i in range(4):
    u = np.array([4, 5])

    if b.shape[0] <= 0:
        b = np.expand_dims(u, 1)
    else: b = np.concatenate(
        (b, np.expand_dims(u, 1)), 1
    )

print(custom_add(a, b))
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

a = np.array([[1, 2], [1, 2], [1, 2], [1, 2]]).T
b = np.array([[4, 5], [4, 5], [4, 5], [4, 5]]).T


print(custom_add(a, b))
# 1 --- 4
# 2 --- 5
# 1 --- 4
# 2 --- 5
# 1 --- 4
# 2 --- 5
# 1 --- 4
# 2 --- 5
# [[5 5 5 5]
#  [7 7 7 7]]