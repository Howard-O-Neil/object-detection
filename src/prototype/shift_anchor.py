from generate_anchors import generate_anchors as gen_anchs
import numpy as np

input = gen_anchs()

width = 4
height = 5

_feat_stride = 16

shift_x = np.arange(0, width) * _feat_stride
shift_y = np.arange(0, height) * _feat_stride

# return index matrix
shift_x, shift_y = np.meshgrid(shift_x, shift_y)

# print(shift_x.ravel())
# print(shift_y.ravel())

# ravel = flatten array
shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                    shift_x.ravel(), shift_y.ravel())).transpose()

# 1 anchor -> 20 shifts
# 9 anchor -> 180 shifts ?
A = input.shape[0] # number of anchor
K = shifts.shape[0] # number of shifts

print(shifts)
print(shifts.reshape((1, K, 4)))
print(shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
all_anchors = (input.reshape((1, A, 4)) +
                shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
all_anchors = all_anchors.reshape((K * A, 4))


print(all_anchors.shape)