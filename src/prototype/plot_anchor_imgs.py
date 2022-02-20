from generate_anchors import generate_anchors as gen_anchs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import PIL

def rescale_img(img_dir):
    perfect_size = 512

    img = tf.expand_dims(
        tf.cast(tf.convert_to_tensor(np.asarray(PIL.Image.open(img_dir))), tf.float32),
        axis=0,
    )

    img_w = img[0].shape[1]
    img_h = img[0].shape[0]

    if img_h >= img_w:
        new_w = perfect_size
        new_h = math.ceil(img_h * (new_w / img_w))
    else:
        new_h = perfect_size
        new_w = math.ceil(img_w * (new_h / img_h))

    return tf.image.resize(img, [new_h, new_w], preserve_aspect_ratio=False)

img_dir = "/home/howard/project/object-detection/images/test/animal_2.jpg"
img = rescale_img(img_dir)[0]

print(f"Image width  : {img.shape[1]}")
print(f"Image height : {img.shape[0]}")

list_feat_stride = [8, 16, 32]

def cal_anchors(_feat_stride):

    input = gen_anchs()

    _conv_scale = 32

    feature_w = img.shape[1] / _conv_scale
    feature_h = img.shape[0] / _conv_scale

    print(f"Feature width  : {feature_w}")
    print(f"Feature height : {feature_h}")

    shift_x = np.arange(0, feature_w) * _feat_stride
    shift_y = np.arange(0, feature_h) * _feat_stride

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

    all_anchors = np.add(
        input.reshape((1, A, 4)),
        shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    )
    all_anchors = all_anchors.reshape((K * A, 4))

    max_distance_w_anchor = np.max(
        np.max(all_anchors[:, [0, 2]], axis=1)
    )
    max_distance_h_anchor = np.max(
        np.max(all_anchors[:, [1, 3]], axis=1)
    )

    print(max_distance_w_anchor)
    print(max_distance_h_anchor)

    return all_anchors

from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as mpatches

# plot canvas (DCI 2K) = (256 x 8, 135 x 8)
fig = plt.figure(figsize=(256., 135.), dpi=8) 

grid = fig.subplots(2, 2)

print("==========================")

for i, ax in enumerate(fig.get_axes()):
    if i == 3: break

    all_anchors = cal_anchors(list_feat_stride[i])

    ax.set_axis_off()

    ax.imshow(img / 255.)

    anchor_plt = np.vstack([
        np.mean(all_anchors[:, [0, 2]], axis=1),
        np.mean(all_anchors[:, [1, 3]], axis=1)
    ]).transpose()
    
    rects = np.vstack([
            all_anchors[:, 0],
            all_anchors[:, 1],
            all_anchors[:, 2] - all_anchors[:, 0], 
            all_anchors[:, 3] - all_anchors[:, 1],
        ]).transpose()

    for k in range(all_anchors.shape[0]):
        reg = mpatches.Rectangle((anchor_plt[k][0], anchor_plt[k][1]), 5, 5, linewidth=10, edgecolor='g', facecolor="none")
        # reg = mpatches.Rectangle((rects[k][0], rects[k][1]), rects[k][2], rects[k][3], linewidth=10, edgecolor='g', facecolor="none")
        
        ax.add_patch(reg)

plt.savefig("/home/howard/project/object-detection/images/plot/test_anchor_generate.png")
