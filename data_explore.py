import os
os.environ["dataset"] = "/home/howard/dataset/VOCdevkit/VOC2012"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mylib.io_utils.VOC_2012 as io_voc_2012

batch_size = 25                    
start_idx = 30

train_list = np.array(io_voc_2012.get_imgs_dataset("trainval")[start_idx:start_idx+batch_size])

[img_classes, img_bboxs] = io_voc_2012.get_bbox_annotations(train_list)

# print("==============")
# for x in img_bboxs:
#     print(x)

# for x in img_classes:
#     print(x)

[imgs, imgs_change_ratio] = io_voc_2012.scale_imgs(train_list)

img_bboxs = io_voc_2012.scale_annotations(img_bboxs, imgs_change_ratio)

# BELOW CODE IS PRETTY SLOW

from mpl_toolkits.axes_grid1 import ImageGrid

# plot canvas (DCI 2K) = (256 x 8, 135 x 8)
fig = plt.figure(figsize=(256., 135.), dpi=8) 

grid_row = int(train_list.shape[0] / 5)
grid_col = int(train_list.shape[0] / grid_row)
grid = ImageGrid(fig, 111,
                 nrows_ncols=(grid_row, grid_col),  # creates 2x2 grid of axes
                 axes_pad=0.01,  # pad between axes in inch.
                 )

print("==========================")
for i, ax in enumerate(grid):
    ax.set_axis_off()
    ax.imshow(imgs[i])

    rects = img_bboxs[i]

    for rect in rects:
        r = mpatches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=5, edgecolor='r', facecolor="none")
        ax.add_patch(r)

plt.savefig("images/plot/test_bbox.png")
