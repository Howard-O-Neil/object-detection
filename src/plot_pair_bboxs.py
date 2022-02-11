import os
os.environ["dataset"] = "/home/howard/dataset/VOCdevkit/VOC2012"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mylib.io_utils.VOC_2012 as io_voc_2012
import mylib.bbox_utils as bbu


# batch_size = len(trainval_list) # construct dataset from full list
batch_size = 25                    
start_idx = 30

trainval_list = np.array(io_voc_2012.get_imgs_dataset("trainval")[start_idx:start_idx+batch_size])

[img_classes, img_bboxs] = io_voc_2012.get_bbox_annotations(trainval_list)
[imgs, imgs_change_ratio] = io_voc_2012.transform_imgs(trainval_list)
img_bboxs = io_voc_2012.scale_annotations(img_bboxs, imgs_change_ratio)

# BELOW CODE IS PRETTY SLOW

from mpl_toolkits.axes_grid1 import ImageGrid

# plot canvas (DCI 2K) = (256 x 8, 135 x 8)
fig = plt.figure(figsize=(256., 135.), dpi=8) 
fig.tight_layout()

grid_row = int(trainval_list.shape[0] / 5)
grid_col = int(trainval_list.shape[0] / grid_row)
# grid = ImageGrid(fig, 111,
#                  nrows_ncols=(grid_row, grid_col),  # creates 2x2 grid of axes
#                  axes_pad=0.01,  # pad between axes in inch.
#                  )
grid = fig.subplots(grid_row, grid_col)

print("==========================")
for i, ax in enumerate(fig.get_axes()):
    ss_res = bbu.selective_search(imgs[i])
    pairs = bbu.pair_bboxs_max(ss_res, img_bboxs[i])

    ax.set_axis_off()
    ax.autoscale_view('tight')
    # display image
    #       float   [0 ... 1]
    #       integer [0 ... 255]
    ax.imshow(imgs[i])

    rects_1 = pairs[:, 0]
    rects_2 = pairs[:, 1]

    for k in range(rects_1.shape[0]):
        rect_reg = rects_1[k]
        reg = mpatches.Rectangle((rect_reg[0], rect_reg[1]), rect_reg[2], rect_reg[3], linewidth=5, edgecolor='g', facecolor="none")

        rect_gt = rects_2[k]
        gt = mpatches.Rectangle((rect_gt[0], rect_gt[1]), rect_gt[2], rect_gt[3], linewidth=5, edgecolor='r', facecolor="none")

        ax.add_patch(gt)
        ax.add_patch(reg)

plt.savefig("../images/plot/test_pair_regions.png")