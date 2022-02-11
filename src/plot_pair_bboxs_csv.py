from cmath import rect
import os

os.environ["dataset"] = "/home/howard/dataset/VOCdevkit/VOC2012"
os.environ["dataset2"] = "/home/howard/project/object-detection/data/pascal_voc2012"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import mylib.io_utils.VOC_2012 as io_voc_2012
import mylib.bbox_utils as bbu

# # Remove 1st col index
x_DF = pd.read_csv(f"""{os.getenv("dataset2")}/bbox_X_MERGED.csv""")
y_DF = pd.read_csv(f"""{os.getenv("dataset2")}/bbox_Y_MERGED.csv""")
imgs_DF = pd.read_csv(f"""{os.getenv("dataset2")}/image_ID_MERGED.csv""")

# batch_size = len(trainval_list) # construct dataset from full list
batch_size = 25
start_idx = 30
trainval_list = np.array(io_voc_2012.get_imgs_dataset("trainval")[start_idx:start_idx+batch_size])

x_DF = np.array(x_DF.values.tolist()).astype(np.float32)[:, 1:]
y_DF = np.array(y_DF.values.tolist()).astype(np.float32)[:, 1:]
imgs_DF = np.array(imgs_DF.values.tolist())[:, 1]

from mpl_toolkits.axes_grid1 import ImageGrid

# plot canvas (DCI 2K) = (256 x 8, 135 x 8)
fig = plt.figure(figsize=(256., 135.), dpi=8) 

grid_row = int(trainval_list.shape[0] / 5)
grid_col = int(trainval_list.shape[0] / grid_row)
# grid = ImageGrid(fig, 111,
#                  nrows_ncols=(grid_row, grid_col),  # creates 2x2 grid of axes
#                  axes_pad=0.01,  # pad between axes in inch.
#                  )
grid = fig.subplots(grid_row, grid_col)


print("==========================")
for i, ax in enumerate(fig.get_axes()):
    img_id = trainval_list[i]
    filter_ids = np.where(imgs_DF == img_id, True, False)
    [imgs, _] = io_voc_2012.scale_imgs([img_id])

    ax.set_axis_off()

    # display image
    #       float   [0 ... 1]
    #       integer [0 ... 255]
    ax.imshow(imgs[0] / 255.)

    rects_1 = x_DF[filter_ids]
    rects_2 = y_DF[filter_ids]

    for k in range(rects_1.shape[0]):
        rect_reg = rects_1[k]
        reg = mpatches.Rectangle((rect_reg[0], rect_reg[1]), rect_reg[2], rect_reg[3], linewidth=5, edgecolor='g', facecolor="none")

        rect_gt = rects_2[k]
        gt = mpatches.Rectangle((rect_gt[0], rect_gt[1]), rect_gt[2], rect_gt[3], linewidth=5, edgecolor='r', facecolor="none")

        ax.add_patch(gt)
        ax.add_patch(reg)

plt.savefig("../images/plot/test_pair_regions_csv.png")
