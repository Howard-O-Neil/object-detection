import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from mylib.io_utils import VOC_2012 as io_voc_2012
from mylib import bbox_utils as bbu
from mylib import img_utils as imu

from model.model_imagenet.VGG_16 import Pretrain_VGG16

import os
os.environ["dataset"] = "/home/howard/dataset/VOCdevkit/VOC2012"
os.environ["dataset2"] = "/home/howard/project/object-detection/data/pascal_voc2012"

# batch_size = len(trainval_list) # construct dataset from full list
batch_size = 25
start_idx = 30
trainval_list = np.array(io_voc_2012.get_imgs_dataset("trainval")[start_idx:start_idx+batch_size])

x_DF = pd.read_csv(f"""{os.getenv("dataset2")}/bbox_X_MERGED.csv""")
y_DF = pd.read_csv(f"""{os.getenv("dataset2")}/bbox_Y_MERGED.csv""")
imgs_DF = pd.read_csv(f"""{os.getenv("dataset2")}/image_ID_MERGED.csv""")

x_DF = np.array(x_DF.values.tolist()).astype(np.int32)[:, 1:]
y_DF = np.array(y_DF.values.tolist()).astype(np.int32)[:, 1:]
imgs_DF = np.array(imgs_DF.values.tolist())[:, 1]

img_id = trainval_list[0]
[imgs, _] = io_voc_2012.transform_imgs([img_id])

filter_ids = np.where(imgs_DF == img_id, True, False)

batch_y = y_DF[filter_ids]
extracted_bbox = imu.extract_bbox(imgs[0], batch_y[0])

from mpl_toolkits.axes_grid1 import ImageGrid

# plot canvas (DCI 2K) = (256 x 8, 135 x 8)
fig = plt.figure(figsize=(256., 135.), dpi=8) 
fig.tight_layout()

grid_row = 1
grid_col = 2
# grid = ImageGrid(fig, 111,
#                  nrows_ncols=(grid_row, grid_col),  # creates 2x2 grid of axes
#                  axes_pad=0.3,  # pad between axes in inch.
#                  )

grid = fig.subplots(grid_row, grid_col)
print("==========================")
for i, ax in enumerate(fig.get_axes()):    
    if i == 0:
        ax.set_axis_off()
        ax.imshow(imgs[0] / 255.)
    else:
        ax.set_axis_off()
        ax.imshow(extracted_bbox / 255.)

plt.savefig("../../images/plot/test_bbox_extraction.png")