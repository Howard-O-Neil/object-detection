import sys
sys.path.insert(0, "..")

import numpy as np
import tensorflow as tf
import pandas as pd
import mylib.io_utils.VOC_2012 as io_voc_2012
import mylib.bbox_utils as bbu

import os
MODEL_ID = 1
os.environ["dataset"] = "/home/howard/dataset/VOCdevkit/VOC2012"
os.environ["dataset2"] = "/home/howard/project/object-detection/data/pascal_voc2012"
os.environ["model_path"] = f"/home/howard/project/object-detection/meta/r_cnn/model{MODEL_ID}"
os.environ["log_path"] = "/home/howard/project/object-detection/log/train.log"

os.environ["train_batch_size"] = "32"


trainval_list = io_voc_2012.get_imgs_dataset("trainval")

x_DF = pd.read_csv(f"""{os.getenv("dataset2")}/bbox_X_MERGED.csv""")
y_DF = pd.read_csv(f"""{os.getenv("dataset2")}/bbox_Y_MERGED.csv""")
imgs_DF = pd.read_csv(f"""{os.getenv("dataset2")}/image_ID_MERGED.csv""")

x_DF = np.array(x_DF.values.tolist()).astype(np.float32)[:, 1:]
y_DF = np.array(y_DF.values.tolist()).astype(np.float32)[:, 1:]
imgs_DF = np.array(imgs_DF.values.tolist())[:, 1]

for img_dir in trainval_list:
    if x_DF[np.where(imgs_DF == img_dir, True, False)].shape[0] == 0:
        print(img_dir)