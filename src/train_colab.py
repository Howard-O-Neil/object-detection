import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import mylib.io_utils.VOC_2012 as io_voc_2012

import model.model_imagenet.bbox_predict as bbp
import model.model_imagenet.img_cnn_core as icnn

x_DF = pd.read_csv(f"""{os.getenv("dataset2")}/bbox_X_MERGED.csv""")
y_DF = pd.read_csv(f"""{os.getenv("dataset2")}/bbox_Y_MERGED.csv""")
imgs_DF = pd.read_csv(f"""{os.getenv("dataset2")}/image_ID_MERGED.csv""")

# remove pandas column np.array
x_DF = np.array(x_DF.values.tolist()).astype(np.float32)[:, 1:]
y_DF = np.array(y_DF.values.tolist()).astype(np.float32)[:, 1:]
imgs_DF = np.array(imgs_DF.values.tolist())[:, 1]

trainval_list = np.array(io_voc_2012.get_imgs_dataset("trainval"))

split_index = int(trainval_list.shape[0] * 0.8)
train_list = trainval_list[0:split_index]
val_list = trainval_list[split_index:trainval_list.shape[0]]

image_cnn_core = icnn.CNN_core().get_model()

bbox_model = bbp.Bbox_predict(version=1)
bbox_model.assign_cnn_model(image_cnn_core)
bbox_model.assign_img_list_train(train_list)
bbox_model.assign_img_list_validation(val_list)
bbox_model.assign_bbox_dataset(x_DF, y_DF, imgs_DF)

try:
    bbox_model.train_loop(transfer=False)

# Google Colab support throw KeyboardInterrupt error when interupt kernel
except KeyboardInterrupt:
    bbox_model.save_model()

    print("===== SAVE MODEL SUCCESSFULLY =====")
    exit(0)
else:
    print("UNKNOWN ERROR!!!")