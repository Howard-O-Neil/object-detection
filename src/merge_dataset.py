from functools import cmp_to_key
import os
from cv2 import sort

os.environ["dataset"] = "/home/howard/project/object-detection/data/pascal_voc2012"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mylib.io_utils.VOC_2012 as io_voc_2012
import mylib.bbox_utils as bbu

import pandas as pd

list_x = []
list_y = []
list_imgs = []

for file in os.listdir(os.getenv("dataset")):
    if (
        file.startswith("bbox_X_")
        or file.startswith("bbox_Y_")
        or file.startswith("image_ID_")
    ):
        continue
    if file.startswith("bbox_X"):
        list_x.append(file)
    elif file.startswith("bbox_Y"):
        list_y.append(file)
    else:
        list_imgs.append(file)

list_x_DF = []
for k in range(0, len(list_x)):
    df = pd.read_csv(f"""{os.getenv("dataset")}/bbox_X{k}.csv""")
    df.drop(["Unnamed: 0"], axis=1, inplace=True)

    list_x_DF.append(df)
x_DF = pd.concat(list_x_DF, axis=0)
x_DF.reset_index(drop=True, inplace=True)

list_y_DF = []
for k in range(0, len(list_y)):
    df = pd.read_csv(f"""{os.getenv("dataset")}/bbox_Y{k}.csv""")
    df.drop(["Unnamed: 0"], axis=1, inplace=True)

    list_y_DF.append(df)
y_DF = pd.concat(list_y_DF, axis=0)
y_DF.reset_index(drop=True, inplace=True)

list_imgs_DF = []
for k in range(0, len(list_imgs)):
    df = pd.read_csv(f"""{os.getenv("dataset")}/image_ID{k}.csv""")
    df.drop(["Unnamed: 0"], axis=1, inplace=True)

    list_imgs_DF.append(df)

imgs_DF = pd.concat(list_imgs_DF, axis=0)
imgs_DF.reset_index(drop=True, inplace=True)

imgs_DF.to_csv(f"../data/pascal_voc2012/image_ID_MERGED.csv")
x_DF.to_csv(f"../data/pascal_voc2012/bbox_X_MERGED.csv")
y_DF.to_csv(f"../data/pascal_voc2012/bbox_Y_MERGED.csv")