from functools import cmp_to_key
import os
from cv2 import sort

os.environ["dataset_version"] = "/home/howard/project/object-detection/data/pascal_voc2012/dataset_v2"

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

def compare_path(item1: str, item2: str):
    return_value = 1

    if len(item1) > len(item2):
        return_value = 1
    elif len(item1) < len(item2):
        return_value = -1
    else:
        if item1 > item2:
            return_value = 1
        elif item1 < item2:
            return_value = -1
        else:
            return_value = 0

    return return_value

def merge_DF(list_dirs):
    list_DF = []
    
    for dir in list_dirs:
        df = pd.read_csv(f"""{os.getenv("dataset_version")}/{dir}""")
        df.drop(["Unnamed: 0"], axis=1, inplace=True)

        list_DF.append(df)
    
    DF = pd.concat(list_DF, axis=0)
    DF.reset_index(drop=True, inplace=True)

    return DF

for file in os.listdir(os.getenv("dataset_version")):
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

list_x.sort(key=cmp_to_key(compare_path))
list_y.sort(key=cmp_to_key(compare_path))
list_imgs.sort(key=cmp_to_key(compare_path))

x_DF = merge_DF(list_x)
y_DF = merge_DF(list_y)
imgs_DF = merge_DF(list_imgs)

x_DF.to_csv(f"""{os.getenv("dataset_version")}/bbox_X_MERGED.csv""")
y_DF.to_csv(f"""{os.getenv("dataset_version")}/bbox_Y_MERGED.csv""")
imgs_DF.to_csv(f"""{os.getenv("dataset_version")}/image_ID_MERGED.csv""")
