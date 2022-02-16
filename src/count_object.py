import os
os.environ["dataset"] = "/home/howard/dataset/VOCdevkit/VOC2012"
os.environ["dataset2"] = "/home/howard/project/object-detection/data/pascal_voc2012/dataset_v2"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mylib.io_utils.VOC_2012 as io_voc_2012
import mylib.bbox_utils as bbu
import mylib.img_utils as imu
import pandas as pd

trainval_list = io_voc_2012.get_imgs_dataset("trainval")

[img_classes, img_bboxs] = io_voc_2012.get_bbox_annotations(trainval_list)

num_objects = 0

for bboxs in img_bboxs:
    num_objects += bboxs.shape[0]

print(num_objects)
