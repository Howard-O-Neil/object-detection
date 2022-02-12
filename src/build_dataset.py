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

# load all dataset cause crash ram
batch_size = 50

full_batch = (len(trainval_list) // batch_size) + 1
if len(trainval_list) % batch_size == 0:
    full_batch -= 1

data = np.array([])
for k in range(full_batch):
    offset = k * batch_size

    [img_classes, img_bboxs] = io_voc_2012.get_bbox_annotations(trainval_list[offset:offset + batch_size])
    [imgs, imgs_change_ratio] = io_voc_2012.transform_imgs(trainval_list[offset:offset + batch_size])
    img_bboxs = io_voc_2012.scale_annotations(img_bboxs, imgs_change_ratio)

    subdata = np.array([])
    imgs_ids = np.array([])
    for i in range(len(img_bboxs)):
        ss_res = bbu.selective_search(imgs[i])
        pairs = bbu.pair_bboxs_max(ss_res, img_bboxs[i], filter_rate=0.4)

        if pairs.shape[0] <= 0:
            continue

        if len(subdata.shape) <= 1:
            subdata = pairs
        else: subdata = np.concatenate((subdata, pairs), axis=0)
        
        print(trainval_list[offset + i])
        for u in range(pairs.shape[0]):
            imgs_ids = np.append(imgs_ids, trainval_list[offset + i])

    if subdata.shape[0] <= 0:
        continue

    DF_imgids = pd.DataFrame(imgs_ids)
    DF_imgids.to_csv(f"""{os.getenv("dataset2")}/image_ID{k}.csv""")

    DF_inputx = pd.DataFrame(subdata[:, 0])
    DF_inputy = pd.DataFrame(subdata[:, 1])
    DF_inputx.to_csv(f"""{os.getenv("dataset2")}/bbox_X{k}.csv""")
    DF_inputy.to_csv(f"""{os.getenv("dataset2")}/bbox_Y{k}.csv""")

    print(f"===== DONE BATCH {k} =====")