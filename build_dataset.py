import os
os.environ["dataset"] = "/home/howard/dataset/VOCdevkit/VOC2012"

import tensorflow as tf
import numpy as np

import mylib.io_utils.VOC_2012 as io_voc_2012
import mylib.bbox_utils as bbu

trainval_list = io_voc_2012.get_imgs_dataset("trainval")

# batch_size = len(trainval_list) # construct dataset from full list
batch_size = 2

[img_classes, img_bboxs] = io_voc_2012.get_bbox_annotations(trainval_list[0:batch_size])
[imgs, imgs_change_ratio] = io_voc_2012.scale_imgs(trainval_list[0:batch_size])
img_bboxs = io_voc_2012.scale_annotations(img_bboxs, imgs_change_ratio)

ss_res = bbu.selective_search(imgs[0])

print(bbu.pair_bboxs_max(ss_res, img_bboxs[0]))
# print(bbu.pair_bboxs_overlapse(ss_res, img_bboxs[0]))
# # print(ss_res)