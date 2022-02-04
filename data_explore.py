import os
os.environ["dataset"] = "/home/howard/dataset/VOCdevkit/VOC2012"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mylib.image_utils.VOC_2012 as img_voc_2012
import mylib.io_utils.VOC_2012 as io_voc_2012

train_list = io_voc_2012.get_imgs_dataset("train")

[img_classes, img_bboxs] = io_voc_2012.get_bbox_annotations(train_list[0:2])

# print("==============")
# for x in img_bboxs:
#     print(x)

# for x in img_classes:
#     print(x)

[imgs, imgs_change_ratio] = io_voc_2012.scale_imgs(train_list[0:2])

img_bboxs = io_voc_2012.scale_annotations(img_bboxs, imgs_change_ratio)

# print("==============")
# for x in img_bboxs:
#     print(x)

fig, ax = plt.subplots(len(img_bboxs), 1)

for i in range(0, len(img_bboxs)):
    ax[i].imshow(imgs[i])

    rects = img_bboxs[i]

    for rect in rects:
        r = mpatches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r', facecolor="none")
        ax[i].add_patch(r)

plt.savefig("images/plot/test_bbox.png")
