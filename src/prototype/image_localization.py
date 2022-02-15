import sys
sys.path.append("..")

import os

MODEL_ID = 1
os.environ["dataset"] = "/home/howard/dataset/VOCdevkit/VOC2012"
os.environ["dataset2"] = "/home/howard/project/object-detection/data/pascal_voc2012"
os.environ["model_path"] = f"/home/howard/project/object-detection/meta/r_cnn/model{MODEL_ID}"
os.environ["log_path"] = "/home/howard/project/object-detection/log/train.log"
os.environ["img_path"] = "/home/howard/project/object-detection/images/test"
os.environ["plot"] = "/home/howard/project/object-detection/images/plot"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import mylib.io_utils.VOC_2012 as io_voc_2012

import model.model_imagenet.bbox_predict as bbp
import model.model_imagenet.img_cnn_core as icnn
import model.model_imagenet.VGG_16 as vgg
import cv2

vgg_16 = vgg.Pretrain_VGG16()
image_cnn_core = icnn.CNN_core().get_model()

bbox_model = bbp.Bbox_predict(version=1)
bbox_model.assign_cnn_model(image_cnn_core)
bbox_model.load_model()

img_path = f"""{os.getenv("img_path")}/traffic_1.jpeg"""

image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

scale_img = tf.image.resize(
    image, [500, 500], method="bilinear", preserve_aspect_ratio=True
).numpy()

print(scale_img.shape)

plt.imshow(scale_img / 255.)
plt.savefig(f"""{os.getenv("plot")}/test_ss.png""")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

ss.setBaseImage(scale_img)
ss.switchToSelectiveSearchFast() # reduce number of boxes

boxes = ss.process()

print(boxes)

fig, ax = plt.subplots(1)

plt.imshow(scale_img / 255.)
for i, rect in enumerate(boxes):
    r = mpatches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r', facecolor="none")
    ax.add_patch(r)

plt.savefig(f"""{os.getenv("plot")}/test_ss_1.png""")

print(boxes.shape)
scores = vgg_16.predict(img_path, boxes[0:100])
print(scores)
print(scores.shape)

fig, ax = plt.subplots(1)

new_boxes = bbox_model.predict_bboxs(img_path, boxes)
plt.imshow(scale_img / 255.)
for i, rect in enumerate(new_boxes):
    r = mpatches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r', facecolor="none")
    ax.add_patch(r)

plt.savefig(f"""{os.getenv("plot")}/test_ss_2.png""")
