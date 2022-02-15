import sys
sys.path.append("..")

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model.model_imagenet.MOBILE_NET_V2 import Pretrain_mobilenet_v2
from model.model_imagenet.imagenet_labels import Imagenet_labels
import cv2

import os
os.environ["imagenet_dataset"] = "/home/howard/project/object-detection/data/imagenet"

lb = Imagenet_labels()
md = Pretrain_mobilenet_v2()


img_dir = "/home/howard/project/object-detection/images/test/animal_1.jpg"
img_tensor = tf.image.resize(
    tf.cast(
        tf.convert_to_tensor(np.asarray(PIL.Image.open(img_dir))),
        tf.dtypes.float32,
    ),
    [224, 224],
    method="bilinear",
    preserve_aspect_ratio=False,
).numpy()

prediction = md.predict_labels(img_tensor)
boxes_label = lb.labels[prediction[:, 0].astype(np.int32)]

print("===== PREDICTION RESULTS =====")
print(boxes_label)