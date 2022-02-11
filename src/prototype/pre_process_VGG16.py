import tensorflow as tf
import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import tensorflow.keras.applications.vgg16 as pretrain_vgg
import cv2

img_dir = "/home/howard/project/object-detection/images/test/traffic_1.jpeg"
img_tensor = tf.image.resize(
    tf.cast(
        tf.convert_to_tensor(np.asarray(PIL.Image.open(img_dir))),
        tf.dtypes.float32,
    ),
    [500, 500],
    method="bilinear",
    preserve_aspect_ratio=True,
).numpy()


preprocess = pretrain_vgg.preprocess_input(img_tensor)

plt.imshow(img_tensor)
plt.savefig("/home/howard/project/object-detection/images/plot/test_vgg_preprocess.png")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

ss.setBaseImage(preprocess)
ss.switchToSelectiveSearchFast()  # reduce number of boxes

boxes = ss.process()

fig, ax = plt.subplots(1)

plt.imshow(preprocess)
for i, rect in enumerate(boxes):
    r = mpatches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r', facecolor="none")
    ax.add_patch(r)

plt.savefig(f"/home/howard/project/object-detection/images/plot/test_vgg_preprocess_selective_search.png")
