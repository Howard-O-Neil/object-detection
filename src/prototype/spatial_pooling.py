import math
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import PIL

img_dir = "/home/howard/project/object-detection/images/test/animal_2.jpg"

perfect_size = 9 # (7*7)

img = tf.expand_dims(
    tf.cast(tf.convert_to_tensor(np.asarray(PIL.Image.open(img_dir))), tf.float32),
    axis=0,
)
print(img.shape)

img_w = img[0].shape[1]
img_h = img[0].shape[0]

if img_h >= img_w:
    new_w = perfect_size
    new_h = math.ceil(img_h * (new_w / img_w))
else:
    new_h = perfect_size
    new_w = math.ceil(img_w * (new_h / img_h))

img = tf.image.resize(img, [new_h, new_w], preserve_aspect_ratio=True)

img_w = img[0].shape[1]
img_h = img[0].shape[0]

# print((new_h, new_w))

bin_w = 7
bin_h = 7

_pool_size = (math.floor(img_h / bin_h), math.floor(img_w / bin_w))
_pool_stride = (math.floor(img_h / bin_h), math.floor(img_w / bin_w))

print(f"Pool size   : {_pool_size}")
print(f"Pool stride : {_pool_stride}")
max_pool = keras.layers.MaxPooling2D(pool_size=_pool_size, strides=_pool_stride)

maxpool_img = max_pool(img)
print(img.shape)
print(maxpool_img.shape)

# plot canvas (DCI 2K) = (256 x 8, 135 x 8)
fig = plt.figure(figsize=(256., 135.), dpi=8)

grid_row = 1
grid_col = 2

grid = fig.subplots(grid_row, grid_col)

fig.get_axes()[0].imshow(img[0].numpy() / 255.)
fig.get_axes()[1].imshow(maxpool_img[0].numpy() / 255.)

plt.savefig("/home/howard/project/object-detection/images/plot/test_maxpool_img.png")
