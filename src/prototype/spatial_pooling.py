import math
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import PIL

img_dir = "/home/howard/project/object-detection/images/test/animal_2.jpg"

perfect_size = 45  # 7 + 1

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

bin_w = 40
bin_h = 40

window_h = math.ceil(img_h / bin_h)
window_w = math.ceil(img_w / bin_w)

pad_h = ((window_h * bin_h) - img_h + 1) / 2
pad_w = ((window_w * bin_w) - img_w + 1) / 2

_pool_size_stride = (window_h, window_w)

pad_img = keras.layers.ZeroPadding2D(
    padding=(
        (math.ceil(pad_h), math.floor(pad_h)),
        (math.ceil(pad_w), math.floor(pad_w)),
    )
)(img)

max_pool = keras.layers.MaxPooling2D(
    pool_size=_pool_size_stride, strides=_pool_size_stride
)

maxpool_img = max_pool(pad_img)
print(img.shape)
print(maxpool_img.shape)

# plot canvas (DCI 2K) = (256 x 8, 135 x 8)
fig = plt.figure(figsize=(256.0, 135.0), dpi=8)

grid_row = 1
grid_col = 3

grid = fig.subplots(grid_row, grid_col)

fig.get_axes()[0].imshow(img[0].numpy() / 255.0)
fig.get_axes()[1].imshow(pad_img[0].numpy() / 255.0)
fig.get_axes()[2].imshow(maxpool_img[0].numpy() / 255.0)

plt.savefig("/home/howard/project/object-detection/images/plot/test_maxpool_img.png")
