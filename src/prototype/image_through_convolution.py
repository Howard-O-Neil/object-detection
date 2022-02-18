import math
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import PIL

img_dir = "/home/howard/project/object-detection/images/test/animal_4.jpg"

perfect_size = 512  # (7*7)

img = tf.expand_dims(
    tf.cast(tf.convert_to_tensor(np.asarray(PIL.Image.open(img_dir))), tf.float32),
    axis=0,
)

img = tf.image.resize(img, [10, 10])

conv = keras.layers.Convolution2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)

conv_img = conv(img)

print(img.shape)
print(conv_img.shape)
