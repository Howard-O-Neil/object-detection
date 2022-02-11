from tensorflow import keras
import tensorflow as tf
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define Sequential model with 3 layers
# initializer = tf.keras.initializers.HeNormal()

initializer = tf.keras.initializers.Zeros()

model = keras.Sequential(
    [
        keras.layers.Dense(2, activation="relu", name="layer1", kernel_initializer=initializer, bias_initializer=initializer),
        keras.layers.Dense(3, activation="relu", name="layer2", kernel_initializer=initializer, bias_initializer=initializer),
        keras.layers.Dense(4, activation="relu", name="layer3", kernel_initializer=initializer, bias_initializer=initializer),
    ]
)
model.compile()

# This is not a training loop, variables change only apply gradient on optimizer
for i in range(0, 100):
    x = tf.ones([1, 3, 3, 3])
    y = model(x)

print(y.numpy())

print(model.summary())

dataset = tf.data.Dataset.range(3)
dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
dataset = dataset.repeat(2)

print(list(dataset.as_numpy_iterator()))
print(list(dataset.as_numpy_iterator()))
print(list(dataset.as_numpy_iterator()))