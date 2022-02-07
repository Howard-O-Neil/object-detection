from tensorflow import keras
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define Sequential model with 3 layers
initializer = tf.keras.initializers.HeNormal()

model = keras.Sequential(
    [
        keras.layers.Dense(2, activation="relu", name="layer1", kernel_initializer=initializer, bias_initializer=initializer),
        keras.layers.Dense(3, activation="relu", name="layer2", kernel_initializer=initializer, bias_initializer=initializer),
        keras.layers.Dense(4, name="layer3", kernel_initializer=initializer, bias_initializer=initializer),
    ]
)
model.compile()
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)

# print(y.numpy())