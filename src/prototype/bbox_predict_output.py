import sys
sys.path.append("..")

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import model.model_imagenet.bbox_predict as bbp

_lamda = 0.5

model = keras.Sequential(
    [
        bbp.Kaming_he_dense(2048, _lamda),
        bbp.Kaming_he_dense(2048, _lamda),
        bbp.Kaming_he_dense(1024, _lamda),
        bbp.Kaming_he_dense(1024, _lamda),
        bbp.Kaming_he_dense(4, _lamda, activation=False),
    ]
)
model.build((None, 4096)) # Fake input shape

inputs = np.random.normal(3, 2.5, size=(100, 4096))
print(model(inputs).shape)
print(model.layers[-1])