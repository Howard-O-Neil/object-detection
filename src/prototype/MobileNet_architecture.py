import tensorflow as tf
import tensorflow.keras as keras

model = keras.applications.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
model.build((224, 224))
print(model.summary())
