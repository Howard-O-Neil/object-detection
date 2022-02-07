from math import e
import tensorflow as tf

class Pretrain_VGG16:
    def __init__(self):
        self.VGG_16_model = tf.keras.applications.VGG16(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
    
    def get_model(self):
        return self.VGG_16_model