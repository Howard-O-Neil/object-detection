import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import PIL

real_img_dir = "/home/howard/project/object-detection/images/test/animal_1.jpg"

img = tf.image.resize(
        tf.expand_dims(
            tf.cast(
                tf.convert_to_tensor(np.asarray(PIL.Image.open(real_img_dir))), tf.float32
            ),
            axis=0,
        ), [224, 224]
    )

mobile_net_model = keras.applications.MobileNetV2(
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_shape=None,
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

print(np.max(mobile_net_model(img).numpy(), axis=1)) 
# Result more than 0.9, model really confident

img = tf.expand_dims(tf.zeros([224, 224, 3]), axis=0)

print(np.max(mobile_net_model(img).numpy(), axis=1))
# Result below 0.1, model don't extract any features at all
# + Padding with zeros may work in preserving image size without affecting to model accuracy
# + Max pool may work in preserving features