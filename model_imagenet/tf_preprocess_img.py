import tensorflow as tf

def tf_preprocess_images(data, is_train):
    return tf.image.resize(
        data,  # normalize RGBs
        [224, 224], method="bicubic"
    )