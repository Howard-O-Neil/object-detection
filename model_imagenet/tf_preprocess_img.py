import tensorflow as tf

def tf_preprocess_images(data, is_train):
    data_f = tf.cast(data, tf.float32)
    data_shape = tf.shape(data_f)
    data_size = tf.squeeze(tf.slice(data_shape, [0], [1]))

    # 3072 = 3 * 32 * 32
    # origional image size = (32 x 32)
    # each pixel contain a list of (R, G, B)
    imgs = tf.reshape(data_f, [data_size, 3, 32, 32])

    # tranpose, allow displaying with matplotlib
    proper_imgs = tf.transpose(imgs, [0, 2, 3, 1])

    imgs_norm = tf.image.resize(
        tf.divide(proper_imgs, 255.0),  # normalize RGBs
        [224, 224], method="bicubic"
    )

    gray_scale = tf.reshape(
        tf.reduce_mean(imgs_norm, axis=3),
        [-1, 224, 224, 1],
    )

    if is_train:
        gray_scale = imgs_norm

    return gray_scale