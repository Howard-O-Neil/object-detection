import tensorflow as tf
import numpy as np
import PIL
import tensorflow.keras.applications.vgg16 as pretrain_vgg


def extract_bbox(img, bbox):
    x = np.int32(bbox[0])
    y = np.int32(bbox[1])
    w = np.int32(bbox[2])
    h = np.int32(bbox[3])

    bbox_img = img[y : y + h, x : x + w]
    # img_tensor = tf.divide(tf.convert_to_tensor(bbox_img), 255.0)
    img_tensor = tf.convert_to_tensor(bbox_img)

    return tf.image.resize(
        img_tensor, [224, 224], method="bilinear", preserve_aspect_ratio=False
    ).numpy()


def imagenet_load(img_dirs):
    res = []

    for img_dir in img_dirs:
        img_tensor = tf.cast(
            tf.convert_to_tensor(np.asarray(PIL.Image.open(img_dir))),
            tf.dtypes.float32,
        )
        scale_img = tf.image.resize(
            pretrain_vgg.preprocess_input(img_tensor),
            [500, 500],
            method="bilinear",
            preserve_aspect_ratio=True,
        ).numpy()

        res.append(scale_img)

    return res


def extract_bboxs(img, bboxs):
    if len(bboxs.shape) <= 1:
        bboxs = np.expand_dims(bboxs, axis=0)

    res = np.array([])
    for i in range(bboxs.shape[0]):
        bbox = bboxs[i]
        x = np.int32(bbox[0])
        y = np.int32(bbox[1])
        w = np.int32(bbox[2])
        h = np.int32(bbox[3])

        bbox_img = img[y : y + h, x : x + w]
        # img_tensor = tf.divide(tf.convert_to_tensor(bbox_img), 255.0)
        img_tensor = tf.convert_to_tensor(bbox_img)

        if len(res.shape) <= 1:
            res = np.expand_dims(
                tf.image.resize(
                    img_tensor,
                    [224, 224],
                    method="bilinear",
                    preserve_aspect_ratio=False,
                ).numpy(),
                axis=0,
            )
        else:
            res = np.concatenate(
                (
                    res,
                    np.expand_dims(
                        tf.image.resize(
                            img_tensor,
                            [224, 224],
                            method="bilinear",
                            preserve_aspect_ratio=False,
                        ).numpy(),
                        axis=0,
                    ),
                ),
                axis=0,
            )
    return res
