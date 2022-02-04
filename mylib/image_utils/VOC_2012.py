import numpy as np
import tensorflow as tf

def convert_to_bbox(str_arr, img_size):
    x = np.float32(str_arr[0]) * img_size[0]
    y = np.float32(str_arr[1]) * img_size[1]
    w = np.float32(str_arr[2]) * img_size[0]
    h = np.float32(str_arr[3]) * img_size[1]

    return np.array([x, y, w, h])

def get_bbox_infos(datapoint, img_size):
    return np.array([
        [
            convert_to_bbox(datapoint["ground_truth"]["detections"][detect_idx]["bounding_box"], img_size),
            datapoint["ground_truth"]["detections"][detect_idx]["label"]
        ]
        for detect_idx in range(len(datapoint["ground_truth"]["detections"]))
    ])

def rescale_img(datapoint):
    path = datapoint["filepath"]
    
    img_byte = tf.io.read_file(path)
    img_tensor = tf.io.decode_image(img_byte, channels=3, dtype=tf.dtypes.float32)
    img_resize = tf.image.resize(img_tensor, [500, 500], preserve_aspect_ratio=True)

    return img_resize

# def calculate_iou(bbox_1, bbox_2):
