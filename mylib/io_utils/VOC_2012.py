import os
import xml.etree.cElementTree as ET
import numpy as np
import tensorflow as tf
import PIL

CATEGORIES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# mode = train | val | trainval
def get_imgs_dataset(mode):
    parent_dir = os.getenv("dataset")
    filename = os.path.join(parent_dir, f"ImageSets/Segmentation/{mode}.txt")

    with open(filename, 'r') as f:
        return f.read().splitlines()

def get_bbox_annotations(img_strs):
    img_bboxs = []
    img_classes = []

    for img_str in img_strs:

        parent_dir = os.getenv("dataset")
        filename = os.path.join(parent_dir, f"Annotations/{img_str}.xml")

        tree = ET.ElementTree(file=filename)
    
        bboxs = np.array([[0., 0., 0., 0.]])
        classes = np.array([0])
        for child in tree.iter("object"):
            class_id = np.int32(CATEGORIES.index(child.find("name").text.lower()))
            x_min = np.float32(child.find("bndbox").find("xmin").text)
            x_max = np.float32(child.find("bndbox").find("xmax").text)
            y_min = np.float32(child.find("bndbox").find("ymin").text)
            y_max = np.float32(child.find("bndbox").find("ymax").text)

            bboxs = np.concatenate((
                bboxs,
                np.array([[x_min, y_min, x_max - x_min, y_max - y_min]])
            ), axis=0)
            classes = np.concatenate((classes, [class_id]), axis=0)

        img_bboxs.append(bboxs[1:])
        img_classes.append(classes[1:])

    return [img_classes, img_bboxs]

def scale_imgs(img_strs):
    imgs = []
    imgs_change_ratio = []

    for img_str in img_strs:
        parent_dir = os.getenv("dataset")
        annotations_dir = os.path.join(parent_dir, f"Annotations/{img_str}.xml")
        image_dir = os.path.join(parent_dir, f"JPEGImages/{img_str}.jpg")

        print(image_dir)
        tree = ET.ElementTree(file=annotations_dir)

        original_w = np.float32(tree.find("size").find("width").text)
        original_h = np.float32(tree.find("size").find("height").text)

        img_tensor = tf.cast(
            tf.convert_to_tensor(np.asarray(PIL.Image.open(image_dir))),
            tf.dtypes.float32
        )
        img_tensor = tf.divide(img_tensor, 255.)
        scale_img = tf.image.resize(img_tensor, [500, 500], \
            method="bilinear", preserve_aspect_ratio=True).numpy()

        ratio_w = np.divide(scale_img.shape[1], original_w)
        ratio_h = np.divide(scale_img.shape[0], original_h)

        imgs.append(scale_img)
        imgs_change_ratio.append(np.array([ratio_w, ratio_h]))

    return [imgs, imgs_change_ratio]

def scale_annotations(bboxs, imgs_change_ratio):
    res = []
    for i, bbox in enumerate(bboxs):

        bbox[:, 0] *= imgs_change_ratio[i][1]
        bbox[:, 2] *= imgs_change_ratio[i][1]
        bbox[:, 1] *= imgs_change_ratio[i][0]
        bbox[:, 3] *= imgs_change_ratio[i][0]

        res.append(bbox)
    return res

