# THIS CAN BE CODE ON NOTEBOOK

import pickle
import numpy
import numpy as np

import model_cifa10.file_utils as file_utils

# test local only
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_ID = 1

LABEL_PATH = "data/imagenet/LOC_synset_mapping.txt"
labels = numpy.array([["", ""]])

f = os.open(LABEL_PATH, os.O_RDONLY)
res = os.read(f, os.path.getsize(LABEL_PATH)).decode("utf-8")

for label_str in res.split("\n")[0:-1]:
    arr = numpy.expand_dims(numpy.array(label_str.split(" ", 1)), 0)
    labels = np.concatenate((labels, arr), axis=0)

labels = labels[1:, :]


# t_train_size = 100
# t_test_size = 32

_, data, _ = file_utils.read_data(\
    "data/cifa10", is_train=True)

# data = data[5000:5000+t_train_size]
# labels = labels[5000:5000+t_train_size]

_, t_data, _ = file_utils.read_data(\
    "data/cifa10", is_train=False)

# not required training
# required download weights from google apis
from model_imagenet.model import predict_model
import model_imagenet.tf_preprocess_img as tf_preproc

def compute_labels(data, labels):
    predict_res = predict_model.predict_on_batch(tf_preproc.tf_preprocess_images(data[5:5+16], True))
    return labels[numpy.argmax(predict_res, 1)]

def compute_probs(data):
    predict_res = predict_model.predict_on_batch(tf_preproc.tf_preprocess_images(data[5:5+16], True))
    return numpy.amax(predict_res, 1)
