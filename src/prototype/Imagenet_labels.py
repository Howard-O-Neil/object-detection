import sys
sys.path.append("../")

import os
os.environ["imagenet_dataset"] = "/home/howard/project/object-detection/data/imagenet"

from model.model_imagenet.imagenet_labels import Imagenet_labels

lb = Imagenet_labels()

start_idx = 900
for label in lb.labels[start_idx:start_idx+100].tolist():
    print(f"{label[0]} {label[1]}")
