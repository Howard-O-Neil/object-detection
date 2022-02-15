import os
import numpy as np

class Imagenet_labels:
    def __init__(self):
        LABEL_PATH = os.path.join(os.getenv("imagenet_dataset"), "LOC_synset_mapping.txt")
        self.labels = np.array([["", ""]])

        f = os.open(LABEL_PATH, os.O_RDONLY)
        res = os.read(f, os.path.getsize(LABEL_PATH)).decode("utf-8")

        for label_str in res.split("\n")[0:-1]:
            arr = np.expand_dims(np.array(label_str.split(" ", 1)), 0)
            self.labels = np.concatenate((self.labels, arr), axis=0)

        self.labels = self.labels[1:, :]
