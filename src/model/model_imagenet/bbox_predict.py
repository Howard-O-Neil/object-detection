import tensorflow as tf
from tensorflow import keras

class Bbox_predict:
    def __init__(self):
        
        self.model = keras.Sequential(
            [
                keras.layers.Dense(2048, activation='relu'),
                keras.layers.Dense(2048, activation='relu'),
                keras.layers.Dense(1024, activation='relu'),
                keras.layers.Dense(1024, activation='relu'),
                keras.layers.Dense(4, activation='relu'),
            ]
        )

    # def construct_model