import tensorflow as tf
from tensorflow import keras
from model.model_imagenet import VGG_16

# input_1 = img_model.get_layer("input_1")

# block1_conv1 = img_model.get_layer("block1_conv1")
# block1_conv2 = img_model.get_layer("block1_conv2")
# block1_pool = img_model.get_layer("block1_pool")

# block2_conv1 = img_model.get_layer("block2_conv1")
# block2_conv2 = img_model.get_layer("block2_conv2")
# block2_pool = img_model.get_layer("block2_pool")

# block3_conv1 = img_model.get_layer("block3_conv1")
# block3_conv2 = img_model.get_layer("block3_conv2")
# block3_conv3 = img_model.get_layer("block3_conv3")
# block3_pool = img_model.get_layer("block3_pool")

# block4_conv1 = img_model.get_layer("block4_conv1")
# block4_conv2 = img_model.get_layer("block4_conv2")
# block4_conv3 = img_model.get_layer("block4_conv3")
# block4_pool = img_model.get_layer("block4_pool")

# block5_conv1 = img_model.get_layer("block5_conv1")
# block5_conv2 = img_model.get_layer("block5_conv2")
# block5_conv3 = img_model.get_layer("block5_conv3")
# block5_pool = img_model.get_layer("block5_pool")

# flatten_pool = img_model.get_layer("flatten")
# fc1 = img_model.get_layer("fc1")
# fc2 = img_model.get_layer("fc2")

class CNN_core(VGG_16.Pretrain_VGG16):
    def __init__(self):
        super(CNN_core, self).__init__()
        img_model = self.VGG_16_model
        self.img_cnn = keras.Sequential(
            [
                img_model.get_layer("block1_conv1"),
                img_model.get_layer("block1_conv2"),
                img_model.get_layer("block1_pool"),
                img_model.get_layer("block2_conv1"),
                img_model.get_layer("block2_conv2"),
                img_model.get_layer("block2_pool"),
                img_model.get_layer("block3_conv1"),
                img_model.get_layer("block3_conv2"),
                img_model.get_layer("block3_conv3"),
                img_model.get_layer("block3_pool"),
                img_model.get_layer("block4_conv1"),
                img_model.get_layer("block4_conv2"),
                img_model.get_layer("block4_conv3"),
                img_model.get_layer("block4_pool"),
                img_model.get_layer("block5_conv1"),
                img_model.get_layer("block5_conv2"),
                img_model.get_layer("block5_conv3"),
                img_model.get_layer("block5_pool"),
                img_model.get_layer("flatten"),
                # img_model.get_layer("fc1"),
                # img_model.get_layer("fc2")
            ]
        )

        for layer in self.img_cnn.layers:
            layer.trainable = False
        
        self.img_cnn.compile()
    
    def get_model(self):
        return self.img_cnn