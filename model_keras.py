from model_imagenet.model import predict_model as pm

layer = pm.get_layer('block5_conv3').output
print(layer)
# print(pm.summary())